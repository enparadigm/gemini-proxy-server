from __future__ import annotations

import array
import asyncio
import base64
import json
import os
import secrets
import time
import wave

from loguru import logger


def _maybe_extract_pcm_base64(obj):
    """
    Best-effort extraction of audio/pcm base64 payloads from arbitrary JSON shapes.

    Returns a list of base64 strings (may be empty).
    """
    out = []

    def walk(x):
        if isinstance(x, dict):
            mime = x.get("mime_type") or x.get("mimeType") or x.get("mime")
            # Common patterns:
            # - {"mime_type":"audio/pcm","data":"..."}
            # - {"inlineData":{"mimeType":"audio/pcm","data":"..."}}
            # - {"mimeType":"audio/pcm","data":"..."}
            if isinstance(mime, str) and "audio/pcm" in mime.lower():
                data = (
                    x.get("data") or x.get("audio") or x.get("pcm") or x.get("audioData")
                )
                if isinstance(data, str) and data:
                    out.append(data)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return out


class ConversationRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        record_dir: str,
        sample_rate: int,
        sample_width_bytes: int,
        mixdown: str,
        user_input_sample_rate: int,
        bot_input_sample_rate: int,
    ):
        self.enabled = enabled
        self.record_dir = record_dir

        self.sample_rate = sample_rate
        self.sample_width_bytes = sample_width_bytes
        self.mixdown = mixdown
        self.user_input_sample_rate = user_input_sample_rate
        self.bot_input_sample_rate = bot_input_sample_rate

        self.session_id = f"{int(time.time())}-{secrets.token_hex(4)}"
        self.base_dir = os.path.join(self.record_dir, self.session_id)

        self._lock = asyncio.Lock()
        self._start_ts = None

        self._user_pcm_path = os.path.join(self.base_dir, "user_timeline.pcm")
        self._bot_pcm_path = os.path.join(self.base_dir, "bot_timeline.pcm")
        self._events_path = os.path.join(self.base_dir, "events.jsonl")
        self._wav_path = os.path.join(self.base_dir, "conversation.wav")

        self._user_bytes_written = 0
        self._bot_bytes_written = 0

        self._user_f = None
        self._bot_f = None
        self._events_f = None

    async def open(self):
        if not self.enabled:
            return
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            self._user_f = open(self._user_pcm_path, "ab", buffering=0)
            self._bot_f = open(self._bot_pcm_path, "ab", buffering=0)
            self._events_f = open(self._events_path, "a", encoding="utf-8")
            logger.info("📝 Recording enabled. Session dir: {}", self.base_dir)
        except Exception as e:
            self.enabled = False
            logger.warning("⚠️ Recording disabled (failed to open files): {}", e)

    async def close(self):
        if not self.enabled:
            return
        async with self._lock:
            for f in (self._user_f, self._bot_f, self._events_f):
                try:
                    if f:
                        f.close()
                except Exception:
                    pass
            self._user_f = None
            self._bot_f = None
            self._events_f = None

    async def _write_event(self, event: dict):
        if not self.enabled or not self._events_f:
            return
        self._events_f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events_f.flush()

    def _pcm_silence(self, n_bytes: int) -> bytes:
        return b"\x00" * n_bytes

    def _resample_pcm16_mono(self, pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
        if self.sample_width_bytes != 2:
            return pcm_bytes
        if not pcm_bytes or src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
            return pcm_bytes
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
            if not pcm_bytes:
                return b""

        src = array.array("h")
        src.frombytes(pcm_bytes)
        src_len = len(src)
        if src_len == 0:
            return b""

        dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
        dst = array.array("h")
        for i in range(dst_len):
            pos = i * (src_rate / dst_rate)
            left = int(pos)
            if left >= src_len - 1:
                sample = src[src_len - 1]
            else:
                frac = pos - left
                sample = int(src[left] * (1.0 - frac) + src[left + 1] * frac)
            dst.append(sample)
        return dst.tobytes()

    async def _append_timeline(self, direction: str, ts: float, pcm_bytes: bytes):
        if not self.enabled:
            return
        if self.sample_width_bytes <= 0:
            return

        if self._start_ts is None:
            self._start_ts = ts

        target_offset_samples = int(max(0.0, ts - self._start_ts) * self.sample_rate)
        target_offset_bytes = target_offset_samples * self.sample_width_bytes

        async with self._lock:
            if direction == "client→api":
                f = self._user_f
                bytes_written = self._user_bytes_written
            else:
                f = self._bot_f
                bytes_written = self._bot_bytes_written

            if not f:
                return

            if target_offset_bytes > bytes_written:
                gap = target_offset_bytes - bytes_written
                f.write(self._pcm_silence(gap))
                bytes_written += gap

            f.write(pcm_bytes)
            bytes_written += len(pcm_bytes)

            if direction == "client→api":
                self._user_bytes_written = bytes_written
            else:
                self._bot_bytes_written = bytes_written

    async def observe(self, direction: str, msg, is_binary: bool):
        if not self.enabled:
            return

        ts = time.time()
        event = {
            "ts": ts,
            "direction": direction,
            "type": "binary" if is_binary else "text",
            "byte_len": (
                len(msg)
                if isinstance(msg, (bytes, bytearray))
                else (len(msg.encode("utf-8")) if isinstance(msg, str) else None)
            ),
        }

        if is_binary:
            event["note"] = "binary frame (not parsed)"
            await self._write_event(event)
            return

        if not isinstance(msg, str):
            event["note"] = f"unexpected msg type {type(msg)}"
            await self._write_event(event)
            return

        try:
            payload = json.loads(msg)
        except Exception as e:
            event["note"] = f"non-json text ({e})"
            event["text_prefix"] = msg[:2000]
            await self._write_event(event)
            return

        b64_chunks = _maybe_extract_pcm_base64(payload)
        event["pcm_b64_chunks"] = len(b64_chunks)
        await self._write_event(event)

        if not b64_chunks:
            return

        for i, b64 in enumerate(b64_chunks):
            try:
                pcm = base64.b64decode(b64, validate=False)
            except Exception as e:
                await self._write_event(
                    {
                        "ts": time.time(),
                        "direction": direction,
                        "type": "text",
                        "note": f"base64 decode failed chunk={i} ({e})",
                    }
                )
                continue

            if self.sample_width_bytes > 1 and (len(pcm) % self.sample_width_bytes) != 0:
                await self._write_event(
                    {
                        "ts": time.time(),
                        "direction": direction,
                        "type": "text",
                        "note": f"pcm length not aligned to sample width chunk={i}",
                        "pcm_len": len(pcm),
                        "sample_width_bytes": self.sample_width_bytes,
                    }
                )

            src_rate = (
                self.user_input_sample_rate
                if direction == "client→api"
                else self.bot_input_sample_rate
            )
            pcm_resampled = self._resample_pcm16_mono(
                pcm_bytes=pcm,
                src_rate=src_rate,
                dst_rate=self.sample_rate,
            )
            if src_rate != self.sample_rate:
                await self._write_event(
                    {
                        "ts": time.time(),
                        "direction": direction,
                        "type": "text",
                        "note": "resampled_pcm",
                        "src_rate": src_rate,
                        "dst_rate": self.sample_rate,
                        "src_len": len(pcm),
                        "dst_len": len(pcm_resampled),
                    }
                )

            await self._append_timeline(direction, ts, pcm_resampled)

    def _interleave_stereo_pcm16le(self, left: bytes, right: bytes) -> bytes:
        out = bytearray()
        n = min(len(left), len(right))
        for i in range(0, n, 2):
            out.extend(left[i : i + 2])
            out.extend(right[i : i + 2])
        return bytes(out)

    async def finalize_wav(self):
        if not self.enabled:
            return None

        await self.close()

        try:
            user_size = (
                os.path.getsize(self._user_pcm_path)
                if os.path.exists(self._user_pcm_path)
                else 0
            )
            bot_size = (
                os.path.getsize(self._bot_pcm_path)
                if os.path.exists(self._bot_pcm_path)
                else 0
            )
            logger.info("🎧 Finalizing WAV. user_bytes={} bot_bytes={}", user_size, bot_size)

            if self.sample_width_bytes != 2:
                logger.warning(
                    "⚠️ WAV finalize assumes PCM16LE (2 bytes). Adjust RECORD_SAMPLE_WIDTH_BYTES if needed."
                )

            max_size = max(user_size, bot_size)
            if max_size == 0:
                logger.warning("⚠️ No audio captured; skipping WAV write.")
                return None

            with wave.open(self._wav_path, "wb") as wf, open(
                self._user_pcm_path, "rb"
            ) as uf, open(self._bot_pcm_path, "rb") as bf:
                wf.setnchannels(1 if self.mixdown == "mono" else 2)
                wf.setsampwidth(self.sample_width_bytes)
                wf.setframerate(self.sample_rate)

                block_bytes = 2 * 4800
                read_pos = 0
                while read_pos < max_size:
                    u = uf.read(block_bytes) or b""
                    b = bf.read(block_bytes) or b""
                    read_pos += block_bytes

                    if len(u) < block_bytes:
                        u = u + self._pcm_silence(block_bytes - len(u))
                    if len(b) < block_bytes:
                        b = b + self._pcm_silence(block_bytes - len(b))

                    if self.mixdown == "mono":
                        ua = array.array("h")
                        ba = array.array("h")
                        ua.frombytes(u)
                        ba.frombytes(b)
                        mixed = array.array(
                            "h", [int((ua[i] + ba[i]) / 2) for i in range(len(ua))]
                        )
                        wf.writeframes(mixed.tobytes())
                    else:
                        wf.writeframes(self._interleave_stereo_pcm16le(u, b))

            logger.info("✅ Wrote WAV: {}", self._wav_path)
            return self._wav_path
        except Exception:
            logger.exception("❌ Failed to finalize WAV")
            return None

