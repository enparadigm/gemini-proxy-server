import asyncio
import base64
import json
import os
import secrets
import ssl
import time
import wave
import array

import certifi
import google.auth
from google.auth.transport.requests import Request
import websockets
from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol

REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
API_WS_URL = (
    f"wss://{REGION}-aiplatform.googleapis.com/"
    "ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
)
PORT = 8080
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


RECORD_ENABLED = os.getenv("RECORD_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
RECORD_DIR = os.getenv("RECORD_DIR", "recordings")
RECORD_SAMPLE_RATE = int(os.getenv("RECORD_SAMPLE_RATE", "24000"))
RECORD_SAMPLE_WIDTH_BYTES = int(os.getenv("RECORD_SAMPLE_WIDTH_BYTES", "2"))  # PCM16LE default
RECORD_MIXDOWN = os.getenv("RECORD_MIXDOWN", "stereo").lower()  # "stereo" or "mono"
RECORD_USER_INPUT_SAMPLE_RATE = int(os.getenv("RECORD_USER_INPUT_SAMPLE_RATE", "16000"))
RECORD_BOT_INPUT_SAMPLE_RATE = int(os.getenv("RECORD_BOT_INPUT_SAMPLE_RATE", "24000"))


def get_access_token():
    print("🔑 Getting access token via google.auth.default()")
    creds, _ = google.auth.default(quota_project_id=PROJECT_ID)
    if not creds.valid:
        print("🔄 Refreshing access token…")
        creds.refresh(Request())
    print("✅ Got access token", f"(project: {PROJECT_ID})" if PROJECT_ID else "")
    return creds.token


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
                data = x.get("data") or x.get("audio") or x.get("pcm") or x.get("audioData")
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
        sample_rate: int,
        sample_width_bytes: int,
        mixdown: str,
        user_input_sample_rate: int,
        bot_input_sample_rate: int,
    ):
        self.sample_rate = sample_rate
        self.sample_width_bytes = sample_width_bytes
        self.mixdown = mixdown
        self.user_input_sample_rate = user_input_sample_rate
        self.bot_input_sample_rate = bot_input_sample_rate

        self.session_id = f"{int(time.time())}-{secrets.token_hex(4)}"
        self.base_dir = os.path.join(RECORD_DIR, self.session_id)

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
        if not RECORD_ENABLED:
            return
        os.makedirs(self.base_dir, exist_ok=True)
        # Binary timeline PCM, silence inserted by timestamp alignment.
        self._user_f = open(self._user_pcm_path, "ab", buffering=0)
        self._bot_f = open(self._bot_pcm_path, "ab", buffering=0)
        self._events_f = open(self._events_path, "a", encoding="utf-8")
        print(f"📝 Recording enabled. Session dir: {self.base_dir}")

    async def close(self):
        if not RECORD_ENABLED:
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
        if not RECORD_ENABLED or not self._events_f:
            return
        self._events_f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._events_f.flush()

    def _pcm_silence(self, n_bytes: int) -> bytes:
        # PCM silence is 0 for signed PCM16LE.
        return b"\x00" * n_bytes

    def _resample_pcm16_mono(self, pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
        # Keep implementation intentionally simple and dependency-free.
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
        # Linear interpolation to keep speech natural enough.
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
        """
        Write PCM bytes into a per-direction timeline aligned to wall-clock timestamps.
        We insert silence if there is a gap between the current timeline position and ts.
        """
        if not RECORD_ENABLED:
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
        """
        direction: "client→api" or "api→client"
        msg: str or bytes
        """
        if not RECORD_ENABLED:
            return

        ts = time.time()

        # Always log basic envelope for debugging.
        event = {
            "ts": ts,
            "direction": direction,
            "type": "binary" if is_binary else "text",
            "byte_len": len(msg) if isinstance(msg, (bytes, bytearray)) else (len(msg.encode("utf-8")) if isinstance(msg, str) else None),
        }

        if is_binary:
            event["note"] = "binary frame (not parsed)"
            await self._write_event(event)
            return

        if not isinstance(msg, str):
            event["note"] = f"unexpected msg type {type(msg)}"
            await self._write_event(event)
            return

        # JSON parse for PCM chunks.
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

            # Minimal sanity check: sample_width alignment (best effort).
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

            src_rate = self.user_input_sample_rate if direction == "client→api" else self.bot_input_sample_rate
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
        # Both left and right are PCM16LE mono.
        out = bytearray()
        n = min(len(left), len(right))
        # Each frame is 2 bytes.
        for i in range(0, n, 2):
            out.extend(left[i : i + 2])
            out.extend(right[i : i + 2])
        return bytes(out)

    async def finalize_wav(self):
        if not RECORD_ENABLED:
            return None

        await self.close()

        # Create a single WAV file from the two timeline PCM files.
        try:
            user_size = os.path.getsize(self._user_pcm_path) if os.path.exists(self._user_pcm_path) else 0
            bot_size = os.path.getsize(self._bot_pcm_path) if os.path.exists(self._bot_pcm_path) else 0
            print(f"🎧 Finalizing WAV. user_bytes={user_size} bot_bytes={bot_size}")

            if self.sample_width_bytes != 2:
                print("⚠️ WAV finalize assumes PCM16LE (2 bytes). Adjust RECORD_SAMPLE_WIDTH_BYTES if needed.")

            max_size = max(user_size, bot_size)
            if max_size == 0:
                print("⚠️ No audio captured; skipping WAV write.")
                return None

            with wave.open(self._wav_path, "wb") as wf, open(self._user_pcm_path, "rb") as uf, open(self._bot_pcm_path, "rb") as bf:
                if self.mixdown == "mono":
                    wf.setnchannels(1)
                else:
                    wf.setnchannels(2)
                wf.setsampwidth(self.sample_width_bytes)
                wf.setframerate(self.sample_rate)

                block_bytes = 2 * 4800  # ~0.2s @ 24kHz mono PCM16
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
                        # Mix mono by averaging int16 samples.
                        ua = array.array("h")
                        ba = array.array("h")
                        ua.frombytes(u)
                        ba.frombytes(b)
                        mixed = array.array("h", [int((ua[i] + ba[i]) / 2) for i in range(len(ua))])
                        wf.writeframes(mixed.tobytes())
                    else:
                        wf.writeframes(self._interleave_stereo_pcm16le(u, b))

            print(f"✅ Wrote WAV: {self._wav_path}")
            return self._wav_path
        except Exception as e:
            print(f"❌ Failed to finalize WAV: {e}")
            return None


async def proxy_loop(src, dst, label: str, recorder: ConversationRecorder | None = None):
    print(f"{label}: start proxy loop")
    try:
        async for msg in src:
            # Normalize bytes → text when possible so the browser sees strings
            if isinstance(msg, bytes):
                try:
                    text = msg.decode("utf-8")
                    msg = text
                except UnicodeDecodeError:
                    # Truly binary (e.g., audio) – just forward as‑is
                    if recorder:
                        await recorder.observe(label, msg, is_binary=True)
                    await dst.send(msg)
                    continue

            if recorder:
                await recorder.observe(label, msg, is_binary=False)
            await dst.send(msg)
    except ConnectionClosed as e:
        print(f"{label}: connection closed ({e.code}, {e.reason})")
    except Exception as e:
        print(f"{label}: unexpected error: {e}")
    finally:
        try:
            await dst.close()
        except Exception:
            pass
        print(f"{label}: dst closed")


async def handle_client(client_ws: WebSocketServerProtocol):
    print("🔌 New WebSocket client connection from browser")
    recorder = ConversationRecorder(
        sample_rate=RECORD_SAMPLE_RATE,
        sample_width_bytes=RECORD_SAMPLE_WIDTH_BYTES,
        mixdown=RECORD_MIXDOWN,
        user_input_sample_rate=RECORD_USER_INPUT_SAMPLE_RATE,
        bot_input_sample_rate=RECORD_BOT_INPUT_SAMPLE_RATE,
    )
    await recorder.open()

    # 1) Read the first message from the browser (GeminiLiveAPI sends service_url)
    try:
        first_msg = await asyncio.wait_for(client_ws.recv(), timeout=10.0)
        print(f"📥 First message from client: {first_msg!r}")
    except Exception as e:
        print(f"⏱️ Timeout or error waiting for first client message: {e}")
        await client_ws.close(code=1008, reason="No initial message")
        await recorder.finalize_wav()
        return

    # 2) Get access token
    try:
        token = get_access_token()
    except Exception as e:
        print(f"❌ Error getting access token: {e}")
        await client_ws.close(code=1011, reason="Auth failed")
        await recorder.finalize_wav()
        return

    headers = {"Authorization": f"Bearer {token}"}
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    # 3) Connect to Gemini
    print(f"🌐 Connecting to Gemini API at {API_WS_URL}")
    try:
        async with websockets.connect(
            API_WS_URL, additional_headers=headers, ssl=ssl_ctx
        ) as api_ws:
            print("✅ Connected to Gemini API")

            # NOTE: we intentionally do NOT forward first_msg; it just contained service_url
            client_to_api = asyncio.create_task(
                proxy_loop(client_ws, api_ws, "client→api", recorder=recorder)
            )
            api_to_client = asyncio.create_task(
                proxy_loop(api_ws, client_ws, "api→client", recorder=recorder)
            )

            done, pending = await asyncio.wait(
                [client_to_api, api_to_client],
                return_when=asyncio.FIRST_COMPLETED,
            )
            print("⚠️ One side of the proxy finished, cancelling the other")
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    except ConnectionClosed as e:
        print(f"❌ Upstream (Gemini) connection closed: {e.code} {e.reason}")
        if not client_ws.closed:
            await client_ws.close(code=e.code, reason=e.reason)
    except Exception as e:
        print(f"❌ Failed to connect to Gemini API: {e}")
        if not client_ws.closed:
            await client_ws.close(code=1011, reason="Upstream failure")
    finally:
        await recorder.finalize_wav()


async def main():
    async with websockets.serve(handle_client, "0.0.0.0", PORT):
        print(f"🚀 Proxy running on ws://0.0.0.0:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
