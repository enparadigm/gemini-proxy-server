import asyncio
import json
import os
import ssl

import certifi
import google.auth
from google.auth.transport.requests import Request
from loguru import logger
import websockets
from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol

from conversation_recorder import ConversationRecorder
from usage_tracker import UsageTracker

REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
API_WS_URL = (
    f"wss://{REGION}-aiplatform.googleapis.com/"
    "ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
)
PORT = 8080
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


RECORD_ENABLED = os.getenv("RECORD_ENABLED", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
RECORD_DIR = os.getenv("RECORD_DIR", "recordings")
RECORD_SAMPLE_RATE = int(os.getenv("RECORD_SAMPLE_RATE", "24000"))
RECORD_SAMPLE_WIDTH_BYTES = int(
    os.getenv("RECORD_SAMPLE_WIDTH_BYTES", "2")
)  # PCM16LE default
RECORD_MIXDOWN = os.getenv("RECORD_MIXDOWN", "stereo").lower()  # "stereo" or "mono"
RECORD_USER_INPUT_SAMPLE_RATE = int(os.getenv("RECORD_USER_INPUT_SAMPLE_RATE", "16000"))
RECORD_BOT_INPUT_SAMPLE_RATE = int(os.getenv("RECORD_BOT_INPUT_SAMPLE_RATE", "24000"))


def get_access_token():
    logger.info("🔑 Getting access token via google.auth.default()")
    creds, _ = google.auth.default(quota_project_id=PROJECT_ID)
    if not creds.valid:
        logger.info("🔄 Refreshing access token…")
        creds.refresh(Request())
    logger.info(
        "✅ Got access token {}", f"(project: {PROJECT_ID})" if PROJECT_ID else ""
    )
    return creds.token


async def proxy_loop(
    src,
    dst,
    label: str,
    recorder: ConversationRecorder | None = None,
    usage_tracker: UsageTracker | None = None,
):
    logger.info("{}: start proxy loop", label)
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

            # ── Intercept api→client messages for usageMetadata ──────────
            if usage_tracker and label == "api→client":
                try:
                    parsed = json.loads(msg)
                    usage = parsed.get("usageMetadata")
                    if usage:
                        usage_tracker.record_turn(usage)
                        logger.info(
                            "📊 Turn {}: input={} output={} total={} tokens",
                            len(usage_tracker.turns),
                            usage.get("promptTokenCount", "?"),
                            usage.get("candidatesTokenCount", "?"),
                            usage.get("totalTokenCount", "?"),
                        )
                except (json.JSONDecodeError, AttributeError):
                    pass  # not JSON or no usageMetadata — forward as-is
            # ─────────────────────────────────────────────────────────────

            if recorder:
                await recorder.observe(label, msg, is_binary=False)
            await dst.send(msg)
    except ConnectionClosed as e:
        logger.warning("{}: connection closed ({}, {})", label, e.code, e.reason)
    except Exception as e:
        logger.error("{}: unexpected error: {}", label, e)
    finally:
        try:
            await dst.close()
        except Exception:
            pass
        logger.info("{}: dst closed", label)


async def handle_client(client_ws: WebSocketServerProtocol):
    logger.info("🔌 New WebSocket client connection from browser")
    recorder = ConversationRecorder(
        enabled=RECORD_ENABLED,
        record_dir=RECORD_DIR,
        sample_rate=RECORD_SAMPLE_RATE,
        sample_width_bytes=RECORD_SAMPLE_WIDTH_BYTES,
        mixdown=RECORD_MIXDOWN,
        user_input_sample_rate=RECORD_USER_INPUT_SAMPLE_RATE,
        bot_input_sample_rate=RECORD_BOT_INPUT_SAMPLE_RATE,
    )
    await recorder.open()

    usage_tracker = UsageTracker(session_id=recorder.session_id)

    # 1) Read the first message from the browser (GeminiLiveAPI sends service_url)
    try:
        first_msg = await asyncio.wait_for(client_ws.recv(), timeout=10.0)
        logger.info("📥 First message from client: {!r}", first_msg)
    except Exception as e:
        logger.warning("⏱️ Timeout or error waiting for first client message: {}", e)
        await client_ws.close(code=1008, reason="No initial message")
        await recorder.finalize_wav()
        return

    # 2) Get access token
    try:
        token = get_access_token()
    except Exception as e:
        logger.error("❌ Error getting access token: {}", e)
        await client_ws.close(code=1011, reason="Auth failed")
        await recorder.finalize_wav()
        return

    headers = {"Authorization": f"Bearer {token}"}
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    # 3) Connect to Gemini
    logger.info("🌐 Connecting to Gemini API at {}", API_WS_URL)
    try:
        async with websockets.connect(
            API_WS_URL, additional_headers=headers, ssl=ssl_ctx
        ) as api_ws:
            logger.info("✅ Connected to Gemini API")

            # NOTE: we intentionally do NOT forward first_msg; it just contained service_url
            client_to_api = asyncio.create_task(
                proxy_loop(client_ws, api_ws, "client→api", recorder=recorder)
            )
            api_to_client = asyncio.create_task(
                proxy_loop(
                    api_ws,
                    client_ws,
                    "api→client",
                    recorder=recorder,
                    usage_tracker=usage_tracker,
                )
            )

            done, pending = await asyncio.wait(
                [client_to_api, api_to_client],
                return_when=asyncio.FIRST_COMPLETED,
            )
            logger.warning("⚠️ One side of the proxy finished, cancelling the other")
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    except ConnectionClosed as e:
        logger.warning(
            "❌ Upstream (Gemini) connection closed: {} {}", e.code, e.reason
        )
        if not client_ws.closed:
            await client_ws.close(code=e.code, reason=e.reason)
    except Exception as e:
        logger.error("❌ Failed to connect to Gemini API: {}", e)
        if not client_ws.closed:
            await client_ws.close(code=1011, reason="Upstream failure")
    finally:
        await recorder.finalize_wav()
        usage_tracker.print_report()

        try:
            cost_path = os.path.join(recorder.base_dir, "cost_report.json")
            with open(cost_path, "w", encoding="utf-8") as f:
                json.dump(
                    usage_tracker.calculate_cost(),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info("💰 Wrote cost report: {}", cost_path)
        except Exception:
            logger.exception("Failed to write cost_report.json")


async def main():
    async with websockets.serve(handle_client, "0.0.0.0", PORT):
        logger.info("🚀 Proxy running on ws://0.0.0.0:{}", PORT)
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
