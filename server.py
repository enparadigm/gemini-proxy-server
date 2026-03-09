import asyncio
import os
import ssl

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


def get_access_token():
    print("🔑 Getting access token via google.auth.default()")
    creds, _ = google.auth.default(quota_project_id=PROJECT_ID)
    if not creds.valid:
        print("🔄 Refreshing access token…")
        creds.refresh(Request())
    print("✅ Got access token", f"(project: {PROJECT_ID})" if PROJECT_ID else "")
    return creds.token


async def proxy_loop(src, dst, label: str):
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
                    await dst.send(msg)
                    continue

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

    # 1) Read the first message from the browser (GeminiLiveAPI sends service_url)
    try:
        first_msg = await asyncio.wait_for(client_ws.recv(), timeout=10.0)
        print(f"📥 First message from client: {first_msg!r}")
    except Exception as e:
        print(f"⏱️ Timeout or error waiting for first client message: {e}")
        await client_ws.close(code=1008, reason="No initial message")
        return

    # 2) Get access token
    try:
        token = get_access_token()
    except Exception as e:
        print(f"❌ Error getting access token: {e}")
        await client_ws.close(code=1011, reason="Auth failed")
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
                proxy_loop(client_ws, api_ws, "client→api")
            )
            api_to_client = asyncio.create_task(
                proxy_loop(api_ws, client_ws, "api→client")
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


async def main():
    async with websockets.serve(handle_client, "0.0.0.0", PORT):
        print(f"🚀 Proxy running on ws://0.0.0.0:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
