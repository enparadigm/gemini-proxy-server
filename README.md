### Gemini WebSocket Proxy Server

This project runs a small WebSocket proxy in front of the Gemini Multimodal Live API.  
The proxy:
- Accepts WebSocket connections from the browser.
- Obtains an OAuth2 access token using `google.auth.default()`.
- Opens a WebSocket connection to the Gemini API and forwards messages in both directions.

The proxy listens on `ws://localhost:80` by default.

You can also pass a **project ID** into the proxy via the `GOOGLE_CLOUD_PROJECT` environment variable. The proxy will use this as the quota/billing project when obtaining access tokens.

Similarly, you can control the **region** of the Gemini Live API endpoint via `GOOGLE_CLOUD_REGION` (default is `us-central1`). For example, to use an EU endpoint you can set it to `europe-west4`.

Run the proxy:
```bash
python3 server.py
```

---

### Running the demo frontend

After the proxy is running, set up and run the demo frontend:

1. **Clone the demo repo and go to the sample app directory**

   ```bash
   git clone https://github.com/GoogleCloudPlatform/generative-ai.git
   cd generative-ai/gemini/multimodal-live-api/native-audio-websocket-demo-apps/plain-js-demo-app/frontend
   ```

2. **Run the Python HTTP server to host the frontend app on port 8001**

   ```bash
   python3 -m http.server 8001
   ```

3. **Open the frontend in your browser**
   - Go to `http://localhost:8001`.

4. **Configure the project and proxy**
   - When prompted in the UI, provide the **project ID** (for example `gemini-rag-demo-486916`) — this is used by the sample frontend.
   - Ensure the frontend is configured to use the proxy WebSocket URL:
     - `ws://localhost:8080`

With this setup:
- The browser connects from `http://localhost:8001` → `ws://localhost:80` (the proxy).
- The proxy obtains an access token using ADC / your service account, using `GOOGLE_CLOUD_PROJECT` as the quota project when set.
- The proxy then connects securely to the Gemini API and forwards messages between the browser and Gemini.

