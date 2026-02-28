import os
import asyncio
import json
import websockets

class Session:
    def __init__(self, ws, session_id, url):
        self.ws = ws
        self.session_id = session_id
        self.url = url

    def close(self):
        # Clean up any resources, LiveKit tracks, etc.
        pass

    async def send_disconnect(self, reason: str):
        await self.ws.send(json.dumps({"event": "disconnect", "reason": reason}))

    async def process_binary_message(self, data: bytes):
        # Handle Genesys audio payload
        print(f"Received {len(data)} bytes")
        # Forward to LiveKit if needed

    async def process_text_message(self, message: str):
        payload = json.loads(message)
        print(f"Received JSON message: {payload}")
        # Handle 'start', 'media', 'stop'

# Session map
sessions = {}

async def handle_ws(ws):
    session_id = ws.request.headers.get("audiohook-session-id", "unknown")
    session = Session(ws, session_id, ws.request.path)
    sessions[ws] = session
    print("Session created:", session_id)

    try:
        async for message in ws:
            if isinstance(message, bytes):
                await session.process_binary_message(message)
            else:
                await session.process_text_message(message)
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        await session.send_disconnect("session closed")
        session.close()
        sessions.pop(ws, None)
        print("Session deleted:", session_id)

async def main():
    # port = int(os.environ.get("PORT", 8765))
    port = int(8765)
    async with websockets.serve(handle_ws, "0.0.0.0", port):
        print(f"WSS server running on port {port}")
        await asyncio.Future()  # keep alive

if __name__ == "__main__":
    asyncio.run(main())