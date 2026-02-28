import os
import asyncio
import json
import websockets

from livekit import rtc
from livekit.api.access_token import AccessToken
from livekit.api.access_token import VideoGrants
import numpy as np
from scipy.signal import resample
# import audioop
from livekit.rtc import AudioStream

LIVEKIT_URL = "wss://voice-agent-7t8ve31g.livekit.cloud"
API_KEY = "APIbQVgTaAddcUQ"
API_SECRET = "hi5iBYBCmk65N4S4k8Rfyj9IzzdBEzCbTjy0KGSRzHB"

def create_livekit_token(identity: str, room: str) -> str:
    token = AccessToken(api_key=API_KEY, api_secret=API_SECRET)
    token.with_identity(identity)
    token.with_grants(VideoGrants(room_join=True, room=room))
    return token.to_jwt()

async def create_room(identity: str, room: str):
    token = create_livekit_token(identity, room)

    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    return room

def lin2ulaw(pcm16_bytes: bytes) -> bytes:
    """Convert PCM16 bytes to 8-bit µ-law bytes."""
    pcm = np.frombuffer(pcm16_bytes, dtype=np.int16)
    pcm = np.clip(pcm, -32768, 32767)
    magnitude = np.abs(pcm)
    exponent = np.floor(np.log2(magnitude + 1e-9)).astype(np.int16)
    mantissa = (magnitude >> (exponent - 4)) & 0x0F
    ulaw = ((pcm < 0).astype(np.int16) << 7) | ((exponent & 0x07) << 4) | mantissa
    ulaw = (ulaw + 128).astype(np.uint8)
    return ulaw.tobytes()

def ulaw2lin(ulaw_bytes: bytes) -> np.ndarray:
    """Convert 8-bit µ-law bytes to PCM16 numpy array."""
    ulaw = np.frombuffer(ulaw_bytes, dtype=np.uint8).astype(np.int16)
    ulaw = ulaw - 128
    magnitude = ((ulaw & 0x0F) << 3) + 0x84
    magnitude <<= ((ulaw & 0x70) >> 4)
    pcm16 = np.where(ulaw & 0x80, 0x84 - magnitude, magnitude - 0x84)
    return pcm16

# --- Helper: Resample PCM16 bytes ---
def resample_audio(pcm16_bytes: bytes, in_rate: int, out_rate: int) -> bytes:
    audio = np.frombuffer(pcm16_bytes, dtype=np.int16)
    n_samples = int(len(audio) * out_rate / in_rate)
    resampled = resample(audio, n_samples).astype(np.int16)
    return resampled.tobytes()

FRAME_SIZE = 160  # 20ms @ 8kHz
FRAME_DURATION = 0.02  # 20ms

# --- Forward LiveKit agent audio → Genesys ---
async def forward_agent_audio(track, ws):
    audio_stream = AudioStream(track)
    send_buffer = bytearray()
    async for event in audio_stream:
        frame = event.frame  # AudioFrame
        pcm16_48k = frame.data  # PCM16 @ 48kHz

        pcm16_8k = resample_audio(pcm16_48k, 48000, 8000)
        pcmu_bytes = lin2ulaw(pcm16_8k)

        send_buffer.extend(pcmu_bytes)

        # print("PCMU bytes: ", len(pcmu_bytes))

        # Send only when we have 20ms (160 bytes)
        while len(send_buffer) >= FRAME_SIZE:
            chunk = bytes(send_buffer[:FRAME_SIZE])
            print("chunk len: ", len(chunk))
            del send_buffer[:FRAME_SIZE]

            await ws.send(chunk)
            await asyncio.sleep(FRAME_DURATION)

class Session:
    def __init__(self, ws, session_id, url):
        self.ws = ws
        self.session_id = session_id
        self.url = url
        self.send_seq = 1
        self.livekit_room = None
        self.local_audio_source = None

    def close(self):
        # Clean up any resources, LiveKit tracks, etc.
        pass

    async def send_disconnect(self, reason: str):
        await self.ws.send(json.dumps({"event": "disconnect", "reason": reason}))

    async def process_binary_message(self, data: bytes):
        # Handle Genesys audio payload
        print(f"Received {len(data)} bytes")

        if not self.livekit_room or not self.local_audio_source:
            print(f"[{self.session_id}] Received audio before OPEN. Ignoring.")
            return

        # 1. Convert PCMU → PCM16
        pcm16_array = ulaw2lin(data)  # returns numpy int16 array

        # 2. Resample to 48 kHz (LiveKit expects 48kHz)
        pcm16_8k = pcm16_array.tobytes()
        pcm16_48k = resample_audio(pcm16_8k, 8000, 48000)

        # 3. Build an AudioFrame
        frame_size = 960 * 2  # 2 bytes per sample, mono
        for i in range(0, len(pcm16_48k), frame_size):
            chunk = pcm16_48k[i:i + frame_size]
            if len(chunk) < frame_size:
                continue  # skip incomplete frames

            # 4️⃣ Build AudioFrame
            audio_frame = rtc.AudioFrame.create(48000, 1, 960)  # 48 kHz, 1 channel, 960 samples
            np.copyto(
                np.frombuffer(audio_frame.data, dtype=np.int16),
                np.frombuffer(chunk, dtype=np.int16)
            )

            # 5️⃣ Capture frame via AudioSource
            await self.local_audio_source.capture_frame(audio_frame)

        print(f"[{self.session_id}] Published {len(data)} bytes ({len(pcm16_48k) // 2} samples) to LiveKit")

    async def process_text_message(self, message: str):
        payload = json.loads(message)
        print(f"Received JSON message: {payload}")
        # Handle 'start', 'media', 'stop'

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        msg_type = payload.get("type")

        response = {
            "version": payload.get("version", "2"),
            "seq": self.send_seq,
            "clientseq": payload["seq"],
            "id": payload["id"],
        }

        if msg_type == "open":
            print("AudioHook stream opened")

            token = create_livekit_token(self.session_id, f"room_{self.session_id}")
            self.livekit_room = rtc.Room()

            def on_track_subscribed(track, pub, participant):
                print("track subscribed: %s", pub.sid)
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    asyncio.create_task(forward_agent_audio(track, self.ws))

            self.livekit_room.on("track_subscribed", on_track_subscribed)

            await self.livekit_room.connect(LIVEKIT_URL, token)

            # self.livekit_room = await create_room(self.session_id, f"room_{self.session_id}")
            # Create local track for Genesys caller → LiveKit agent
            self.local_audio_source = rtc.AudioSource(48000, 1)
            local_track = rtc.LocalAudioTrack.create_audio_track("caller", self.local_audio_source)
            await self.livekit_room.local_participant.publish_track(local_track)
            print("published caller track")

            # Subscribe to agent audio → Genesys
            # @self.livekit_room.on("track_subscribed")

            response.update({
                "type": "opened",
                "parameters": {
                    "media": payload["parameters"]["media"],
                    "supportedLanguages": [
                        "en-US", "en-GB", "fi-FI", "sv-SE"
                    ]
                }
            })

            print(f"Forward JSON message: {response}")

            await self.ws.send(json.dumps(response))

            # Increment server seq for next message
            self.send_seq += 1
        elif msg_type == "ping":
            response.update({
                "type": "pong",
                "parameters": {}
            })

            print(f"Forward JSON message: {response}")

            await self.ws.send(json.dumps(response))

            # Increment server seq for next message
            self.send_seq += 1
        elif msg_type == "close":
            print("Stream closing")
            await self.livekit_room.disconnect()

        elif msg_type == "error":
            print("Stream closing")
            await self.livekit_room.disconnect()

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