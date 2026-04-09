import os
import asyncio
import json
import websockets

from livekit import rtc
from livekit.api.access_token import AccessToken
from livekit.api.access_token import VideoGrants
import numpy as np
from scipy.signal import resample
import audioop
from livekit.rtc import AudioStream
from math import gcd
from scipy.signal import resample_poly

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

def lin2ulaw(pcm16: np.ndarray) -> bytes:
    """Convert int16 ndarray to µ-law bytes."""
    return audioop.lin2ulaw(pcm16.tobytes(), 2)

def ulaw2lin(ulaw_bytes: bytes) -> np.ndarray:
    """Convert µ-law bytes to int16 ndarray."""
    pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
    return np.frombuffer(pcm_bytes, dtype=np.int16)

# --- Helper: Resample PCM16 bytes ---
def resample_audio(audio: np.ndarray, in_rate: int, out_rate: int) -> bytes:
    """Resample int16 ndarray from in_rate to out_rate."""
    g = gcd(in_rate, out_rate)
    up = out_rate // g
    down = in_rate // g
    resampled = resample_poly(audio, up, down).astype(np.int16)
    return np.clip(resampled, -32768, 32767).astype(np.int16)  # clip before cast

FRAME_DURATION = 0.2  # 200ms
FRAME_SIZE_TO_GENESYS = 8000 * FRAME_DURATION

# --- Forward LiveKit agent audio → Genesys ---
async def forward_agent_audio(track, ws, send_gate: asyncio.Event, on_turn_end):
    audio_stream = AudioStream(track)
    send_buffer = bytearray()
    next_send_time = None

    silent_frames = 0
    SILENCE_FRAMES = 50   # 50 * 20ms = 1 second of silence
    has_spoken = False    # don't close before agent says anything

    async for event in audio_stream:
        frame = event.frame
        pcm16_48k = np.frombuffer(frame.data, dtype=np.int16)
        pcm16_8k = resample_audio(pcm16_48k, 48000, 8000)
        pcmu_bytes = lin2ulaw(pcm16_8k)

        is_silent = np.max(np.abs(pcm16_48k)) < 10

        if is_silent:
            if has_spoken:
                silent_frames += 1
                if silent_frames >= SILENCE_FRAMES:
                    print("Agent done speaking — sending close")
                    silent_frames = 0
                    has_spoken = False
                    # await on_turn_end()  # triggers send_close()
                    # return
        else:
            has_spoken = True
            silent_frames = 0

        send_buffer.extend(pcmu_bytes)

        print(
            f"send_buffer size before drain: {len(send_buffer)}, frames to send: {len(send_buffer) // FRAME_SIZE_TO_GENESYS}")  # ← here

        while len(send_buffer) >= FRAME_SIZE_TO_GENESYS:
            if not send_gate.is_set():
                print("Send-gate not set! Ignoring")
                send_buffer.clear()  # discard stale audio
                next_send_time = None  # reset pacing
                break

            now = asyncio.get_event_loop().time()

            # Initialize or enforce 20ms spacing
            if next_send_time is None:
                next_send_time = now
            elif next_send_time > now:
                await asyncio.sleep(next_send_time - now)

            chunk = bytes(send_buffer[:FRAME_SIZE_TO_GENESYS])
            del send_buffer[:FRAME_SIZE_TO_GENESYS]

            try:
                print(
                    f"ws send size: {len(chunk)}")
                await ws.send(chunk)
            except Exception:
                return

            next_send_time += FRAME_DURATION  # advance by exactly 20ms

class Session:
    def __init__(self, ws, session_id, url):
        self.ws = ws
        self.session_id = session_id
        self.url = url
        self.send_seq = 1
        self.livekit_room = None
        self.local_audio_source = None
        self.audio_buffer = np.array([], dtype=np.int16)
        self.forwarding_active = False  # in __init__
        self.send_audio_event = asyncio.Event()
        self.client_seq = None

    def close(self):
        # Clean up any resources, LiveKit tracks, etc.
        pass

    async def send_disconnect(self, reason: str = "completed"):
        msg = {
            "version": "2",
            "type": "disconnect",
            "seq": self.send_seq,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {
                "reason": reason,
            }
        }
        print(f"Forward JSON message: {msg}")
        await self.ws.send(json.dumps(msg))
        self.send_seq += 1
        # await self.ws.send(json.dumps({"type": "disconnect", "reason": reason}))

    async def process_binary_message(self, data: bytes):
        # Handle Genesys audio payload
        print(f"Received {len(data)} bytes")

        if not self.livekit_room or not self.local_audio_source:
            print(f"[{self.session_id}] Received audio before OPEN. Ignoring.")
            return

        # 1. PCMU → PCM16 @ 8kHz
        pcm16_8k = ulaw2lin(data)

        # 2. Resample 8kHz → 48kHz
        pcm16_48k = resample_audio(pcm16_8k, 8000, 48000)

        # 3. Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, pcm16_48k])

        # 3. Build an AudioFrame
        SAMPLES_PER_FRAME = 48000 * FRAME_DURATION
        while len(self.audio_buffer) >= SAMPLES_PER_FRAME:
            chunk = self.audio_buffer[:SAMPLES_PER_FRAME]
            self.audio_buffer = self.audio_buffer[SAMPLES_PER_FRAME:]

            audio_frame = rtc.AudioFrame.create(48000, 1, SAMPLES_PER_FRAME)
            np.copyto(
                np.frombuffer(audio_frame.data, dtype=np.int16),
                chunk
            )
            await self.local_audio_source.capture_frame(audio_frame)
            print("DEBUG: Captured one 20ms frame to LiveKit")  # Add this
        # frame_size = 960 * 2  # 2 bytes per sample, mono
        # for i in range(0, len(pcm16_48k), frame_size):
        #     chunk = pcm16_48k[i:i + frame_size]
        #     if len(chunk) < frame_size:
        #         continue  # skip incomplete frames
        #
        #     # 4️⃣ Build AudioFrame
        #     audio_frame = rtc.AudioFrame.create(48000, 1, 960)  # 48 kHz, 1 channel, 960 samples
        #     np.copyto(
        #         np.frombuffer(audio_frame.data, dtype=np.int16),
        #         np.frombuffer(chunk, dtype=np.int16)
        #     )
        #
        #     # 5️⃣ Capture frame via AudioSource
        #     await self.local_audio_source.capture_frame(audio_frame)

        print(f"buffer={len(self.audio_buffer)} samples pending")

    async def process_text_message(self, message: str):
        payload = json.loads(message)
        print(f"Received JSON message: {payload}")
        # Handle 'start', 'media', 'stop'
        self.client_seq = payload["seq"]

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
            self.send_audio_event.set()

            token = create_livekit_token(self.session_id, f"room_{self.session_id}")
            self.livekit_room = rtc.Room()

            def on_track_subscribed(track, pub, participant):
                print("track subscribed: %s", pub.sid)
                # if track.kind == rtc.TrackKind.KIND_AUDIO:
                #     asyncio.create_task(forward_agent_audio(track, self.ws))
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    if not self.forwarding_active:
                        self.forwarding_active = True
                        asyncio.ensure_future(forward_agent_audio(
                            track,
                            self.ws,
                            self.send_audio_event,
                            self.send_disconnect
                        ))
                    else:
                        print("WARNING: duplicate track_subscribed, ignoring")

            self.livekit_room.on("track_subscribed", on_track_subscribed)

            await self.livekit_room.connect(LIVEKIT_URL, token)

            # self.livekit_room = await create_room(self.session_id, f"room_{self.session_id}")
            # Create local track for Genesys caller → LiveKit agent
            self.local_audio_source = rtc.AudioSource(48000, 1)
            local_track = rtc.LocalAudioTrack.create_audio_track("genesys-client", self.local_audio_source)

            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE

            await self.livekit_room.local_participant.publish_track(local_track, options)
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
        elif msg_type == "playback_completed":
            # self.send_audio_event.set()  # Genesys is done, stop sending
            print("Playback completed — starting audio send")
            # await self.send_disconnect("session closed")
        elif msg_type == "playback_started":
            # self.send_audio_event.clear()
            print("Playback started — stopping audio send")
        elif msg_type == "close" or msg_type == "disconnect":
            print("Stream closing (close)", msg_type)
            response.update({
                "type": "closed"
            })
            await self.ws.send(json.dumps(response))
            await self.livekit_room.disconnect()
        elif msg_type == "error":
            print("Stream closing (error)")
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
        # await session.send_disconnect("session closed")
        # session.close()
        # sessions.pop(ws, None)
        print("Session completed:", session_id)

async def main():
    # port = int(os.environ.get("PORT", 8765))
    port = int(8765)
    async with websockets.serve(handle_ws, "0.0.0.0", port):
        print(f"WSS server running on port {port}")
        await asyncio.Future()  # keep alive

if __name__ == "__main__":
    asyncio.run(main())