import asyncio
import websockets
import json
import base64
from livekit import rtc
from livekit.api.access_token import AccessToken
from livekit.api.access_token import VideoGrants
import numpy as np
from scipy.signal import resample

LIVEKIT_URL = "wss://voice-agent-7t8ve31g.livekit.cloud"
API_KEY = "APIbQVgTaAddcUQ"
API_SECRET = "hi5iBYBCmk65N4S4k8Rfyj9IzzdBEzCbTjy0KGSRzHB"
ROOM_NAME = "genesys-room"

def resample_audio(pcm_data, input_rate=8000, output_rate=48000):
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    num_output_samples = int(len(samples) * output_rate / input_rate)
    resampled = resample(samples, num_output_samples)
    return resampled.astype(np.int16).tobytes()

def create_livekit_token(identity: str, room: str) -> str:
    token = AccessToken(api_key=API_KEY, api_secret=API_SECRET)
    token.with_identity(identity)
    token.with_grants(VideoGrants(room_join=True, room=room))
    return token.to_jwt()

async def create_room():
    token = create_livekit_token("bridge-bot", "genesys-room")

    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    return room

async def handle_connection(ws):
    print("Client connected")
    room = await create_room()

    # ---- Publish track (Caller → Agent) ----
    source = rtc.AudioSource(48000, 1)
    local_track = rtc.LocalAudioTrack.create_audio_track("caller", source)
    await room.local_participant.publish_track(local_track)

    # ---- Subscribe to agent track (Agent → Caller) ----
    async def livekit_to_genesys():
        @room.on("track_subscribed")
        def on_track(track, pub, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.create_task(forward_agent_audio(track))

        async def forward_agent_audio(track):
            async for frame in track:
                pcm_48k = frame.data

                # Downsample to 8kHz
                pcm_8k = resample_audio(pcm_48k, 48000, 8000)

                # 20ms chunks (320 bytes)
                chunk_size = 320
                for i in range(0, len(pcm_8k), chunk_size):
                    chunk = pcm_8k[i:i+chunk_size]
                    if len(chunk) == chunk_size:
                        payload = base64.b64encode(chunk).decode()

                        msg = {
                            "event": "media",
                            "media": {"payload": payload}
                        }

                        await ws.send(json.dumps(msg))

    asyncio.create_task(livekit_to_genesys())

    # ---- Genesys → LiveKit ----
    try:
        async for message in ws:
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                print("Call started:", data)
            elif event == "media":
                # Forward payload to LiveKit agent
                payload_b64 = data["media"]["payload"]
                # Convert base64 → PCM if needed, send to LiveKit track
                print("Received media:", len(payload_b64))
            elif data.get("event") == "media":
                payload = base64.b64decode(data["media"]["payload"])

                # Upsample 8k → 48k
                pcm_48k = resample_audio(payload, 8000, 48000)

                frame_size = 960 * 2  # 20ms @ 48kHz
                for i in range(0, len(pcm_48k), frame_size):
                    frame = pcm_48k[i:i+frame_size]
                    if len(frame) == frame_size:
                        await source.capture_frame(frame)
            elif event == "stop":
                print("Call ended")
                await room.disconnect()
            else:
                print("Unknown event:", data)

            await ws.send(json.dumps({"status": "ok"}))
    except Exception as e:
        print("Error:", e)

async def main():
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        print("Bridge running on 8765")
        await asyncio.Future()

asyncio.run(main())