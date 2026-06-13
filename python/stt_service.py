"""Local speech-to-text over WebSocket (faster-whisper on CPU)."""
import asyncio
import json
import struct

import numpy as np
import websockets
from faster_whisper import WhisperModel
from websockets.exceptions import ConnectionClosed

PORT = 8766
HOST = "127.0.0.1"
_model = None


def load_model():
    global _model
    if _model is None:
        print("Loading Whisper tiny.en (first run downloads ~75MB)...")
        _model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return _model


def resample_mono(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    pcm = pcm.astype(np.float64).flatten()
    if pcm.size == 0:
        return np.array([], dtype=np.float32)
    if orig_sr == target_sr:
        return pcm.astype(np.float32)
    duration = len(pcm) / orig_sr
    n_out = max(1, int(duration * target_sr))
    x_old = np.linspace(0.0, len(pcm) - 1, len(pcm))
    x_new = np.linspace(0.0, len(pcm) - 1, n_out)
    return np.interp(x_new, x_old, pcm).astype(np.float32)


async def handle_client(websocket):
    model = load_model()
    try:
        async for message in websocket:
            if not isinstance(message, (bytes, bytearray)) or len(message) < 8:
                continue
            try:
                sr = struct.unpack_from("<I", message, 0)[0]
                pcm = np.frombuffer(message[4:], dtype=np.float32)
                if pcm.size < 512:
                    await websocket.send(json.dumps({"text": ""}))
                    continue
                pcm16k = resample_mono(pcm, sr, 16000)
                if pcm16k.size < 4000:
                    await websocket.send(json.dumps({"text": ""}))
                    continue
                segments, _ = model.transcribe(
                    pcm16k,
                    language="en",
                    beam_size=1,
                    vad_filter=False,
                    without_timestamps=True,
                )
                text = "".join(s.text for s in segments).strip()
                await websocket.send(json.dumps({"text": text}))
            except Exception as e:
                try:
                    await websocket.send(json.dumps({"text": "", "error": str(e)}))
                except ConnectionClosed:
                    break
    except ConnectionClosed:
        return


async def main():
    load_model()
    print(f"STT ready — ws://{HOST}:{PORT}")
    async with websockets.serve(handle_client, HOST, PORT, max_size=50 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
