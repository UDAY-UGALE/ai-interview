import argparse
import asyncio
import urllib.parse
import json
import sys
import time

import sounddevice as sd
import websockets

from config import load_defaults


CLIENT_DEFAULTS = load_defaults()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream microphone audio to InterviewCopilot.")
    parser.add_argument("--ws-url", default=CLIENT_DEFAULTS.ws_url)
    parser.add_argument("--session-id", default=CLIENT_DEFAULTS.session_id)
    parser.add_argument(
        "--token",
        default=CLIENT_DEFAULTS.token,
        help="Shared secret for a deployed backend (server-side APP_AUTH_TOKEN).",
    )
    parser.add_argument("--samplerate", type=int, default=16000)
    parser.add_argument("--chunk-ms", type=int, default=20, choices=(10, 20, 30))
    parser.add_argument("--device", default=None, help="Input device index or name.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-meter", action="store_true", help="Hide the live mic level meter.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    blocksize = int(args.samplerate * args.chunk_ms / 1000)
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    loop = asyncio.get_running_loop()
    meter = AudioMeter(enabled=not args.no_meter)

    print_selected_device(args.device)

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)

        chunk = bytes(indata)
        meter.observe(chunk)
        loop.call_soon_threadsafe(enqueue_audio, chunk)

    def enqueue_audio(chunk: bytes) -> None:
        if audio_queue.full():
            return
        audio_queue.put_nowait(chunk)

    ws_url = f"{args.ws_url}?session_id={urllib.parse.quote(args.session_id)}"
    if args.token:
        ws_url += f"&token={urllib.parse.quote(args.token)}"

    async with websockets.connect(ws_url, max_size=None) as websocket:
        receiver = asyncio.create_task(print_server_messages(websocket))

        print(f"Streaming mic audio to {args.ws_url}. Press Ctrl+C to stop.")
        with sd.RawInputStream(
            samplerate=args.samplerate,
            blocksize=blocksize,
            channels=1,
            dtype="int16",
            device=args.device,
            callback=audio_callback,
        ):
            while True:
                await websocket.send(await audio_queue.get())

        await receiver


async def print_server_messages(websocket) -> None:
    async for message in websocket:
        if isinstance(message, bytes):
            continue

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            print(message)
            continue

        message_type = payload.get("type")
        if message_type == "ready":
            print(
                "Backend ready "
                f"(sample_rate={payload.get('sample_rate')}, frame_ms={payload.get('frame_ms')})."
            )
        elif message_type == "transcript":
            print(
                "[transcript] "
                f"{payload.get('text')} "
                f"({payload.get('segment_seconds')}s, {payload.get('stt_latency_ms')}ms STT)"
            )
        elif message_type == "speech_segment":
            print(f"[speech segment] {payload.get('segment_seconds')}s; transcribing...")
        elif message_type == "error":
            print(f"[error] {payload.get('message')}", file=sys.stderr)
        else:
            print(payload)


class AudioMeter:
    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._last_print = time.monotonic()
        self._peak_rms = 0

    def observe(self, pcm: bytes) -> None:
        if not self._enabled:
            return

        self._peak_rms = max(self._peak_rms, pcm16_rms(pcm))
        now = time.monotonic()
        if now - self._last_print < 1:
            return

        bars = min(40, self._peak_rms // 120)
        print(f"[mic rms] {self._peak_rms:5d} {'#' * bars}")
        self._peak_rms = 0
        self._last_print = now


def print_selected_device(device) -> None:
    try:
        info = sd.query_devices(device, "input")
    except Exception as exc:
        print(f"Could not read selected input device: {exc}", file=sys.stderr)
        return

    print(f"Input device: {info.get('name', 'unknown')}")


def pcm16_rms(frame: bytes) -> int:
    sample_count = len(frame) // 2
    if sample_count == 0:
        return 0

    total = 0
    for index in range(0, len(frame) - 1, 2):
        sample = int.from_bytes(frame[index : index + 2], byteorder="little", signed=True)
        total += sample * sample

    return int((total / sample_count) ** 0.5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
