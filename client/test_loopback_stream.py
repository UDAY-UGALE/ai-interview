"""
Streams SYSTEM OUTPUT audio (what you hear through your speakers/headphones -
e.g. the other person's voice in a Teams/Zoom/Meet call) to InterviewCopilot,
instead of your own microphone.

Uses the `soundcard` library's WASAPI loopback support (Windows-only here;
soundcard also supports loopback-equivalent capture on Linux/pulseaudio and
macOS/coreaudio). This taps the audio right at your chosen output device,
before it becomes physical sound, so:
  - It only picks up what your SYSTEM is playing (the meeting audio).
  - It will NOT pick up your own mic/voice at all.
  - It works identically whether you're on speakers, wired headphones, or
    (mostly) Bluetooth headsets -- see README note on Bluetooth profiles.

NOTE: `sounddevice`'s WasapiSettings does NOT support loopback (that's a
long-standing open feature request upstream: python-sounddevice#281). This
script intentionally uses `soundcard` instead, which implements it directly.

Usage:
    python client\\test_loopback_stream.py --list-devices
    python client\\test_loopback_stream.py --device 0
"""

import argparse
import asyncio
import queue
import sys
import threading
import time
import json

import numpy as np
import soundcard as sc
import websockets


DEFAULT_WS_URL = "ws://127.0.0.1:8000/ws/audio"
TARGET_SAMPLE_RATE = 16000
CHUNK_MS = 20  # must match AUDIO_FRAME_MS on the backend
BLOCKSIZE = int(TARGET_SAMPLE_RATE * CHUNK_MS / 1000)  # 320 frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream system/loopback (speaker) audio to InterviewCopilot."
    )
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--session-id", default="default")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Index of the OUTPUT device to loop back, from --list-devices. "
        "Defaults to your system's current default output device.",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-meter", action="store_true", help="Hide the live level meter.")
    return parser.parse_args()


def list_devices() -> None:
    speakers = sc.all_speakers()
    default = sc.default_speaker()
    for index, speaker in enumerate(speakers):
        marker = ">" if speaker.name == default.name else " "
        print(f"{marker} {index}  {speaker.name}")
    print(
        "\nPick the OUTPUT device you actually hear meeting audio through "
        "(the one marked '>' is your current Windows default), then pass "
        "its index via --device."
    )


def resolve_speaker(device_index):
    speakers = sc.all_speakers()
    if device_index is None:
        return sc.default_speaker()
    if not (0 <= device_index < len(speakers)):
        raise ValueError(
            f"--device {device_index} is out of range (0..{len(speakers) - 1}). "
            "Run --list-devices to see valid indices."
        )
    return speakers[device_index]


class LoopbackCapture:
    """Runs soundcard's blocking record loop on a background thread and
    hands 16kHz mono PCM16 chunks to an asyncio queue on the event loop."""

    def __init__(self, speaker, loop: asyncio.AbstractEventLoop,
                 audio_queue: "asyncio.Queue[bytes]", meter: "AudioMeter") -> None:
        self._speaker = speaker
        self._loop = loop
        self._queue = audio_queue
        self._meter = meter
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        mic = sc.get_microphone(self._speaker.id, include_loopback=True)
        # Requesting TARGET_SAMPLE_RATE directly: WASAPI shared-mode lets the
        # Windows audio engine resample for us, so we don't need our own
        # resampler like a raw-PortAudio approach would.
        try:
            with mic.recorder(samplerate=TARGET_SAMPLE_RATE, blocksize=BLOCKSIZE) as recorder:
                while not self._stop_event.is_set():
                    data = recorder.record(numframes=BLOCKSIZE)  # float32, (frames, channels)
                    mono = data.mean(axis=1) if data.ndim == 2 and data.shape[1] > 1 else data.reshape(-1)
                    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
                    chunk = pcm16.tobytes()
                    self._meter.observe(chunk)
                    self._loop.call_soon_threadsafe(self._enqueue, chunk)
        except Exception as exc:  # surface capture errors on the main thread
            print(f"[loopback capture error] {exc}", file=sys.stderr)
            self._loop.call_soon_threadsafe(self._stop_event.set)

    def _enqueue(self, chunk: bytes) -> None:
        if self._queue.full():
            return
        self._queue.put_nowait(chunk)


async def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    speaker = resolve_speaker(args.device)
    print(f"Loopback device: {speaker.name}")
    print(f"Capturing at {TARGET_SAMPLE_RATE} Hz mono (resampled by Windows audio engine).")

    ws_url = f"{args.ws_url}?session_id={args.session_id}"

    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    loop = asyncio.get_running_loop()
    meter = AudioMeter(enabled=not args.no_meter)
    capture = LoopbackCapture(speaker, loop, audio_queue, meter)

    # Audio capture runs continuously across reconnects (started once, here)
    # so a network blip doesn't cause an audio glitch -- only the websocket
    # send loop needs to reconnect; captured chunks just queue up (bounded,
    # oldest dropped) until the connection comes back.
    capture.start()
    try:
        while True:
            try:
                async with websockets.connect(ws_url, max_size=None) as websocket:
                    print(f"Streaming system audio to {ws_url}. Press Ctrl+C to stop.")
                    receiver = asyncio.create_task(print_server_messages(websocket))
                    try:
                        while True:
                            await websocket.send(await audio_queue.get())
                    finally:
                        receiver.cancel()
            except (websockets.exceptions.ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                print(f"[connection lost] {exc}; retrying in 2s...", file=sys.stderr)
                await asyncio.sleep(2)
    finally:
        capture.stop()


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
            confidence = payload.get("confidence")
            tag = " [LOW CONFIDENCE]" if payload.get("low_confidence") else ""
            print(
                "[transcript] "
                f"{payload.get('text')} "
                f"({payload.get('segment_seconds')}s, {payload.get('stt_latency_ms')}ms STT, "
                f"confidence={confidence}){tag}"
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
        print(f"[system rms] {self._peak_rms:5d} {'#' * bars}")
        self._peak_rms = 0
        self._last_print = now


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
