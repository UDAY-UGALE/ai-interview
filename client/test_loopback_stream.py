"""
Streams SYSTEM OUTPUT audio (what you hear through your speakers/headphones -
e.g. the other person's voice in a Teams/Zoom/Meet call) to InterviewCopilot,
instead of your own microphone.

Capture goes through `client/wasapi_loopback.py`, which talks to Windows Core
Audio directly. This taps the audio at your chosen output endpoint, before it
becomes physical sound, so:
  - It only picks up what your SYSTEM is playing (the meeting audio).
  - It will NOT pick up your own mic/voice at all.
  - It works identically on speakers, wired headphones, HDMI, USB and
    (mostly) Bluetooth headsets -- see README note on Bluetooth profiles.

Two libraries were tried before this one and both are dead ends on Windows:
`sounddevice`/PortAudio still has no loopback support (python-sounddevice#281),
and `soundcard` 0.4.x mis-declares WAVEFORMATEXTENSIBLE to CFFI, which corrupts
the endpoint's channel mask and can abort the process with
STATUS_HEAP_CORRUPTION (0xC0000374) depending on the driver. The gory details
are in the `wasapi_loopback` module docstring.

Usage:
    python client\\test_loopback_stream.py --list-devices
    python client\\test_loopback_stream.py --device 0
    python client\\test_loopback_stream.py --device "Headphones"
"""

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
import urllib.parse

import numpy as np
import websockets

import wasapi_loopback as wl
from config import load_defaults


# Resolved by client/config.py: CLI flag > env var > config file > localhost.
# The same build therefore serves a developer on 127.0.0.1 and an installed
# client pointed at a deployed backend.
CLIENT_DEFAULTS = load_defaults()
TARGET_SAMPLE_RATE = 16000
CHUNK_MS = 20  # must match AUDIO_FRAME_MS on the backend -- size of each
                # chunk actually SENT over the websocket, one per message.
WIRE_BLOCKSIZE = int(TARGET_SAMPLE_RATE * CHUNK_MS / 1000)  # 320 frames

# How much slack to keep in the WASAPI capture buffer, in milliseconds.
#
# This used to be the size of each read, because `soundcard` re-resampled on
# every single record() call and a late 20ms read would overflow WASAPI's ring
# buffer -- the "data discontinuity in recording" warning -- handing back
# corrupted PCM that the recognizer transcribed into plausible garbage.
#
# Capture now reads whatever WASAPI already has ready and resamples in numpy,
# so a read is a memcpy plus one convolution and there is no per-call deadline
# to miss. The flag survives because the underlying trade-off does: it now
# sizes the endpoint buffer, i.e. how far behind the app may fall before
# Windows starts dropping frames. Larger is safer on a loaded machine and
# costs nothing in latency, because audio is still forwarded as soon as it
# arrives rather than being held for a full gulp.
DEFAULT_RECORD_CHUNK_MS = 100

# How often to check whether Windows switched the default playback device.
# Only polled when the user did not pin a device with --device.
DEFAULT_DEVICE_POLL_SECONDS = 2.0

log = logging.getLogger("loopback")


class CaptureUnavailable(RuntimeError):
    """Every capture strategy failed. Carries what was tried, and why."""

    def __init__(self, attempts: list[str],
                 endpoints: "list[wl.AudioEndpoint] | None" = None):
        self.attempts = attempts
        if endpoints is None:
            # Best effort: if enumeration is what failed, say so rather than
            # letting a second failure hide the first.
            try:
                endpoints = wl.enumerate_output_endpoints()
            except wl.WasapiError:
                endpoints = []
        lines = ["Could not capture system audio. Attempts, in order:"]
        lines += [f"  - {a}" for a in attempts]
        lines.append("")
        lines.append("Active output endpoints Windows reports:")
        lines += [f"  {e.index}  {e}" for e in endpoints] or ["  (none)"]
        lines += [
            "",
            "What to try:",
            "  * Check a playback device is enabled in Windows Sound settings.",
            "  * Pick a specific device: --list-devices, then --device <n>.",
            "  * If the device is held in exclusive mode by another app, close"
            " it or untick 'Allow applications to take exclusive control'.",
            "  * Restart the Windows Audio service if it is not running.",
            "  * As a last resort, enable Stereo Mix in Recording devices and"
            " pass --allow-stereo-mix.",
        ]
        super().__init__("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream system/loopback (speaker) audio to InterviewCopilot."
    )
    parser.add_argument("--ws-url", default=CLIENT_DEFAULTS.ws_url)
    parser.add_argument("--session-id", default=CLIENT_DEFAULTS.session_id)
    parser.add_argument(
        "--token",
        default=CLIENT_DEFAULTS.token,
        help="Shared secret for a deployed backend (server-side APP_AUTH_TOKEN). "
        "Defaults to the INTERVIEW_TOKEN environment variable.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="The OUTPUT device to loop back: an index from --list-devices, a "
        "device-name substring, or a Windows endpoint ID. Defaults to your "
        "system's current default output device, and follows it if you switch "
        "devices while running.",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-meter", action="store_true", help="Hide the live level meter.")
    parser.add_argument(
        "--record-chunk-ms",
        type=int,
        default=DEFAULT_RECORD_CHUNK_MS,
        help=(
            "How much slack to keep in the WASAPI capture buffer, in ms "
            f"(default {DEFAULT_RECORD_CHUNK_MS}). Larger = more headroom "
            "before a loaded machine starts dropping frames. Unlike previous "
            "versions this no longer adds latency in front of each answer."
        ),
    )
    parser.add_argument(
        "--allow-stereo-mix",
        action="store_true",
        help="Permit falling back to a Stereo Mix recording device if WASAPI "
        "loopback cannot be opened on any output endpoint. Off by default: "
        "Stereo Mix is usually absent or disabled, and its routing is not "
        "always the audio you expect.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log capture diagnostics (device, formats, fallbacks) in detail.",
    )
    return parser.parse_args()


def list_devices() -> None:
    endpoints = wl.enumerate_output_endpoints()
    if not endpoints:
        print("No active audio output devices found. Check Windows Sound settings.")
        return
    for endpoint in endpoints:
        marker = ">" if endpoint.is_default else " "
        print(f"{marker} {endpoint.index}  {endpoint.name}")
    print(
        "\nPick the OUTPUT device you actually hear meeting audio through "
        "(the one marked '>' is your current Windows default), then pass "
        "its index via --device. A name substring works too, and is stable "
        "across reboots in a way indices are not."
    )


class LoopbackCapture:
    """Runs WASAPI loopback capture on a background thread and hands 16kHz
    mono PCM16 chunks to an asyncio queue on the event loop.

    Audio is captured at whatever format the endpoint natively runs at
    (typically 48kHz stereo float32, but 44.1kHz, mono and multichannel
    endpoints all work), downmixed and resampled to 16kHz mono here, then
    sliced into WIRE_BLOCKSIZE (20ms) pieces so the backend and its VAD see
    exactly the cadence they always did.

    The thread owns a reopen loop: if the endpoint is invalidated (unplugged,
    disabled, reconfigured) or the user switches Windows' default playback
    device, capture reopens on the right endpoint instead of dying.
    """

    def __init__(self, selector, loop: asyncio.AbstractEventLoop,
                 audio_queue: "asyncio.Queue[bytes]", meter: "AudioMeter",
                 record_chunk_ms: int = DEFAULT_RECORD_CHUNK_MS,
                 allow_stereo_mix: bool = False) -> None:
        self._selector = selector
        self._loop = loop
        self._queue = audio_queue
        self._meter = meter
        self._buffer_ms = max(50, int(record_chunk_ms))
        self._allow_stereo_mix = allow_stereo_mix
        self._follow_default = selector is None
        self._stop_event = threading.Event()
        self._dropped_chunks = 0
        self._pending = bytearray()
        self._fatal: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="loopback-capture",
                                        daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)

    @property
    def failure(self) -> BaseException | None:
        """Set when capture gave up for good, so main() can report it."""
        return self._fatal

    # -- device selection -------------------------------------------------

    def _candidates(self, notes: list[str]):
        """Capture strategies to try, best first.

        Alternate output endpoints are only considered when the user did not
        name a device: silently moving to a different speaker would be as
        confusing as silently recording the microphone. Stereo Mix is opt-in
        for the same reason, and is the only non-loopback entry here -- a
        real microphone is never a candidate.

        Enumeration itself can fail (the audio service restarting, a device
        vanishing mid-list), so each source is guarded separately and records
        why it produced nothing rather than aborting the whole search.
        """
        candidates = []
        primary = None
        try:
            primary = wl.resolve_endpoint(self._selector)
            candidates.append((primary, True, "WASAPI loopback"))
        except (wl.WasapiError, ValueError) as exc:
            notes.append(f"selecting --device {self._selector!r}: {exc}")

        if self._follow_default:
            try:
                for endpoint in wl.enumerate_output_endpoints():
                    if primary is None or endpoint.endpoint_id != primary.endpoint_id:
                        candidates.append(
                            (endpoint, True,
                             "WASAPI loopback (alternate output endpoint)"))
            except wl.WasapiError as exc:
                notes.append(f"enumerating output endpoints: {exc}")

        if self._allow_stereo_mix:
            try:
                stereo_mix = wl.find_stereo_mix()
            except wl.WasapiError as exc:
                stereo_mix = None
                notes.append(f"looking for Stereo Mix: {exc}")
            if stereo_mix is None:
                notes.append("Stereo Mix: no such recording device is enabled")
            else:
                candidates.append((stereo_mix, False, "Stereo Mix recording device"))

        return candidates

    def _open(self):
        """Open the first strategy that works, or raise CaptureUnavailable.

        `notes` records why a candidate source offered nothing; `failures`
        records candidates that were actually tried and did not open. Only
        the latter are worth telling the user about on a success.
        """
        notes: list[str] = []
        failures: list[str] = []
        for endpoint, loopback, backend in self._candidates(notes):
            try:
                recorder = wl.LoopbackRecorder(
                    endpoint, buffer_ms=self._buffer_ms, loopback=loopback
                )
                recorder.__enter__()
            except (wl.WasapiError, OSError, ValueError) as exc:
                failures.append(f"{backend} on {endpoint.name!r}: {exc}")
                log.warning("%s initialization failed for %r: %s",
                            backend, endpoint.name, exc)
                log.info("Attempting fallback ...")
                continue

            report = wl.CaptureReport(
                endpoint=endpoint,
                backend=backend,
                native_rate=recorder.format.sample_rate,
                native_channels=recorder.format.channels,
                native_format=recorder.format.sample_format,
                target_rate=TARGET_SAMPLE_RATE,
                attempts=failures,
            )
            return recorder, report

        raise CaptureUnavailable(
            notes + failures or ["no capture strategy was available"])

    # -- capture ----------------------------------------------------------

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop_event.is_set():
            try:
                self._session()
                backoff = 0.5
            except CaptureUnavailable as exc:
                self._fatal = exc
                print(f"[loopback capture error]\n{exc}", file=sys.stderr)
                self._loop.call_soon_threadsafe(self._stop_event.set)
                return
            except wl.WasapiError as exc:
                if self._stop_event.is_set():
                    return
                reason = ("the device went away"
                          if exc.device_invalidated else str(exc))
                print(f"[audio] capture interrupted ({reason}); "
                      f"reopening in {backoff:.1f}s...", file=sys.stderr)
                self._stop_event.wait(backoff)
                backoff = min(backoff * 2, 5.0)
            except Exception as exc:  # never let the thread die silently
                self._fatal = exc
                print(f"[loopback capture error] {exc!r}", file=sys.stderr)
                log.exception("unexpected capture failure")
                self._loop.call_soon_threadsafe(self._stop_event.set)
                return

    def _session(self) -> None:
        """One open-capture-close cycle. Returns when it is time to reopen."""
        recorder, report = self._open()
        try:
            resampler = wl.Resampler(recorder.format.sample_rate, TARGET_SAMPLE_RATE)
            report.resampler = resampler.description
            for line in report.lines():
                print(line)
            if report.attempts:
                print(f"(after {len(report.attempts)} failed attempt(s); "
                      "run with --verbose for details)")
            sys.stdout.flush()

            self._pump(recorder, resampler)
        finally:
            recorder.__exit__(None, None, None)

    def _pump(self, recorder: wl.LoopbackRecorder, resampler: wl.Resampler) -> None:
        frame_bytes = WIRE_BLOCKSIZE * 2  # int16 = 2 bytes/frame
        current_default = wl.default_output_endpoint_id() if self._follow_default else ""
        next_poll = time.monotonic() + DEFAULT_DEVICE_POLL_SECONDS

        while not self._stop_event.is_set():
            frames = recorder.read(timeout=0.05)
            if frames.shape[0]:
                mono = wl.downmix_to_mono(frames)
                resampled = resampler.process(mono)
                if resampled.size:
                    self._pending.extend(wl.float_to_pcm16(resampled))

                    # Re-slice into fixed 20ms pieces -- the size the
                    # websocket protocol and the backend's VAD expect --
                    # keeping any remainder for the next read instead of
                    # dropping or misaligning it.
                    while len(self._pending) >= frame_bytes:
                        chunk = bytes(self._pending[:frame_bytes])
                        del self._pending[:frame_bytes]
                        self._meter.observe(chunk)
                        self._loop.call_soon_threadsafe(self._enqueue, chunk)
            else:
                # Nothing ready yet. A short sleep keeps this thread off a
                # spin loop without adding meaningful latency.
                self._stop_event.wait(0.005)

            if self._follow_default and time.monotonic() >= next_poll:
                next_poll = time.monotonic() + DEFAULT_DEVICE_POLL_SECONDS
                latest = wl.default_output_endpoint_id()
                if latest and latest != current_default:
                    print("[audio] Windows switched the default playback device; "
                          "following it.", file=sys.stderr)
                    return  # outer loop reopens on the new endpoint

    def _enqueue(self, chunk: bytes) -> None:
        if self._queue.full():
            # Drop the OLDEST buffered audio, not this new chunk. A full
            # queue means the websocket is behind; keeping the stale head
            # and discarding fresh audio makes the backlog permanent and
            # feeds the recognizer audio from seconds ago. Dropping from the
            # front keeps the stream current -- and says so, because silent
            # audio loss is indistinguishable downstream from the
            # interviewer having said nothing.
            try:
                self._queue.get_nowait()
                self._dropped_chunks += 1
                if self._dropped_chunks % 50 == 1:
                    print(
                        f"[audio] backlog: dropped {self._dropped_chunks} chunks "
                        f"({self._dropped_chunks * CHUNK_MS}ms) to stay live",
                        file=sys.stderr,
                    )
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(chunk)


async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_devices:
        list_devices()
        return

    # Resolve the selection before opening a socket or starting a thread, so
    # a typo in --device is one clear line rather than a background failure.
    try:
        wl.resolve_endpoint(args.device)
    except (wl.WasapiError, ValueError) as exc:
        print(f"[audio] {exc}", file=sys.stderr)
        return

    ws_url =f"{args.ws_url}?session_id={urllib.parse.quote(args.session_id)}"
    if args.token:
        ws_url += f"&token={urllib.parse.quote(args.token)}"

    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
    loop = asyncio.get_running_loop()
    meter = AudioMeter(enabled=not args.no_meter)
    capture = LoopbackCapture(
        args.device, loop, audio_queue, meter,
        record_chunk_ms=args.record_chunk_ms,
        allow_stereo_mix=args.allow_stereo_mix,
    )

    # Audio capture runs continuously across reconnects (started once, here)
    # so a network blip doesn't cause an audio glitch -- only the websocket
    # send loop needs to reconnect; captured chunks just queue up (bounded,
    # oldest dropped) until the connection comes back.
    capture.start()
    try:
        while True:
            if capture.failure is not None:
                return  # capture already printed why; nothing left to stream
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
            tail = "" if payload.get("is_final", True) else " (continues)"
            print(
                f"[speech segment] {payload.get('segment_seconds')}s "
                f"({payload.get('speech_seconds')}s voiced), "
                f"utterance {payload.get('utterance_id')}{tail}; transcribing..."
            )
        elif message_type == "transcript_dropped":
            # Visible on purpose: these are the transcripts that used to
            # reach the LLM as questions nobody asked.
            print(
                f"[dropped:{payload.get('reason')}] {payload.get('text')!r} "
                f"(confidence={payload.get('confidence')}, "
                f"{payload.get('speech_seconds')}s voiced)"
            )
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
    """RMS level of a PCM16 frame, for the on-screen meter.

    numpy rather than a Python loop over every sample: this runs on every
    20ms frame on the capture thread, and the loop version cost ~165us a
    frame -- CPU spent competing with the WASAPI read deadline it exists to
    protect.
    """
    samples = np.frombuffer(frame, dtype=np.int16)
    if samples.size == 0:
        return 0
    return int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
