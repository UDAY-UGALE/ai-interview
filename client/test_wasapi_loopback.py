"""
Test suite for WASAPI loopback capture (client/wasapi_loopback.py).

Most of it runs offline with no audio device at all: struct layout, format
parsing, resampling, downmixing and device selection are all pure functions
over bytes and arrays. Those cases are the ones that would have caught the
bug this module exists to fix, so they run everywhere -- including CI on a
machine with no sound card.

The `device` and `runtime` groups need a real output endpoint and are
skipped (not failed) when there isn't one. `runtime` includes a 30-second
continuous capture, so it is opt-in via --runtime.

Usage (from the project root):
    python client\\test_wasapi_loopback.py
    python client\\test_wasapi_loopback.py --list
    python client\\test_wasapi_loopback.py --runtime
    python client\\test_wasapi_loopback.py --only resampler
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import struct
import sys
import time
import traceback

import numpy as np

import wasapi_loopback as wl


# --------------------------------------------------------------------------
# Tiny test harness -- same shape as the other client/test_*.py suites
# --------------------------------------------------------------------------

CASES: list[tuple[str, str, str, callable]] = []


def case(group: str, name: str, description: str):
    def register(fn):
        CASES.append((group, name, description, fn))
        return fn
    return register


class Skip(Exception):
    """Raised by a case that cannot run in this environment."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close_to(actual: float, expected: float, tolerance: float, what: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{what}: got {actual:.4f}, expected {expected:.4f} +/- {tolerance:.4f}"
        )


# --------------------------------------------------------------------------
# Helpers: build WAVEFORMATEX blobs exactly as GetMixFormat would
# --------------------------------------------------------------------------

def make_mix_format(rate=48000, channels=2, bits=32, extensible=True,
                    is_float=True, channel_mask=0x3) -> ctypes.Array:
    """A byte-exact WAVEFORMATEX(TENSIBLE), allocated at its true size.

    Sized `18 + cbSize` like the real CoTaskMemAlloc block, so anything that
    reads past the declared end runs off the end of this buffer too and gets
    caught by the bounds checks below rather than by a heap abort.
    """
    block_align = channels * bits // 8
    tag = wl.WAVE_FORMAT_EXTENSIBLE if extensible else (
        wl.WAVE_FORMAT_IEEE_FLOAT if is_float else wl.WAVE_FORMAT_PCM)
    cb_size = 22 if extensible else 0
    header = struct.pack(
        "<HHIIHHH", tag, channels, rate, rate * block_align,
        block_align, bits, cb_size,
    )
    if extensible:
        subtype = (wl.KSDATAFORMAT_SUBTYPE_IEEE_FLOAT if is_float
                   else wl.KSDATAFORMAT_SUBTYPE_PCM)
        header += struct.pack("<HI", bits, channel_mask)
        header += ctypes.string_at(ctypes.byref(subtype), 16)

    check(len(header) == 18 + cb_size,
          f"test helper built {len(header)} bytes, expected {18 + cb_size}")
    buffer = (ctypes.c_char * len(header))()
    buffer.raw = header
    return buffer


def sine(freq: float, rate: int, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_frequency(signal: np.ndarray, rate: int) -> float:
    windowed = signal * np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    return float(np.fft.rfftfreq(signal.size, 1 / rate)[int(np.argmax(spectrum))])


# --------------------------------------------------------------------------
# struct layout -- the regression group for the 0xC0000374 bug
# --------------------------------------------------------------------------

@case("layout", "packing", "WAVEFORMATEX/EXTENSIBLE match Windows' packed layout")
def test_packing():
    check(ctypes.sizeof(wl.WAVEFORMATEX) == 18,
          f"WAVEFORMATEX is {ctypes.sizeof(wl.WAVEFORMATEX)} bytes, must be 18")
    check(ctypes.sizeof(wl.WAVEFORMATEXTENSIBLE) == 40,
          f"WAVEFORMATEXTENSIBLE is {ctypes.sizeof(wl.WAVEFORMATEXTENSIBLE)} bytes, must be 40")
    for field, offset in (("Format", 0), ("wValidBitsPerSample", 18),
                          ("dwChannelMask", 20), ("SubFormat", 24)):
        actual = getattr(wl.WAVEFORMATEXTENSIBLE, field).offset
        check(actual == offset,
              f"WAVEFORMATEXTENSIBLE.{field} at offset {actual}, must be {offset}")


@case("layout", "no-mask-clobber",
      "writing wValidBitsPerSample leaves dwChannelMask intact")
def test_no_mask_clobber():
    """The exact regression.

    soundcard's unpacked struct put `Samples` at offset 20, so setting
    wValidBitsPerSample=32 overwrote dwChannelMask -- turning a valid stereo
    mask (0x3) into 0x20, a mask naming one channel for a two-channel
    format. That inconsistency is what the audio engine's converter choked
    on. With correct packing the two fields cannot collide.
    """
    buffer = make_mix_format(channels=2, channel_mask=0x3)
    view = ctypes.cast(buffer, ctypes.POINTER(wl.WAVEFORMATEXTENSIBLE))[0]
    check(view.dwChannelMask == 0x3, "precondition: mask should start at 0x3")

    view.wValidBitsPerSample = 32

    check(view.dwChannelMask == 0x3,
          f"dwChannelMask was clobbered to 0x{view.dwChannelMask:X} "
          "(this is the soundcard bug)")
    check(bin(view.dwChannelMask).count("1") == view.Format.nChannels,
          "channel mask popcount must match nChannels")
    parsed = wl._parse_format(ctypes.cast(buffer, ctypes.c_void_p))
    check(parsed.is_float and parsed.channels == 2,
          "SubFormat must still parse as 2-channel float after the write")


@case("layout", "no-overread", "parsing never reads past the declared allocation")
def test_no_overread():
    """A plain WAVEFORMATEX has cbSize=0 and is only 18 bytes long.

    soundcard cast such a block to its 44-byte struct and read SubFormat
    from offsets 28..43 -- 26 bytes past the end. Parsing must instead
    honour cbSize and stay inside the block.
    """
    buffer = make_mix_format(extensible=False, is_float=True, bits=32)
    check(len(buffer) == 18, f"plain WAVEFORMATEX helper is {len(buffer)} bytes")

    # A guard page would be ideal; a poisoned trailer is the portable
    # equivalent -- if parsing consults it, the values it derives change.
    guarded = (ctypes.c_char * 64)()
    guarded.raw = buffer.raw + b"\xAB" * 46
    parsed = wl._parse_format(ctypes.cast(guarded, ctypes.c_void_p))
    check(parsed.is_float, "plain IEEE_FLOAT tag must parse as float")
    check(parsed.channel_mask == 0,
          f"non-extensible format has no channel mask, got 0x{parsed.channel_mask:X}")


# --------------------------------------------------------------------------
# format parsing
# --------------------------------------------------------------------------

@case("format", "extensible-float32", "48kHz stereo float32 extensible")
def test_extensible_float():
    parsed = wl._parse_format(ctypes.cast(make_mix_format(), ctypes.c_void_p))
    check(parsed.sample_rate == 48000, f"rate {parsed.sample_rate}")
    check(parsed.channels == 2, f"channels {parsed.channels}")
    check(parsed.is_float and parsed.bits_per_sample == 32, "expected float32")
    check(parsed.block_align == 8, f"block_align {parsed.block_align}")
    check(parsed.sample_format == "float32", parsed.sample_format)


@case("format", "rates", "44.1k / 48k / 96k / 192k and 16k all parse")
def test_rates():
    for rate in (16000, 44100, 48000, 88200, 96000, 192000):
        parsed = wl._parse_format(
            ctypes.cast(make_mix_format(rate=rate), ctypes.c_void_p))
        check(parsed.sample_rate == rate, f"{rate} parsed as {parsed.sample_rate}")


@case("format", "channels", "mono, stereo, 5.1 and 7.1 all parse")
def test_channels():
    for channels, mask in ((1, 0x4), (2, 0x3), (6, 0x3F), (8, 0x63F)):
        parsed = wl._parse_format(ctypes.cast(
            make_mix_format(channels=channels, channel_mask=mask), ctypes.c_void_p))
        check(parsed.channels == channels,
              f"{channels}ch parsed as {parsed.channels}")
        check(parsed.channel_mask == mask, f"mask 0x{parsed.channel_mask:X}")


@case("format", "pcm-depths", "int16 / int24 / int32 PCM parse")
def test_pcm_depths():
    for bits in (16, 24, 32):
        parsed = wl._parse_format(ctypes.cast(
            make_mix_format(bits=bits, is_float=False), ctypes.c_void_p))
        check(not parsed.is_float, f"int{bits} parsed as float")
        check(parsed.bits_per_sample == bits, f"got {parsed.bits_per_sample}")


@case("format", "unsupported", "an unknown format tag raises, never crashes")
def test_unsupported_format():
    header = struct.pack("<HHIIHHH", 0x0161, 2, 48000, 192000, 8, 32, 0)
    buffer = (ctypes.c_char * 18)()
    buffer.raw = header
    try:
        wl._parse_format(ctypes.cast(buffer, ctypes.c_void_p))
    except wl.WasapiError as exc:
        check("not PCM" in str(exc) or "tag" in str(exc), f"unclear message: {exc}")
        return
    raise AssertionError("an unknown wFormatTag must raise WasapiError")


@case("format", "truncated-extensible",
      "EXTENSIBLE with a short cbSize raises instead of over-reading")
def test_truncated_extensible():
    header = struct.pack("<HHIIHHH", wl.WAVE_FORMAT_EXTENSIBLE, 2, 48000,
                         384000, 8, 32, 0)  # claims EXTENSIBLE, cbSize=0
    buffer = (ctypes.c_char * 18)()
    buffer.raw = header
    try:
        wl._parse_format(ctypes.cast(buffer, ctypes.c_void_p))
    except wl.WasapiError as exc:
        check("cbSize" in str(exc), f"unclear message: {exc}")
        return
    raise AssertionError("a truncated EXTENSIBLE header must raise WasapiError")


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------

@case("resampler", "rates", "every plausible device rate converts to 16kHz")
def test_resampler_rates():
    for src in (8000, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000):
        resampler = wl.Resampler(src, 16000)
        signal = sine(440, src, 1.0)
        out = resampler.process(signal)
        expected = signal.size * 16000 / src
        close_to(out.size, expected, max(64, expected * 0.02),
                 f"{src}->16000 output length")


@case("resampler", "pitch", "a 440Hz tone stays 440Hz through conversion")
def test_resampler_pitch():
    for src in (44100, 48000, 96000):
        resampler = wl.Resampler(src, 16000)
        out = resampler.process(sine(440, src, 2.0))
        peak = dominant_frequency(out, 16000)
        close_to(peak, 440.0, 8.0, f"{src}->16000 dominant frequency")


@case("resampler", "anti-alias", "content above 8kHz is removed, not folded down")
def test_resampler_antialias():
    """Without a filter a 15kHz tone at 48kHz folds to 1kHz -- right in the
    middle of the speech band, and audible as a whistle the recognizer would
    happily transcribe around."""
    resampler = wl.Resampler(48000, 16000)
    loud = sine(15000, 48000, 1.0, amplitude=0.9)
    out = resampler.process(loud)
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    check(rms < 0.02,
          f"15kHz tone survived downsampling at RMS {rms:.4f}; it should be "
          "attenuated to near silence, not aliased into the speech band")


@case("resampler", "continuity", "chunked input matches whole input")
def test_resampler_continuity():
    """Filter history and fractional phase must carry across calls; if they
    don't, every chunk boundary is a click and the sample count drifts."""
    signal = sine(300, 48000, 1.0)
    whole = wl.Resampler(48000, 16000).process(signal)

    chunked_resampler = wl.Resampler(48000, 16000)
    pieces = []
    # Deliberately ragged chunk sizes: WASAPI packets are not uniform.
    position = 0
    for size in [480, 1000, 137, 4800, 96, 2400] * 40:
        if position >= signal.size:
            break
        pieces.append(chunked_resampler.process(signal[position:position + size]))
        position += size
    chunked = np.concatenate(pieces)

    check(abs(chunked.size - whole.size) <= 2,
          f"chunked produced {chunked.size} samples, whole produced {whole.size}")
    n = min(chunked.size, whole.size)
    error = float(np.max(np.abs(chunked[:n] - whole[:n])))
    check(error < 1e-3,
          f"chunked and whole output differ by {error:.6f}; state is not "
          "carried across chunk boundaries")


@case("resampler", "passthrough", "a 16kHz device needs no conversion")
def test_resampler_passthrough():
    resampler = wl.Resampler(16000, 16000)
    check("none" in resampler.description, resampler.description)
    signal = sine(440, 16000, 0.5)
    out = resampler.process(signal)
    check(out.size == signal.size, f"{out.size} != {signal.size}")
    check(np.array_equal(out, signal), "passthrough must not alter samples")


@case("resampler", "empty", "empty and tiny inputs are safe")
def test_resampler_empty():
    resampler = wl.Resampler(48000, 16000)
    check(resampler.process(np.zeros(0, dtype=np.float32)).size == 0,
          "empty input must give empty output")
    for _ in range(50):
        resampler.process(np.zeros(1, dtype=np.float32))  # must not raise


@case("resampler", "invalid", "a nonsense rate is rejected up front")
def test_resampler_invalid():
    for src, dst in ((0, 16000), (48000, 0), (-48000, 16000)):
        try:
            wl.Resampler(src, dst)
        except ValueError:
            continue
        raise AssertionError(f"Resampler({src}, {dst}) should have raised")


# --------------------------------------------------------------------------
# downmix and PCM conversion
# --------------------------------------------------------------------------

@case("mixdown", "channels", "mono, stereo and multichannel downmix correctly")
def test_downmix():
    mono = wl.downmix_to_mono(np.array([[0.5], [0.25]], dtype=np.float32))
    check(mono.shape == (2,), f"mono shape {mono.shape}")
    close_to(float(mono[0]), 0.5, 1e-6, "1ch passthrough")

    stereo = wl.downmix_to_mono(np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32))
    close_to(float(stereo[0]), 0.5, 1e-6, "stereo average")
    close_to(float(stereo[1]), 0.5, 1e-6, "stereo average")

    surround = wl.downmix_to_mono(np.ones((4, 6), dtype=np.float32) * 0.3)
    check(surround.shape == (4,), f"5.1 shape {surround.shape}")
    close_to(float(surround[0]), 0.3, 1e-6, "6ch average")

    flat = wl.downmix_to_mono(np.array([0.1, 0.2], dtype=np.float32))
    check(flat.shape == (2,), "already-1D input must pass through")


@case("mixdown", "pcm16", "float to PCM16 clips instead of wrapping")
def test_pcm16():
    """Wrapping is the dangerous failure: +1.2 becoming a large negative
    sample is a loud click, and downstream it looks like real audio."""
    raw = wl.float_to_pcm16(np.array([0.0, 1.0, -1.0, 2.5, -2.5], dtype=np.float32))
    samples = np.frombuffer(raw, dtype="<i2")
    check(samples.size == 5, f"got {samples.size} samples")
    check(samples[0] == 0, f"0.0 -> {samples[0]}")
    check(samples[1] == 32767, f"1.0 -> {samples[1]}")
    check(samples[2] == -32767, f"-1.0 -> {samples[2]}")
    check(samples[3] == 32767, f"2.5 must clip to 32767, got {samples[3]}")
    check(samples[4] == -32767, f"-2.5 must clip to -32767, got {samples[4]}")


@case("mixdown", "frame-size", "20ms at 16kHz is exactly 640 bytes")
def test_frame_size():
    """The backend's VAD frames the stream by byte count, so this is a wire
    protocol constant, not an implementation detail."""
    raw = wl.float_to_pcm16(np.zeros(320, dtype=np.float32))
    check(len(raw) == 640, f"320 frames -> {len(raw)} bytes, expected 640")


# --------------------------------------------------------------------------
# device selection
# --------------------------------------------------------------------------

class FakeEnumeration:
    """Swap in a synthetic device list, so selection logic is testable on
    any machine -- including the multi-device layouts this developer's PC
    does not have."""

    def __init__(self, names_and_defaults):
        self.endpoints = [
            wl.AudioEndpoint(endpoint_id=f"{{0.0.0.0}}.{{fake-{i}}}",
                             name=name, is_default=default, index=i)
            for i, (name, default) in enumerate(names_and_defaults)
        ]
        self._saved = None

    def __enter__(self):
        self._saved = wl.enumerate_output_endpoints
        wl.enumerate_output_endpoints = lambda: list(self.endpoints)
        return self

    def __exit__(self, *exc):
        wl.enumerate_output_endpoints = self._saved


TYPICAL_LAYOUT = [
    ("Speakers (Realtek(R) Audio)", True),
    ("Headphones (2- Realtek(R) Audio)", False),
    ("LG HDR 4K (NVIDIA High Definition Audio)", False),
    ("Headset (Jabra Evolve2 65 Hands-Free)", False),
    ("Speakers (USB Audio Device)", False),
]


@case("selection", "by-index", "--device N picks the Nth listed endpoint")
def test_select_by_index():
    with FakeEnumeration(TYPICAL_LAYOUT):
        for i, (name, _) in enumerate(TYPICAL_LAYOUT):
            check(wl.resolve_endpoint(i).name == name, f"index {i}")
            check(wl.resolve_endpoint(str(i)).name == name, f"string index {i}")


@case("selection", "by-name", "--device accepts a name substring")
def test_select_by_name():
    with FakeEnumeration(TYPICAL_LAYOUT):
        check("Jabra" in wl.resolve_endpoint("jabra").name, "bluetooth headset")
        check("HDR 4K" in wl.resolve_endpoint("NVIDIA").name, "HDMI endpoint")
        check("USB" in wl.resolve_endpoint("usb audio").name, "USB endpoint")


@case("selection", "by-endpoint-id", "--device accepts a Core Audio endpoint ID")
def test_select_by_endpoint_id():
    with FakeEnumeration(TYPICAL_LAYOUT) as fake:
        wanted = fake.endpoints[2]
        check(wl.resolve_endpoint(wanted.endpoint_id).name == wanted.name,
              "endpoint ID lookup")


@case("selection", "default", "no --device follows the Windows default")
def test_select_default():
    with FakeEnumeration(TYPICAL_LAYOUT):
        check(wl.resolve_endpoint(None).name.startswith("Speakers (Realtek"),
              "should pick the endpoint marked default")

    # Default moved to the headset, as it does when one connects.
    moved = [(n, n.startswith("Headset")) for n, _ in TYPICAL_LAYOUT]
    with FakeEnumeration(moved):
        check(wl.resolve_endpoint(None).name.startswith("Headset"),
              "should follow the default when it moves")


@case("selection", "single-device", "a one-output machine resolves cleanly")
def test_select_single():
    with FakeEnumeration([("Speakers (Realtek(R) Audio)", True)]):
        check(wl.resolve_endpoint(None).index == 0, "default on single device")
        check(wl.resolve_endpoint(0).index == 0, "index 0 on single device")


@case("selection", "errors", "bad selections raise actionable errors")
def test_select_errors():
    with FakeEnumeration(TYPICAL_LAYOUT):
        for bad in (99, -1, "99"):
            try:
                wl.resolve_endpoint(bad)
            except ValueError as exc:
                check("--list-devices" in str(exc), f"unhelpful message: {exc}")
            else:
                raise AssertionError(f"--device {bad} should have raised")

        try:
            wl.resolve_endpoint("nonexistent device")
        except ValueError as exc:
            check("matched no output device" in str(exc), str(exc))
        else:
            raise AssertionError("an unmatched name should raise")

        try:
            wl.resolve_endpoint("Realtek")  # matches Speakers and Headphones
        except ValueError as exc:
            check("matches several" in str(exc), str(exc))
        else:
            raise AssertionError("an ambiguous name should raise")


@case("selection", "no-devices", "a machine with no outputs reports it clearly")
def test_select_no_devices():
    with FakeEnumeration([]):
        try:
            wl.resolve_endpoint(None)
        except wl.WasapiError as exc:
            check("no active audio output endpoints" in str(exc), str(exc))
            return
        raise AssertionError("an empty device list must raise WasapiError")


# --------------------------------------------------------------------------
# live device tests
# --------------------------------------------------------------------------

def require_endpoint() -> wl.AudioEndpoint:
    endpoint = wl.default_output_endpoint()
    if endpoint is None:
        raise Skip("no active output endpoint on this machine")
    return endpoint


@case("device", "enumerate", "real endpoints carry stable Core Audio IDs")
def test_enumerate_real():
    endpoints = wl.enumerate_output_endpoints()
    if not endpoints:
        raise Skip("no active output endpoints")
    ids = set()
    for endpoint in endpoints:
        check(endpoint.endpoint_id.startswith("{"),
              f"unexpected endpoint ID {endpoint.endpoint_id!r}")
        check(endpoint.endpoint_id not in ids, "endpoint IDs must be unique")
        ids.add(endpoint.endpoint_id)
        check(endpoint.name.strip() != "", "endpoint must have a name")
    check(sum(e.is_default for e in endpoints) <= 1,
          "at most one endpoint can be the default")
    # Stable across calls: the same IDs come back in the same order.
    again = wl.enumerate_output_endpoints()
    check([e.endpoint_id for e in again] == [e.endpoint_id for e in endpoints],
          "enumeration order must be stable within a session")


@case("device", "stereo-mix", "Stereo Mix is detected if present (absence is fine)")
def test_stereo_mix_detection():
    stereo_mix = wl.find_stereo_mix()
    inputs = wl.enumerate_input_endpoints()
    if stereo_mix is None:
        print(f"      (no Stereo Mix among {len(inputs)} capture endpoints -- "
              "normal on most machines)")
    else:
        check(stereo_mix in inputs, "Stereo Mix must come from the input list")
        print(f"      (found {stereo_mix.name!r})")


@case("device", "open-close", "a loopback stream opens at native format and closes")
def test_open_close():
    endpoint = require_endpoint()
    with wl.LoopbackRecorder(endpoint) as recorder:
        fmt = recorder.format
        check(fmt.sample_rate >= 8000, f"implausible rate {fmt.sample_rate}")
        check(fmt.channels >= 1, f"implausible channel count {fmt.channels}")
        check(fmt.block_align == fmt.channels * fmt.bits_per_sample // 8,
              f"block_align {fmt.block_align} inconsistent with {fmt.describe()}")
        print(f"      (native: {fmt.describe()})")


@case("device", "restart", "capture can be stopped and restarted repeatedly")
def test_restart():
    """Re-opening is what happens after a device change, so a leak or a
    missed CoUninitialize here would show up as a failure to recover."""
    endpoint = require_endpoint()
    for attempt in range(5):
        with wl.LoopbackRecorder(endpoint) as recorder:
            recorder.read(timeout=0.05)
        time.sleep(0.02)


@case("device", "switch", "capture can move between endpoints")
def test_switch_devices():
    endpoints = wl.enumerate_output_endpoints()
    if len(endpoints) < 2:
        raise Skip(f"only {len(endpoints)} output endpoint(s) on this machine")
    for endpoint in endpoints[:3]:
        with wl.LoopbackRecorder(endpoint) as recorder:
            recorder.read(timeout=0.05)
            print(f"      ({endpoint.name}: {recorder.format.describe()})")


@case("device", "bad-endpoint", "a vanished endpoint raises, never crashes")
def test_bad_endpoint():
    """The disconnected-device case: an ID that no longer resolves must come
    back as a catchable WasapiError, not a native abort."""
    ghost = wl.AudioEndpoint(
        endpoint_id="{0.0.0.00000000}.{00000000-0000-0000-0000-000000000000}",
        name="Device That Went Away", index=0,
    )
    try:
        with wl.LoopbackRecorder(ghost):
            pass
    except wl.WasapiError as exc:
        check(exc.hresult != 0, "WasapiError must carry an HRESULT")
        return
    raise AssertionError("opening a nonexistent endpoint must raise WasapiError")


@case("device", "read-shape", "reads come back as (frames, native channels)")
def test_read_shape():
    endpoint = require_endpoint()
    with wl.LoopbackRecorder(endpoint) as recorder:
        channels = recorder.format.channels
        for _ in range(20):
            frames = recorder.read(timeout=0.05)
            check(frames.ndim == 2, f"expected 2D, got shape {frames.shape}")
            check(frames.shape[1] == channels,
                  f"expected {channels} channels, got {frames.shape[1]}")
            check(frames.dtype == np.float32, f"expected float32, got {frames.dtype}")
            check(np.all(np.isfinite(frames)), "capture produced NaN or inf")


@case("device", "silence-pacing", "an idle endpoint still paces the stream")
def test_silence_pacing():
    """WASAPI hands back no packets at all while nothing is playing. If that
    stalled the reader, the backend's VAD would see a frozen clock and the
    whole pipeline would hang on a quiet moment."""
    endpoint = require_endpoint()
    with wl.LoopbackRecorder(endpoint) as recorder:
        rate = recorder.format.sample_rate
        collected = 0
        start = time.perf_counter()
        while time.perf_counter() - start < 2.0:
            frames = recorder.read(timeout=0.05)
            collected += frames.shape[0]
            if frames.shape[0] == 0:
                time.sleep(0.005)
        elapsed = time.perf_counter() - start
        observed = collected / elapsed
        check(observed > rate * 0.85,
              f"only {observed:.0f} frames/s against a {rate} Hz device; "
              "an idle endpoint is starving the stream")


# --------------------------------------------------------------------------
# runtime (long) tests
# --------------------------------------------------------------------------

@case("device", "suspend-recovery",
      "a long stall fills bounded silence instead of exhausting memory")
def test_suspend_recovery():
    """After a laptop suspends, wall-clock jumps by minutes or hours. Filling
    that literally would ask for a multi-gigabyte array on resume."""
    endpoint = require_endpoint()
    with wl.LoopbackRecorder(endpoint) as recorder:
        recorder.read(timeout=0.05)
        # Pretend the process was frozen for an hour.
        recorder._silence_clock = time.perf_counter() - 3600.0
        frames = recorder.read(timeout=0.05)
        ceiling = int(recorder._MAX_SILENCE_FILL_SECONDS
                      * recorder.format.sample_rate)
        check(frames.shape[0] <= ceiling,
              f"filled {frames.shape[0]} frames after a 1h stall; "
              f"the cap is {ceiling}")


@case("runtime", "sustained", "30 seconds of continuous capture at 16kHz mono")
def test_sustained_capture():
    endpoint = require_endpoint()
    duration = 30.0
    with wl.LoopbackRecorder(endpoint) as recorder:
        fmt = recorder.format
        resampler = wl.Resampler(fmt.sample_rate, 16000)
        print(f"      capturing {duration:.0f}s from {endpoint.name} "
              f"({fmt.describe()}) ...")
        emitted = 0
        frame_bytes = 640
        pending = bytearray()
        wire_frames = 0
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            frames = recorder.read(timeout=0.05)
            if frames.shape[0] == 0:
                time.sleep(0.005)
                continue
            out = resampler.process(wl.downmix_to_mono(frames))
            if out.size:
                check(np.all(np.isfinite(out)), "resampler produced NaN or inf")
                emitted += out.size
                pending.extend(wl.float_to_pcm16(out))
                while len(pending) >= frame_bytes:
                    del pending[:frame_bytes]
                    wire_frames += 1
        elapsed = time.perf_counter() - start

    observed = emitted / elapsed
    drift = 100 * (observed - 16000) / 16000
    print(f"      {emitted} samples in {elapsed:.1f}s = {observed:.0f}/s "
          f"({drift:+.2f}% drift), {wire_frames} wire frames, "
          f"{recorder.discontinuities} discontinuities")
    check(abs(drift) < 2.0,
          f"output rate drifted {drift:+.2f}% from 16000 Hz over {elapsed:.0f}s")
    check(wire_frames > duration * 45,
          f"only {wire_frames} 20ms frames in {elapsed:.0f}s")


@case("runtime", "restart-cycle", "stop and restart 10 times without leaking")
def test_restart_cycle():
    endpoint = require_endpoint()
    for cycle in range(10):
        with wl.LoopbackRecorder(endpoint) as recorder:
            resampler = wl.Resampler(recorder.format.sample_rate, 16000)
            deadline = time.perf_counter() + 0.4
            while time.perf_counter() < deadline:
                frames = recorder.read(timeout=0.05)
                if frames.shape[0]:
                    resampler.process(wl.downmix_to_mono(frames))
                else:
                    time.sleep(0.005)


# --------------------------------------------------------------------------
# fallback hierarchy (client/test_loopback_stream.py)
# --------------------------------------------------------------------------

import test_loopback_stream as tls  # noqa: E402  (after wl, by design)


class FakeRecorder:
    """A LoopbackRecorder stand-in that fails for chosen endpoint names.

    Lets the fallback ladder be exercised on a machine with one working
    speaker, which is otherwise the one configuration that can never reach
    the interesting branches.
    """

    failing: set[str] = set()
    opened: list[tuple[str, bool]] = []

    def __init__(self, endpoint, buffer_ms=200, loopback=True, **kwargs):
        self.endpoint = endpoint
        self.loopback = loopback
        self.discontinuities = 0
        self._format = wl.StreamFormat(48000, 2, 32, True, 0x3, 8)

    @property
    def format(self):
        return self._format

    def __enter__(self):
        if self.endpoint.name in FakeRecorder.failing:
            raise wl.WasapiError(wl.AUDCLNT_E_DEVICE_IN_USE,
                                 "IAudioClient::Initialize(loopback)")
        FakeRecorder.opened.append((self.endpoint.name, self.loopback))
        return self

    def __exit__(self, *exc):
        return False

    def read(self, timeout=0.5):
        return np.zeros((0, 2), dtype=np.float32)


class FakeCaptureWorld:
    """Patch out every device-touching entry point the fallback ladder uses."""

    def __init__(self, layout=TYPICAL_LAYOUT, failing=(), stereo_mix=None):
        self.enumeration = FakeEnumeration(layout)
        self.failing = set(failing)
        self.stereo_mix = stereo_mix
        self._saved = {}

    def __enter__(self):
        self.enumeration.__enter__()
        self._saved = {
            "LoopbackRecorder": wl.LoopbackRecorder,
            "find_stereo_mix": wl.find_stereo_mix,
            "tls_enumerate": tls.wl.enumerate_output_endpoints,
        }
        FakeRecorder.failing = self.failing
        FakeRecorder.opened = []
        wl.LoopbackRecorder = FakeRecorder
        wl.find_stereo_mix = lambda: self.stereo_mix
        return self

    def __exit__(self, *exc):
        wl.LoopbackRecorder = self._saved["LoopbackRecorder"]
        wl.find_stereo_mix = self._saved["find_stereo_mix"]
        self.enumeration.__exit__()


def make_capture(selector=None, allow_stereo_mix=False):
    """A LoopbackCapture built for _open() only -- no loop, queue or thread."""
    capture = tls.LoopbackCapture.__new__(tls.LoopbackCapture)
    capture._selector = selector
    capture._buffer_ms = 100
    capture._allow_stereo_mix = allow_stereo_mix
    capture._follow_default = selector is None
    return capture


STEREO_MIX = wl.AudioEndpoint(endpoint_id="{0.0.1.0}.{stereo-mix}",
                              name="Stereo Mix (Realtek(R) Audio)", index=0)


@case("fallback", "primary", "a healthy endpoint is used with no fallback")
def test_fallback_primary():
    with FakeCaptureWorld():
        recorder, report = make_capture()._open()
        check(report.backend == "WASAPI loopback", report.backend)
        check(report.attempts == [], f"unexpected failures: {report.attempts}")
        check(recorder.loopback is True, "primary path must use loopback")
        check(report.native_rate == 48000 and report.native_channels == 2,
              "report should carry the negotiated native format")


@case("fallback", "alternate-endpoint",
      "a dead default falls through to another output endpoint")
def test_fallback_alternate():
    dead = TYPICAL_LAYOUT[0][0]
    with FakeCaptureWorld(failing=[dead]):
        recorder, report = make_capture()._open()
        check(recorder.endpoint.name != dead, "must not use the failing endpoint")
        check("alternate" in report.backend, report.backend)
        check(len(report.attempts) == 1, f"attempts: {report.attempts}")
        check(dead in report.attempts[0], report.attempts[0])
        check(recorder.loopback is True, "fallback must still be loopback capture")


@case("fallback", "pinned-device-never-switches",
      "an explicitly chosen device is never silently swapped")
def test_fallback_pinned():
    """If the user named a device and it fails, capturing a *different*
    speaker would be as wrong as capturing the microphone -- it would record
    audio they are not listening to, with no indication."""
    dead = TYPICAL_LAYOUT[0][0]
    with FakeCaptureWorld(failing=[dead]):
        try:
            make_capture(selector=0)._open()
        except tls.CaptureUnavailable as exc:
            check(dead in str(exc), "the error must name the device that failed")
            return
        raise AssertionError("a pinned device that fails must not fall through")


@case("fallback", "stereo-mix-opt-in",
      "Stereo Mix is used only when --allow-stereo-mix is passed")
def test_fallback_stereo_mix():
    every_name = [name for name, _ in TYPICAL_LAYOUT]

    # Without the flag: no Stereo Mix, even though one exists.
    with FakeCaptureWorld(failing=every_name, stereo_mix=STEREO_MIX):
        try:
            make_capture()._open()
        except tls.CaptureUnavailable:
            pass
        else:
            raise AssertionError("Stereo Mix must not be used without the flag")

    # With the flag: used, and opened as a capture device, not a loopback.
    with FakeCaptureWorld(failing=every_name, stereo_mix=STEREO_MIX):
        recorder, report = make_capture(allow_stereo_mix=True)._open()
        check("Stereo Mix" in report.backend, report.backend)
        check(recorder.loopback is False,
              "Stereo Mix is a real capture endpoint, not a loopback")
        check(len(report.attempts) == len(every_name),
              f"every output endpoint should have been tried: {report.attempts}")


@case("fallback", "missing-stereo-mix",
      "--allow-stereo-mix on a machine without one still fails cleanly")
def test_fallback_missing_stereo_mix():
    with FakeCaptureWorld(failing=[n for n, _ in TYPICAL_LAYOUT], stereo_mix=None):
        try:
            make_capture(allow_stereo_mix=True)._open()
        except tls.CaptureUnavailable as exc:
            check("no such recording device" in str(exc), str(exc))
            return
        raise AssertionError("a missing Stereo Mix must not silently succeed")


@case("fallback", "never-microphone", "a microphone is never a capture candidate")
def test_fallback_never_microphone():
    """The one outcome worse than failing: streaming the candidate's own
    voice to the backend while they believe it is the interviewer's."""
    microphone = wl.AudioEndpoint(endpoint_id="{0.0.1.0}.{mic}",
                                  name="Microphone (Realtek(R) Audio)", index=0)
    with FakeCaptureWorld(failing=[n for n, _ in TYPICAL_LAYOUT],
                          stereo_mix=None) as world:
        # Even if a microphone somehow reaches the Stereo Mix slot, the flag
        # is off, so it must never be opened.
        world.stereo_mix = microphone
        capture = make_capture()
        try:
            capture._open()
        except tls.CaptureUnavailable:
            pass
        check(FakeRecorder.opened == [],
              f"nothing should have opened, but did: {FakeRecorder.opened}")


@case("fallback", "error-report", "an unrecoverable failure explains itself")
def test_fallback_error_report():
    with FakeCaptureWorld(failing=[n for n, _ in TYPICAL_LAYOUT]):
        try:
            make_capture()._open()
        except tls.CaptureUnavailable as exc:
            text = str(exc)
            for required in ("Attempts, in order:",       # what was tried
                             "WASAPI loopback",            # attempted backend
                             "exclusive mode",             # reason for failure
                             "Active output endpoints",    # what exists
                             "What to try:"):              # suggested action
                check(required in text, f"error report is missing {required!r}")
            return
        raise AssertionError("expected CaptureUnavailable")


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    parser.add_argument("--only", default=None,
                        help="Run one group (layout, format, resampler, mixdown, "
                             "selection, device, runtime) or one case name.")
    parser.add_argument("--runtime", action="store_true",
                        help="Include the long runtime group (takes ~35s).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the capture layer's own log output. The "
                             "fallback cases provoke warnings on purpose, so "
                             "they are silenced unless asked for.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.CRITICAL,
                        format="[%(levelname)s] %(name)s: %(message)s")

    if args.list:
        for group, name, description, _ in CASES:
            print(f"{group:10s} {name:20s} {description}")
        return 0

    selected = []
    for group, name, description, fn in CASES:
        if args.only:
            if args.only not in (group, name):
                continue
        elif group == "runtime" and not args.runtime:
            continue
        selected.append((group, name, description, fn))

    passed = failed = skipped = 0
    current_group = None
    for group, name, description, fn in selected:
        if group != current_group:
            print(f"\n== {group} ==")
            current_group = group
        try:
            fn()
        except Skip as exc:
            print(f"  SKIP {name:22s} {description}\n       ({exc})")
            skipped += 1
        except Exception as exc:
            print(f"  FAIL {name:22s} {description}")
            for line in traceback.format_exception_only(type(exc), exc):
                print(f"       {line.rstrip()}")
            failed += 1
        else:
            print(f"  ok   {name:22s} {description}")
            passed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    if not args.runtime and not args.only:
        print("(runtime group skipped; pass --runtime for the 30s capture test)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
