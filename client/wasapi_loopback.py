"""Windows WASAPI loopback capture, implemented directly against Core Audio.

Why this module exists instead of `soundcard`
---------------------------------------------
`soundcard` 0.4.x describes WAVEFORMATEX/WAVEFORMATEXTENSIBLE to CFFI without
Windows' 1-byte struct packing. The real headers wrap those structs in
`pshpack1.h`, so `WAVEFORMATEX` is 18 bytes and `WAVEFORMATEXTENSIBLE` is 40;
unpacked, the compiler pads them to 20 and 44 and every field after `cbSize`
lands at the wrong offset:

    field            true offset   soundcard's offset
    Samples                  18                   20
    dwChannelMask            20                   24
    SubFormat                24                   28
    sizeof                   40                   44

`IAudioClient::GetMixFormat` returns a `CoTaskMemAlloc` block of exactly
`18 + cbSize` = 40 bytes, so that mismatch has two consequences:

  * reading `SubFormat` touches bytes 40..43, four bytes past the end of the
    allocation -- soundcard's own comment that "the last four bytes seem to
    vary randomly" is that out-of-bounds read observing heap metadata; and
  * writing `Samples.wValidBitsPerSample` lands on `dwChannelMask` instead,
    rewriting a valid stereo mask of 0x3 (FRONT_LEFT|FRONT_RIGHT) into 0x20
    (BACK_RIGHT alone) -- a mask whose popcount no longer matches nChannels.

That malformed format is then handed to `IAudioClient::Initialize` together
with AUTOCONVERTPCM, which is precisely the flag that asks the audio engine
to build a channel remap matrix from `dwChannelMask`. A mask naming fewer
channels than `nChannels` makes the converter and the endpoint's APO chain
disagree about buffer geometry, and the resulting overrun surfaces as
STATUS_HEAP_CORRUPTION (0xC0000374). Whether it is *detected* depends on heap
layout and on which APOs a given driver loads, which is why identical code
aborts on one PC and appears to work on another.

What this module does instead
-----------------------------
  * Correctly packed (`_pack_ = 1`) structures, asserted at import.
  * Opens the endpoint at its **native mix format**, byte for byte as
    `GetMixFormat` returned it. The format is never written to, so no
    malformed channel mask can reach the driver.
  * Passes only AUDCLNT_STREAMFLAGS_LOOPBACK -- no AUTOCONVERTPCM and no
    sample-rate-converter flags, so the engine's converter is never engaged.
  * Resamples to 16 kHz mono in Python, where a bug is a traceback rather
    than a native abort.
  * Identifies devices by Core Audio endpoint ID, never by a positional index.

Public API
----------
    enumerate_output_endpoints()   -> list[AudioEndpoint]
    default_output_endpoint()      -> AudioEndpoint | None
    resolve_endpoint(selector)     -> AudioEndpoint
    LoopbackRecorder(endpoint)     -> context manager, native-format frames
    Resampler(src_rate, dst_rate)  -> stateful float32 rate conversion
"""

from __future__ import annotations

import ctypes
import logging
import math
import time
from ctypes import POINTER, byref, c_void_p
from ctypes.wintypes import BYTE, DWORD, LPCWSTR, LPWSTR, UINT, WORD
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

_ole32 = ctypes.windll.ole32

# COM methods are declared returning c_long rather than ctypes.HRESULT on
# purpose: HRESULT makes ctypes raise OSError on any failure, and several
# calls here have failures that are ordinary control flow (no default device,
# a device that vanished mid-enumeration). We check every HRESULT explicitly.
_HRESULT = ctypes.c_long


# --------------------------------------------------------------------------
# COM plumbing
# --------------------------------------------------------------------------

class GUID(ctypes.Structure):
    _fields_ = [("Data1", DWORD), ("Data2", WORD), ("Data3", WORD),
                ("Data4", BYTE * 8)]

    def __init__(self, text: str | None = None):
        super().__init__()
        if text is not None:
            if _ole32.CLSIDFromString(ctypes.c_wchar_p(text), byref(self)) != 0:
                raise ValueError(f"bad GUID {text!r}")


CLSID_MMDeviceEnumerator = GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
IID_IMMDeviceEnumerator = GUID("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
IID_IAudioClient = GUID("{1CB9AD4C-DBFA-4C32-B178-C2F568A703B2}")
IID_IAudioCaptureClient = GUID("{C8ADBD64-E71E-48A0-A4DE-185C395CD317}")

CLSCTX_ALL = 23
EDATAFLOW_RENDER = 0
EDATAFLOW_CAPTURE = 1
EROLE_CONSOLE = 0
DEVICE_STATE_ACTIVE = 0x00000001

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_NOPERSIST = 0x00080000
AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY = 0x1
AUDCLNT_BUFFERFLAGS_SILENT = 0x2

S_OK = 0
S_FALSE = 1
RPC_E_CHANGED_MODE = 0x80010106
COINIT_MULTITHREADED = 0x0
COINIT_APARTMENTTHREADED = 0x2

AUDCLNT_E_DEVICE_INVALIDATED = 0x88890004
AUDCLNT_E_UNSUPPORTED_FORMAT = 0x88890008
AUDCLNT_E_DEVICE_IN_USE = 0x8889000A
AUDCLNT_E_SERVICE_NOT_RUNNING = 0x88890010
AUDCLNT_E_ENDPOINT_CREATE_FAILED = 0x8889001A
AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED = 0x8889000F

_ERROR_DETAIL = {
    AUDCLNT_E_DEVICE_INVALIDATED:
        "the device was removed, disabled or reconfigured",
    AUDCLNT_E_UNSUPPORTED_FORMAT:
        "the endpoint does not support the requested format",
    AUDCLNT_E_DEVICE_IN_USE:
        "the device is held in exclusive mode by another application",
    AUDCLNT_E_SERVICE_NOT_RUNNING:
        "the Windows Audio service is not running",
    AUDCLNT_E_ENDPOINT_CREATE_FAILED:
        "the audio endpoint could not be created",
    AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED:
        "exclusive mode is not permitted for this endpoint",
}


class WasapiError(RuntimeError):
    """A Core Audio call failed. `hresult` carries the raw HRESULT."""

    def __init__(self, hresult: int, context: str):
        self.hresult = hresult & 0xFFFFFFFF
        self.context = context
        message = f"{context} failed (HRESULT 0x{self.hresult:08X})"
        detail = _ERROR_DETAIL.get(self.hresult)
        if detail:
            message += f": {detail}"
        super().__init__(message)

    @property
    def device_invalidated(self) -> bool:
        """True when reopening on a fresh endpoint is the right response."""
        return self.hresult == AUDCLNT_E_DEVICE_INVALIDATED


def _check(hresult: int, context: str) -> None:
    if (hresult & 0xFFFFFFFF) not in (S_OK, S_FALSE):
        raise WasapiError(hresult, context)


class ComInitialized:
    """Per-thread `CoInitializeEx`.

    COM apartments are per thread, not per process. Capture runs on its own
    thread, so that thread joins an apartment itself instead of depending on
    whichever thread happened to import this module -- and leaves it again on
    the way out, so repeated start/stop cycles don't leak apartment refs.
    """

    def __init__(self, multithreaded: bool = True):
        self._flags = (COINIT_MULTITHREADED if multithreaded
                       else COINIT_APARTMENTTHREADED)
        self._owned = False

    def __enter__(self) -> "ComInitialized":
        hr = _ole32.CoInitializeEx(None, self._flags) & 0xFFFFFFFF
        if hr in (S_OK, S_FALSE):
            self._owned = True
        elif hr == RPC_E_CHANGED_MODE:
            # The thread is already in the other apartment kind. Usable, but
            # not ours to tear down.
            self._owned = False
        else:
            raise WasapiError(hr, "CoInitializeEx")
        return self

    def __exit__(self, *exc) -> None:
        if self._owned:
            _ole32.CoUninitialize()
            self._owned = False


def _vtbl(ptr: c_void_p, index: int, *argtypes):
    """Bind method `index` of the COM vtable behind `ptr`."""
    vtable = ctypes.cast(ptr, POINTER(POINTER(c_void_p)))[0]
    return ctypes.WINFUNCTYPE(_HRESULT, c_void_p, *argtypes)(vtable[index])


def _release(ptr: c_void_p) -> None:
    if ptr:
        ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(
            ctypes.cast(ptr, POINTER(POINTER(c_void_p)))[0][2]
        )(ptr)


# --------------------------------------------------------------------------
# Audio format structures -- packed exactly as Windows declares them
# --------------------------------------------------------------------------

class WAVEFORMATEX(ctypes.Structure):
    """mmreg.h, declared inside pshpack1.h -- 18 bytes, 1-byte aligned."""
    _pack_ = 1
    _fields_ = [
        ("wFormatTag", WORD),
        ("nChannels", WORD),
        ("nSamplesPerSec", DWORD),
        ("nAvgBytesPerSec", DWORD),
        ("nBlockAlign", WORD),
        ("wBitsPerSample", WORD),
        ("cbSize", WORD),
    ]


class WAVEFORMATEXTENSIBLE(ctypes.Structure):
    """mmreg.h -- 40 bytes, 1-byte aligned."""
    _pack_ = 1
    _fields_ = [
        ("Format", WAVEFORMATEX),
        ("wValidBitsPerSample", WORD),
        ("dwChannelMask", DWORD),
        ("SubFormat", GUID),
    ]


# The whole point of this module. If these ever drift, loopback capture is
# writing into someone else's heap block again -- fail at import, loudly.
assert ctypes.sizeof(WAVEFORMATEX) == 18, "WAVEFORMATEX must pack to 18 bytes"
assert ctypes.sizeof(WAVEFORMATEXTENSIBLE) == 40, "WAVEFORMATEXTENSIBLE must pack to 40 bytes"
assert WAVEFORMATEXTENSIBLE.dwChannelMask.offset == 20
assert WAVEFORMATEXTENSIBLE.SubFormat.offset == 24

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

KSDATAFORMAT_SUBTYPE_PCM = GUID("{00000001-0000-0010-8000-00AA00389B71}")
KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = GUID("{00000003-0000-0010-8000-00AA00389B71}")


def _guid_bytes(guid: GUID) -> bytes:
    return ctypes.string_at(byref(guid), ctypes.sizeof(GUID))


@dataclass(frozen=True)
class StreamFormat:
    """The format an endpoint actually runs at, as the device reported it."""
    sample_rate: int
    channels: int
    bits_per_sample: int
    is_float: bool
    channel_mask: int
    block_align: int

    def describe(self) -> str:
        kind = "float" if self.is_float else "int"
        return (f"{self.sample_rate} Hz, {self.channels} ch, "
                f"{kind}{self.bits_per_sample}, "
                f"channel mask 0x{self.channel_mask:X}")

    @property
    def sample_format(self) -> str:
        return ("float" if self.is_float else "int") + str(self.bits_per_sample)


def _parse_format(pwfx: c_void_p) -> StreamFormat:
    """Read a WAVEFORMATEX* without ever writing to it.

    The extended fields are only dereferenced when `cbSize` proves they are
    present in the allocation, so this can never read past the end of the
    block `GetMixFormat` handed us -- the bug that started all this.
    """
    base = ctypes.cast(pwfx, POINTER(WAVEFORMATEX))[0]
    tag, bits, channels = base.wFormatTag, base.wBitsPerSample, base.nChannels
    mask = 0

    if tag == WAVE_FORMAT_EXTENSIBLE:
        if base.cbSize < 22:
            raise WasapiError(
                AUDCLNT_E_UNSUPPORTED_FORMAT,
                f"mix format claims WAVE_FORMAT_EXTENSIBLE but cbSize={base.cbSize}",
            )
        ext = ctypes.cast(pwfx, POINTER(WAVEFORMATEXTENSIBLE))[0]
        mask = ext.dwChannelMask
        subformat = _guid_bytes(ext.SubFormat)
        if subformat == _guid_bytes(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT):
            is_float = True
        elif subformat == _guid_bytes(KSDATAFORMAT_SUBTYPE_PCM):
            is_float = False
        else:
            raise WasapiError(AUDCLNT_E_UNSUPPORTED_FORMAT,
                              "mix format uses an unrecognised KSDATAFORMAT subtype")
    elif tag == WAVE_FORMAT_IEEE_FLOAT:
        is_float = True
    elif tag == WAVE_FORMAT_PCM:
        is_float = False
    else:
        raise WasapiError(AUDCLNT_E_UNSUPPORTED_FORMAT,
                          f"mix format tag 0x{tag:04X} is neither PCM nor float")

    supported = (32, 64) if is_float else (8, 16, 24, 32)
    if bits not in supported:
        kind = "float" if is_float else "int"
        raise WasapiError(AUDCLNT_E_UNSUPPORTED_FORMAT,
                          f"{kind}{bits} samples are not supported")
    if channels < 1:
        raise WasapiError(AUDCLNT_E_UNSUPPORTED_FORMAT,
                          f"mix format reports {channels} channels")

    return StreamFormat(
        sample_rate=int(base.nSamplesPerSec),
        channels=int(channels),
        bits_per_sample=int(bits),
        is_float=is_float,
        channel_mask=int(mask),
        block_align=int(base.nBlockAlign),
    )


# --------------------------------------------------------------------------
# Device enumeration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioEndpoint:
    """A Windows audio endpoint, identified by its stable Core Audio ID.

    `endpoint_id` is the string Windows itself uses (e.g.
    ``{0.0.0.00000000}.{guid}``). It survives reboots, index reshuffles and
    driver reinstalls, none of which a positional index does. `index` is a
    presentation detail for the CLI and is only meaningful within one
    enumeration.
    """
    endpoint_id: str
    name: str
    is_default: bool = False
    index: int = -1

    def __str__(self) -> str:
        return f"{self.name}{' (default)' if self.is_default else ''}"


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", DWORD)]


PKEY_Device_FriendlyName = ("{A45C254E-DF1C-4EFD-8020-67D146A850E0}", 14)

VT_LPWSTR = 31


def _propvariant_string(store: c_void_p, fmtid: str, pid: int) -> str:
    # PROPVARIANT is 16 bytes on x86 and 24 on x64; 32 is safely over-sized.
    # vt sits at offset 0 and the union at offset 8 on both.
    propvariant = (ctypes.c_byte * 32)()
    key = PROPERTYKEY(fmtid=GUID(fmtid), pid=pid)

    get_value = _vtbl(store, 5, POINTER(PROPERTYKEY), c_void_p)
    if get_value(store, byref(key), ctypes.cast(propvariant, c_void_p)) != S_OK:
        return ""
    try:
        if ctypes.cast(propvariant, POINTER(WORD))[0] != VT_LPWSTR:
            return ""
        value = ctypes.cast(ctypes.byref(propvariant, 8), POINTER(c_void_p))[0]
        return ctypes.wstring_at(value) if value else ""
    finally:
        _ole32.PropVariantClear(ctypes.cast(propvariant, c_void_p))


class _Enumerator:
    """Short-lived IMMDeviceEnumerator wrapper.

    Built per operation rather than cached, so every listing is the live one:
    endpoints appear and disappear while the app runs.
    """

    def __init__(self):
        self.ptr = c_void_p()
        hr = _ole32.CoCreateInstance(
            byref(CLSID_MMDeviceEnumerator), None, CLSCTX_ALL,
            byref(IID_IMMDeviceEnumerator), byref(self.ptr),
        )
        _check(hr, "CoCreateInstance(MMDeviceEnumerator)")

    def __enter__(self) -> "_Enumerator":
        return self

    def __exit__(self, *exc) -> None:
        _release(self.ptr)
        self.ptr = c_void_p()

    def default_device(self, flow: int = EDATAFLOW_RENDER,
                       role: int = EROLE_CONSOLE) -> c_void_p:
        """The current default endpoint, or a null pointer if there is none.

        A machine with every output disabled genuinely has no default; that
        is a state to report, not an exception to raise.
        """
        device = c_void_p()
        get_default = _vtbl(self.ptr, 4, ctypes.c_int, ctypes.c_int,
                            POINTER(c_void_p))
        if get_default(self.ptr, flow, role, byref(device)) != S_OK:
            return c_void_p()
        return device

    def device_by_id(self, endpoint_id: str) -> c_void_p:
        device = c_void_p()
        get_device = _vtbl(self.ptr, 5, LPCWSTR, POINTER(c_void_p))
        _check(get_device(self.ptr, endpoint_id, byref(device)),
               f"IMMDeviceEnumerator::GetDevice({endpoint_id})")
        return device

    def collection(self, flow: int, state_mask: int = DEVICE_STATE_ACTIVE) -> c_void_p:
        collection = c_void_p()
        enum_endpoints = _vtbl(self.ptr, 3, ctypes.c_int, DWORD, POINTER(c_void_p))
        _check(enum_endpoints(self.ptr, flow, state_mask, byref(collection)),
               "IMMDeviceEnumerator::EnumAudioEndpoints")
        return collection


def _device_id(device: c_void_p) -> str:
    out = LPWSTR()
    _check(_vtbl(device, 5, POINTER(LPWSTR))(device, byref(out)),
           "IMMDevice::GetId")
    try:
        return ctypes.wstring_at(out)
    finally:
        _ole32.CoTaskMemFree(out)


def _device_name(device: c_void_p) -> str:
    store = c_void_p()
    open_store = _vtbl(device, 4, DWORD, POINTER(c_void_p))
    if open_store(device, 0, byref(store)) != S_OK:  # STGM_READ
        return "Unknown device"
    try:
        return _propvariant_string(store, *PKEY_Device_FriendlyName) or "Unknown device"
    finally:
        _release(store)


def _enumerate(flow: int) -> list[AudioEndpoint]:
    with ComInitialized(), _Enumerator() as enum:
        default_id = ""
        default = enum.default_device(flow)
        if default:
            try:
                default_id = _device_id(default)
            finally:
                _release(default)

        collection = enum.collection(flow)
        try:
            count = UINT()
            _check(_vtbl(collection, 3, POINTER(UINT))(collection, byref(count)),
                   "IMMDeviceCollection::GetCount")
            item = _vtbl(collection, 4, UINT, POINTER(c_void_p))

            endpoints: list[AudioEndpoint] = []
            for i in range(count.value):
                device = c_void_p()
                if item(collection, i, byref(device)) != S_OK:
                    continue  # unplugged between GetCount and Item
                try:
                    endpoint_id = _device_id(device)
                    endpoints.append(AudioEndpoint(
                        endpoint_id=endpoint_id,
                        name=_device_name(device),
                        is_default=(endpoint_id == default_id),
                        index=len(endpoints),
                    ))
                finally:
                    _release(device)
            return endpoints
        finally:
            _release(collection)


def enumerate_output_endpoints() -> list[AudioEndpoint]:
    """Every active render endpoint, in Windows' own enumeration order.

    Covers speakers, headphones, HDMI/DisplayPort, USB and Bluetooth alike:
    to Core Audio they are all render endpoints, and none needs a special
    case here.
    """
    return _enumerate(EDATAFLOW_RENDER)


def enumerate_input_endpoints() -> list[AudioEndpoint]:
    """Every active capture endpoint (microphones, line-in, Stereo Mix)."""
    return _enumerate(EDATAFLOW_CAPTURE)


# Stereo Mix is a driver-provided capture endpoint whose name is localised
# and vendor-specific, so matching it means matching names. These are the
# forms Realtek, VIA, Conexant and Windows' own translations actually ship.
_STEREO_MIX_NAMES = (
    "stereo mix", "stereomix", "stereo-mix", "what u hear", "what you hear",
    "wave out mix", "waveout mix", "loopback", "mixage stéréo", "stereomix",
    "stereo-mixning", "summe", "stereo mix ", "mezcla estéreo",
)


def find_stereo_mix() -> AudioEndpoint | None:
    """A Stereo Mix style capture endpoint, if the driver exposes one.

    Returns None far more often than not: most modern drivers ship Stereo
    Mix disabled or omit it entirely, which is exactly why it is a last
    resort and never the default path.
    """
    for endpoint in enumerate_input_endpoints():
        lowered = endpoint.name.lower()
        if any(name in lowered for name in _STEREO_MIX_NAMES):
            return endpoint
    return None


def default_output_endpoint() -> AudioEndpoint | None:
    """The endpoint Windows is playing through right now, or None."""
    for endpoint in enumerate_output_endpoints():
        if endpoint.is_default:
            return endpoint
    return None


def default_output_endpoint_id() -> str:
    """The current default render endpoint's ID, or "" if there is none.

    Cheap enough to poll: this is how the capture loop notices that the user
    switched from speakers to a headset while the app was running.
    """
    try:
        with ComInitialized(), _Enumerator() as enum:
            device = enum.default_device(EDATAFLOW_RENDER)
            if not device:
                return ""
            try:
                return _device_id(device)
            finally:
                _release(device)
    except WasapiError:
        return ""


def resolve_endpoint(selector: "int | str | None") -> AudioEndpoint:
    """Turn a CLI `--device` value into a real endpoint.

    Accepts a logical index (as printed by --list-devices), a Core Audio
    endpoint ID, or a case-insensitive name substring. None means "whatever
    Windows is using right now", which is re-resolved on every call rather
    than remembered.
    """
    endpoints = enumerate_output_endpoints()
    if not endpoints:
        raise WasapiError(
            AUDCLNT_E_ENDPOINT_CREATE_FAILED,
            "no active audio output endpoints found -- check that a playback "
            "device is enabled in Windows Sound settings",
        )

    if selector is None:
        for endpoint in endpoints:
            if endpoint.is_default:
                return endpoint
        return endpoints[0]

    if isinstance(selector, bool):
        raise TypeError("--device must be an index, endpoint ID or name")

    if isinstance(selector, int):
        if not 0 <= selector < len(endpoints):
            raise ValueError(
                f"--device {selector} is out of range (0..{len(endpoints) - 1}). "
                "Run --list-devices to see valid indices."
            )
        return endpoints[selector]

    text = str(selector).strip()
    if text.isdigit():
        return resolve_endpoint(int(text))
    for endpoint in endpoints:
        if endpoint.endpoint_id == text:
            return endpoint
    matches = [e for e in endpoints if text.lower() in e.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(repr(e.name) for e in matches)
        raise ValueError(f"--device {text!r} matches several devices: {names}")
    raise ValueError(f"--device {text!r} matched no output device. Run --list-devices.")


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

class Resampler:
    """Streaming rate conversion for one mono float32 channel.

    An anti-alias FIR at the source rate, then fractional linear
    interpolation. Because everything above the destination Nyquist is gone
    *before* interpolation, the interpolator's error sits well below the
    noise floor of the speech this feeds, at the cost of one convolution per
    chunk and no native code at all.

    Filter history and fractional read position persist across calls, so
    consecutive chunks join without a click or a dropped sample.
    """

    def __init__(self, src_rate: int, dst_rate: int):
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self._step = self.src_rate / self.dst_rate
        self._passthrough = self.src_rate == self.dst_rate

        if self._passthrough:
            self._taps = np.zeros(0, dtype=np.float32)
            self._history = np.zeros(0, dtype=np.float32)
        else:
            self._taps = self._design_lowpass()
            self._history = np.zeros(len(self._taps) - 1, dtype=np.float32)

        self._phase = 0.0
        self._tail = np.zeros(0, dtype=np.float32)

    # A Hamming-windowed sinc has a transition band of roughly 3.3 * fs / N.
    # Sizing N from the width we can afford, rather than from a fixed
    # multiplier, is what keeps a 192 kHz endpoint as alias-free as a 48 kHz
    # one instead of degrading with the ratio.
    _TRANSITION_CONSTANT = 3.3

    def _design_lowpass(self) -> np.ndarray:
        """Windowed-sinc low-pass whose stopband starts below the
        destination Nyquist, at any source rate."""
        nyquist = min(self.src_rate, self.dst_rate) / 2.0
        cutoff = nyquist * 0.90  # -6 dB point, leaving room for the skirt

        # Fit the upper half of the transition band between `cutoff` and
        # `nyquist`, so nothing that could fold back survives it.
        max_transition = 2.0 * (nyquist - cutoff)
        numtaps = int(np.clip(
            round(self._TRANSITION_CONSTANT * self.src_rate / max_transition),
            63, 1023,
        )) | 1  # odd => exactly linear phase

        n = np.arange(numtaps, dtype=np.float64) - (numtaps - 1) / 2.0
        fc = cutoff / self.src_rate
        taps = 2 * fc * np.sinc(2 * fc * n) * np.hamming(numtaps)
        taps /= taps.sum()
        return taps.astype(np.float32)

    @property
    def description(self) -> str:
        if self._passthrough:
            return "none (device already at target rate)"
        return (f"{len(self._taps)}-tap windowed-sinc low-pass + linear "
                f"interpolation ({self.src_rate} -> {self.dst_rate} Hz)")

    def process(self, mono: np.ndarray) -> np.ndarray:
        """Resample one chunk. Returns float32; may be empty for tiny inputs."""
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        if self._passthrough:
            return mono
        if mono.size == 0 and self._tail.size == 0:
            return np.zeros(0, dtype=np.float32)

        padded = np.concatenate((self._history, mono))
        if padded.size < self._taps.size:
            self._history = padded
            return np.zeros(0, dtype=np.float32)
        filtered = np.convolve(padded, self._taps, mode="valid").astype(np.float32)
        self._history = padded[-(self._taps.size - 1):]

        # One sample of overlap is retained so the next chunk can interpolate
        # across the boundary instead of restarting at it.
        available = np.concatenate((self._tail, filtered))
        if available.size < 2:
            self._tail = available
            return np.zeros(0, dtype=np.float32)

        # Every output needs both available[left] and available[left + 1], so
        # read positions must stay strictly below the final index; whatever
        # is left over is carried into the next chunk rather than dropped.
        limit = available.size - 1
        count = int(math.ceil((limit - self._phase) / self._step))
        if count <= 0:
            self._tail = available
            return np.zeros(0, dtype=np.float32)

        positions = self._phase + self._step * np.arange(count, dtype=np.float64)
        left = np.floor(positions).astype(np.intp)
        frac = (positions - left).astype(np.float32)
        out = available[left] * (1.0 - frac) + available[left + 1] * frac

        next_position = positions[-1] + self._step
        consumed = min(int(math.floor(next_position)), available.size - 1)
        self._phase = next_position - consumed
        self._tail = available[consumed:]
        return out.astype(np.float32)


def downmix_to_mono(frames: np.ndarray) -> np.ndarray:
    """Average every channel into one.

    A plain mean rather than a weighted surround downmix: this feeds speech
    recognition, where any channel may carry dialogue, and a 5.1 centre
    weighting would attenuate the rest for no benefit.
    """
    if frames.ndim == 1:
        return frames.astype(np.float32, copy=False)
    if frames.shape[1] == 1:
        return frames[:, 0].astype(np.float32, copy=False)
    return frames.mean(axis=1, dtype=np.float32)


def float_to_pcm16(mono: np.ndarray) -> bytes:
    """Clip to [-1, 1] and convert to little-endian int16."""
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# --------------------------------------------------------------------------
# Loopback capture
# --------------------------------------------------------------------------

class LoopbackRecorder:
    """WASAPI loopback capture on one render endpoint, at its native format.

    Use as a context manager on the thread that will read from it: the COM
    apartment and the IAudioClient both belong to that thread.

        with LoopbackRecorder(endpoint) as rec:
            frames = rec.read()   # float32 (frames, channels) at rec.format
    """

    # Ceiling on synthesised silence per read, in seconds. See `read`.
    _MAX_SILENCE_FILL_SECONDS = 1.0

    def __init__(self, endpoint: AudioEndpoint, buffer_ms: int = 200,
                 silence_granularity_ms: int = 40, loopback: bool = True):
        self.endpoint = endpoint
        # False opens the endpoint as an ordinary capture device, which is
        # only correct for something that already carries the output mix
        # (Stereo Mix). Never point it at a microphone.
        self.loopback = loopback
        self._buffer_ms = max(50, int(buffer_ms))
        # How long the endpoint must stay quiet before `read` starts filling
        # the gap with silence. Small enough that an idle device still paces
        # the stream smoothly, large enough not to race normal packet jitter.
        self._silence_after = max(10, int(silence_granularity_ms)) / 1000.0
        self._com: ComInitialized | None = None
        self._client = c_void_p()
        self._capture = c_void_p()
        self._format: StreamFormat | None = None
        self._started = False
        self._first_packet = True
        self._silence_clock: float | None = None
        self._discontinuities = 0

    @property
    def format(self) -> StreamFormat:
        if self._format is None:
            raise RuntimeError("recorder is not open")
        return self._format

    @property
    def discontinuities(self) -> int:
        """How many times WASAPI reported a gap in the captured stream."""
        return self._discontinuities

    # -- lifecycle --------------------------------------------------------

    def __enter__(self) -> "LoopbackRecorder":
        self._com = ComInitialized()
        self._com.__enter__()
        try:
            self._open()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _open(self) -> None:
        with _Enumerator() as enum:
            device = enum.device_by_id(self.endpoint.endpoint_id)
            try:
                activate = _vtbl(device, 3, POINTER(GUID), DWORD, c_void_p,
                                 POINTER(c_void_p))
                _check(activate(device, byref(IID_IAudioClient), CLSCTX_ALL,
                                None, byref(self._client)),
                       "IMMDevice::Activate(IAudioClient)")
            finally:
                _release(device)

        # Ask the device what it is already running at and take that answer
        # verbatim. The pointer goes straight to Initialize without a single
        # field being written, so no malformed format -- in particular no
        # channel mask inconsistent with nChannels -- can reach the driver
        # or the endpoint's APO chain.
        pwfx = c_void_p()
        _check(_vtbl(self._client, 8, POINTER(c_void_p))(self._client, byref(pwfx)),
               "IAudioClient::GetMixFormat")
        try:
            self._format = _parse_format(pwfx)
            log.info("Native mix format for %r: %s",
                     self.endpoint.name, self._format.describe())

            # Shared mode, loopback flag only. Deliberately no AUTOCONVERTPCM
            # or SRC_DEFAULT_QUALITY: those engage the audio engine's
            # channel-and-rate converter, the component that reads
            # dwChannelMask and the one implicated in the 0xC0000374 aborts.
            # Rate and channel conversion happen in Python instead.
            flags = AUDCLNT_STREAMFLAGS_NOPERSIST
            if self.loopback:
                flags |= AUDCLNT_STREAMFLAGS_LOOPBACK
            duration = int(self._buffer_ms * 10_000)  # ms -> hectonanoseconds
            initialize = _vtbl(self._client, 3, ctypes.c_int, DWORD,
                               ctypes.c_longlong, ctypes.c_longlong,
                               c_void_p, c_void_p)
            mode = "loopback" if self.loopback else "capture"
            _check(initialize(self._client, AUDCLNT_SHAREMODE_SHARED, flags,
                              duration, 0, pwfx, None),
                   f"IAudioClient::Initialize({mode})")
        finally:
            _ole32.CoTaskMemFree(pwfx)

        _check(_vtbl(self._client, 14, POINTER(GUID), POINTER(c_void_p))(
                   self._client, byref(IID_IAudioCaptureClient), byref(self._capture)),
               "IAudioClient::GetService(IAudioCaptureClient)")

        # Bind the hot-path methods once; these run on every read.
        self._get_buffer = _vtbl(
            self._capture, 3, POINTER(POINTER(BYTE)), POINTER(UINT),
            POINTER(DWORD), c_void_p, c_void_p,
        )
        self._release_buffer = _vtbl(self._capture, 4, UINT)
        self._next_packet_size = _vtbl(self._capture, 5, POINTER(UINT))

        _check(_vtbl(self._client, 10)(self._client), "IAudioClient::Start")
        self._started = True
        self._silence_clock = time.perf_counter()

    def __exit__(self, *exc) -> None:
        if self._started:
            try:
                _vtbl(self._client, 11)(self._client)  # Stop
            except Exception:  # teardown must not mask the original failure
                log.debug("IAudioClient::Stop failed during teardown", exc_info=True)
            self._started = False
        _release(self._capture)
        self._capture = c_void_p()
        _release(self._client)
        self._client = c_void_p()
        if self._com is not None:
            self._com.__exit__(None, None, None)
            self._com = None

    # -- reading ----------------------------------------------------------

    def _decode(self, data_ptr, frames: int) -> np.ndarray:
        """Copy one WASAPI packet out as float32 (frames, channels).

        The copy is deliberate and immediate: the pointer is only valid until
        ReleaseBuffer, so a numpy view onto it would outlive its buffer.
        """
        fmt = self.format
        raw = ctypes.string_at(data_ptr, frames * fmt.block_align)

        if fmt.is_float:
            dtype = "<f4" if fmt.bits_per_sample == 32 else "<f8"
            samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        elif fmt.bits_per_sample == 16:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif fmt.bits_per_sample == 32:
            samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif fmt.bits_per_sample == 24:
            # Three packed bytes widened into the top of an int32, which sign
            # extends for free; the constant divisor undoes the shift.
            packed = np.frombuffer(raw, dtype=np.uint8)
            packed = packed[: (packed.size // 3) * 3].reshape(-1, 3)
            widened = np.zeros((packed.shape[0], 4), dtype=np.uint8)
            widened[:, 1:] = packed
            samples = widened.view("<i4").reshape(-1).astype(np.float32) / 2147483648.0
        else:  # 8-bit PCM is unsigned by definition
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

        usable = (samples.size // fmt.channels) * fmt.channels
        return samples[:usable].reshape(-1, fmt.channels).astype(np.float32, copy=True)

    def read(self, timeout: float = 0.5) -> np.ndarray:
        """Drain whatever WASAPI has ready, as float32 (frames, channels).

        A loopback endpoint delivers no packets at all while nothing is
        playing, so an idle device would otherwise stall the caller forever.
        After about one buffer's worth of quiet this synthesises exactly as
        many zero frames as wall-clock says are missing, which keeps the
        downstream timeline honest without inventing audio while real
        packets are still flowing.
        """
        fmt = self.format
        blocks: list[np.ndarray] = []
        captured = 0
        deadline = time.perf_counter() + max(0.0, timeout)

        while True:
            packet = UINT()
            hr = self._next_packet_size(self._capture, byref(packet))
            if (hr & 0xFFFFFFFF) != S_OK:
                raise WasapiError(hr, "IAudioCaptureClient::GetNextPacketSize")
            if packet.value == 0:
                break

            data_ptr = POINTER(BYTE)()
            frames = UINT()
            flags = DWORD()
            hr = self._get_buffer(self._capture, byref(data_ptr), byref(frames),
                                  byref(flags), None, None)
            if (hr & 0xFFFFFFFF) != S_OK:
                raise WasapiError(hr, "IAudioCaptureClient::GetBuffer")

            try:
                count = frames.value
                if count:
                    if (flags.value & AUDCLNT_BUFFERFLAGS_SILENT) or not data_ptr:
                        block = np.zeros((count, fmt.channels), dtype=np.float32)
                    else:
                        block = self._decode(data_ptr, count)
                    blocks.append(block)
                    captured += block.shape[0]
            finally:
                # Always release, even if decoding raised: an unreleased
                # packet wedges the capture stream permanently.
                self._release_buffer(self._capture, frames.value)

            if flags.value & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY and not self._first_packet:
                self._discontinuities += 1
                if self._discontinuities in (1, 10) or self._discontinuities % 100 == 0:
                    log.warning(
                        "WASAPI reported a gap in captured audio (%d so far); "
                        "the machine may be under load",
                        self._discontinuities,
                    )
            self._first_packet = False

            if time.perf_counter() >= deadline:
                break

        now = time.perf_counter()
        if captured:
            self._silence_clock = now
            return np.concatenate(blocks, axis=0)

        idle = now - (self._silence_clock or now)
        if idle < self._silence_after:
            return np.zeros((0, fmt.channels), dtype=np.float32)
        self._silence_clock = now
        # Cap the fill. After a laptop suspends for an hour `idle` is 3600s,
        # and honouring it literally would ask for a multi-gigabyte array --
        # turning a resumed machine into a MemoryError. A second of silence
        # is all the downstream timeline needs to keep moving; the rest of
        # the gap is real lost time, not audio we can invent.
        seconds = min(idle, self._MAX_SILENCE_FILL_SECONDS)
        if idle > self._MAX_SILENCE_FILL_SECONDS:
            log.warning("capture stalled for %.1fs (system suspend or audio "
                        "service restart); filling %.1fs of silence",
                        idle, seconds)
        return np.zeros((int(seconds * fmt.sample_rate), fmt.channels),
                        dtype=np.float32)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

@dataclass
class CaptureReport:
    """What the capture layer actually negotiated, for logging and errors."""
    endpoint: AudioEndpoint
    backend: str
    native_rate: int = 0
    native_channels: int = 0
    native_format: str = ""
    target_rate: int = 0
    resampler: str = ""
    attempts: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        return [
            f"Selected logical device: {self.endpoint.index}",
            f"Device name: {self.endpoint.name}",
            f"Endpoint ID: {self.endpoint.endpoint_id}",
            "Host API: WASAPI (Windows Core Audio)",
            "Endpoint type: Playback (captured via loopback)",
            f"Native sample rate: {self.native_rate}",
            f"Native channels: {self.native_channels}",
            f"Native sample format: {self.native_format}",
            f"Capture backend: {self.backend}",
            f"Output format: {self.target_rate} Hz mono PCM16",
            f"Resampler: {self.resampler}",
        ]
