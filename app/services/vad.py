from collections import deque
from dataclasses import dataclass
from typing import Literal, Protocol


try:
    import webrtcvad
except ImportError:
    webrtcvad = None


VadBackend = Literal["energy", "webrtc"]


@dataclass(frozen=True, slots=True)
class AudioSegment:
    pcm: bytes
    sample_rate: int
    duration_seconds: float


class SpeechDetector(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        ...


class EnergySpeechDetector:
    def __init__(self, threshold: int) -> None:
        self._threshold = threshold

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return _pcm16_rms(frame) >= self._threshold


class WebRtcSpeechDetector:
    def __init__(self, mode: int) -> None:
        if webrtcvad is None:
            raise RuntimeError(
                "VAD_BACKEND=webrtc requires installing webrtcvad-wheels. "
                "Use VAD_BACKEND=energy on Windows/Python 3.14 unless you have C++ build tools."
            )
        self._vad = webrtcvad.Vad(mode)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return self._vad.is_speech(frame, sample_rate)


class SpeechSegmenter:
    def __init__(
        self,
        *,
        sample_rate: int,
        frame_ms: int,
        vad_backend: VadBackend,
        vad_mode: int,
        energy_threshold: int,
        min_segment_seconds: float,
        max_segment_seconds: float,
        end_silence_ms: int,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("VAD frame size must be 10, 20, or 30 ms.")
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("VAD sample rate must be 8, 16, 32, or 48 kHz.")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._detector: SpeechDetector = _build_detector(
            backend=vad_backend,
            mode=vad_mode,
            energy_threshold=energy_threshold,
        )
        self._frame_bytes = int(sample_rate * (frame_ms / 1000) * 2)
        self._min_frames = max(1, int((min_segment_seconds * 1000) / frame_ms))
        self._max_frames = max(self._min_frames, int((max_segment_seconds * 1000) / frame_ms))
        self._end_silence_frames = max(1, int(end_silence_ms / frame_ms))
        self._preroll_frames = max(1, int(200 / frame_ms))

        self._pending = bytearray()
        self._triggered = False
        self._silence_frames = 0
        self._preroll: deque[bytes] = deque(maxlen=self._preroll_frames)
        self._current: list[bytes] = []

    def accept(self, chunk: bytes) -> list[AudioSegment]:
        self._pending.extend(chunk)
        segments: list[AudioSegment] = []

        while len(self._pending) >= self._frame_bytes:
            frame = bytes(self._pending[: self._frame_bytes])
            del self._pending[: self._frame_bytes]

            segment = self._process_frame(frame)
            if segment:
                segments.append(segment)

        return segments

    def flush(self) -> AudioSegment | None:
        if not self._triggered or len(self._current) < self._min_frames:
            self._reset_active_segment()
            return None
        return self._emit_segment()

    def _process_frame(self, frame: bytes) -> AudioSegment | None:
        is_speech = self._detector.is_speech(frame, self.sample_rate)

        if not self._triggered:
            self._preroll.append(frame)
            if is_speech:
                self._triggered = True
                self._current = list(self._preroll)
                self._silence_frames = 0
            return None

        self._current.append(frame)
        if is_speech:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if len(self._current) >= self._max_frames:
            return self._emit_segment()

        if (
            self._silence_frames >= self._end_silence_frames
            and len(self._current) >= self._min_frames
        ):
            return self._emit_segment()

        return None

    def _emit_segment(self) -> AudioSegment:
        pcm = b"".join(self._current)
        duration_seconds = len(self._current) * self.frame_ms / 1000
        self._reset_active_segment()
        return AudioSegment(
            pcm=pcm,
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds,
        )

    def _reset_active_segment(self) -> None:
        self._triggered = False
        self._silence_frames = 0
        self._current = []
        self._preroll.clear()


def _build_detector(*, backend: VadBackend, mode: int, energy_threshold: int) -> SpeechDetector:
    if backend == "webrtc":
        return WebRtcSpeechDetector(mode)
    return EnergySpeechDetector(energy_threshold)


def _pcm16_rms(frame: bytes) -> int:
    sample_count = len(frame) // 2
    if sample_count == 0:
        return 0

    total = 0
    for index in range(0, len(frame) - 1, 2):
        sample = int.from_bytes(frame[index : index + 2], byteorder="little", signed=True)
        total += sample * sample

    return int((total / sample_count) ** 0.5)
