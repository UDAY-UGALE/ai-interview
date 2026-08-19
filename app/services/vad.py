"""Voice activity detection and utterance segmentation.

The job of this module is to turn a continuous stream of PCM frames into
SEGMENTS worth sending to speech-to-text, and -- just as importantly -- to
say which segments belong to the SAME spoken utterance.

Two ideas the previous version was missing, both of which caused real
downstream damage:

1. A segment is only worth a transcription request if it actually contains
   enough VOICED audio. The old segmenter emitted a segment for any frame
   that crossed a fixed energy threshold, padded out to min_segment_seconds
   with silence. A 100ms keyboard click or a burst of call comfort-noise
   therefore became a 0.8s "speech" segment that was 0.7s silence -- and
   Whisper, handed near-silence, does not return nothing: it invents
   plausible text ("Thank you.", "Tch.", "."). That is where both the STT
   request explosion (429s) and most of the garbage transcripts came from.

2. One utterance can span several segments. A long question gets force-cut
   at max_segment_seconds, and a mid-sentence breath ends a segment early.
   Every segment now carries an `utterance_id`, so the rest of the pipeline
   can tell "more of the same question" apart from "a new thing was said"
   instead of guessing from timing alone.

The detector also tracks a rolling noise floor rather than trusting one
fixed threshold. System-loopback audio from a video call has a continuously
varying noise floor (codec comfort noise, other participants' room tone),
which a fixed threshold either sits under (constant false triggering, what
was happening) or over (missed quiet speech).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Protocol


try:
    import webrtcvad
except ImportError:
    webrtcvad = None

try:
    import numpy as np
except ImportError:
    np = None


VadBackend = Literal["energy", "webrtc"]


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """One chunk of audio worth transcribing.

    `utterance_id` groups segments that belong to the same continuous piece
    of speech: a long question force-cut at max_segment_seconds, or split by
    a breath too short to count as the end of the utterance, keeps the same
    id across all of its pieces. `is_final` marks the piece that ended on a
    real end-of-speech silence -- i.e. the interviewer actually stopped
    talking, as opposed to us cutting them off mid-sentence.
    """

    pcm: bytes
    sample_rate: int
    duration_seconds: float
    # Seconds of the segment the detector actually scored as voiced.
    # `duration_seconds` includes leading pre-roll and trailing silence, so
    # this -- not the total -- is what says whether the segment is worth an
    # STT request at all.
    speech_seconds: float = 0.0
    utterance_id: int = 0
    # 0-based position of this segment within its utterance.
    index_in_utterance: int = 0
    # True when this segment ended because speech genuinely stopped; False
    # when it was force-cut at max_segment_seconds and more of the same
    # utterance is still coming.
    is_final: bool = True
    # time.monotonic() when the last frame of this segment arrived, so
    # downstream stages can measure real end-of-speech -> answer latency.
    captured_at: float = field(default_factory=time.monotonic)

    @property
    def speech_ratio(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.speech_seconds / self.duration_seconds


class SpeechDetector(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        ...


class EnergySpeechDetector:
    """Energy VAD with an adaptive noise floor and hysteresis.

    `threshold` is treated as a FLOOR for the trigger level, not as the
    trigger level itself: the detector tracks the quietest recent frames as
    the ambient noise level and requires speech to stand a real margin above
    THAT. On a call whose background noise sits above the configured
    threshold (the common case for system-loopback audio) a fixed threshold
    fires continuously; this adapts instead.

    Hysteresis (a lower level to stay in speech than to enter it) keeps
    ordinary within-word dips -- stop consonants, the quiet part of a
    diphthong -- from being scored as end-of-speech and chopping a question
    into fragments.
    """

    def __init__(
        self,
        threshold: int,
        *,
        adaptive: bool = True,
        noise_margin: float = 2.5,
        release_ratio: float = 0.6,
        noise_window_frames: int = 250,  # ~5s of 20ms frames
    ) -> None:
        self._floor = max(1, threshold)
        self._adaptive = adaptive
        self._noise_margin = max(noise_margin, 1.0)
        self._release_ratio = release_ratio
        self._recent: deque[float] = deque(maxlen=noise_window_frames)
        self._noise_level = float(threshold) / self._noise_margin
        self._in_speech = False
        self._speech_run = 0
        # Nobody speaks for 30 seconds without a single gap below the
        # release level. Past this, "speech" is really a noisy line that the
        # detector latched onto on its very first frame and has been unable
        # to re-measure since -- so it starts re-measuring.
        self._max_speech_run = max(noise_window_frames * 6, 1)

    @property
    def noise_level(self) -> float:
        return self._noise_level

    @property
    def trigger_level(self) -> float:
        return max(float(self._floor), self._noise_level * self._noise_margin)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        rms = pcm16_rms(frame)

        # The noise floor is measured ONLY from frames that are not already
        # speech, and speech frames are kept out of the window entirely.
        # Letting speech feed the estimate is self-defeating: during a long
        # question the floor climbs toward the level of the voice it is
        # supposed to be detecting, the trigger climbs with it, and the
        # detector goes deaf part-way through -- measured at roughly five
        # seconds in, which is well inside the length of a real interview
        # question.
        if self._adaptive and (not self._in_speech or self._speech_run > self._max_speech_run):
            self._recent.append(rms)
            if len(self._recent) >= 25:
                # 20th percentile of recent quiet frames is roughly "what
                # this line sounds like when nobody is talking".
                ordered = sorted(self._recent)
                quiet = ordered[max(0, int(len(ordered) * 0.2))]
                # Track downward fast (a talker stopping should free the
                # floor quickly) and upward slowly.
                alpha = 0.25 if quiet < self._noise_level else 0.02
                self._noise_level += alpha * (quiet - self._noise_level)

        trigger = self.trigger_level
        if self._in_speech:
            self._in_speech = rms >= trigger * self._release_ratio
        else:
            self._in_speech = rms >= trigger

        self._speech_run = self._speech_run + 1 if self._in_speech else 0
        return self._in_speech


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
        min_speech_ms: int = 320,
        onset_frames: int = 3,
        preroll_ms: int = 300,
        carryover_ms: int = 200,
        adaptive_threshold: bool = True,
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
            adaptive=adaptive_threshold,
        )
        self._frame_bytes = int(sample_rate * (frame_ms / 1000) * 2)
        self._min_frames = max(1, int((min_segment_seconds * 1000) / frame_ms))
        self._max_frames = max(self._min_frames, int((max_segment_seconds * 1000) / frame_ms))
        self._end_silence_frames = max(1, int(end_silence_ms / frame_ms))
        self._preroll_frames = max(1, int(preroll_ms / frame_ms))
        # How much real voiced audio a segment needs before it is worth an
        # STT request. This is the single most effective filter in the
        # pipeline: without it every click and noise blip becomes a
        # transcription request that can only come back as invented text.
        self._min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        # Consecutive voiced frames needed to open a segment -- one loud
        # frame is a click, not speech.
        self._onset_frames = max(1, onset_frames)
        # Audio kept from the end of a force-cut segment and prepended to the
        # next one, so a word split across the cut is whole in at least one
        # of the two pieces instead of mangled in both.
        self._carryover_frames = max(0, int(carryover_ms / frame_ms))

        self._pending = bytearray()
        self._triggered = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._onset_run = 0
        self._preroll: deque[bytes] = deque(maxlen=self._preroll_frames)
        self._current: list[bytes] = []
        self._carryover: list[bytes] = []
        self._utterance_id = 0
        self._index_in_utterance = 0
        # Set while an utterance is open, so a force-cut segment and the
        # continuation after it share one id.
        self._utterance_open = False
        self.dropped_segments = 0

    @property
    def noise_level(self) -> float:
        return getattr(self._detector, "noise_level", 0.0)

    @property
    def speech_active(self) -> bool:
        """Is someone talking right now?

        True from the moment a segment opens until it is emitted. This is the
        signal that tells the rest of the pipeline "more words are coming"
        without having to guess from timers -- and timers cannot do this job,
        because the wait between two transcripts of one spoken question is
        mostly the time it takes to SAY the next sentence, which looks
        identical to the speaker having stopped.
        """
        return self._triggered

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
        """Emit whatever is buffered (connection closing / end of stream)."""
        if not self._triggered or self._speech_frames < self._min_speech_frames:
            self._reset_active_segment()
            self._close_utterance()
            return None
        return self._emit_segment(is_final=True)

    def _process_frame(self, frame: bytes) -> AudioSegment | None:
        is_speech = self._detector.is_speech(frame, self.sample_rate)

        if not self._triggered:
            self._preroll.append(frame)
            self._onset_run = self._onset_run + 1 if is_speech else 0
            if self._onset_run >= self._onset_frames:
                self._triggered = True
                # Pre-roll carries the frames just BEFORE the trigger, which
                # is where the first consonant of the first word lives, plus
                # any carryover from a force-cut of the same utterance.
                self._current = self._carryover + list(self._preroll)
                self._carryover = []
                self._silence_frames = 0
                self._speech_frames = self._onset_run
                self._onset_run = 0
            elif self._utterance_open:
                # Silence between segments of an open utterance: once it runs
                # past the end-of-speech window the utterance is genuinely
                # over, and the next trigger starts a new one.
                self._silence_frames += 1
                if self._silence_frames >= self._end_silence_frames:
                    self._close_utterance()
            return None

        self._current.append(frame)
        if is_speech:
            self._silence_frames = 0
            self._speech_frames += 1
        else:
            self._silence_frames += 1

        # Force-cut: the interviewer is still going. Emit what we have so it
        # can start transcribing NOW, and keep the utterance open so the
        # continuation is understood as part of the same question.
        if len(self._current) >= self._max_frames:
            return self._emit_segment(is_final=False)

        if self._silence_frames >= self._end_silence_frames:
            if self._speech_frames >= self._min_speech_frames:
                return self._emit_segment(is_final=True)
            # Not enough real speech in there -- a click, a door, a codec
            # artifact. Throw it away instead of paying for a transcription
            # request that can only come back as a hallucination.
            self.dropped_segments += 1
            self._reset_active_segment()
            self._close_utterance()
            return None

        return None

    def _emit_segment(self, *, is_final: bool) -> AudioSegment:
        pcm = b"".join(self._current)
        duration_seconds = len(self._current) * self.frame_ms / 1000
        speech_seconds = self._speech_frames * self.frame_ms / 1000

        if not self._utterance_open:
            self._utterance_id += 1
            self._index_in_utterance = 0
            self._utterance_open = True
        index = self._index_in_utterance
        self._index_in_utterance += 1

        carryover: list[bytes] = []
        if not is_final and self._carryover_frames:
            carryover = self._current[-self._carryover_frames :]

        self._reset_active_segment()
        self._carryover = carryover
        if is_final:
            self._close_utterance()

        return AudioSegment(
            pcm=pcm,
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds,
            speech_seconds=speech_seconds,
            utterance_id=self._utterance_id,
            index_in_utterance=index,
            is_final=is_final,
        )

    def _reset_active_segment(self) -> None:
        self._triggered = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._onset_run = 0
        self._current = []
        self._preroll.clear()

    def _close_utterance(self) -> None:
        self._utterance_open = False
        self._carryover = []
        self._index_in_utterance = 0


def _build_detector(
    *, backend: VadBackend, mode: int, energy_threshold: int, adaptive: bool = True
) -> SpeechDetector:
    if backend == "webrtc":
        return WebRtcSpeechDetector(mode)
    return EnergySpeechDetector(energy_threshold, adaptive=adaptive)


def pcm16_rms(frame: bytes) -> float:
    """RMS of a PCM16 frame.

    Vectorised through numpy when it is available: the pure-Python version
    costs ~165us per 20ms frame, which is ~1% of a core per audio stream
    spent inside the event loop -- small on its own, but it is paid on every
    single frame while STT and LLM calls are trying to make progress on that
    same loop.
    """
    if not frame:
        return 0.0
    if np is not None:
        samples = np.frombuffer(frame, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    return _pcm16_rms_python(frame)


def _pcm16_rms_python(frame: bytes) -> float:
    sample_count = len(frame) // 2
    if sample_count == 0:
        return 0.0

    total = 0
    for index in range(0, len(frame) - 1, 2):
        sample = int.from_bytes(frame[index : index + 2], byteorder="little", signed=True)
        total += sample * sample

    return (total / sample_count) ** 0.5


# Kept as an alias -- the old private name is imported by the test clients.
_pcm16_rms = pcm16_rms
