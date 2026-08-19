"""The provider-agnostic half of speech-to-text.

Everything here is deliberately free of any specific vendor: the result
type, the service contract, and the audio/prompt helpers every provider
needs. A new backend (NVIDIA Riva, a NIM endpoint, a local model, whatever
comes next) only has to implement `transcribe_pcm16` and return a
`TranscriptionResult` -- nothing upstream of it changes.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.services.transcript_quality import looks_like_stt_hallucination  # noqa: F401  (re-export)


def _truncate_prompt_bytes(text: str, max_bytes: int = 850) -> str:
    """Groq's Whisper prompt field has a hard cap around 896 bytes.

    Plain Python slicing (text[:900]) counts CHARACTERS, not bytes -- if the
    text contains multi-byte UTF-8 characters (smart quotes, em-dashes,
    bullet points -- common in resumes/JDs pasted from Word or extracted
    from a PDF), the actual UTF-8 byte length can exceed the character
    count, silently pushing past the real limit even though a character-
    count slice looked safe. Truncate by real UTF-8 byte length instead,
    with a safety margin below the documented ~896 cap.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # Truncating raw bytes can land mid-character; decode with errors=
    # "ignore" to drop any incomplete trailing multi-byte sequence cleanly.
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class MissingGroqApiKeyError(RuntimeError):
    pass


class MissingSTTConfigError(RuntimeError):
    """A provider was selected but is not usable as configured (missing key,
    missing optional dependency, unreachable endpoint)."""


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    # Rough 0..1 confidence estimate. Not a calibrated probability -- a
    # relative signal for "does this transcript look trustworthy".
    confidence: float = 1.0
    # Whether `confidence` came from the provider at all. This matters more
    # than it looks: when a provider returns no usable per-segment scores,
    # `confidence` falls back to 1.0, and treating that as "the model is
    # certain" is what let every hallucinated fragment through the filters
    # at full confidence. Downstream code must check this before trusting
    # the number.
    confidence_known: bool = False
    # Average probability the audio was silence/noise rather than speech
    # (Whisper's no_speech_prob, or a provider equivalent). Used to
    # hard-filter the "invent plausible text on silence" failure mode.
    no_speech_prob: float = 0.0
    provider: str = ""
    model: str = ""


@runtime_checkable
class STTService(Protocol):
    """The entire contract a speech-to-text backend has to satisfy."""

    name: str

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        ...


def _field(segment: Any, name: str, default: float = 0.0) -> float:
    """Read a field from a provider segment that may be an object OR a dict.

    This is not defensive padding -- it is a fix. Groq's SDK types
    `Transcription` with only `text`; `segments` arrives as an untyped extra
    field, i.e. a list of plain dicts. The previous code read it with
    `getattr(s, "avg_logprob", 0.0)`, which on a dict always returns the
    default. Every transcript therefore scored confidence exactly 1.0 and
    no_speech_prob exactly 0.0, so the low-confidence flag never showed and
    the hallucination filter never fired even once.
    """
    if isinstance(segment, dict):
        value = segment.get(name, default)
    else:
        value = getattr(segment, name, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def analyze_segments(segments: list) -> tuple[float, float, bool]:
    """Turn per-segment decoder stats into (confidence, no_speech_prob, known).

    avg_logprob is near 0 when the model is confident and strongly negative
    when it is guessing; no_speech_prob is its own estimate that the audio
    was not speech at all. Segments are weighted by duration when timings
    are available, so one short garbage segment tacked onto a good long one
    cannot dominate the score.
    """
    if not segments:
        return 1.0, 0.0, False

    total_weight = 0.0
    logprob_sum = 0.0
    no_speech_sum = 0.0
    saw_logprob = False

    for segment in segments:
        start = _field(segment, "start", 0.0)
        end = _field(segment, "end", 0.0)
        weight = max(end - start, 0.0) or 1.0

        if isinstance(segment, dict):
            has_logprob = "avg_logprob" in segment
        else:
            has_logprob = hasattr(segment, "avg_logprob")
        saw_logprob = saw_logprob or has_logprob

        logprob_sum += _field(segment, "avg_logprob", 0.0) * weight
        no_speech_sum += _field(segment, "no_speech_prob", 0.0) * weight
        total_weight += weight

    if total_weight <= 0:
        return 1.0, 0.0, False

    avg_logprob = logprob_sum / total_weight
    avg_no_speech = no_speech_sum / total_weight

    logprob_confidence = max(0.0, min(1.0, 1.0 + avg_logprob))
    confidence = max(0.0, min(1.0, logprob_confidence * (1.0 - avg_no_speech)))
    return confidence, avg_no_speech, saw_logprob


# Kept under the old private name; several modules imported it directly.
_analyze_segments = analyze_segments


def pcm16_to_wav(pcm: bytes, *, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


_pcm16_to_wav = pcm16_to_wav
