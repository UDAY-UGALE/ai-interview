"""Speech-to-text, as a replaceable component.

Everything the rest of the app needs is re-exported here, so callers import
from `app.services.stt` and never from a vendor-specific module. Choosing a
backend is `build_stt_service(settings)`; adding one means writing a class
with a single `transcribe_pcm16` method and one line in `registry.py`.
"""

from app.services.stt.base import (
    MissingGroqApiKeyError,
    MissingSTTConfigError,
    STTService,
    TranscriptionResult,
    _analyze_segments,
    _pcm16_to_wav,
    _truncate_prompt_bytes,
    analyze_segments,
    looks_like_stt_hallucination,
    pcm16_to_wav,
)
from app.services.stt.composite import FallbackSTTService, RacingSTTService
from app.services.stt.registry import build_stt_service


__all__ = [
    "DeepgramSTTService",
    "FallbackSTTService",
    "MissingGroqApiKeyError",
    "MissingSTTConfigError",
    "RacingSTTService",
    "STTService",
    "TranscriptionResult",
    "analyze_segments",
    "build_stt_service",
    "looks_like_stt_hallucination",
    "pcm16_to_wav",
    "_analyze_segments",
    "_pcm16_to_wav",
    "_truncate_prompt_bytes",
]


def __getattr__(name: str):
    """Lazily expose the concrete provider classes.

    Importing them eagerly would pull in every optional dependency (Riva's
    gRPC stack, faster-whisper's CTranslate2 runtime) just to import the
    package, which is exactly what makes an optional backend feel
    mandatory.
    """
    if name in ("GroqSTTService", "OpenAIWhisperSTTService"):
        from app.services.stt import whisper_api

        return getattr(whisper_api, name)
    if name == "FasterWhisperSTTService":
        from app.services.stt.local import FasterWhisperSTTService

        return FasterWhisperSTTService
    if name in ("NvidiaRivaSTTService", "NvidiaNimSTTService"):
        from app.services.stt import nvidia

        return getattr(nvidia, name)
    if name in ("DeepgramSTTService", "DeepgramStreamingSession", "StreamingTranscript"):
        from app.services.stt import deepgram

        return getattr(deepgram, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
