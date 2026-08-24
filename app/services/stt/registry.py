"""One place that decides which STT backend a session gets.

Swapping providers -- Groq Whisper today, an NVIDIA model next -- is a
one-line `.env` change and nothing outside this file has to know. Routes ask
for `build_stt_service(settings, prompt=...)` and get something satisfying
the `STTService` protocol; they never name a vendor.
"""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.services.stt.base import MissingSTTConfigError, STTService
from app.services.stt.composite import FallbackSTTService, RacingSTTService


logger = logging.getLogger(__name__)

# build_stt_service() runs once per audio websocket, so an unconditional
# warning would repeat on every reconnect for the whole interview.
_warned_fallback_unavailable = False


def _warn_fallback_unavailable() -> None:
    global _warned_fallback_unavailable
    if _warned_fallback_unavailable:
        return
    _warned_fallback_unavailable = True
    logger.warning(
        "STT_FALLBACK_ENABLED is true but OPENAI_API_KEY is not set, so there is "
        "NO fallback recognizer: a failed or timed-out transcription loses that "
        "segment outright. Set OPENAI_API_KEY to enable the fallback, or set "
        "STT_FALLBACK_ENABLED=false to make the absence explicit."
    )


def build_stt_service(
    settings: Settings,
    *,
    prompt: str | None = None,
    keyterms: list[str] | None = None,
) -> STTService:
    """Build the STT service for one session, including any fallback wiring.

    `prompt` biases the Whisper-family backends; `keyterms` biases Deepgram.
    They carry the same information (this session's vocabulary) in the shape
    each provider accepts, and both are optional -- a caller that supplies
    neither gets the provider's configured defaults.
    """
    primary = _build_primary(settings, prompt=prompt, keyterms=keyterms)

    fallback = _build_fallback(settings, prompt=prompt)
    if fallback is None:
        return primary

    if settings.stt_race_providers:
        return RacingSTTService(primary, fallback)
    return FallbackSTTService(primary, fallback)


def _build_primary(
    settings: Settings, *, prompt: str | None, keyterms: list[str] | None = None
) -> STTService:
    provider = settings.stt_provider

    if provider == "deepgram":
        from app.services.stt.deepgram import DeepgramSTTService

        return DeepgramSTTService.from_settings(settings, keyterms=keyterms)

    if provider == "faster_whisper":
        from app.services.stt.local import FasterWhisperSTTService

        return FasterWhisperSTTService(
            model_size=settings.faster_whisper_model,
            device=settings.faster_whisper_device,
            compute_type=settings.faster_whisper_compute_type,
            language=settings.stt_language,
            prompt=prompt,
            cpu_threads=settings.faster_whisper_cpu_threads,
        )

    if provider == "nvidia":
        return _build_nvidia(settings, prompt=prompt)

    if provider == "openai":
        from app.services.stt.whisper_api import OpenAIWhisperSTTService

        if not settings.openai_api_key:
            raise MissingSTTConfigError("Set OPENAI_API_KEY to use STT_PROVIDER=openai.")
        return OpenAIWhisperSTTService(
            api_key=settings.openai_api_key,
            model=settings.stt_model or "whisper-1",
            language=settings.stt_language,
            prompt=prompt,
        )

    from app.services.stt.whisper_api import GroqSTTService

    return GroqSTTService.from_settings(settings, prompt=prompt)


def _build_nvidia(settings: Settings, *, prompt: str | None) -> STTService:
    if settings.nvidia_stt_mode == "nim":
        from app.services.stt.nvidia import NvidiaNimSTTService

        if not settings.nvidia_stt_base_url:
            raise MissingSTTConfigError(
                "Set NVIDIA_STT_BASE_URL (e.g. http://localhost:9000/v1) to use "
                "STT_PROVIDER=nvidia with NVIDIA_STT_MODE=nim."
            )
        return NvidiaNimSTTService(
            base_url=settings.nvidia_stt_base_url,
            model=settings.nvidia_stt_model,
            api_key=settings.nvidia_api_key,
            language=settings.stt_language,
            prompt=prompt,
        )

    from app.services.stt.nvidia import NvidiaRivaSTTService

    return NvidiaRivaSTTService(
        server=settings.nvidia_riva_server,
        model=settings.nvidia_stt_model,
        language=settings.nvidia_stt_language,
        api_key=settings.nvidia_api_key,
        function_id=settings.nvidia_riva_function_id,
        use_ssl=settings.nvidia_riva_use_ssl,
        prompt=prompt,
    )


def _build_fallback(settings: Settings, *, prompt: str | None) -> STTService | None:
    """The fallback only exists to survive a primary-provider failure.

    It is deliberately never the same provider as the primary (a fallback to
    the thing that just rate-limited you does not help), and it is skipped
    entirely unless a key for it is configured.
    """
    if not settings.stt_fallback_enabled:
        return None
    if settings.stt_provider == "openai":
        return None

    # Deepgram primary falls back to Groq Whisper when one is available.
    # This is what "keep Whisper available during the migration" actually
    # means in running code: switching STT_PROVIDER to deepgram does not
    # leave a session with no recognizer at all if Deepgram has a bad
    # minute, and it does not require an OpenAI key that this deployment
    # has never had. Checked before the OpenAI branch because the Groq key
    # is the one that is actually configured here.
    if settings.stt_provider == "deepgram" and settings.groq_api_key:
        from app.services.stt.whisper_api import GroqSTTService

        try:
            return GroqSTTService.from_settings(settings, prompt=prompt)
        except MissingSTTConfigError:
            logger.warning("Groq Whisper STT fallback is unavailable; continuing without one")
            return None

    if not settings.openai_api_key:
        # STT_FALLBACK_ENABLED=true reads as "there is redundancy here", and
        # without this line there is nothing anywhere to say otherwise: the
        # function just returns None and a failed transcription silently
        # loses that segment. Say it once, out loud, rather than letting the
        # configuration imply a safety net that does not exist.
        _warn_fallback_unavailable()
        return None

    from app.services.stt.whisper_api import OpenAIWhisperSTTService

    try:
        return OpenAIWhisperSTTService(
            api_key=settings.openai_api_key,
            language=settings.stt_language,
            prompt=prompt,
        )
    except MissingSTTConfigError:
        logger.warning("OpenAI STT fallback is unavailable; continuing without one")
        return None
