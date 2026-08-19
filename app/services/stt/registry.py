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


def build_stt_service(settings: Settings, *, prompt: str | None = None) -> STTService:
    """Build the STT service for one session, including any fallback wiring."""
    primary = _build_primary(settings, prompt=prompt)

    fallback = _build_fallback(settings, prompt=prompt)
    if fallback is None:
        return primary

    if settings.stt_race_providers:
        return RacingSTTService(primary, fallback)
    return FallbackSTTService(primary, fallback)


def _build_primary(settings: Settings, *, prompt: str | None) -> STTService:
    provider = settings.stt_provider

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
    if settings.stt_provider == "openai" or not settings.openai_api_key:
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
