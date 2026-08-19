"""Whisper-over-HTTP backends: Groq and OpenAI.

Both expose the same OpenAI-shaped `audio.transcriptions.create` API, so
they differ only in client construction and defaults.
"""

from __future__ import annotations

import logging

from groq import AsyncGroq

from app.core.config import Settings
from app.services.stt.base import (
    MissingGroqApiKeyError,
    MissingSTTConfigError,
    TranscriptionResult,
    _truncate_prompt_bytes,
    analyze_segments,
    pcm16_to_wav,
)


try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


logger = logging.getLogger(__name__)


class GroqSTTService:
    name = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        language: str | None,
        prompt: str | None = None,
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._language = language
        self._prompt = prompt

    @classmethod
    def from_settings(cls, settings: Settings, *, prompt: str | None = None) -> "GroqSTTService":
        if not settings.groq_api_key:
            raise MissingGroqApiKeyError("Set GROQ_API_KEY in your environment or .env file.")

        return cls(
            api_key=settings.groq_api_key,
            model=settings.stt_model,
            language=settings.stt_language,
            prompt=prompt if prompt is not None else settings.stt_prompt,
        )

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        wav_bytes = pcm16_to_wav(pcm, sample_rate=sample_rate)
        transcription_args = {
            "file": ("speech.wav", wav_bytes, "audio/wav"),
            "model": self._model,
            "response_format": "verbose_json",
            "temperature": 0.0,
        }
        if self._language:
            transcription_args["language"] = self._language
        if self._prompt:
            # Groq's Whisper prompt param is capped around 896 bytes; keep it
            # short so it actually gets used as a vocabulary/context bias
            # instead of being rejected outright (see _truncate_prompt_bytes).
            transcription_args["prompt"] = _truncate_prompt_bytes(self._prompt)

        response = await self._client.audio.transcriptions.create(**transcription_args)
        text = getattr(response, "text", "") or ""
        segments = getattr(response, "segments", None) or []
        confidence, no_speech_prob, known = analyze_segments(segments)
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            confidence_known=known,
            no_speech_prob=no_speech_prob,
            provider=self.name,
            model=self._model,
        )


class OpenAIWhisperSTTService:
    """Fallback STT using OpenAI's Whisper endpoint -- same interface as
    GroqSTTService, so it is a drop-in swap/fallback."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-1",
        language: str | None = None,
        prompt: str | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise MissingSTTConfigError(
                "Install the openai package to use OpenAIWhisperSTTService."
            )
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._language = language
        self._prompt = prompt

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        wav_bytes = pcm16_to_wav(pcm, sample_rate=sample_rate)
        transcription_args = {
            "file": ("speech.wav", wav_bytes, "audio/wav"),
            "model": self._model,
            "response_format": "verbose_json",
            "temperature": 0.0,
        }
        if self._language:
            transcription_args["language"] = self._language
        if self._prompt:
            transcription_args["prompt"] = _truncate_prompt_bytes(self._prompt)

        response = await self._client.audio.transcriptions.create(**transcription_args)
        text = getattr(response, "text", "") or ""
        segments = getattr(response, "segments", None) or []
        confidence, no_speech_prob, known = analyze_segments(segments)
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            confidence_known=known,
            no_speech_prob=no_speech_prob,
            provider=self.name,
            model=self._model,
        )
