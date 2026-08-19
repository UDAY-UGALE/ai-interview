"""NVIDIA speech-to-text backends.

Two deployment shapes are supported, because "the NVIDIA STT model" means
different transports depending on where it runs:

* `riva`  -- gRPC to a Riva ASR server. This covers a self-hosted Riva or
  ASR NIM container (`localhost:50051`) AND NVIDIA's hosted NVCF functions
  (`grpc.nvcf.nvidia.com:443` plus a function-id and API key). Models:
  Parakeet CTC/RNNT/TDT, Conformer. Needs `pip install nvidia-riva-client`.

* `nim`   -- HTTP to an OpenAI-compatible `/v1/audio/transcriptions`
  endpoint, which recent ASR NIM builds and build.nvidia.com expose. Needs
  nothing beyond the `openai` package that is already a dependency.

Both return real per-result confidence, which matters: the pipeline's
hallucination filtering is only as good as the confidence signal it gets,
and a backend that reports nothing gets treated as "unknown" rather than as
"certain" (see TranscriptionResult.confidence_known).

Riva is imported lazily so neither the package nor a running server is
needed unless this provider is actually selected.
"""

from __future__ import annotations

import asyncio
import logging
import math

from app.services.stt.base import (
    MissingSTTConfigError,
    TranscriptionResult,
    pcm16_to_wav,
)


try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


logger = logging.getLogger(__name__)


def _confidence_from_riva(raw: float | None) -> tuple[float, bool]:
    """Normalise Riva's confidence into 0..1.

    Riva reports confidence differently per model family: CTC models return
    a 0..1 score, while RNNT/TDT models return a log-probability (<= 0) and
    some builds return 0.0 to mean "not scored". Map each into the same
    0..1 space, and report whether the number means anything at all.
    """
    if raw is None:
        return 1.0, False
    if raw == 0.0:
        # Ambiguous: could be "unscored" or a perfect log-prob. Treat it as
        # unknown rather than as maximum confidence -- assuming certainty is
        # precisely the failure this pipeline is being fixed for.
        return 1.0, False
    if raw < 0:
        # Log-probability: exp() it back into a probability-like score.
        return max(0.0, min(1.0, math.exp(raw))), True
    return max(0.0, min(1.0, float(raw))), True


class NvidiaRivaSTTService:
    """Riva ASR over gRPC (self-hosted server, ASR NIM container, or NVCF)."""

    name = "nvidia_riva"

    def __init__(
        self,
        *,
        server: str,
        model: str | None = None,
        language: str = "en-US",
        api_key: str | None = None,
        function_id: str | None = None,
        use_ssl: bool = False,
        prompt: str | None = None,
        automatic_punctuation: bool = True,
    ) -> None:
        try:
            import riva.client  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise MissingSTTConfigError(
                "STT_PROVIDER=nvidia with NVIDIA_STT_MODE=riva needs the Riva client: "
                "pip install nvidia-riva-client"
            ) from exc

        self._riva = riva.client
        self._model = model or ""
        self._language = language
        self._prompt = prompt

        metadata = []
        if function_id:
            metadata.append(("function-id", function_id))
        if api_key:
            metadata.append(("authorization", f"Bearer {api_key}"))

        auth = riva.client.Auth(
            uri=server,
            use_ssl=use_ssl or bool(function_id),
            metadata_args=[list(pair) for pair in metadata] or None,
        )
        self._service = riva.client.ASRService(auth)
        self._automatic_punctuation = automatic_punctuation

    def _build_config(self, sample_rate: int):
        config = self._riva.RecognitionConfig(
            encoding=self._riva.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=sample_rate,
            language_code=self._language,
            max_alternatives=1,
            enable_automatic_punctuation=self._automatic_punctuation,
            audio_channel_count=1,
        )
        if self._model:
            config.model = self._model
        if self._prompt:
            # Riva biases decoding through word boosting rather than a free
            # text prompt -- feed the vocabulary hint in as boosted phrases
            # so misheard framework names get the same help Whisper's prompt
            # param gave them.
            try:
                self._riva.add_word_boosting_to_config(
                    config, _boost_phrases(self._prompt), 4.0
                )
            except Exception:  # pragma: no cover - client version differences
                logger.debug("Riva word boosting unavailable; continuing without it")
        return config

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        # The Riva client is synchronous/blocking, so it runs in a worker
        # thread -- otherwise it stalls the event loop that is simultaneously
        # receiving audio and streaming an answer.
        return await asyncio.to_thread(self._transcribe_sync, pcm, sample_rate)

    def _transcribe_sync(self, pcm: bytes, sample_rate: int) -> TranscriptionResult:
        config = self._build_config(sample_rate)
        # Riva's offline recognize wants a container; WAV keeps the header
        # self-describing rather than relying on config alone.
        response = self._service.offline_recognize(
            pcm16_to_wav(pcm, sample_rate=sample_rate), config
        )

        parts: list[str] = []
        confidences: list[float] = []
        known_any = False
        for result in getattr(response, "results", []) or []:
            alternatives = getattr(result, "alternatives", None) or []
            if not alternatives:
                continue
            best = alternatives[0]
            text = (getattr(best, "transcript", "") or "").strip()
            if text:
                parts.append(text)
            confidence, known = _confidence_from_riva(getattr(best, "confidence", None))
            if known:
                confidences.append(confidence)
                known_any = True

        confidence = sum(confidences) / len(confidences) if confidences else 1.0
        return TranscriptionResult(
            text=" ".join(parts).strip(),
            confidence=confidence,
            confidence_known=known_any,
            no_speech_prob=0.0,
            provider=self.name,
            model=self._model or "riva-default",
        )


class NvidiaNimSTTService:
    """An ASR NIM (or build.nvidia.com endpoint) speaking the OpenAI
    `/v1/audio/transcriptions` shape. Reuses the OpenAI SDK pointed at the
    NIM base URL rather than hand-rolling an HTTP client."""

    name = "nvidia_nim"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        language: str | None = "en",
        prompt: str | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise MissingSTTConfigError(
                "Install the openai package to use STT_PROVIDER=nvidia with "
                "NVIDIA_STT_MODE=nim."
            )
        # A locally hosted NIM usually needs no key, but the SDK requires a
        # non-empty string.
        self._client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self._model = model
        self._language = language
        self._prompt = prompt

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        wav_bytes = pcm16_to_wav(pcm, sample_rate=sample_rate)
        args = {
            "file": ("speech.wav", wav_bytes, "audio/wav"),
            "model": self._model,
            "response_format": "verbose_json",
        }
        if self._language:
            args["language"] = self._language

        try:
            response = await self._client.audio.transcriptions.create(**args)
        except Exception:
            # Not every NIM build implements verbose_json; fall back to plain
            # text rather than failing the whole segment.
            args["response_format"] = "json"
            response = await self._client.audio.transcriptions.create(**args)

        text = getattr(response, "text", "") or ""
        segments = getattr(response, "segments", None) or []
        # Reuse the Whisper-shaped scoring when the endpoint provides it.
        from app.services.stt.base import analyze_segments

        confidence, no_speech_prob, known = analyze_segments(segments)
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            confidence_known=known,
            no_speech_prob=no_speech_prob,
            provider=self.name,
            model=self._model,
        )


def _boost_phrases(prompt: str, limit: int = 60) -> list[str]:
    """Turn the vocabulary-bias prompt into a Riva word-boosting list."""
    seen: list[str] = []
    for raw in prompt.replace("\n", " ").split(","):
        phrase = raw.strip(" .:;-")
        if 1 < len(phrase) <= 40 and phrase not in seen:
            seen.append(phrase)
        if len(seen) >= limit:
            break
    return seen
