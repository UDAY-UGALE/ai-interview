import asyncio
import io
import logging
import wave
from dataclasses import dataclass
from functools import lru_cache

from groq import AsyncGroq

from app.core.config import Settings


try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import numpy as np
except ImportError:
    np = None


logger = logging.getLogger(__name__)


def _truncate_prompt_bytes(text: str, max_bytes: int = 850) -> str:
    """Groq's Whisper prompt field has a hard cap around 896 bytes.
    Plain Python slicing (text[:900]) counts CHARACTERS, not bytes -- if the
    text contains multi-byte UTF-8 characters (smart quotes, em-dashes,
    bullet points -- common in resumes/JDs pasted from Word or extracted
    from a PDF), the actual UTF-8 byte length can exceed the character
    count, silently pushing past the real limit even though a character-
    count slice looked safe. Truncate by real UTF-8 byte length instead,
    with a safety margin below the documented ~896 cap."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # Truncating raw bytes can land mid-character; decode with errors=
    # "ignore" to drop any incomplete trailing multi-byte sequence cleanly.
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class MissingGroqApiKeyError(RuntimeError):
    pass


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    # Rough 0..1 confidence estimate derived from Whisper's per-segment
    # avg_logprob/no_speech_prob. Not a calibrated probability -- just a
    # relative signal for "does this transcript look trustworthy".
    confidence: float = 1.0
    # Raw average no_speech_prob across segments (0..1) -- used separately
    # from `confidence` to hard-filter Whisper's well-known hallucination
    # failure mode: inventing plausible-sounding text on silence/noise/
    # background chatter instead of returning nothing.
    no_speech_prob: float = 0.0


def _analyze_segments(segments: list) -> tuple[float, float]:
    """Whisper's verbose_json response includes per-segment avg_logprob
    (near 0 = confident, very negative = uncertain) and no_speech_prob
    (probability the segment is silence/noise, not real speech). Returns
    (confidence, avg_no_speech_prob)."""
    if not segments:
        return 1.0, 0.0

    logprobs = [getattr(s, "avg_logprob", 0.0) or 0.0 for s in segments]
    no_speech = [getattr(s, "no_speech_prob", 0.0) or 0.0 for s in segments]

    avg_logprob = sum(logprobs) / len(logprobs)
    avg_no_speech = sum(no_speech) / len(no_speech)

    logprob_confidence = max(0.0, min(1.0, 1.0 + avg_logprob))
    confidence = max(0.0, min(1.0, logprob_confidence * (1.0 - avg_no_speech)))
    return confidence, avg_no_speech


def looks_like_stt_hallucination(text: str, *, min_repeats: int = 3) -> bool:
    """Whisper sometimes gets stuck looping the same short phrase when the
    audio is noisy/ambiguous (a well-documented failure mode) instead of
    transcribing real words. Detect a phrase of 2-4 words repeated back to
    back several times in a row and treat it as garbage rather than a real
    question."""
    words = text.lower().split()
    for n in (2, 3, 4):
        needed = n * min_repeats
        if len(words) < needed:
            continue
        for start in range(len(words) - needed + 1):
            phrase = words[start : start + n]
            if all(
                words[start + k * n : start + (k + 1) * n] == phrase
                for k in range(min_repeats)
            ):
                return True
    return False


class GroqSTTService:
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
        wav_bytes = _pcm16_to_wav(pcm, sample_rate=sample_rate)
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
        confidence, no_speech_prob = _analyze_segments(segments)
        return TranscriptionResult(text=text, confidence=confidence, no_speech_prob=no_speech_prob)


class OpenAIWhisperSTTService:
    """Fallback STT using OpenAI's Whisper endpoint -- same interface as
    GroqSTTService, so it's a drop-in swap/fallback."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-1",
        language: str | None = None,
        prompt: str | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("Install the openai package to use OpenAIWhisperSTTService.")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._language = language
        self._prompt = prompt

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        wav_bytes = _pcm16_to_wav(pcm, sample_rate=sample_rate)
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
        confidence, no_speech_prob = _analyze_segments(segments)
        return TranscriptionResult(text=text, confidence=confidence, no_speech_prob=no_speech_prob)


class FallbackSTTService:
    """Wraps a primary STT service with a secondary one. If the primary call
    fails (e.g. a network blip to Groq's API), automatically retries the
    same audio against the fallback provider instead of just erroring out
    on that segment."""

    def __init__(self, primary, fallback) -> None:
        self._primary = primary
        self._fallback = fallback

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        try:
            return await self._primary.transcribe_pcm16(pcm, sample_rate=sample_rate)
        except Exception:
            if self._fallback is None:
                raise
            logger.warning("Primary STT provider failed; retrying with fallback", exc_info=True)
            return await self._fallback.transcribe_pcm16(pcm, sample_rate=sample_rate)


class RacingSTTService:
    """Calls the primary and fallback STT providers CONCURRENTLY and returns
    whichever finishes first (successfully). Trades an extra STT API call
    every time for a much lower worst-case latency when one provider is
    having a slow/shaky moment -- vs. FallbackSTTService, which only tries
    the second provider sequentially after the first fails or times out."""

    def __init__(self, primary, fallback) -> None:
        self._primary = primary
        self._fallback = fallback

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        tasks = [
            asyncio.create_task(self._primary.transcribe_pcm16(pcm, sample_rate=sample_rate)),
            asyncio.create_task(self._fallback.transcribe_pcm16(pcm, sample_rate=sample_rate)),
        ]
        pending = tasks
        last_error: Exception | None = None

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is None:
                    for remaining in pending:
                        remaining.cancel()
                    return task.result()
                last_error = exc

        raise last_error or RuntimeError("Both STT providers failed")


@lru_cache(maxsize=1)
def _load_faster_whisper_model(
    model_size: str, device: str, compute_type: str, cpu_threads: int
) -> "WhisperModel":
    # Cached process-wide: the model is expensive to load (downloads once,
    # then loads weights into RAM every process start), so we load it ONCE
    # and reuse it across every websocket connection/session, not per call.
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads or 0,
    )


class FasterWhisperSTTService:
    """Fully local Whisper transcription via faster-whisper/CTranslate2 --
    no network round-trip at all. Local/dev-machine option: the machine
    running this needs enough RAM+CPU+storage to hold and run the model
    (roughly 1-2GB for large-v3-turbo int8), and it competes for that
    machine's own resources. Not recommended for a shared/cloud deployment
    unless you've specifically sized the server for it -- use provider=groq
    there instead, which needs no local resources."""

    def __init__(
        self,
        *,
        model_size: str,
        device: str,
        compute_type: str,
        language: str | None,
        prompt: str | None = None,
        cpu_threads: int = 0,
    ) -> None:
        if WhisperModel is None:
            raise RuntimeError(
                "Install faster-whisper (pip install faster-whisper) to use "
                "STT_PROVIDER=faster_whisper."
            )
        if np is None:
            raise RuntimeError("Install numpy to use STT_PROVIDER=faster_whisper.")

        self._model = _load_faster_whisper_model(model_size, device, compute_type, cpu_threads)
        self._language = language
        self._prompt = prompt

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        # faster-whisper's transcribe() is a blocking, CPU-bound call -- run
        # it in a worker thread so it doesn't stall the asyncio event loop
        # (and therefore the rest of the app: other sessions, HTTP routes).
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, pcm)

    def _transcribe_sync(self, pcm: bytes) -> TranscriptionResult:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments_iter, _info = self._model.transcribe(
            audio,
            language=self._language,
            initial_prompt=_truncate_prompt_bytes(self._prompt) if self._prompt else None,
            vad_filter=False,  # we already run our own VAD upstream
            beam_size=1,  # greedy decoding -- fastest option; raise for more accuracy
        )
        segments = list(segments_iter)  # materialize the generator
        text = " ".join(segment.text.strip() for segment in segments).strip()
        confidence, no_speech_prob = _analyze_segments(segments)
        return TranscriptionResult(text=text, confidence=confidence, no_speech_prob=no_speech_prob)


def _pcm16_to_wav(pcm: bytes, *, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()
