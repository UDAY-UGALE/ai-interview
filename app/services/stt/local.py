"""Local, no-network transcription via faster-whisper / CTranslate2."""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from app.services.stt.base import (
    MissingSTTConfigError,
    TranscriptionResult,
    _truncate_prompt_bytes,
    analyze_segments,
)


try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import numpy as np
except ImportError:
    np = None


logger = logging.getLogger(__name__)


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
    """Fully local Whisper transcription -- no network round trip at all.

    Local/dev-machine option: the machine running this needs enough
    RAM+CPU+storage to hold and run the model (roughly 1-2GB for
    large-v3-turbo int8), and it competes for that machine's own resources.
    Not recommended for a shared/cloud deployment unless the server has been
    sized for it.
    """

    name = "faster_whisper"

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
            raise MissingSTTConfigError(
                "Install faster-whisper (pip install faster-whisper) to use "
                "STT_PROVIDER=faster_whisper."
            )
        if np is None:
            raise MissingSTTConfigError("Install numpy to use STT_PROVIDER=faster_whisper.")

        self._model = _load_faster_whisper_model(model_size, device, compute_type, cpu_threads)
        self._model_size = model_size
        self._language = language
        self._prompt = prompt

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        # transcribe() is a blocking, CPU-bound call -- run it in a worker
        # thread so it does not stall the asyncio event loop (and therefore
        # the rest of the app: other sessions, HTTP routes).
        return await asyncio.to_thread(self._transcribe_sync, pcm)

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
        confidence, no_speech_prob, known = analyze_segments(segments)
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            confidence_known=known,
            no_speech_prob=no_speech_prob,
            provider=self.name,
            model=self._model_size,
        )
