import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import get_settings
from app.core.redis_client import get_session_store
from app.services.question_gate import get_question_pipeline
from app.services.stt import (
    FallbackSTTService,
    FasterWhisperSTTService,
    GroqSTTService,
    MissingGroqApiKeyError,
    OpenAIWhisperSTTService,
    RacingSTTService,
    TranscriptionResult,
    _truncate_prompt_bytes,
    looks_like_stt_hallucination,
)
from app.services.vad import AudioSegment, SpeechSegmenter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket) -> None:
    settings = get_settings()
    session_id = websocket.query_params.get("session_id", "default")
    await websocket.accept()

    try:
        stt_prompt = await _build_stt_prompt(settings, session_id)

        if settings.stt_provider == "faster_whisper":
            # Fully local -- no Groq API key needed for this path at all.
            stt = FasterWhisperSTTService(
                model_size=settings.faster_whisper_model,
                device=settings.faster_whisper_device,
                compute_type=settings.faster_whisper_compute_type,
                language=settings.stt_language,
                prompt=stt_prompt,
                cpu_threads=settings.faster_whisper_cpu_threads,
            )
        else:
            stt = GroqSTTService.from_settings(settings, prompt=stt_prompt)

        # If a network blip or outage takes down the primary STT provider,
        # fall back to OpenAI Whisper -- only wired in if that key is set.
        # Default mode: sequential (try primary, only call OpenAI if it fails/
        # times out). Opt-in STT_RACE_PROVIDERS=true mode: call both at once
        # and use whichever returns first -- lower worst-case latency on a
        # shaky connection, at the cost of always paying for two STT calls.
        if settings.openai_api_key:
            fallback = OpenAIWhisperSTTService(
                api_key=settings.openai_api_key,
                language=settings.stt_language,
                prompt=stt_prompt,
            )
            if settings.stt_race_providers:
                stt = RacingSTTService(stt, fallback)
            else:
                stt = FallbackSTTService(stt, fallback)
    except (MissingGroqApiKeyError, RuntimeError) as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1008)
        return

    segmenter = SpeechSegmenter(
        sample_rate=settings.audio_sample_rate,
        frame_ms=settings.audio_frame_ms,
        vad_backend=settings.vad_backend,
        vad_mode=settings.vad_mode,
        energy_threshold=settings.vad_energy_threshold,
        min_segment_seconds=settings.segment_min_seconds,
        max_segment_seconds=settings.segment_max_seconds,
        end_silence_ms=settings.segment_end_silence_ms,
    )
    queue: asyncio.Queue[Optional[AudioSegment]] = asyncio.Queue()
    worker = asyncio.create_task(
        _transcription_worker(
            websocket=websocket,
            stt=stt,
            queue=queue,
            session_id=session_id,
            stt_timeout_seconds=settings.stt_timeout_seconds,
            confidence_threshold=settings.stt_confidence_threshold,
            no_speech_threshold=settings.stt_no_speech_threshold,
        )
    )

    await websocket.send_json(
        {
            "type": "ready",
            "session_id": session_id,
            "sample_rate": settings.audio_sample_rate,
            "frame_ms": settings.audio_frame_ms,
        }
    )

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect

            chunk = message.get("bytes")
            if not chunk:
                continue

            for segment in segmenter.accept(chunk):
                await _send_json_safe(
                    websocket,
                    {
                        "type": "speech_segment",
                        "segment_seconds": round(segment.duration_seconds, 2),
                    },
                )
                await queue.put(segment)
    except WebSocketDisconnect:
        logger.info("Audio websocket disconnected")
    finally:
        flushed = segmenter.flush()
        if flushed:
            await _send_json_safe(
                websocket,
                {
                    "type": "speech_segment",
                    "segment_seconds": round(flushed.duration_seconds, 2),
                },
            )
            await queue.put(flushed)

        await queue.put(None)
        await worker


async def _transcription_worker(
    websocket: WebSocket,
    stt,
    queue: asyncio.Queue[Optional[AudioSegment]],
    session_id: str,
    stt_timeout_seconds: int,
    confidence_threshold: float,
    no_speech_threshold: float,
) -> None:
    while True:
        segment = await queue.get()
        if segment is None:
            return

        started_at = time.perf_counter()
        try:
            result: TranscriptionResult = await asyncio.wait_for(
                stt.transcribe_pcm16(segment.pcm, sample_rate=segment.sample_rate),
                timeout=stt_timeout_seconds,
            )
        except TimeoutError:
            logger.exception("STT transcription timed out")
            await _send_json_safe(
                websocket,
                {
                    "type": "error",
                    "message": (
                        f"STT timed out after {stt_timeout_seconds}s. "
                        "Check your network/Groq API status."
                    ),
                },
            )
            continue
        except Exception as exc:
            logger.exception("STT transcription failed")
            await _send_json_safe(
                websocket,
                {
                    "type": "error",
                    "message": f"STT failed ({type(exc).__name__}): {exc}",
                },
            )
            continue

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        text = result.text.strip()
        if not text:
            continue

        # Hard-filter Whisper's known hallucination failure modes instead of
        # just flagging them: (1) high no_speech_prob means the model itself
        # is signalling this was probably silence/noise, not real speech, yet
        # it still emitted plausible-sounding text; (2) a short phrase looped
        # several times in a row is a classic "got stuck" artifact. Either
        # one means this text should never reach the LLM as a real question.
        if result.no_speech_prob >= no_speech_threshold:
            logger.info(
                "Dropping likely-hallucinated transcript (no_speech_prob=%.2f): %s",
                result.no_speech_prob,
                text,
            )
            continue
        if looks_like_stt_hallucination(text):
            logger.info("Dropping repetitive/looping transcript: %s", text)
            continue

        low_confidence = result.confidence < confidence_threshold
        if low_confidence:
            logger.info(
                "Low-confidence transcript (%.2f): %s", result.confidence, text
            )
        else:
            logger.info("Transcript: %s", text)

        await get_question_pipeline().submit_transcript(
            session_id=session_id,
            text=text,
            confidence=result.confidence,
        )
        await _send_json_safe(
            websocket,
            {
                "type": "transcript",
                "text": text,
                "segment_seconds": round(segment.duration_seconds, 2),
                "stt_latency_ms": latency_ms,
                "confidence": round(result.confidence, 2),
                "low_confidence": low_confidence,
            },
        )


async def _send_json_safe(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_json(payload)
    except (RuntimeError, WebSocketDisconnect):
        logger.debug("Could not send websocket payload; connection is closed")


async def _build_stt_prompt(settings, session_id: str) -> str:
    """Combine the generic tech-vocabulary bias with this session's actual
    resume/JD text, so Whisper is biased toward terms that are actually
    likely to come up (e.g. specific frameworks/versions from the JD) --
    this is what fixes misheard jargon like "React 19" -> "reactivity"."""
    parts = [settings.stt_prompt] if settings.stt_prompt else []
    try:
        context = await get_session_store().get_context(session_id)
        if context.job_description:
            parts.append(context.job_description[:500])
        if context.resume_text:
            parts.append(context.resume_text[:300])
    except Exception:
        logger.exception("Could not load session context for STT prompt biasing")
    # Byte-safe truncation, not text[:900] -- Groq's actual limit is ~896
    # UTF-8 bytes, and resume/JD text (especially from a PDF upload) often
    # contains multi-byte characters (smart quotes, em-dashes, bullets) that
    # make a character-count slice silently exceed the real byte limit.
    return _truncate_prompt_bytes(" ".join(parts))
