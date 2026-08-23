import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.auth import websocket_token_ok
from app.core.config import get_settings
from app.core.redis_client import get_session_store
from app.services.question_gate import get_question_pipeline
from app.services.stt import (
    MissingGroqApiKeyError,
    MissingSTTConfigError,
    TranscriptionResult,
    _truncate_prompt_bytes,
    build_stt_service,
)
from app.services.transcript_quality import assess_transcript
from app.services.vad import AudioSegment, SpeechSegmenter, UtteranceClosed

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket) -> None:
    if not await websocket_token_ok(websocket):
        return

    settings = get_settings()
    session_id = websocket.query_params.get("session_id", "default")
    await websocket.accept()

    try:
        stt_prompt = await _build_stt_prompt(settings, session_id)
        # Which provider this is (Groq Whisper, an NVIDIA model, a local
        # one) is decided entirely inside the registry -- this route only
        # knows it has something with transcribe_pcm16.
        stt = build_stt_service(settings, prompt=stt_prompt)
    except (MissingGroqApiKeyError, MissingSTTConfigError, RuntimeError) as exc:
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
        min_speech_ms=settings.segment_min_speech_ms,
        onset_frames=settings.vad_onset_frames,
        preroll_ms=settings.vad_preroll_ms,
        carryover_ms=settings.segment_carryover_ms,
        adaptive_threshold=settings.vad_adaptive_threshold,
        calibration_ms=settings.vad_calibration_ms,
    )

    # Two stages, deliberately decoupled:
    #   segments -> [_transcription_dispatcher] -> in-flight tasks -> [_result_consumer]
    # The dispatcher starts an STT call the moment a segment exists (bounded
    # by a semaphore); the consumer delivers each result as soon as THAT
    # result is ready, in spoken order. Keeping these separate is the whole
    # point: the previous version only drained a finished call when the
    # in-flight count hit its cap, so a completed transcript sat undelivered
    # until two MORE segments happened to arrive -- adding several seconds to
    # every question, and gluing the tail of a question onto whatever noise
    # eventually unblocked it.
    segment_queue: asyncio.Queue[AudioSegment | UtteranceClosed | None] = asyncio.Queue()
    result_queue: asyncio.Queue[
        tuple[AudioSegment, asyncio.Task, float] | UtteranceClosed | None
    ] = asyncio.Queue()

    dispatcher = asyncio.create_task(
        _transcription_dispatcher(
            stt=stt,
            segment_queue=segment_queue,
            result_queue=result_queue,
            max_concurrent_segments=settings.stt_max_concurrent_segments,
        )
    )
    consumer = asyncio.create_task(
        _result_consumer(
            websocket=websocket,
            result_queue=result_queue,
            session_id=session_id,
            settings=settings,
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

    speech_active = False

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
                        "speech_seconds": round(segment.speech_seconds, 2),
                        "utterance_id": segment.utterance_id,
                        "is_final": segment.is_final,
                    },
                )
                await segment_queue.put(segment)

            # An utterance that ended without a final segment -- the speaker
            # stopped inside a force-cut, or the tail was too quiet to be
            # worth transcribing -- has no transcript coming to report its
            # end. Nothing contradicted "more is coming", so the gate held a
            # finished question for the full awaiting-more cap (measured:
            # ~20s instead of ~1.4s) until the speaker happened to say
            # something else. Queued rather than reported directly so it
            # lands after the transcripts of that utterance's earlier
            # segments (see UtteranceClosed).
            for closed_id in segmenter.take_closed_utterances():
                await segment_queue.put(UtteranceClosed(utterance_id=closed_id))

            # Tell the gate whenever the interviewer starts or stops talking.
            # Without this the gate can only infer "are they still going?"
            # from the delay between transcripts -- and most of that delay is
            # the time taken to SPEAK the next sentence, which is
            # indistinguishable from having finished. That is what caused a
            # scenario question's setup sentences to be discarded one by one
            # before the actual question arrived.
            if segmenter.speech_active != speech_active:
                speech_active = segmenter.speech_active
                await get_question_pipeline().set_speech_active(
                    session_id=session_id, active=speech_active
                )
    except WebSocketDisconnect:
        logger.info("Audio websocket disconnected")
    finally:
        flushed = segmenter.flush()
        if flushed:
            await segment_queue.put(flushed)
        for closed_id in segmenter.take_closed_utterances():
            await segment_queue.put(UtteranceClosed(utterance_id=closed_id))

        # Whatever state the VAD was left in, the interviewer is not
        # talking once the socket is gone -- otherwise a buffer could sit
        # held open forever waiting for a continuation.
        if speech_active:
            await get_question_pipeline().set_speech_active(
                session_id=session_id, active=False
            )

        await segment_queue.put(None)
        await dispatcher
        await result_queue.put(None)
        await consumer
        logger.info(
            "Audio session %s closed (%d sub-threshold segments never sent to STT)",
            session_id,
            segmenter.dropped_segments,
        )
        # The one failure that otherwise leaves NO trace anywhere: audio that
        # stood above the measured noise floor but never reached the speech
        # trigger. No segment means no STT call, no transcript, no event -- so
        # a question missed this way was previously undiagnosable after the
        # fact. Reported at WARNING because it is actionable: it means
        # VAD_ENERGY_THRESHOLD is too high for this line.
        if segmenter.sub_threshold_runs:
            logger.warning(
                "Audio session %s: %d run(s) of audio were above the noise floor but "
                "below the speech trigger (%.0f RMS) and were never transcribed; "
                "loudest reached %.0f RMS. If questions went unanswered, lower "
                "VAD_ENERGY_THRESHOLD.",
                session_id,
                segmenter.sub_threshold_runs,
                segmenter.trigger_level,
                segmenter.sub_threshold_peak,
            )


async def _transcription_dispatcher(
    *,
    stt,
    segment_queue: asyncio.Queue[AudioSegment | UtteranceClosed | None],
    result_queue: asyncio.Queue[tuple[AudioSegment, asyncio.Task, float] | UtteranceClosed | None],
    max_concurrent_segments: int,
) -> None:
    """Start an STT call per segment, at most `max_concurrent_segments` at a
    time, and hand each in-flight call straight to the consumer in the order
    the audio was spoken.

    A long utterance is force-cut into several segments, and those calls
    genuinely overlap in wall-clock time instead of queueing behind each
    other -- but the concurrency cap keeps one talkative session from firing
    an unbounded number of parallel requests at the provider.
    """
    limiter = asyncio.Semaphore(max_concurrent_segments)

    async def _run(segment: AudioSegment) -> TranscriptionResult:
        async with limiter:
            return await stt.transcribe_pcm16(segment.pcm, sample_rate=segment.sample_rate)

    while True:
        segment = await segment_queue.get()
        if segment is None:
            return

        # Not audio: an end-of-utterance marker, passed straight through so
        # the consumer sees it in spoken order relative to the segments
        # around it. No STT call, nothing to await.
        if isinstance(segment, UtteranceClosed):
            await result_queue.put(segment)
            continue

        started_at = time.perf_counter()
        task = asyncio.create_task(_run(segment))
        await result_queue.put((segment, task, started_at))


async def _result_consumer(
    *,
    websocket: WebSocket,
    result_queue: asyncio.Queue[tuple[AudioSegment, asyncio.Task, float] | UtteranceClosed | None],
    session_id: str,
    settings,
) -> None:
    """Deliver transcription results in spoken order, each one the moment it
    is ready."""
    while True:
        item = await result_queue.get()
        if item is None:
            return

        if isinstance(item, UtteranceClosed):
            # Every transcript spoken before this point has already been
            # delivered above, so it is now safe to say this utterance is
            # over and nothing more of it is coming.
            await get_question_pipeline().close_utterance(
                session_id=session_id, utterance_id=item.utterance_id
            )
            continue

        segment, task, started_at = item
        await _handle_transcription_result(
            websocket=websocket,
            session_id=session_id,
            segment=segment,
            task=task,
            started_at=started_at,
            settings=settings,
        )


async def _handle_transcription_result(
    *,
    websocket: WebSocket,
    session_id: str,
    segment: AudioSegment,
    task: asyncio.Task,
    started_at: float,
    settings,
) -> None:
    # The task may already have been running for a while by the time we get
    # here (it started at `started_at`, concurrently with whatever segment(s)
    # were drained before this one) -- base the timeout on time remaining
    # since it actually started, so the real end-to-end cap per segment stays
    # stt_timeout_seconds regardless of queueing position.
    stt_timeout_seconds = settings.stt_timeout_seconds
    remaining = max(0.1, stt_timeout_seconds - (time.perf_counter() - started_at))
    try:
        result: TranscriptionResult = await asyncio.wait_for(task, timeout=remaining)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning("STT transcription timed out after %ss", stt_timeout_seconds)
        await _close_utterance(session_id, segment)
        await _send_json_safe(
            websocket,
            {
                "type": "error",
                "message": (
                    f"STT timed out after {stt_timeout_seconds}s. "
                    "Check your network/STT provider status."
                ),
            },
        )
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("STT transcription failed")
        await _close_utterance(session_id, segment)
        await _send_json_safe(
            websocket,
            {
                "type": "error",
                "message": f"STT failed ({type(exc).__name__}): {exc}",
            },
        )
        return

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    text = result.text.strip()
    if not text:
        await _close_utterance(session_id, segment)
        return

    # Whisper-family models return TEXT for silence rather than nothing, so
    # every transcript is checked against three independent signals before it
    # is allowed to become a question: how much of the segment was actually
    # voiced (from the VAD), what the model thought of its own output, and
    # whether the words look like words at all.
    if result.confidence_known and result.no_speech_prob >= settings.stt_no_speech_threshold:
        logger.info(
            "Dropping likely-hallucinated transcript (no_speech_prob=%.2f): %s",
            result.no_speech_prob,
            text,
        )
        await _close_utterance(session_id, segment)
        return

    verdict = assess_transcript(
        text,
        confidence=result.confidence,
        confidence_known=result.confidence_known,
        speech_seconds=segment.speech_seconds,
        min_confidence=settings.stt_drop_confidence_threshold,
    )
    if not verdict.keep:
        logger.info("Dropping transcript (%s): %s", verdict.reason, text)
        await _close_utterance(session_id, segment)
        await _send_json_safe(
            websocket,
            {
                "type": "transcript_dropped",
                "text": text,
                "reason": verdict.reason,
                "confidence": round(result.confidence, 2),
                "speech_seconds": round(segment.speech_seconds, 2),
            },
        )
        return

    low_confidence = result.confidence_known and (
        result.confidence < settings.stt_confidence_threshold
    )
    logger.info(
        "Transcript [utt %s%s] (%.0fms STT, conf=%.2f%s): %s",
        segment.utterance_id,
        "" if segment.is_final else ", partial",
        latency_ms,
        result.confidence,
        "" if result.confidence_known else " unscored",
        text,
    )

    await get_question_pipeline().submit_transcript(
        session_id=session_id,
        text=text,
        confidence=result.confidence,
        confidence_known=result.confidence_known,
        utterance_id=segment.utterance_id,
        utterance_final=segment.is_final,
        spoken_at=segment.captured_at,
        stt_latency_ms=latency_ms,
    )
    await _send_json_safe(
        websocket,
        {
            "type": "transcript",
            "text": text,
            "segment_seconds": round(segment.duration_seconds, 2),
            "speech_seconds": round(segment.speech_seconds, 2),
            "stt_latency_ms": latency_ms,
            "confidence": round(result.confidence, 2),
            "low_confidence": low_confidence,
            "utterance_id": segment.utterance_id,
            "is_final": segment.is_final,
        },
    )


async def _close_utterance(session_id: str, segment: AudioSegment) -> None:
    """Tell the gate this utterance produced nothing usable.

    Without this, a question whose FINAL segment was dropped (transcription
    failed, or came back as noise) would leave the gate waiting for a
    continuation that is never coming, holding a real question unanswered.
    """
    await get_question_pipeline().close_utterance(
        session_id=session_id, utterance_id=segment.utterance_id
    )


async def _send_json_safe(websocket: WebSocket, payload: dict) -> None:
    try:
        await websocket.send_json(payload)
    except (RuntimeError, WebSocketDisconnect):
        logger.debug("Could not send websocket payload; connection is closed")


async def _build_stt_prompt(settings, session_id: str) -> str:
    """Combine the generic tech-vocabulary bias with this session's actual
    resume/JD text, so the recognizer is biased toward terms that are likely
    to come up (e.g. specific frameworks/versions from the JD) -- this is
    what fixes misheard jargon like "React 19" -> "reactivity"."""
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
