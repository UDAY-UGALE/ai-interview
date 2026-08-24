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
from app.services.session_vocabulary import as_whisper_prompt
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
        # One vocabulary, two shapes: Whisper takes a prompt string, Deepgram
        # takes a keyterm list. Both carry the same session terms, so
        # switching provider does not silently change what the recognizer is
        # biased toward.
        vocabulary, _lookup = await get_question_pipeline().session_vocabulary(session_id)
        stt_prompt = await _build_stt_prompt(settings, session_id, vocabulary)
        # Which provider this is (Groq Whisper, Deepgram, an NVIDIA model, a
        # local one) is decided entirely inside the registry -- this route
        # only knows it has something with transcribe_pcm16.
        stt = build_stt_service(settings, prompt=stt_prompt, keyterms=vocabulary)
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

    # Optional live Deepgram socket, running ALONGSIDE the VAD rather than
    # replacing it. The division of labour is deliberate and measured:
    #
    #   * the VAD keeps producing speech_active, which is the gate's only
    #     factual "they are talking right now" signal and the thing that
    #     stops a scenario question's setup being discarded. It is good at
    #     that job and nothing here changes it.
    #   * Deepgram does the transcription, and delivers the final transcript
    #     essentially AT end-of-speech: measured p50 20ms after the speaker
    #     stopped, against ~764ms for the segment path (420ms of VAD
    #     end-silence before a segment even closes, plus ~344ms of Whisper
    #     round trip).
    #
    # While this is on, segments are still cut and reported to the overlay
    # but are NOT sent for transcription -- otherwise every question would
    # be transcribed and submitted twice.
    streaming = await _maybe_start_streaming(
        settings=settings,
        session_id=session_id,
        websocket=websocket,
        keyterms=vocabulary,
    )

    await websocket.send_json(
        {
            "type": "ready",
            "session_id": session_id,
            "sample_rate": settings.audio_sample_rate,
            "frame_ms": settings.audio_frame_ms,
            "stt_provider": "deepgram-stream" if streaming else getattr(stt, "name", ""),
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

            if streaming is not None:
                await streaming.send_audio(chunk)

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
                # Deepgram is already transcribing this audio off the live
                # socket; sending the segment as well would transcribe and
                # submit every question twice.
                if streaming is None:
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
        if streaming is not None:
            await streaming.aclose()
        flushed = segmenter.flush()
        if flushed and streaming is None:
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


async def _maybe_start_streaming(*, settings, session_id: str, websocket, keyterms):
    """Open the live Deepgram socket, if it is enabled and usable.

    Returns None when streaming is off or unavailable, and the caller then
    runs the ordinary VAD-segment path unchanged. A streaming failure must
    never take the session down: the segment path is a complete, working
    recognizer on its own, so anything that goes wrong here degrades to it
    rather than to silence.
    """
    if not settings.deepgram_streaming:
        return None
    if not settings.deepgram_api_key:
        logger.warning(
            "DEEPGRAM_STREAMING=true but DEEPGRAM_API_KEY is not set; "
            "falling back to the segment transcription path."
        )
        return None

    from app.services.stt.deepgram import DeepgramStreamingSession, StreamingTranscript

    pipeline = get_question_pipeline()

    async def _on_transcript(transcript: StreamingTranscript) -> None:
        if not transcript.text:
            return
        if not transcript.is_final:
            # Interim hypotheses are shown, never acted on. They can still
            # change, and acting on one is how a half-question gets
            # answered -- the exact defect the rest of this work removes.
            await _send_json_safe(
                websocket,
                {
                    "type": "transcript_interim",
                    "text": transcript.text,
                    "stt_provider": transcript.provider,
                    "stt_is_final": False,
                    "stt_confidence": round(transcript.confidence, 3),
                    "stt_endpoint_reason": transcript.endpoint_reason,
                },
            )
            return

        verdict = assess_transcript(
            transcript.text,
            confidence=transcript.confidence,
            confidence_known=transcript.confidence_known,
            # No VAD segment to measure, so the voiced-duration signal is
            # not available here. Passing None rather than a made-up number
            # keeps the filter honest: it falls back to the confidence and
            # word-shape checks, both of which Deepgram supports.
            speech_seconds=None,
            min_confidence=settings.stt_drop_confidence_threshold,
        )
        if not verdict.keep:
            logger.info("Dropping streaming transcript (%s): %s", verdict.reason, transcript.text)
            await _send_json_safe(
                websocket,
                {
                    "type": "transcript_dropped",
                    "text": transcript.text,
                    "reason": verdict.reason,
                    "confidence": round(transcript.confidence, 2),
                },
            )
            return

        await pipeline.submit_transcript(
            session_id=session_id,
            text=transcript.text,
            confidence=transcript.confidence,
            confidence_known=transcript.confidence_known,
            utterance_final=True,
            stt_latency_ms=transcript.latency_ms,
            stt_provider=transcript.provider,
        )
        await _send_json_safe(
            websocket,
            {
                "type": "transcript",
                "text": transcript.text,
                "stt_latency_ms": transcript.latency_ms,
                "confidence": round(transcript.confidence, 2),
                "low_confidence": transcript.confidence_known
                and transcript.confidence < settings.stt_confidence_threshold,
                "is_final": True,
                "stt_provider": transcript.provider,
                "stt_model": settings.deepgram_model,
                "stt_is_final": True,
                "stt_confidence": round(transcript.confidence, 3),
                "stt_endpoint_reason": transcript.endpoint_reason,
            },
        )

    session = DeepgramStreamingSession(
        settings=settings,
        sample_rate=settings.audio_sample_rate,
        keyterms=keyterms,
        on_transcript=_on_transcript,
    )
    try:
        await session.start()
    except Exception:
        logger.warning(
            "Could not open the Deepgram streaming socket; using the segment "
            "transcription path instead.",
            exc_info=True,
        )
        return None
    return session


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
        stt_provider=result.provider,
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
            # Which recognizer actually produced this. Recorded per
            # transcript rather than per session because the fallback path
            # can switch provider mid-interview, and a comparison that
            # cannot tell which engine produced which line is not a
            # comparison.
            "stt_provider": result.provider,
            "stt_model": result.model,
            "stt_is_final": segment.is_final,
            "stt_confidence": round(result.confidence, 3),
            "stt_endpoint_reason": "vad_end_silence" if segment.is_final else "vad_force_cut",
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


async def _build_stt_prompt(settings, session_id: str, vocabulary: list[str]) -> str:
    """Bias the recognizer toward the terms this interview will actually use.

    This used to paste the job description and the resume into the prompt as
    PROSE, and it did not survive contact with a real session. The prompt has
    a hard ~850 UTF-8 byte cap, the fixed vocabulary already spent 433 of
    them, and the resume was appended last -- so with a 500-character JD
    loaded the total came to 1,114 bytes and the truncation removed the
    resume ENTIRELY. Measured directly: "resume text reached the recognizer:
    False". The candidate's own project and tool names, which are both the
    likeliest words to be spoken and the likeliest to be misheard, were
    getting no biasing at all while the code comment claimed they were the
    fix for misheard jargon.

    A term list says the same thing in a fraction of the space, and the
    ordering guarantees that if anything is dropped it is the generic tail
    rather than this session's own words. See app/services/session_vocabulary.
    """
    if vocabulary:
        return as_whisper_prompt(vocabulary)

    # No resume, no JD, nothing discussed yet -- fall back to the static
    # vocabulary, which is what a fresh session had before any of this.
    return _truncate_prompt_bytes(settings.stt_prompt or "")
