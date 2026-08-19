import asyncio
import difflib
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TypedDict

from app.core.config import Settings, get_settings
from app.core.redis_client import InterviewSessionContext, get_session_store
from app.services.answer_hub import answer_hub
from app.services.llm import MissingLLMConfigError, build_llm_client, client_supports_vision
from app.services.transcript_quality import lexical_word_ratio


try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    END = START = StateGraph = None


logger = logging.getLogger(__name__)


class GateState(TypedDict):
    text: str
    should_answer: bool
    reason: str


@dataclass(slots=True)
class GateResult:
    should_answer: bool
    reason: str
    # The part of the buffered transcript that is actually the question.
    # Live speech-to-text hands the gate a mix of the real question and
    # whatever fragments landed around it; `focus` is what gets sent to the
    # LLM, so a question never arrives wrapped in unrelated noise. Empty
    # means "the whole transcript".
    focus: str = ""
    # Whether the text looks like a finished thought. Drives whether the
    # answer fires immediately or waits a moment for the rest of a sentence.
    complete: bool = True
    # Canonical interview intent when one was recognised ("tell me about
    # yourself" and its many mis-transcriptions all resolve to one label),
    # so an imperfect transcript can still be answered for what was meant.
    intent: str = ""

    def answer_text(self, transcript: str) -> str:
        return self.focus or transcript


@dataclass(slots=True)
class _PendingTranscript:
    parts: list[str] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    # Tracks a currently-streaming _generate_answer call separately from
    # `task` (the debounce/gate task) -- _process_after_debounce clears
    # `task` to None before it starts generating an answer (see below), so
    # without this a new transcript arriving mid-answer had nothing left to
    # cancel and would let a second answer generate concurrently, streaming
    # interleaved tokens into the same overlay session as the first.
    answer_task: asyncio.Task[None] | None = None
    first_seen: float = 0.0
    min_confidence: float = 1.0
    # When the most recent fragment arrived. The buffer is held open based
    # on silence since THIS, not on total elapsed time -- an unrelated
    # fragment that nothing follows gets discarded quickly instead of
    # loitering for seconds and being glued onto the next real question.
    last_part_at: float = 0.0
    # Id of the VAD utterance the most recent fragment came from, and
    # whether that utterance is still open (the segment was force-cut
    # mid-sentence rather than ending on real silence). While it is open we
    # KNOW more of the same sentence is coming, so there is nothing to
    # guess about and no reason to answer yet.
    utterance_id: int | None = None
    awaiting_more_of_utterance: bool = False
    # True while the VAD says the interviewer is talking RIGHT NOW. The
    # single most useful signal the gate has: it distinguishes "they went
    # quiet, act on what we have" from "they are three words into the next
    # sentence" -- which are identical as far as any timer can tell, because
    # the wait between two transcripts of one question is mostly the time
    # taken to say the next sentence.
    speech_active: bool = False
    # Memoised tier-2 classifier verdicts, keyed by the exact text that was
    # classified -- the debounce loop re-evaluates the same buffer every
    # cycle, and without this each cycle paid for another network call.
    intent_cache: dict[str, GateResult] = field(default_factory=dict)


class QuestionAnswerPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = get_session_store()
        self._pending: dict[str, _PendingTranscript] = {}
        self._lock = asyncio.Lock()
        self._gate_graph = _build_gate_graph()

    async def submit_transcript(
        self,
        *,
        session_id: str,
        text: str,
        confidence: float = 1.0,
        confidence_known: bool = False,
        utterance_id: int | None = None,
        utterance_final: bool = True,
        spoken_at: float | None = None,
    ) -> None:
        """Feed one transcription result into the pipeline.

        `utterance_id` / `utterance_final` come from the VAD and are what
        turn a stream of independent transcription results back into
        utterances. Several results with the same id are pieces of ONE thing
        the interviewer said (a long question force-cut for transcription);
        a new id is a genuinely new thing. Without that distinction the only
        available signal was elapsed time, which is why unrelated fragments
        ended up concatenated into questions nobody asked.
        """
        cleaned = _clean_transcript(text)
        if not cleaned:
            return

        await answer_hub.broadcast_json(
            session_id,
            {
                "type": "transcript",
                "session_id": session_id,
                "text": cleaned,
                "confidence": round(confidence, 2),
                "low_confidence": (
                    confidence_known and confidence < self._settings.stt_confidence_threshold
                ),
            },
        )

        now = time.monotonic()
        async with self._lock:
            pending = self._pending.setdefault(session_id, _PendingTranscript())

            # Buffer lifetime is owned entirely by the decision loop, which
            # runs every question_debounce_ms and drops anything it is not
            # still deliberately waiting on. So by the time a genuinely
            # unrelated fragment arrives, the earlier one is already gone and
            # cannot be inherited -- while a fragment the loop is actively
            # holding for (a question that trails off mid-sentence) is not
            # yanked out from under it here.
            if not pending.parts:
                pending.first_seen = spoken_at or now
            pending.parts.append(cleaned)
            pending.min_confidence = min(pending.min_confidence, confidence)
            pending.last_part_at = now
            pending.utterance_id = utterance_id
            pending.awaiting_more_of_utterance = not utterance_final
            if pending.task and not pending.task.done():
                pending.task.cancel()
            # Deliberately does NOT touch pending.answer_task here. Cancelling
            # a streaming answer/screen-analysis the instant ANY new speech
            # arrives -- before knowing whether it's a real new question or
            # just noise/a stray mistranscribed fragment -- means a single
            # bad STT segment can kill a perfectly good in-progress answer.
            # That's exactly what happened in practice: with audio capture
            # producing frequent garbage transcripts, almost no answer or
            # screen analysis survived long enough to finish. Cancellation of
            # a running answer now only happens in _process_after_debounce,
            # once the gate has actually CONFIRMED this new text is a real,
            # different question -- not on mere arrival.
            pending.task = asyncio.create_task(self._process_after_debounce(session_id))

    async def set_speech_active(self, *, session_id: str, active: bool) -> None:
        """Report whether the interviewer is talking right now (from the VAD).

        While this is true the gate will not discard a buffered part-question,
        no matter how long the pause since the last transcript -- because that
        pause is them speaking the next sentence, not them finishing.
        """
        async with self._lock:
            pending = self._pending.setdefault(session_id, _PendingTranscript())
            pending.speech_active = active
            if active:
                # Speech resuming counts as activity: it keeps the buffer
                # alive through the whole of the next sentence, rather than
                # it expiring while that sentence is still being spoken.
                pending.last_part_at = time.monotonic()

    async def close_utterance(self, *, session_id: str, utterance_id: int | None) -> None:
        """Report that an utterance ended without producing usable text.

        The gate deliberately keeps waiting while the VAD says an utterance
        is still open. If the last piece of that utterance is dropped
        (transcription failed, timed out, or came back as noise), nothing
        would ever contradict that and the buffered question would sit
        unanswered until a timeout expired. This closes the loop instead.
        """
        async with self._lock:
            pending = self._pending.get(session_id)
            if pending is None or not pending.awaiting_more_of_utterance:
                return
            if utterance_id is not None and pending.utterance_id != utterance_id:
                return
            pending.awaiting_more_of_utterance = False

    async def ask_directly(self, *, session_id: str, text: str) -> None:
        """Manual override: skip STT/the gate entirely and answer exactly
        this text. Used when the interviewer's question was misheard and
        the person corrects it by hand (or types a question in) -- cancels
        any in-flight/pending automatic answer for this session first."""
        cleaned = _clean_transcript(text)
        if not cleaned:
            return

        async with self._lock:
            pending = self._pending.setdefault(session_id, _PendingTranscript())
            if pending.task and not pending.task.done():
                pending.task.cancel()
            if pending.answer_task and not pending.answer_task.done():
                pending.answer_task.cancel()
            pending.parts.clear()
            pending.first_seen = 0.0
            pending.min_confidence = 1.0
            # Stored in answer_task (not task) so an ambient/noise transcript
            # arriving right after this -- submit_transcript() only ever
            # cancels `task`, never `answer_task` directly -- can't kill this
            # manually-triggered answer the way it could when this used to
            # live in `task`.
            pending.answer_task = asyncio.create_task(
                self._generate_answer(session_id, cleaned, confidence=1.0, forced=True)
            )

    async def _process_after_debounce(self, session_id: str) -> None:
        """Decide when the buffered speech is a question worth answering.

        Runs as a loop rather than by re-scheduling itself, and every wait it
        performs is for a specific reason it can name:

        * the VAD says this utterance is still open -> more words are coming,
          there is nothing to decide yet
        * the text is answerable and looks finished -> answer NOW, no grace
          period (this is where seconds of the old latency lived: a short but
          complete question was held back on the chance it might continue)
        * the text is answerable but trails off mid-sentence -> hold briefly
          for the rest
        * the text is not answerable -> hold only as long as a continuation
          could plausibly still arrive, then drop it, so it can never be
          merged into a later question
        """
        try:
            while True:
                await asyncio.sleep(self._settings.question_debounce_ms / 1000)

                async with self._lock:
                    pending = self._pending.setdefault(session_id, _PendingTranscript())
                    transcript = _combine_transcript_parts(pending.parts)

                    if not transcript:
                        pending.task = None
                        return

                    now = time.monotonic()
                    since_last_part = now - pending.last_part_at if pending.last_part_at else 0.0
                    awaiting_more = pending.awaiting_more_of_utterance
                    speech_active = pending.speech_active
                    # Measured from the last thing said, not from the first.
                    # A 4-5 sentence scenario question takes longer to speak
                    # than any sane total budget, and capping on total
                    # elapsed time threw away the first half of it mid-
                    # question: the LLM then answered "how would you debug
                    # this?" having never seen the system, the symptom or
                    # the numbers. Silence is the only honest signal that a
                    # buffer is stuck rather than still being spoken.
                    timed_out = since_last_part >= self._settings.question_max_wait_seconds
                    gate = self._run_gate(transcript)
                    cached_intent = pending.intent_cache.get(transcript)

                # Tier 2 (fast intent classifier): only for the genuinely
                # ambiguous bucket the rule gate couldn't confidently decide
                # either way (reason=no_question_signal). Runs OUTSIDE the
                # lock since it's a network call -- holding a pipeline-wide
                # lock during it would serialize every other session's
                # question processing behind one slow classifier call.
                # Skipped entirely while the utterance is still open: there
                # is no point classifying half a sentence, and that was a
                # network round trip on the critical path of every question.
                if (
                    not awaiting_more
                    and not gate.should_answer
                    and gate.reason == "no_question_signal"
                    and self._settings.fast_intent_enabled
                ):
                    if cached_intent is not None:
                        gate = cached_intent
                    else:
                        gate = await self._run_fast_intent_classifier(transcript, session_id)
                        async with self._lock:
                            pending = self._pending.setdefault(session_id, _PendingTranscript())
                            pending.intent_cache[transcript] = gate

                async with self._lock:
                    pending = self._pending.setdefault(session_id, _PendingTranscript())

                    keep_waiting = _should_keep_waiting(
                        gate=gate,
                        settings=self._settings,
                        transcript=transcript,
                        awaiting_more=awaiting_more,
                        speech_active=speech_active,
                        since_last_part=since_last_part,
                        timed_out=timed_out,
                        looks_like_content=_is_content_clause(transcript),
                    )
                    if keep_waiting:
                        continue

                    pending.parts.clear()
                    pending.task = None
                    pending.intent_cache.clear()
                    pending.utterance_id = None
                    pending.awaiting_more_of_utterance = False
                    confidence = pending.min_confidence
                    pending.min_confidence = 1.0
                    pipeline_ms = (
                        int((time.monotonic() - pending.first_seen) * 1000)
                        if pending.first_seen
                        else 0
                    )
                    pending.first_seen = 0.0
                    pending.last_part_at = 0.0
                    prior_answer_task = pending.answer_task
                break

            # A still-streaming prior answer is only ever touched here, once
            # the gate has actually CONFIRMED this text is a real question --
            # never on mere arrival of new speech (see submit_transcript).
            # If should_answer is False (including the timed-out-while-
            # still-incomplete case), this transcript isn't a real answerable
            # question at all -- e.g. a stray mistranscribed noise fragment
            # -- so the prior answer is left completely undisturbed instead
            # of being killed by something that was never going to be
            # answered anyway.
            if gate.should_answer and prior_answer_task is not None and not prior_answer_task.done():
                if gate.reason == "follow_up":
                    # A follow-up is ABOUT the answer that's still streaming --
                    # cancelling it would leave _resolve_followup() with
                    # nothing to attach to (history is only written once an
                    # answer finishes normally). Wait for it instead.
                    try:
                        await prior_answer_task
                    except asyncio.CancelledError:
                        pass
                else:
                    # A confirmed real, DIFFERENT question -- an actual
                    # barge-in. Cancel the old answer now, not before.
                    prior_answer_task.cancel()

            # Run the answer as its own tracked task (rather than just
            # awaiting it inline) so a barge-in -- new speech arriving
            # while this is streaming -- has something to actually
            # cancel. Without this handle, submit_transcript() had no
            # way to stop an in-flight answer, so interrupting speech
            # started a SECOND, fully concurrent answer instead of
            # replacing the first -- both streamed answer_token events
            # into the same overlay session and interleaved.
            answer_task = asyncio.create_task(
                self._generate_answer(
                    session_id,
                    transcript,
                    confidence=confidence,
                    pipeline_ms=pipeline_ms,
                    precomputed_gate=gate,
                )
            )
            # Only tracked in pending.answer_task when it's a REAL answer
            # (should_answer=True). When should_answer is False, this task
            # just broadcasts the negative gate decision and returns
            # instantly -- nothing worth protecting from interruption -- and
            # registering it here would overwrite (and orphan) the handle to
            # an actual still-streaming answer from earlier, the same bug
            # class this whole answer_task tracking exists to prevent.
            if gate.should_answer:
                async with self._lock:
                    pending = self._pending.setdefault(session_id, _PendingTranscript())
                    pending.answer_task = answer_task

            try:
                await answer_task
            finally:
                if gate.should_answer:
                    async with self._lock:
                        pending = self._pending.setdefault(session_id, _PendingTranscript())
                        if pending.answer_task is answer_task:
                            pending.answer_task = None
        except asyncio.CancelledError:
            raise
        except Exception:
            # This runs as a detached task that nobody awaits, so without
            # this the session would just stop answering -- silently, with
            # no error anywhere -- for the rest of the interview.
            logger.exception("Question decision loop failed; clearing buffer")
            async with self._lock:
                pending = self._pending.setdefault(session_id, _PendingTranscript())
                pending.parts.clear()
                pending.task = None
                pending.first_seen = 0.0
                pending.last_part_at = 0.0
                pending.intent_cache.clear()
            await answer_hub.broadcast_json(
                session_id,
                {
                    "type": "error",
                    "session_id": session_id,
                    "message": "Question pipeline error; the buffer was reset.",
                },
            )

    async def _run_fast_intent_classifier(self, transcript: str, session_id: str) -> GateResult:
        """Tier 2 of the gate: for utterances the rule gate genuinely can't
        classify either way. Uses a small, fast LLM call to make a real
        semantic ANSWER/WAIT/IGNORE judgment instead of silently dropping
        anything that doesn't match a hand-written pattern -- this is what
        actually generalizes to phrasings we haven't seen before, instead of
        needing a new regex/word-list entry every time a new one slips
        through (which is how 'df vs du', 'why', and multi-sentence scenario
        questions all originally got missed)."""
        try:
            history = await self._store.get_history(session_id)
        except Exception:
            history = []

        recent = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in (history or [])[-2:])
        prompt = (
            "You are classifying a fragment of live speech captured during a technical "
            "interview call. Decide exactly ONE of:\n"
            "ANSWER -- this is a complete question or request that should be answered now.\n"
            "WAIT -- the speaker seems to be mid-thought and will likely continue; don't "
            "answer yet. This includes a complete sentence that only SETS UP a question "
            "that has not been asked yet (e.g. 'The production server's disk is full.' "
            "before 'How would you troubleshoot it?').\n"
            "IGNORE -- small talk, filler, background noise, or not addressed to the "
            "candidate at all; never answer this.\n\n"
            + (f"Recent context:\n{recent}\n\n" if recent else "")
            + f'Fragment: "{transcript}"\n\n'
            "Reply with exactly one word: ANSWER, WAIT, or IGNORE."
        )

        async def _classify() -> str:
            llm = build_llm_client(
                self._settings,
                provider="groq",
                model=self._settings.fast_intent_model,
                max_tokens=5,
                temperature=0.0,
            )
            decision = ""
            async for token in llm.stream_chat([{"role": "user", "content": prompt}]):
                decision += token
                if len(decision) > 15:
                    break
            return decision.strip().upper()

        try:
            # Bounded: this sits on the critical path between the interviewer
            # finishing and the answer starting, so a slow classifier call
            # must degrade to "no signal" rather than hold the answer.
            decision = await asyncio.wait_for(
                _classify(), timeout=self._settings.fast_intent_timeout_seconds
            )
        except asyncio.CancelledError:
            raise
        except (TimeoutError, asyncio.TimeoutError):
            logger.info("Fast intent classifier timed out; treating as no signal")
            return GateResult(False, "no_question_signal")
        except Exception:
            logger.exception("Fast intent classifier failed; defaulting to no-answer")
            return GateResult(False, "no_question_signal")

        if "ANSWER" in decision:
            return GateResult(True, "fast_intent_answer")
        if "WAIT" in decision:
            return GateResult(False, "fast_intent_wait")
        return GateResult(False, "fast_intent_ignore")

    async def _generate_answer(
        self,
        session_id: str,
        transcript: str,
        *,
        confidence: float = 1.0,
        forced: bool = False,
        pipeline_ms: int = 0,
        precomputed_gate: GateResult | None = None,
    ) -> None:
        answer_started = False
        started_at = time.perf_counter()

        try:
            if forced:
                # A real GateResult, not just a broadcast: the code below
                # branches on gate.reason, so leaving it unset made every
                # manual override (the /ask endpoint, used to correct a
                # misheard question by hand) die with UnboundLocalError
                # before it reached the LLM.
                gate = GateResult(True, "manual_override", focus=transcript)
                await answer_hub.broadcast_json(
                    session_id,
                    {
                        "type": "question_gate",
                        "session_id": session_id,
                        "text": transcript,
                        "should_answer": True,
                        "reason": "manual_override",
                    },
                )
            else:
                gate = precomputed_gate if precomputed_gate is not None else self._run_gate(transcript)
                await answer_hub.broadcast_json(
                    session_id,
                    {
                        "type": "question_gate",
                        "session_id": session_id,
                        "text": transcript,
                        "should_answer": gate.should_answer,
                        "reason": gate.reason,
                        # What the gate decided is actually the question,
                        # when that differs from everything it heard.
                        "focus": gate.focus if gate.focus != transcript else "",
                    },
                )
                if not gate.should_answer:
                    return
                # From here on the QUESTION is the clause the gate isolated,
                # not the whole buffer. Fragments that landed either side of
                # it are dropped rather than sent to the LLM as if the
                # interviewer had said them.
                transcript = gate.answer_text(transcript)

            context = await self._store.get_context(session_id)
            history = await self._store.get_history(session_id)

            effective_transcript = transcript

            if gate.reason == "follow_up":
                effective_transcript = _resolve_followup(
                    transcript,
                    history,
                )
            elif gate.reason == "keyword_match":
                effective_transcript = _resolve_keyword_mention(transcript)
            elif gate.intent and _lexical_word_ratio(transcript) < 0.75:
                # A recognised interview intent whose words came through
                # badly. Answering the literal mis-transcription would
                # produce nonsense, so the LLM is told what was MEANT --
                # this is the difference between the assistant failing on a
                # slightly garbled "tell me about yourself" and handling it.
                effective_transcript = _resolve_interview_intent(gate.intent, transcript)

            messages = _build_answer_messages(
                effective_transcript,
                context,
                history,
            )

            provider = context.answer_provider or self._settings.answer_provider
            model = context.answer_model or self._settings.answer_model
            llm = build_llm_client(self._settings, provider=provider, model=model)

            answer_started = True
            await answer_hub.broadcast_json(
                session_id,
                {
                    "type": "answer_start",
                    "session_id": session_id,
                    "question": transcript,
                    "provider": provider,
                    "model": model,
                    "confidence": round(confidence, 2),
                    # Forced True for keyword_match regardless of the STT
                    # confidence score -- this is always a guess (a bare term
                    # mention, not a real question), so the overlay should
                    # visibly flag it as uncertain even when STT itself was
                    # confident about the words it heard.
                    "low_confidence": (
                        confidence < self._settings.stt_confidence_threshold
                        or gate.reason == "keyword_match"
                    ),
                    # Time from the first heard word of this question to right
                    # before the LLM call starts -- i.e. VAD-silence-wait +
                    # STT-already-elapsed + debounce-wait + gate time combined.
                    # Real diagnostic data instead of guessing where time goes.
                    "pipeline_ms": pipeline_ms,
                },
            )

            answer_parts: list[str] = []
            first_token_sent = False
            async for token in llm.stream_chat(messages):
                answer_parts.append(token)
                payload = {
                    "type": "answer_token",
                    "session_id": session_id,
                    "token": token,
                }
                if not first_token_sent:
                    payload["first_token_latency_ms"] = int(
                        (time.perf_counter() - started_at) * 1000
                    )
                    first_token_sent = True
                await answer_hub.broadcast_json(session_id, payload)

            answer = "".join(answer_parts).strip()
            if answer:
                await self._store.add_history(session_id, transcript, answer)

            await answer_hub.broadcast_json(
                session_id,
                {
                    "type": "answer_done",
                    "session_id": session_id,
                    "answer": answer,
                },
            )
        except asyncio.CancelledError:
            if answer_started:
                await answer_hub.broadcast_json(
                    session_id,
                    {"type": "answer_cancelled", "session_id": session_id},
                )
            raise
        except MissingLLMConfigError as exc:
            await answer_hub.broadcast_json(
                session_id,
                {"type": "error", "session_id": session_id, "message": str(exc)},
            )
        except Exception as exc:
            logger.exception("Question/answer pipeline failed")
            # Show the real reason (bounded) instead of a vague message --
            # rate limits, auth errors, and timeouts all look identical if
            # hidden behind "check backend logs", which makes them
            # impossible to diagnose from the overlay alone.
            await answer_hub.broadcast_json(
                session_id,
                {
                    "type": "error",
                    "session_id": session_id,
                    "message": f"Answer generation failed ({type(exc).__name__}): {str(exc)[:200]}",
                },
            )

    async def analyze_screen(
        self,
        *,
        session_id: str,
        image_base64: str,
        media_type: str = "image/png",
        question: str | None = None,
    ) -> None:
        """Manual screen-analysis request from the overlay's 'Analyze
        Screen' button. Cancels any in-flight/pending automatic answer for
        this session (same as ask_directly), then sends the screenshot +
        an optional guiding question to a vision-capable LLM and streams
        the result back through the same answer_start/answer_token/
        answer_done events as a normal spoken question, so the overlay
        renders it identically without any special-casing."""
        display_question = (question or "").strip() or "What's on my screen? Help me with it."

        async with self._lock:
            pending = self._pending.setdefault(session_id, _PendingTranscript())
            if pending.task and not pending.task.done():
                pending.task.cancel()
            if pending.answer_task and not pending.answer_task.done():
                pending.answer_task.cancel()
            pending.parts.clear()
            pending.first_seen = 0.0
            pending.min_confidence = 1.0
            # Stored in answer_task (not task) -- same reasoning as
            # ask_directly: submit_transcript() never cancels answer_task on
            # a mere new transcript, only `task`, so an ambient/noise
            # transcript arriving during the (often several-second) vision
            # call can no longer kill this screen analysis mid-flight the
            # way it could when this used to live in `task`.
            pending.answer_task = asyncio.create_task(
                self._generate_screen_answer(
                    session_id,
                    display_question,
                    image_base64=image_base64,
                    media_type=media_type,
                )
            )

    async def _generate_screen_answer(
        self,
        session_id: str,
        display_question: str,
        *,
        image_base64: str,
        media_type: str,
    ) -> None:
        answer_started = False
        started_at = time.perf_counter()

        try:
            await answer_hub.broadcast_json(
                session_id,
                {
                    "type": "question_gate",
                    "session_id": session_id,
                    "text": display_question,
                    "should_answer": True,
                    "reason": "screen_analysis",
                },
            )

            context = await self._store.get_context(session_id)
            system_prompt, user_text = _build_screen_analysis_prompt(display_question, context)

            llm = build_llm_client(
                self._settings,
                provider=self._settings.vision_provider,
                model=self._settings.vision_model,
                max_tokens=self._settings.vision_max_tokens,
                temperature=self._settings.vision_temperature,
            )
            if not client_supports_vision(llm):
                raise MissingLLMConfigError(
                    f"provider={self._settings.vision_provider} model="
                    f"{self._settings.vision_model} has no vision support; set "
                    "VISION_PROVIDER/VISION_MODEL in .env to a vision-capable model "
                    "(e.g. groq/qwen/qwen3.6-27b, openai/gpt-4o, or "
                    "anthropic/claude-3-5-sonnet-latest)."
                )

            answer_started = True
            await answer_hub.broadcast_json(
                session_id,
                {
                    "type": "answer_start",
                    "session_id": session_id,
                    "question": display_question,
                    "provider": self._settings.vision_provider,
                    "model": self._settings.vision_model,
                    "confidence": 1.0,
                    "low_confidence": False,
                    "pipeline_ms": 0,
                },
            )

            answer_parts: list[str] = []
            first_token_sent = False
            async for token in _strip_think_tags(
                llm.stream_vision_chat(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    image_base64=image_base64,
                    media_type=media_type,
                )
            ):
                answer_parts.append(token)
                payload = {
                    "type": "answer_token",
                    "session_id": session_id,
                    "token": token,
                }
                if not first_token_sent:
                    payload["first_token_latency_ms"] = int(
                        (time.perf_counter() - started_at) * 1000
                    )
                    first_token_sent = True
                await answer_hub.broadcast_json(session_id, payload)

            answer = "".join(answer_parts).strip()
            if answer:
                await self._store.add_history(session_id, f"[Screen] {display_question}", answer)

            await answer_hub.broadcast_json(
                session_id,
                {"type": "answer_done", "session_id": session_id, "answer": answer},
            )
        except asyncio.CancelledError:
            if answer_started:
                await answer_hub.broadcast_json(
                    session_id, {"type": "answer_cancelled", "session_id": session_id}
                )
            raise
        except MissingLLMConfigError as exc:
            await answer_hub.broadcast_json(
                session_id, {"type": "error", "session_id": session_id, "message": str(exc)}
            )
        except Exception as exc:
            logger.exception("Screen analysis failed")
            await answer_hub.broadcast_json(
                session_id,
                {
                    "type": "error",
                    "session_id": session_id,
                    "message": f"Screen analysis failed ({type(exc).__name__}): {str(exc)[:200]}",
                },
            )

    # def _run_gate(self, transcript: str) -> GateResult:
    #     if self._gate_graph is not None:
    #         state = self._gate_graph.invoke(
    #             {"text": transcript, "should_answer": False, "reason": ""}
    #         )
    #         return GateResult(
    #             should_answer=state["should_answer"],
    #             reason=state["reason"],
    #         )

    #     return _classify_question(transcript)


    def _run_gate(
        self,
        transcript: str,
        history=None,
    ) -> GateResult:
        return _classify_question(transcript)


def get_question_pipeline() -> QuestionAnswerPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = QuestionAnswerPipeline(get_settings())
    return _pipeline


_pipeline: QuestionAnswerPipeline | None = None


def _build_gate_graph():
    if StateGraph is None:
        logger.warning("langgraph is not installed; using direct question gate")
        return None

    graph = StateGraph(GateState)
    graph.add_node("classify", _classify_node)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", END)
    return graph.compile()


def _classify_node(state: GateState) -> GateState:
    result = _classify_question(state["text"])
    return {
        "text": state["text"],
        "should_answer": result.should_answer,
        "reason": result.reason,
    }

def _classify_question(text: str) -> GateResult:
    """Decide whether the buffered speech contains a question, and WHICH
    part of it is the question.

    The old version classified the buffer as one blob and anchored its
    pattern checks to the very start of it. Live speech-to-text does not
    cooperate with that: a real question routinely arrives with fragments
    stuck to the front and back of it ("Tt. Uday, tell me about yourself.
    Ct.js." is a verbatim example from a session log). Anchored to the
    start, that reads as "tt..." and matches nothing -- so a perfectly
    clear question was silently dropped. The same blob-shaped assumption in
    reverse let a stray "?" inside garbage promote the entire buffer to a
    question.

    So: split into clauses, judge each one, answer the strongest, and hand
    the LLM only the part that was actually asked.
    """
    normalized = _normalize(text)
    words = normalized.split()

    if not normalized:
        return GateResult(False, "empty")

    if normalized in _SMALL_TALK:
        return GateResult(False, "small_talk")

    # Follow-ups take priority -- a bare "why"/"how" would otherwise collide
    # with the incomplete-fragment fallback below.
    if _is_followup_question(normalized):
        return GateResult(True, "follow_up", focus=text.strip(), complete=True)

    clauses = _split_clauses(text)
    verdicts = [_classify_clause(clause) for clause in clauses]

    # Prefer the last clause with a STRONG signal over a later weak one:
    # "Pote, Java." matching on a bare keyword should not outrank an actual
    # question asked before it.
    chosen = _last_index(
        verdicts, lambda v: v.should_answer and v.reason in _STRONG_GATE_REASONS
    )
    if chosen is None:
        chosen = _last_index(verdicts, lambda v: v.should_answer)

    if chosen is not None:
        verdict = verdicts[chosen]
        return GateResult(
            True,
            verdict.reason,
            focus=_select_focus(clauses, verdicts, chosen, verdict.focus),
            complete=verdict.complete,
            intent=verdict.intent,
        )

    # Nothing in there is answerable yet -- say which kind of "not yet" it
    # is, since the wait policy treats them differently.
    if normalized.endswith("...") or (words and words[-1] in _INCOMPLETE_TRAILING_WORDS):
        return GateResult(False, "incomplete_fragment", complete=False)

    if len(words) <= 2:
        return GateResult(False, "too_short")

    return GateResult(False, "no_question_signal")


# Signals that mean "this really is a question", as opposed to the
# best-effort keyword guess.
_STRONG_GATE_REASONS = frozenset(
    {
        "question_mark",
        "question_prefix",
        "question_phrase",
        "interview_intent",
        "vs_comparison",
        "follow_up",
    }
)


def _classify_clause(clause: str) -> GateResult:
    """Judge one clause on its own."""
    normalized = _normalize(clause)
    if not normalized or normalized in _SMALL_TALK:
        return GateResult(False, "small_talk")

    complete = not _trails_off(normalized)

    # A recognised interview request beats everything else, and is matched
    # by MEANING rather than by exact words -- "tell me about yourself",
    # "tell me about your background", "walk me through your experience"
    # are the same request, and speech-to-text mangles all of them in
    # slightly different ways.
    intent, intent_span = _match_interview_intent(normalized)
    if intent:
        return GateResult(
            True,
            "interview_intent",
            focus=_focus_from_span(clause, normalized, intent_span),
            complete=complete,
            intent=intent,
        )

    # A question mark is a strong signal, but not on its own: speech-to-text
    # puts one on any rising intonation, including on a mistranscribed
    # fragment ("Cora?" was logged repeatedly). Requiring the clause to look
    # like a sentence as well is what stops a stray "?" inside garbage from
    # promoting the whole buffer to a question.
    if "?" in clause and _question_mark_is_credible(normalized):
        return GateResult(True, "question_mark", focus=clause.strip(), complete=True)

    located = _locate_question_start(normalized)
    if located is not None:
        return GateResult(
            True,
            "question_prefix",
            focus=_focus_from_span(clause, normalized, located),
            complete=complete,
        )

    if any(phrase in normalized for phrase in _QUESTION_PHRASES):
        return GateResult(True, "question_phrase", focus=clause.strip(), complete=complete)

    if re.search(r"\b\w[\w./-]*\s+(vs\.?|versus)\s+\w[\w./-]*\b", normalized):
        return GateResult(True, "vs_comparison", focus=clause.strip(), complete=complete)

    # Last resort: not shaped like a question at all, but it names a
    # specific, unambiguous tech term (a language, framework, tool) --
    # garbled/terse STT output often keeps a recognizable keyword intact
    # even when the sentence around it doesn't survive (e.g. "Pote, Java."
    # for a mangled real question). Explicit product tradeoff: this WILL
    # sometimes fire on a keyword mentioned in passing rather than actually
    # being asked about -- accepted in exchange for not missing real terse
    # mentions. _generate_answer marks these low_confidence in the overlay
    # and rewrites the question sent to the LLM (see _resolve_keyword_mention)
    # instead of handing it the raw garbled sentence, so both the visible
    # question and the answer are grounded in the term itself, not "Pote,".
    # ...and only when the clause is genuinely terse or garbled. A
    # well-formed sentence that merely mentions a technology is not someone
    # asking about it -- "We have a Django API in production." is the SETUP
    # of a scenario question, and treating it as a bare "Django?" mention
    # answered the setup instead of the question and threw the facts away.
    if _looks_terse_or_garbled(normalized) and _mentions_tech_keyword(normalized):
        return GateResult(True, "keyword_match", focus=clause.strip(), complete=True)

    return GateResult(False, "no_question_signal", complete=complete)


# Sentence boundaries, but only where the punctuation is followed by
# whitespace or the end of the text -- so "Node.js", "Next.js" and "3.5"
# stay in one piece instead of being shredded into fake clauses.
_CLAUSE_BOUNDARY = re.compile(r"(?<=[.?!])\s+")


def _looks_terse_or_garbled(normalized: str, max_words: int = 5) -> bool:
    """Is this a fragment rather than a sentence?

    Gates the last-resort keyword rule, which exists for terse or damaged
    speech-to-text output ("Pote, Java.") and not for ordinary sentences
    that happen to name a technology.
    """
    words = normalized.split()
    return len(words) <= max_words or lexical_word_ratio(normalized) < 0.75


# Words that introduce an EXAMPLE of a question rather than a question.
# "questions like what is product X", "queries such as how do I reset this".
_EXAMPLE_INTRODUCERS = frozenset({"like", "as", "eg", "e.g.", "example", "examples", "say"})


def _question_mark_is_credible(normalized: str) -> bool:
    """Does this clause earn its question mark?

    A real terse question is either several words long, a known follow-up,
    opens with a question word, or names something specific enough to be
    asked about. A one-word fragment that happens to end in "?" is none of
    those.
    """
    core = _strip_filler_lead(normalized).rstrip("?").strip()
    words = core.split()
    if len(words) >= 3:
        return True
    if _is_followup_question(normalized):
        return True
    if any(core.startswith(prefix.strip()) for prefix in _QUESTION_PREFIXES):
        return True
    return _mentions_tech_keyword(normalized)


def _split_clauses(text: str) -> list[str]:
    clauses = [part.strip() for part in _CLAUSE_BOUNDARY.split(text.strip())]
    return [clause for clause in clauses if clause]


def _last_index(items: list, predicate) -> int | None:
    for index in range(len(items) - 1, -1, -1):
        if predicate(items[index]):
            return index
    return None


def _select_focus(
    clauses: list[str], verdicts: list[GateResult], chosen: int, clause_focus: str
) -> str:
    """Grow the answer text outward from the question clause, over every
    neighbour the interviewer clearly said on purpose.

    This is what separates the cases a naive splitter confuses:

    * a scenario question ("The disk partition is full." + "The app stopped
      working." + "How would you troubleshoot?") needs its setup sentences
    * a multi-part question ("What is LoRA?" + "Why use LoRA?" + "Have you
      ever heard of LoRA?") is ONE question with three parts, and answering
      only the last part answers almost nothing
    * a question with mistranscribed noise stuck to it ("Tt." + "Uday, tell
      me about yourself." + "Ct.js.") must carry none of it

    A neighbour qualifies if it reads like a real sentence OR is itself a
    question. Both mean a person said it deliberately; a noise fragment is
    neither. Length alone is not the test -- "Why use LoRA?" is three words
    and is unmistakably part of the question.
    """

    def _belongs(index: int) -> bool:
        return verdicts[index].should_answer or _is_content_clause(clauses[index])

    start = chosen
    while start > 0 and _belongs(start - 1):
        start -= 1

    end = chosen
    while end + 1 < len(clauses) and _belongs(end + 1):
        end += 1

    parts = clauses[start:chosen] + [clause_focus or clauses[chosen]] + clauses[chosen + 1 : end + 1]
    return " ".join(part.strip() for part in parts if part.strip())


def _is_content_clause(clause: str) -> bool:
    normalized = _normalize(clause)
    if not normalized or normalized in _SMALL_TALK:
        return False
    words = normalized.split()
    return len(words) >= 4 and _lexical_word_ratio(clause) >= 0.7


def _trails_off(normalized: str) -> bool:
    words = normalized.split()
    return normalized.endswith("...") or bool(words and words[-1] in _INCOMPLETE_TRAILING_WORDS)


def _locate_question_start(normalized: str) -> tuple[int, int] | None:
    """Find where a question actually starts inside a clause.

    Returns the (start, end) word-index span to answer, or None. The span
    normally covers the whole clause -- it only starts later when the words
    before the question word are junk rather than content, which is how
    "correct hello what is django" answers "what is django" while "the disk
    is full how would you troubleshoot" keeps its setup.
    """
    words = normalized.split()
    if not words:
        return None

    for start in range(len(words)):
        candidate = " ".join(words[start:])
        core = _strip_filler_lead(candidate)
        # Only interrogative openers are searched for mid-clause. The
        # auxiliary/copula openers ("is ", "are ", "does ") are valid at the
        # very start of a question and nowhere else -- searched mid-clause
        # they match the middle of ordinary statements ("the disk partition
        # IS 100% full"), which turns a scenario setup into a question all
        # by itself.
        prefixes = _QUESTION_PREFIXES if start == 0 else _MIDCLAUSE_QUESTION_PREFIXES
        if not any(core.startswith(prefix) for prefix in prefixes):
            continue
        lead = " ".join(words[:start])
        # An example is not a question. "works well with short questions LIKE
        # what is product X" describes the kind of query the system handles,
        # it doesn't ask it -- but it ends in a question-shaped phrase, so a
        # plain prefix scan answered the setup sentence of a scenario
        # question and threw the real question away.
        #
        # Both positions are checked because "like" is also a filler word:
        # _strip_filler_lead removes it, so the same phrase matches one word
        # earlier with "like" as words[start] rather than words[start - 1].
        # A genuinely filler-led question ("so, like, what is Django?") is
        # unaffected -- leading filler is stripped at start == 0, which this
        # guard never touches.
        if start > 0 and (
            words[start] in _EXAMPLE_INTRODUCERS
            or words[start - 1] in _EXAMPLE_INTRODUCERS
        ):
            continue
        if lead and _is_content_clause(lead):
            return (0, len(words))
        return (start, len(words))
    return None


def _focus_from_span(clause: str, normalized: str, span: tuple[int, int] | None) -> str:
    """Map a word-index span on the normalised text back onto the original
    clause, so the answer keeps its real casing and punctuation."""
    if span is None:
        return clause.strip()
    start, _end = span
    if start <= 0:
        return clause.strip()
    words = clause.split()
    if start >= len(words):
        return clause.strip()
    return " ".join(words[start:]).strip()


# Curated from the same vocabulary STT is biased toward (see Settings.stt_prompt),
# broadened with common CS/interview terms -- keeping the "terms we expect to
# hear" and "terms we'll answer on a bare mention of" lists consistent. Each
# term here is specific/uncommon enough in everyday chatter that a bare
# mention is a safe signal, and match.group(0) IS the clean term -- exact
# single-token matches, no cleanup needed.
_TECH_KEYWORD_PATTERN = re.compile(
    r"\b(?:"
    r"react|next\.?js|typescript|javascript|node\.?js|python|fastapi|django|flask|"
    r"langchain|langgraph|rag|llm|gpt-?\d|claude|groq|redis|postgres(?:ql)?|mongodb|"
    r"chromadb|vector database|embeddings?|docker|kubernetes|k8s|aws|gcp|azure|"
    r"ci/cd|rest api|graphql|websocket|grpc|kafka|microservices?|erpnext|frappe|"
    r"java|golang|rust|sql|html|css|c\+\+|c#|"
    # data structures / algorithms
    r"linked list|hash ?map|hashtable|binary search|big[- ]?o|time complexity|"
    r"dynamic programming|recursion|"
    # OOP
    r"inheritance|polymorphism|encapsulation|abstraction|"
    # databases
    r"indexing|acid|sharding|"
    # ML
    r"overfitting|gradient descent|"
    # testing / process
    r"unit test(?:ing)?|\btdd\b|ci\/cd|agile|scrum|"
    # auth / system design
    r"authentication|authorization|oauth|jwt|load balanc\w*|caching|scalability"
    r")\b",
    re.IGNORECASE,
)
# Terms prone enough to STT mishearing that a single exact pattern won't catch
# common variants -- each maps to a fixed clean label instead of using the
# raw matched span, since the fuzzy match itself isn't fit to hand the LLM as
# a topic name. Add new terms here the same way: (loose pattern, clean label).
_FUZZY_KEYWORD_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # "solid" alone is too common in ordinary interview chatter ("that's a
    # solid answer") to trust bare -- only counts near "principle"/"principal"
    # (a very common STT mishearing of "principles"), specific enough to be safe.
    (re.compile(r"\bsolid\b[\s,]{0,4}princip", re.IGNORECASE), "SOLID design principles"),
    (re.compile(r"\bfine[\s-]?tun\w*", re.IGNORECASE), "fine-tuning a model"),
    (re.compile(r"\bnormali[sz]\w*", re.IGNORECASE), "database normalization"),
    (re.compile(r"\bde ?normali[sz]\w*", re.IGNORECASE), "database denormalization"),
    (re.compile(r"\bcap\b[\s,]{0,4}theorem", re.IGNORECASE), "the CAP theorem"),
)


# The handful of requests every interview opens with, grouped by what they
# actually MEAN. Anything in a group is answered the same way, so a
# transcript that lands on any of them -- or near enough to one -- gets the
# same treatment. Matching is fuzzy on purpose: "tell me about yourself" has
# been observed coming back as "tell me about your self", "tell me about
# yourselves", and worse, and a hardcoded string list only ever covers the
# mis-hearings that have already happened.
_INTERVIEW_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "tell me about yourself",
        (
            "tell me about yourself",
            "tell us about yourself",
            "tell me about your self",
            "introduce yourself",
            "your introduction",
            "start with your introduction",
            "give me your introduction",
            "a brief introduction about yourself",
        ),
    ),
    (
        "your background",
        (
            "tell me about your background",
            "tell us about your background",
            "walk me through your background",
            "walk us through your background",
            "take me through your background",
            "start with your background",
            "give me your background",
        ),
    ),
    (
        "your experience",
        (
            "tell me about your experience",
            "tell us about your experience",
            "walk me through your experience",
            "walk us through your experience",
            "take me through your experience",
            "describe your experience",
            "share your experience",
        ),
    ),
    (
        "your projects",
        (
            "tell me about your project",
            "tell me about a project",
            "tell me about your projects",
            "walk me through your project",
            "explain your project",
            "describe your project",
            "tell me about one project",
        ),
    ),
    (
        "your current role",
        (
            "tell me about your current role",
            "what do you do currently",
            "tell me about your current company",
            "walk me through your current role",
        ),
    ),
)

# How close a window of words has to be to a canonical phrase to count.
# Tuned by hand against real mis-transcriptions: high enough that ordinary
# sentences do not collide with an intent, low enough to survive a wrong or
# dropped word.
_INTENT_MATCH_RATIO = 0.82


def _match_interview_intent(normalized: str) -> tuple[str, tuple[int, int] | None]:
    """Find a canonical interview request anywhere in the text.

    Returns (canonical label, word span) or ("", None). Scanning windows
    rather than checking the start is deliberate: "Uday, tell me about
    yourself." is the same request as "tell me about yourself", and the
    prefix-anchored check missed exactly that case in real sessions.
    """
    words = normalized.split()
    if not words:
        return "", None

    best: tuple[float, str, tuple[int, int]] | None = None

    for label, phrases in _INTERVIEW_INTENTS:
        for phrase in phrases:
            phrase_words = phrase.split()
            size = len(phrase_words)
            # Allow the window to run a word short or long so a dropped or
            # inserted word ("tell me a bit about yourself") still lines up.
            for width in {max(2, size - 1), size, size + 1}:
                for start in range(0, max(1, len(words) - width + 1)):
                    window = words[start : start + width]
                    if not window:
                        continue
                    ratio = difflib.SequenceMatcher(
                        None, " ".join(window), phrase
                    ).ratio()
                    if ratio >= _INTENT_MATCH_RATIO and (best is None or ratio > best[0]):
                        best = (ratio, label, (start, start + len(window)))

    if best is None:
        return "", None
    return best[1], best[2]


def _resolve_interview_intent(intent: str, transcript: str) -> str:
    """Rewrite a badly-transcribed but clearly-recognised interview request
    into what the interviewer actually meant, so the answer is grounded in
    the intent instead of in the transcription error."""
    return (
        f'The interviewer asked about "{intent}" -- the speech-to-text came through '
        f'imperfectly as "{transcript}", so answer the intended question, not the '
        f"literal words. Give a short, natural spoken answer as if asked about "
        f'"{intent}" directly.'
    )


def _lexical_word_ratio(text: str) -> float:
    """Share of tokens that look like real words -- see
    app.services.transcript_quality.lexical_word_ratio."""
    return lexical_word_ratio(text)


def _mentions_tech_keyword(normalized: str) -> bool:
    if _TECH_KEYWORD_PATTERN.search(normalized):
        return True
    return any(pattern.search(normalized) for pattern, _label in _FUZZY_KEYWORD_PATTERNS)


def _resolve_keyword_mention(transcript: str) -> str:
    """Rewrites a keyword_match transcript into a clean instruction naming
    just the detected term, instead of handing the LLM the literal garbled
    sentence (e.g. "Pote, Java.") as if it were the real question -- the
    words around the keyword are often STT noise, not real content."""
    normalized = _normalize(transcript)
    tech_match = _TECH_KEYWORD_PATTERN.search(normalized)
    if tech_match:
        term = tech_match.group(0)
    else:
        term = next(
            (label for pattern, label in _FUZZY_KEYWORD_PATTERNS if pattern.search(normalized)),
            transcript,
        )
    return (
        f'The interviewer\'s exact words were unclear or partly mis-transcribed, but '
        f'"{term}" was mentioned -- most likely asking about your experience or '
        f'knowledge of it. Give a short, natural spoken answer about "{term}" as if '
        f"asked to talk about your experience or understanding of it directly."
    )

def _is_followup_question(text: str) -> bool:
    normalized = _strip_filler_lead(_normalize(text)).rstrip("?").strip()

    followups = {
        "why",
        "and why",
        "why?",
        "and why?",
        "why so",
        "why is that",
        "why not",
        "why though",
        "why exactly",
        "how",
        "and how",
        "how?",
        "and how?",
        "how so",
        "how come",
        "hows that",
        "how's that",
        "what about that",
        "what about that?",
        "what about it",
        "what about it?",
        "can you explain",
        "can you elaborate",
        "please elaborate",
        "elaborate",
        "explain that",
        "explain why",
        "tell me more",
        "go on",
        "and then",
        "what do you mean",
        "example",
        "for example",
        "any example",
        "give an example",
        "can you give an example",
        "such as",
        "really",
        "is that so",
    }

    if normalized in followups:
        return True

    # Not in the list, but structurally a follow-up: a short question whose
    # only subject is a bare pronoun -- "have you ever worked with it", "how
    # does that work". Whatever "it" refers to was said earlier, so this
    # cannot be answered on its own. This is how the trailing part of a
    # multi-part question arrives when the interviewer paused before it
    # ("What is LoRA? Why use it?" ... "Have you ever worked with it?"),
    # and answering it as a fresh question loses the topic completely.
    return _has_dangling_reference(normalized)


# Pronouns that point at something said earlier rather than naming it.
_DANGLING_REFERENTS = frozenset({"it", "that", "this", "them", "those", "these", "one"})


def _has_dangling_reference(normalized: str, max_words: int = 7) -> bool:
    words = _strip_filler_lead(normalized).rstrip("?").strip().split()
    if not (2 <= len(words) <= max_words):
        return False
    if not any(word in _DANGLING_REFERENTS for word in words):
        return False
    # If it names a real topic, it stands on its own and is a new question,
    # not a follow-up ("how does indexing work in that case").
    if _mentions_tech_keyword(normalized):
        return False
    core = " ".join(words)
    return any(core.startswith(prefix.strip()) for prefix in _QUESTION_PREFIXES)


def _resolve_followup(transcript: str, history) -> str:
    """A bare follow-up like 'why?' or 'can you elaborate' only makes sense
    attached to the most recent answer -- on its own, the LLM has nothing to
    go on. Rewrite it into a self-contained instruction that names the
    previous question/answer directly, so it answers the actual follow-up
    instead of guessing or giving something generic."""
    if not history:
        return transcript

    last_turn = history[-1]
    return (
        f'The interviewer just asked a short follow-up: "{transcript}" -- this is '
        f'about what you JUST said, not a new topic. The previous question was '
        f'"{last_turn.question}" and you answered: "{last_turn.answer}". Directly '
        f"justify, expand on, or clarify that specific answer with real additional "
        f"reasoning or detail -- don't repeat the same answer and don't give a "
        f"generic response."
    )


# Verdicts that mean "there is genuinely nothing here" -- these clear the
# buffer at once. Everything else gets a short continuation window, because
# an interviewer really does build a question across sentences.
_DISCARD_IMMEDIATELY = frozenset({"empty", "small_talk", "fast_intent_ignore"})


def _looks_finished(text: str, settings: Settings) -> bool:
    """Does this read as a finished thing somebody said?

    Two signals, either of which is enough. Terminal punctuation is the
    strong one: speech-to-text emits it when IT judged the utterance to have
    ended, which is a genuine end-of-thought signal and not merely
    cosmetic -- Whisper and NVIDIA's models with automatic punctuation both
    do this. Length is the weak backup for a provider that returns no
    punctuation at all: past a few words, a question that also matched a
    gate rule is very unlikely to be half a sentence.
    """
    stripped = text.strip()
    if stripped.endswith((".", "?", "!")):
        return True
    return len(stripped.split()) > settings.question_soft_wait_word_limit


def _should_keep_waiting(
    *,
    gate: GateResult,
    settings: Settings,
    transcript: str,
    awaiting_more: bool,
    since_last_part: float,
    timed_out: bool,
    speech_active: bool = False,
    looks_like_content: bool = False,
) -> bool:
    """Should the pipeline hold this buffer instead of acting on it now?

    Every branch here is a deliberate latency decision, so they are all in
    one place rather than spread through the debounce loop.
    """
    # The VAD force-cut this utterance mid-sentence -- the rest of it is
    # already captured and being transcribed. That is a fact, not a guess,
    # so waiting costs nothing and prevents answering half a question.
    #
    # The cap is the longest the next piece could LEGITIMATELY take to
    # arrive -- the force-cut length plus one transcription round trip --
    # measured from the last fragment. question_max_wait_seconds is the
    # wrong number here: a long question's continuation routinely takes
    # longer than the whole budget, and capping on it made the pipeline
    # answer the first half of a question and then answer it again when the
    # rest showed up. The only case this cap really guards is a continuation
    # that never comes because its transcription failed, and audio_ws
    # reports that directly via close_utterance().
    if awaiting_more:
        return since_last_part < (settings.segment_max_seconds + settings.stt_timeout_seconds)

    # They are talking right now. Whatever is buffered is part of what they
    # are still saying, so nothing gets decided or discarded until they stop.
    # An answerable buffer is the one exception: a genuinely new question
    # asked over the top of an old one should still fire (barge-in).
    if speech_active and not gate.should_answer:
        return True

    if timed_out:
        return False

    if gate.should_answer:
        # Trails off mid-sentence ("what is the difference between") -- give
        # the rest of it a moment to arrive.
        if not gate.complete:
            return since_last_part < settings.question_soft_wait_seconds
        # A follow-up is short by design; waiting on it is never right.
        if gate.reason == "follow_up":
            return False
        # Otherwise: answer immediately if the text looks finished. The old
        # code held EVERY short answerable transcript for a fixed grace
        # period on the chance it might continue, which put ~1.8s in front
        # of exactly the questions that were already unambiguous.
        if _looks_finished(gate.answer_text(transcript), settings):
            return False
        return since_last_part < settings.question_soft_wait_seconds

    # Not answerable. Keep it only while a continuation could still plausibly
    # be coming -- a multi-sentence scenario question builds this way. Once
    # that window passes with nothing following, drop it, so it can never
    # become part of a question asked later.
    if gate.reason in _DISCARD_IMMEDIATELY:
        # One exception: the tier-2 classifier calling a full, well-formed
        # sentence "not addressed to the candidate" is exactly how the setup
        # half of a scenario question looks to it ("The production server's
        # disk partition is 100% full."). Discarding that immediately loses
        # the setup and answers only the last sentence, so real sentences
        # still get the continuation window before being dropped.
        if gate.reason == "fast_intent_ignore" and looks_like_content:
            return since_last_part < settings.utterance_merge_gap_seconds
        return False
    return since_last_part < settings.utterance_merge_gap_seconds


def _strip_filler_lead(normalized: str) -> str:
    """Strip leading filler words repeatedly (e.g. 'okay so' -> ''), so a
    question word right after filler still matches a prefix check."""
    changed = True
    while changed:
        changed = False
        for filler in _FILLER_LEADS:
            if normalized.startswith(filler):
                normalized = normalized[len(filler) :]
                changed = True
    return normalized


def _clean_transcript(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _combine_transcript_parts(parts: list[str]) -> str:
    """Join the fragments of one buffered question back into a sentence.

    Consecutive fragments can genuinely overlap: a force-cut segment replays
    its last 200ms into the next segment (so a word split across the cut
    survives), which means the recognizer can legitimately return that word
    at the end of one fragment AND the start of the next. Left alone that
    reads as a stutter in the question the LLM is asked.
    """
    cleaned = [_clean_transcript(part) for part in parts if _clean_transcript(part)]
    if not cleaned:
        return ""

    combined = cleaned[0]
    for part in cleaned[1:]:
        combined = _join_without_overlap(combined, part)
    return combined.replace(" .", ".").replace(" ?", "?")


def _join_without_overlap(left: str, right: str, max_overlap_words: int = 6) -> str:
    left_words = left.split()
    right_words = right.split()
    if not left_words or not right_words:
        return " ".join(left_words + right_words)

    limit = min(max_overlap_words, len(left_words), len(right_words))
    for size in range(limit, 0, -1):
        tail = [_strip_word(word) for word in left_words[-size:]]
        head = [_strip_word(word) for word in right_words[:size]]
        if tail == head and any(tail):
            return " ".join(left_words + right_words[size:])
    return " ".join(left_words + right_words)


def _strip_word(word: str) -> str:
    return re.sub(r"[^\w]", "", word).lower()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" \t\r\n\"'")


# Per-question context budget. Sized so one question -- system prompt plus
# this context plus the answer -- fits comfortably inside a per-minute token
# allowance, since an interview asks several questions a minute.
_RESUME_CHAR_LIMIT = 2000
_JD_CHAR_LIMIT = 1500
_NOTES_CHAR_LIMIT = 500
_HISTORY_TURNS = 3
_HISTORY_ANSWER_CHAR_LIMIT = 400


def _build_answer_messages(
    question: str,
    context: InterviewSessionContext,
    history,
) -> list[dict[str, str]]:
    # Everything below is re-sent on EVERY question, so its size is paid
    # again for each one. That matters more than it looks: providers meter
    # tokens per minute, and an uncapped resume (a PDF upload is easily
    # 5000+ characters) plus four turns of history was pushing a single
    # question over the per-minute budget -- which shows up as the answer
    # stalling for 20+ seconds while the client waits out a 429, not as an
    # obvious quota error. The caps are generous enough to keep the parts
    # of a resume/JD that actually ground an answer.
    context_blocks = []
    if context.resume_text:
        context_blocks.append(f"Candidate resume:\n{context.resume_text[:_RESUME_CHAR_LIMIT]}")
    if context.job_description:
        context_blocks.append(
            f"Job description:\n{context.job_description[:_JD_CHAR_LIMIT]}"
        )
    if context.notes:
        context_blocks.append(f"Extra notes:\n{context.notes[:_NOTES_CHAR_LIMIT]}")

    recent_history = "\n".join(
        f"Q: {turn.question}\nA: {turn.answer[:_HISTORY_ANSWER_CHAR_LIMIT]}"
        for turn in history[-_HISTORY_TURNS:]
    )

    user_content = f"Interview question:\n{question}"
    if context_blocks:
        user_content += "\n\nSession context:\n" + "\n\n".join(context_blocks)
    if recent_history:
        user_content += "\n\nRecent interview history:\n" + recent_history

    return [
        {
            "role": "system",
            "content": (
                "You are the candidate in a live job interview. The interviewer just "
                "asked you a question out loud. Answer as yourself, using the resume "
                "below as your own background. Your answer is read off a screen and "
                "spoken aloud, so it must be easy to scan AND sound like a real person "
                "talking -- never like a written document.\n\n"
                "FORMAT\n"
                "- Short factual question (one thing asked): 1-3 plain spoken sentences, "
                "no bullets.\n"
                "- Anything with more than one part -- multi-part question, scenario, "
                "troubleshooting, comparison, 'tell me about yourself': use 3-5 bullets "
                "(max 6), each line starting with '- '.\n"
                "- Each bullet is ONE complete spoken sentence of 12-25 words that can be "
                "read aloud word for word. Never a fragment, header, or 'Label:' prefix. "
                "Good: '- I'd check the slow query log first, since 95% DB CPU usually "
                "means one bad query rather than real load.' Bad: '- Check logs' / "
                "'- Step 1: Investigation'.\n"
                "- Cover the asked parts in the order asked -- three things asked means "
                "three bullets. Optional one-line lead-in before the bullets; never a "
                "closing summary line.\n"
                "- No headings, bold, numbered lists, sub-bullets, or blank lines between "
                "bullets. If asked for code, give real correct code in a fenced block "
                "(bullet rules don't apply to the code).\n\n"
                "VOICE\n"
                "- Contractions throughout: I'm, it's, don't, I've, that's.\n"
                "- Vary your openings. Never 'Great question' or any praise of the "
                "question, and never comment on the question's wording, order, or "
                "whether it seems repeated.\n"
                "- NEVER restate or narrate the question back before answering. No 'that "
                "sounds tricky', no 'okay so the disk is full and the app is down'. Go "
                "straight to the answer.\n"
                "- Uneven, not symmetric: one point carries a concrete detail (a real "
                "project, a number, a tool from the resume) and the others are shorter. "
                "Avoid giving every point identical weight and shape.\n"
                "- For troubleshooting: name what you'd actually check and WHY, in the "
                "order you'd really do it -- not a vague checklist of everything "
                "checkable.\n"
                "- State things plainly; don't hedge every claim or over-explain.\n\n"
                "LANGUAGE\n"
                "- Simple, clear, everyday English -- the natural register of a "
                "well-spoken Indian professional on a call. Short direct sentences "
                "someone can follow by listening once.\n"
                "- Use the plain word, never the bookish one: use (not utilize), start "
                "(not commence), help (not facilitate), based on (not predicated on), "
                "try (not endeavor), find out (not ascertain), aware (not cognizant), "
                "many (not myriad), strong (not robust), after that (not subsequently), "
                "before (not prior to), to (not in order to), because (not due to the "
                "fact that). No 'furthermore', 'in conclusion', 'in today's fast-paced "
                "environment', 'leverage'.\n"
                "- Keep technical terms exact and unsimplified -- framework, library, "
                "API and pattern names, and terms like index, connection pool, race "
                "condition, idempotent, REST, websocket. Simplify the words AROUND the "
                "term, never the term.\n\n"
                "CONTENT\n"
                "- Ground answers in the resume and job description below, with real "
                "specifics rather than generic claims.\n"
                "- Technical questions get real technical substance: name the actual "
                "mechanism, approach, or trade-off, the way someone who has built with "
                "it would. Behavioral questions stay purely in the plain human voice.\n"
                "- Only name a specific company, project, tool, metric, or achievement "
                "if it actually appears in the resume/JD. If you have no specific for "
                "this question, speak generally about your skills and approach rather "
                "than inventing a fake specific. Same for technical facts -- if you're "
                "not certain of an exact version number or statistic, describe the "
                "concept confidently without the invented detail.\n"
                "- 'Have you worked with X?' is really 'show me you know X'. Lead with "
                "the substance every time: what X does, how you'd use it, the trade-off "
                "that actually matters, and where it connects to real work in the "
                "resume ('same idea as the caching layer I built'). That is the answer "
                "being assessed.\n"
                "- Spend the answer on substance, not on scope. If the resume genuinely "
                "doesn't cover something, acknowledging that is fine -- but keep it to "
                "ONE short clause and move straight into what you do know. Never spend "
                "a second sentence on it, never repeat it at the end, and drop "
                "'I'd pick it up quickly' / 'I'd be ready to learn it' entirely: that "
                "adds no information and reads as apologising. The technical depth is "
                "what's being judged, so that is what the answer should mostly be.\n"
                "- Don't claim a specific employer, project, metric, or achievement "
                "that isn't in the resume -- that's the one hard line. Speaking "
                "confidently and in depth about any technology is always fine, and is "
                "what these questions are actually asking for.\n"
                "- Answer only what was asked. No disclaimers, no mentioning you're an "
                "AI. If the question is ambiguous, answer the most likely reading "
                "instead of asking for clarification.\n"
                "- Never produce a response that talks ABOUT the question instead of "
                "answering it. Banned with zero exceptions: saying you're 'ready to "
                "answer', asking them to 'go ahead and ask', 'did you mean...', or "
                "pointing out a mix-up. A reply containing no actual answer is always "
                "wrong.\n\n"
                "READING THE QUESTION\n"
                "- The text comes from live speech-to-text and may contain small errors "
                "(a term misheard as a similar-sounding word). Silently infer the most "
                "plausible real term from the resume/JD context and answer that. Never "
                "mention the transcription or ask them to repeat.\n"
                "- If it reads like '<introduction/label> <the real question>' -- e.g. "
                "'The third interview question is how do you handle pressure?' -- drop "
                "the label and answer the real question in full.\n"
                "- That applies ONLY to a label. It is never permission to answer just "
                "the last sentence. A scenario's setup sentences are the facts you must "
                "reason from, not preamble -- a generic answer that ignores the stated "
                "numbers and symptoms is wrong. A multi-part question needs every part "
                "answered. If they restarted mid-question, answer what they landed on."
            ),
        },
        {"role": "user", "content": user_content},
    ]


async def _strip_think_tags(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Strips <think>...</think> reasoning blocks out of a token stream.
    Belt-and-suspenders alongside the reasoning_format="hidden" param passed
    to Groq's reasoning models (Qwen3.x etc) -- if that param is ever
    unhonored (SDK version drift, a future model swap, a provider that has
    no such param at all) this still keeps raw <think> chain-of-thought from
    ever reaching the overlay. Buffers only the few characters needed to
    detect a tag split across chunk boundaries, so it doesn't add any real
    latency to the stream."""
    OPEN, CLOSE = "<think>", "</think>"
    buf = ""
    in_think = False
    async for token in tokens:
        buf += token
        while True:
            if not in_think:
                idx = buf.find(OPEN)
                if idx == -1:
                    keep = len(OPEN) - 1
                    if len(buf) > keep:
                        yield buf[: len(buf) - keep]
                        buf = buf[len(buf) - keep :]
                    break
                if idx > 0:
                    yield buf[:idx]
                buf = buf[idx + len(OPEN) :]
                in_think = True
            else:
                idx = buf.find(CLOSE)
                if idx == -1:
                    keep = len(CLOSE) - 1
                    if len(buf) > keep:
                        buf = buf[len(buf) - keep :]
                    break
                buf = buf[idx + len(CLOSE) :]
                in_think = False
    if not in_think and buf:
        yield buf


def _build_screen_analysis_prompt(
    question: str, context: InterviewSessionContext
) -> tuple[str, str]:
    """Builds the (system_prompt, user_text) pair for a screen-analysis
    vision request -- deliberately separate from _build_answer_messages'
    'sound like a human candidate speaking out loud' system prompt, since a
    screen-analysis answer is read on screen (not spoken) and is usually
    about something concrete visible in the screenshot (code, an error, a
    diagram, a question on screen) rather than an interview question."""
    context_blocks = []
    if context.resume_text:
        context_blocks.append(f"Candidate resume:\n{context.resume_text}")
    if context.job_description:
        context_blocks.append(f"Job description:\n{context.job_description}")

    system_prompt = (
        "You are looking at a screenshot of the user's screen, taken during a live call or "
        "work session. The user needs the actual answer/fix/solution to whatever is on "
        "screen -- not a description of what the screenshot shows.\n\n"
        "Hard rules:\n"
        "- Never open by describing the screenshot (no \"I can see...\", \"this appears to "
        "show...\", \"the screen displays...\"). Start directly with the answer, fix, or "
        "solution itself.\n"
        "- If it's code or an error: give the actual corrected code or the specific root "
        "cause and fix -- not a generic description of what kind of error it looks like.\n"
        "- If it's a question, quiz, form field, or exam prompt shown on screen: answer that "
        "exact question directly, like you're the one taking the test.\n"
        "- If it's a UI, dashboard, chat, or document: address specifically what the user's "
        "question is asking about it, not a general tour of everything visible.\n"
        "- Only describe the screen if there is genuinely nothing to solve or answer and a "
        "description is literally what's being asked for.\n"
        "- Plain conversational text only -- no markdown headers, no bullet points."
    )
    user_text = (
        f"{question}\n\n"
        "Give me the direct answer/solution/fix based on what's in this screenshot -- "
        "don't describe the screenshot back to me."
    )
    if context_blocks:
        user_text += "\n\nSession context:\n" + "\n\n".join(context_blocks)
    return system_prompt, user_text


_SMALL_TALK = {
    "thank you",
    "thanks",
    "hi",
    "hello",
    "hi hello",
    "okay",
    "ok",
    "yes",
    "no",
    "good morning",
    "good afternoon",
    "can you hear me",
}

_QUESTION_PREFIXES = (
    "what ",
    "why ",
    "how ",
    "when ",
    "where ",
    "who ",
    "which ",
    "can you ",
    "could you ",
    "would you ",
    "do you ",
    "did you ",
    "does ",
    "is ",
    "are ",
    "have you ",

    # Interview requests
    "tell me ",
    "tell us ",
    "tell you ",
    "explain ",
    "describe ",
    "walk me ",
    "walk us ",
    "take me ",
    "take us ",
    "give me ",
    "give us ",
    "share ",
    "introduce yourself",
    "start with ",
    "let's start with ",
    "lets start with ",
    "let us start with ",
    "let's begin with ",
    "lets begin with ",
    "begin with ",
    "start by ",
    "start with your ",
    "tell us about ",
    "tell me about ",
    "talk about ",
    "talk us through ",
    "walk me through ",
    "walk us through ",
    "take me through ",
    "take us through ",
    "i'd like to hear ",
    "i would like to hear ",
)

# The subset of _QUESTION_PREFIXES that can only be the start of a question,
# so it is safe to look for part-way through a clause. Deliberately excludes
# "is ", "are ", "does ", "do you ", "did you ", "have you ": those open a
# question only in first position and otherwise sit in the middle of
# perfectly ordinary statements.
_MIDCLAUSE_QUESTION_PREFIXES = (
    "what ",
    "why ",
    "how ",
    "when ",
    "where ",
    "who ",
    "which ",
    "can you ",
    "could you ",
    "would you ",
    "tell me ",
    "tell us ",
    "explain ",
    "describe ",
    "walk me ",
    "walk us ",
    "take me ",
    "take us ",
    "give me ",
    "give us ",
    "talk about ",
    "talk us through ",
    "introduce yourself",
    "i'd like to hear ",
    "i would like to hear ",
)

_QUESTION_PHRASES = (
    "difference between",
    "your experience with",
    "how would you",
    "what would you",
    "why should",
    "write a",
    "design a",
    "implement a",
)

# Common lead-in words/phrases before the real question word -- "So what's
# your approach", "Yeah, how do you", "Okay and why", "So let's tell me
# about...". Stripped before the prefix check so these aren't silently
# missed -- including STACKED combinations ("so" + "let's"), since
# _strip_filler_lead loops until nothing more matches at the start, not
# just a single filler word.
_FILLER_LEADS = (
    "so ", "well ", "okay so ", "ok so ", "alright so ", "now ", "and ",
    "yeah so ", "right so ", "actually ", "also ", "um ", "uh ", "alright ",
    "okay ", "ok ", "yeah ", "right ", "i mean ", "like ",
    "let's ", "lets ", "let us ",
)

# Words a sentence is very unlikely to genuinely end on -- if the transcript's
# last word is one of these, the speaker almost certainly paused mid-sentence
# rather than finished. Used both by the gate (to avoid answering a fragment)
# and by _looks_complete (to decide whether to keep waiting for more speech).
_INCOMPLETE_TRAILING_WORDS = {
    "the", "a", "an", "to", "of", "is", "are", "was", "were", "am",
    "for", "and", "or", "but", "so", "with", "in", "on", "at", "that",
    "this", "which", "who", "what", "why", "how", "when", "where",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "my", "your", "our", "their", "its", "from", "by", "as", "if",
    "than", "then", "not", "no", "about", "between", "into", "over",
}
