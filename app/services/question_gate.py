import asyncio
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


@dataclass(slots=True)
class _PendingTranscript:
    parts: list[str] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    first_seen: float = 0.0
    min_confidence: float = 1.0


class QuestionAnswerPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = get_session_store()
        self._pending: dict[str, _PendingTranscript] = {}
        self._lock = asyncio.Lock()
        self._gate_graph = _build_gate_graph()

    async def submit_transcript(
        self, *, session_id: str, text: str, confidence: float = 1.0
    ) -> None:
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
                "low_confidence": confidence < self._settings.stt_confidence_threshold,
            },
        )

        async with self._lock:
            pending = self._pending.setdefault(session_id, _PendingTranscript())
            if not pending.parts:
                pending.first_seen = time.monotonic()
            pending.parts.append(cleaned)
            pending.min_confidence = min(pending.min_confidence, confidence)
            if pending.task and not pending.task.done():
                pending.task.cancel()
            pending.task = asyncio.create_task(self._process_after_debounce(session_id))

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
            pending.parts.clear()
            pending.first_seen = 0.0
            pending.min_confidence = 1.0
            pending.task = asyncio.create_task(
                self._generate_answer(session_id, cleaned, confidence=1.0, forced=True)
            )

    async def _process_after_debounce(self, session_id: str) -> None:
        try:
            await asyncio.sleep(self._settings.question_debounce_ms / 1000)

            async with self._lock:
                pending = self._pending.setdefault(session_id, _PendingTranscript())
                transcript = _combine_transcript_parts(pending.parts)

                if not transcript:
                    pending.task = None
                    return

                elapsed = (
                    time.monotonic() - pending.first_seen if pending.parts else 0.0
                )
                timed_out = elapsed >= self._settings.question_max_wait_seconds
                gate = self._run_gate(transcript)

            # Tier 2 (fast intent classifier): only for the genuinely
            # ambiguous bucket the rule gate couldn't confidently decide
            # either way (reason=no_question_signal). Runs OUTSIDE the lock
            # since it's a network call -- holding a pipeline-wide lock
            # during it would serialize every other session's question
            # processing behind one slow classifier call. If new speech
            # arrives while this is in flight, submit_transcript() cancels
            # this task the same way it always does, so a stale decision
            # from before the new speech never gets applied.
            if (
                not timed_out
                and not gate.should_answer
                and gate.reason == "no_question_signal"
                and self._settings.fast_intent_enabled
            ):
                gate = await self._run_fast_intent_classifier(transcript, session_id)

            async with self._lock:
                pending = self._pending.setdefault(session_id, _PendingTranscript())

                # A sentence ending in a period isn't necessarily the whole
                # thought -- scenario-style questions are often 2-3 sentences
                # ("The disk partition is full. The app has stopped working.
                # How would you troubleshoot?"). If this doesn't look like a
                # real question yet, but it's also not genuinely nothing
                # (empty/small-talk/a classifier-confirmed IGNORE), KEEP the
                # words buffered and wait for more speech instead of
                # discarding them. Discarding here is what was causing only
                # the LAST sentence of a multi-sentence question to ever
                # reach the LLM -- each earlier sentence got wrongly treated
                # as "finished" (it ends with a period) and thrown away the
                # moment the gate said "not a question".
                still_building = (
                    not timed_out
                    and not gate.should_answer
                    and gate.reason not in ("empty", "small_talk", "fast_intent_ignore")
                )
                if still_building:
                    pending.task = asyncio.create_task(
                        self._process_after_debounce(session_id)
                    )
                    return

                pending.parts.clear()
                pending.task = None
                confidence = pending.min_confidence
                pending.min_confidence = 1.0
                pipeline_ms = int((time.monotonic() - pending.first_seen) * 1000) if pending.first_seen else 0
                pending.first_seen = 0.0

            await self._generate_answer(
                session_id,
                transcript,
                confidence=confidence,
                pipeline_ms=pipeline_ms,
                precomputed_gate=gate,
            )
        except asyncio.CancelledError:
            raise

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
            "answer yet.\n"
            "IGNORE -- small talk, filler, background noise, or not addressed to the "
            "candidate at all; never answer this.\n\n"
            + (f"Recent context:\n{recent}\n\n" if recent else "")
            + f'Fragment: "{transcript}"\n\n'
            "Reply with exactly one word: ANSWER, WAIT, or IGNORE."
        )

        try:
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
            decision = decision.strip().upper()
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
                    },
                )
                if not gate.should_answer:
                    return

            # context = await self._store.get_context(session_id)
            # history = await self._store.get_history(session_id)
            # messages = _build_answer_messages(transcript, context, history)

            context = await self._store.get_context(session_id)
            history = await self._store.get_history(session_id)

            effective_transcript = transcript

            if gate.reason == "follow_up":
                effective_transcript = _resolve_followup(
                    transcript,
                    history,
                )

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
                    "low_confidence": confidence < self._settings.stt_confidence_threshold,
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
            pending.parts.clear()
            pending.first_seen = 0.0
            pending.min_confidence = 1.0
            pending.task = asyncio.create_task(
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

# def _classify_question(text: str) -> GateResult:
#     return _classify_interview_intent(text)

def _classify_question(text: str) -> GateResult:
    normalized = _normalize(text)
    words = normalized.split()

    if not normalized:
        return GateResult(False, "empty")

    if normalized in _SMALL_TALK:
        return GateResult(False, "small_talk")

    # 1. Follow-ups take priority over the incomplete-fragment check below --
    #    a bare "why" or "how so" is exactly the kind of short word that
    #    check would otherwise reject as "trailed off mid-sentence", but for
    #    a standalone follow-up that IS the whole utterance.
    if _is_followup_question(normalized):
        return GateResult(True, "follow_up")

    # 2. Check incomplete speech
    if normalized.endswith("...") or (
        words and words[-1] in _INCOMPLETE_TRAILING_WORDS
    ):
        return GateResult(False, "incomplete_fragment")

    # 3. Very short utterances that are not follow-ups
    if len(words) <= 2 and "?" not in normalized:
        return GateResult(False, "too_short")

    # 4. Explicit question
    if "?" in normalized:
        return GateResult(True, "question_mark")

    # 5. Normal question prefixes
    core = _strip_filler_lead(normalized)

    if any(core.startswith(prefix) for prefix in _QUESTION_PREFIXES):
        return GateResult(True, "question_prefix")

    # 6. Semantic question phrases
    if any(phrase in normalized for phrase in _QUESTION_PHRASES):
        return GateResult(True, "question_phrase")

    # 7. Technical comparison
    if re.search(
        r"\b\w[\w./-]*\s+(vs\.?|versus)\s+\w[\w./-]*\b",
        normalized,
    ):
        return GateResult(True, "vs_comparison")

    return GateResult(False, "no_question_signal")

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

    return normalized in followups


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

def _classify_interview_intent(text: str) -> GateResult:
    normalized = _normalize(text)
    words = normalized.split()

    if not normalized:
        return GateResult(False, "empty")

    if normalized in _SMALL_TALK:
        return GateResult(False, "small_talk")

    # ---------------------------------------------------------
    # 1. Detect clearly incomplete speech FIRST
    # ---------------------------------------------------------
    if normalized.endswith("..."):
        return GateResult(False, "incomplete_fragment")

    if words and words[-1] in _INCOMPLETE_TRAILING_WORDS:
        return GateResult(False, "incomplete_fragment")

    # ---------------------------------------------------------
    # 2. Normalize filler words before classification
    # ---------------------------------------------------------
    core = _strip_filler_lead(normalized)

    # ---------------------------------------------------------
    # 3. Direct interview requests
    # ---------------------------------------------------------
    interview_patterns = (
        "tell me about yourself",
        "tell us about yourself",
        "introduce yourself",
        "your introduction",

        "start with your introduction",
        "let's start with your introduction",
        "lets start with your introduction",

        "start with your background",
        "tell me about your background",
        "tell us about your background",

        "walk me through your background",
        "walk us through your background",

        "take me through your background",

        "tell me about your experience",
        "tell us about your experience",

        "walk me through your experience",
        "walk us through your experience",
    )

    if any(pattern in core for pattern in interview_patterns):
        return GateResult(True, "interview_introduction")

    # ---------------------------------------------------------
    # 4. General interview requests
    # ---------------------------------------------------------
    interview_prefixes = (
        "tell me ",
        "tell us ",
        "explain ",
        "describe ",

        "walk me ",
        "walk us ",

        "take me ",
        "take us ",

        "talk about ",
        "talk us through ",

        "share ",

        "give me ",
        "give us ",

        "introduce ",

        "start with ",
        "start by ",

        "begin with ",

        "let's start ",
        "lets start ",

        "let's begin ",
        "lets begin ",

        "i'd like to hear ",
        "i would like to hear ",
    )

    if any(core.startswith(prefix) for prefix in interview_prefixes):
        return GateResult(True, "interview_request")

    # ---------------------------------------------------------
    # 5. Normal grammatical questions
    # ---------------------------------------------------------
    if "?" in normalized:
        return GateResult(True, "question_mark")

    if any(core.startswith(prefix) for prefix in _QUESTION_PREFIXES):
        return GateResult(True, "question_prefix")

    # ---------------------------------------------------------
    # 6. Existing semantic interview phrases
    # ---------------------------------------------------------
    if any(phrase in normalized for phrase in _QUESTION_PHRASES):
        return GateResult(True, "question_phrase")

    # ---------------------------------------------------------
    # 7. Technical comparisons
    # ---------------------------------------------------------
    if re.search(
        r"\b\w[\w./-]*\s+(vs\.?|versus)\s+\w[\w./-]*\b",
        normalized,
    ):
        return GateResult(True, "vs_comparison")

    # ---------------------------------------------------------
    # 8. No strong signal
    # ---------------------------------------------------------
    return GateResult(False, "no_question_signal")

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
    cleaned = [_clean_transcript(part) for part in parts if _clean_transcript(part)]
    return " ".join(cleaned).replace(" .", ".").replace(" ?", "?")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" \t\r\n\"'")


def _build_answer_messages(
    question: str,
    context: InterviewSessionContext,
    history,
) -> list[dict[str, str]]:
    context_blocks = []
    if context.resume_text:
        context_blocks.append(f"Candidate resume:\n{context.resume_text}")
    if context.job_description:
        context_blocks.append(f"Job description:\n{context.job_description}")
    if context.notes:
        context_blocks.append(f"Extra notes:\n{context.notes}")

    recent_history = "\n".join(
        f"Q: {turn.question}\nA: {turn.answer}" for turn in history[-4:]
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
                "You are the candidate in a live job interview, being interviewed for the "
                "role in the job description below, using the resume below as your own "
                "background. The interviewer just asked you a question out loud. Answer it "
                "the way you -- a real person -- would actually speak, off the cuff, on a "
                "call. This will be read out loud, so it has to sound like an actual human "
                "talking, not a written answer, and not like someone reading a script.\n\n"
                "How a real human answer sounds, that an AI-generated one usually doesn't:\n"
                "- Uses contractions naturally: I'm, it's, don't, I've, that's, wasn't.\n"
                "- Doesn't open every answer the same way. Vary it -- sometimes jump straight "
                "into the answer, sometimes start with a quick framing like 'so' or 'yeah, "
                "that's something I've worked with' -- but never 'Great question' or any "
                "praise of the question.\n"
                "- NEVER restate or narrate the scenario back before answering -- no 'that's a "
                "pretty urgent situation', 'that sounds tricky', 'okay so the disk is full and "
                "the app is down' or any sentence that just repeats/summarizes what was already "
                "said in the question. Skip straight to what you'd actually do or say -- a real "
                "person jumps to the action, they don't narrate the problem back first.\n"
                "- Isn't perfectly balanced or symmetric. Real answers ramble slightly, favor "
                "one point over the others, trail into an example instead of listing three "
                "neat points. Sentence lengths vary -- some short, some longer and looser.\n"
                "- For troubleshooting/what-would-you-do questions specifically: never lay it "
                "out as an enumerated runbook ('First I'd..., then I'd..., I'd also..., if that "
                "doesn't work I'd...'). That step-by-step listing structure is a dead giveaway "
                "of an AI answer, and it's also how it runs out of room and cuts off mid-"
                "sentence. Instead say the ONE first thing you'd actually check or do, briefly "
                "say why, and stop -- like a person naming their first move, not reciting a "
                "full checklist of every step they might ever take.\n"
                "- Leans on one concrete, specific detail (a real project, a number, a tool, "
                "something from the resume) rather than covering every angle evenly. A human "
                "remembers one good example, not a comprehensive summary.\n"
                "- Avoids corporate/textbook phrasing -- no 'in today's fast-paced "
                "environment', 'leverage', 'utilize', 'furthermore', 'in conclusion'. Say "
                "'use' not 'utilize', 'because' not 'due to the fact that'.\n"
                "- Doesn't over-explain or hedge every claim. State things plainly, the way "
                "someone confident in their own experience would.\n\n"
                "Formatting:\n"
                "- Plain conversational text only -- no markdown, no bullet points, no "
                "headings, no numbered lists, no asterisks.\n"
                "- Exception: if the question explicitly asks you to write or give code (a "
                "function, script, query, config file, etc), write the actual real, correct "
                "code directly -- code needs proper structure (newlines, indentation), it "
                "doesn't need to sound like spoken conversation, and the 5-6 line limit below "
                "doesn't apply to the code itself. A short one-line spoken lead-in before the "
                "code is fine, or none at all -- the code is the answer.\n"
                "- HARD LIMIT: 5-6 lines maximum, no exceptions. Count roughly one sentence "
                "per line. If you're about to write a 7th line, stop and cut it instead -- a "
                "shorter answer that makes one point well always beats a longer one that "
                "covers more ground. This is a strict ceiling, not a target to aim under.\n"
                "- One continuous flow only. Never split the answer into multiple paragraphs "
                "with blank lines between them, and never stack more than one full example or "
                "story -- pick the single best point or the single best example and stay with "
                "just that, instead of giving one paragraph for approach, another paragraph for "
                "an example, and a third paragraph for a related point. That structure reads "
                "like a written essay, not something a person would actually say out loud.\n\n"
                "Language:\n"
                "- Speak in simple, clear English -- the kind of natural, everyday English a "
                "well-spoken Indian professional would use on a call. Simple, common words "
                "over big or bookish ones.\n"
                "- Specific words to avoid, with the simple word to use instead: utilize -> "
                "use, commence -> start, facilitate -> help, leverage -> use, drawn/draw upon "
                "-> use/based on, endeavor -> try, ascertain -> find out, cognizant -> aware, "
                "myriad -> many, robust -> strong/solid, notwithstanding -> even so, "
                "henceforth -> from now on, predicated on -> based on, in terms of -> for/with, "
                "subsequently -> after that, prior to -> before, in order to -> to.\n"
                "- Before finishing, mentally check every word in the answer -- if a word is "
                "one you'd only see in writing, a report, or a formal email rather than in an "
                "actual spoken conversation, swap it for the plain everyday version. When in "
                "doubt, pick the shorter, more common word.\n"
                "- Short, direct sentences over long, complicated ones. Someone should be "
                "able to follow the answer easily just by listening once, without needing to "
                "replay it.\n"
                "- Avoid heavy jargon, idioms, or American-slang-heavy phrasing that would "
                "sound unnatural in Indian professional English -- keep technical terms "
                "themselves (framework names, tool names) exactly as they are, but explain "
                "around them in plain words.\n\n"
                "Content:\n"
                "- Ground the answer in the candidate resume and job description below when "
                "relevant -- real specifics from the resume, not generic claims.\n"
                "- Technical depth: when the question is technical (a framework, language, "
                "architecture, algorithm, tool, database, or process), answer like a developer "
                "who has actually built with it -- with real technical substance, not vague "
                "soft-skill talk. Name the actual mechanism, the specific approach, or the "
                "concrete trade-off involved, instead of generic lines like 'I focus on "
                "quality' or 'I have experience with that'. If the resume/JD lists a relevant "
                "stack, ground the technical explanation in it and speak the way someone who "
                "really works in that stack would.\n"
                "- Keep technical vocabulary exact and unsimplified even while everything else "
                "stays in plain, simple language -- framework names, library names, API names, "
                "design pattern names, and technical terms (e.g. 'index', 'connection pool', "
                "'race condition', 'idempotent', 'REST', 'websocket') are said exactly as a "
                "developer would say them. Simplify the sentences and framing around a "
                "technical term, never the term itself.\n"
                "- For behavioral/non-technical questions (teamwork, conflict, motivation), "
                "stay purely in the plain conversational human voice described above -- this "
                "technical-depth rule applies only when the question is actually technical.\n"
                "- Accuracy matters more than sounding impressive. Only name a specific "
                "company, employer, project, tool, metric, or achievement if it actually "
                "appears in the resume/job description below. If no specific detail is "
                "available for this question, speak in general terms about your skills, "
                "experience level, or approach instead of inventing a fake specific one -- "
                "a confident general answer is fine; a fabricated specific is not.\n"
                "- NEVER refuse to answer, and never say things like 'I'm a DevOps engineer, "
                "I've never worked with that', 'that's not something I've done', or 'I don't "
                "have experience with that' just because a topic isn't explicitly listed in "
                "the resume/JD. That accuracy rule above is ONLY about not inventing FALSE "
                "facts about your own background (a company you didn't work at, a project you "
                "didn't do) -- it is never a reason to refuse a technical question or decline "
                "to demonstrate a skill. A real candidate always makes a genuine, confident "
                "attempt: writes the code if asked for code, explains the concept if asked to "
                "explain, solves the problem if asked to solve it -- regardless of whether it's "
                "their primary specialty. No matter what role the resume is for, answer every "
                "question asked, using real engineering knowledge and best judgment.\n"
                "- For technical facts (framework/API/version behavior, benchmarks, exact "
                "figures), only state a specific detail (a version number, a precise stat, "
                "an exact API name) if you're actually confident it's correct. If you're not "
                "sure of the exact specific, describe the general concept and how you'd "
                "verify or apply it instead of guessing a specific-sounding but possibly "
                "wrong detail -- still say it plainly and confidently, just without the "
                "invented specifics.\n"
                "- Answer only what was asked. No disclaimers, no mentioning you're an AI, no "
                "restating the question before answering.\n"
                "- If the question is ambiguous, answer the most likely interpretation "
                "directly instead of asking for clarification.\n"
                "- Never comment on the question itself -- not its wording, not its position "
                "in a list (first, last, next, previous), not whether it seems repeated, out "
                "of order, or inconsistent with earlier context. Whatever surrounds the actual "
                "question is just how it was introduced -- strip that framing out and answer "
                "only the real question underneath it, as a complete, substantive answer.\n"
                "- Mechanically: if the transcript looks like '<some introduction/label> "
                "<the actual question>?', the real question is the clause containing the "
                "question words (what/how/why/tell me/etc) and the question mark. Treat "
                "everything before that clause as noise to discard, and answer only that "
                "clause -- fully, as if it were asked on its own with nothing before it.\n"
                "- You must never produce a response that talks ABOUT the question instead of "
                "answering it. Banned outputs, with zero exceptions, no matter how the question "
                "was worded or introduced: saying you're 'ready to answer', asking the "
                "interviewer to 'go ahead and ask' or 'ask it now', saying 'did you mean...', "
                "pointing out a 'mix-up' or confusion, or any sentence that is about the "
                "question rather than a real answer to it. A response with no actual answer in "
                "it is always wrong, even if the question's phrasing seemed unusual.\n"
                "- Example of what NEVER to do: transcript is 'The third interview question in "
                "the list is how do you handle challenging situations?' -- it would be WRONG to "
                "reply anything like 'I appreciate the setup, but I'm ready to answer the "
                "actual question now' or 'go ahead and ask it whenever you're ready'. The only "
                "correct move is to extract 'how do you handle challenging situations' as the "
                "real question and give a full spoken answer to exactly that, e.g. starting "
                "straight in with something like 'so usually when things get tough I try to "
                "first figure out what's actually going wrong before reacting...' and continuing "
                "into a real, specific answer.\n"
                "- Example for a troubleshooting scenario: question is 'The production server's "
                "disk partition is 100% full and the app has stopped working. How would you "
                "troubleshoot?' WRONG: 'So the production server partition is 100% full and the "
                "application has stopped working, that's a pretty urgent situation. First thing "
                "I'd do is check disk usage with df, then I'd use du to see which directories, "
                "then I'd use find to search for large files, I'd also check the application "
                "logs...' (narrates the scenario back, then reads like a numbered runbook, runs "
                "out of room and cuts off). RIGHT: 'First thing I'd check is disk usage with df "
                "-h to confirm it's actually the partition, then du -sh to find what's eating "
                "the space -- usually old logs or a runaway process. I'd clear whatever's safe "
                "to free space fast and get the app back up, then dig into why it filled up so "
                "it doesn't happen again.' -- one flowing thought, one clear first move, done.\n"
                "- The question text comes from live speech-to-text and may contain small "
                "transcription errors -- a technical term misheard as a similar-sounding word "
                "(e.g. \"React 19\" heard as \"reactivity\"). Silently infer the most plausible "
                "real term using the job description/resume as context and answer that -- "
                "never mention the transcription or ask the interviewer to repeat themselves."
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
# your approach", "Yeah, how do you", "Okay and why". Stripped before the
# prefix check so these aren't silently missed.
_FILLER_LEADS = (
    "so ", "well ", "okay so ", "ok so ", "alright so ", "now ", "and ",
    "yeah so ", "right so ", "actually ", "also ", "um ", "uh ", "alright ",
    "okay ", "ok ", "yeah ", "right ", "i mean ", "like ",
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
