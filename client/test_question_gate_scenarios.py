"""
Offline scenario runner for the question gate / debounce / barge-in logic in
app/services/question_gate.py -- lets you replay how an unpredictable human
interviewer actually talks (no "?", filler-word leads, mid-question thinking
pauses, multi-sentence scenario questions, grammar mistakes, cutting in while
the copilot is still answering) WITHOUT needing a mic, Teams/Zoom, or the
backend server running. It drives QuestionAnswerPipeline directly, in-process,
feeding it fake "transcript" events with real time gaps between them, and
prints every event the real overlay would have received.

By default this is fully offline: the LLM call is replaced with a stub that
returns a canned answer instantly (no GROQ_API_KEY / network / cost needed),
and the fast-intent tier-2 classifier is disabled so results are 100%
deterministic -- this isolates the rule-based gate + debounce + soft-wait +
barge-in logic, which is what actually decides *when* to answer, from answer
*quality* (a separate concern).

Pass --live to use your real .env settings instead (real Groq calls for both
the answer and the fast-intent classifier) -- useful for a final realistic
pass once the offline run looks right, at the cost of real API calls + time.

Usage (run from the project root, where .env lives):
    python client\\test_question_gate_scenarios.py
    python client\\test_question_gate_scenarios.py --list
    python client\\test_question_gate_scenarios.py --only midquestion_pause,barge_in
    python client\\test_question_gate_scenarios.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

# So `import app...` works no matter what directory this is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import question_gate as gate_module  # noqa: E402
from app.services.question_gate import get_question_pipeline  # noqa: E402


@dataclass
class Turn:
    text: str
    # Seconds to wait AFTER the previous turn (or scenario start, for the
    # first turn) before sending this chunk -- this is what stands in for a
    # human's real pause between sentences/thoughts.
    delay_before: float = 0.0
    # Instead of a fixed delay, wait until the PREVIOUS turn's answer has
    # fully finished (not just started) before sending this one -- for
    # scenarios that need a real prior turn in history, a fixed guessed
    # delay is fragile: it has to out-last however long gating + streaming
    # takes, which changes as gate/debounce tuning changes.
    wait_for_settle: bool = False


@dataclass
class Scenario:
    name: str
    note: str
    turns: list[Turn]
    settle_timeout: float = 8.0


SCENARIOS: list[Scenario] = [
    Scenario(
        name="clean_question_mark",
        note=(
            "Baseline: a normal, fully-formed question with a '?'. NOTE: at "
            "only 4 words this still falls under question_soft_wait_word_limit "
            "(default 7) and gets held for the ~1.8s grace window like any "
            "other short answerable transcript -- the soft-complete tier is "
            "deliberately not keyed on '?', so short-and-punctuated isn't "
            "treated as safer than short-and-unpunctuated. Confirms the gate "
            "still says should_answer=True immediately; only the FIRING is "
            "delayed. If short, obviously-complete questions like this feel "
            "sluggish in real use, that delay is the dial to lower "
            "(QUESTION_SOFT_WAIT_SECONDS / QUESTION_SOFT_WAIT_WORD_LIMIT)."
        ),
        turns=[Turn("What is dependency injection?")],
    ),
    Scenario(
        name="no_question_mark",
        note=(
            "Same intent, but STT never produced a '?' -- very common in "
            "practice. Should still answer via the question-prefix/phrase "
            "rules, not by looking for punctuation."
        ),
        turns=[Turn("what is the difference between let and var in javascript")],
    ),
    Scenario(
        name="filler_lead",
        note=(
            "Real interviewers rarely start clean -- 'so', 'um', 'okay so' "
            "before the actual question. Filler should be stripped so the "
            "question-word prefix still matches."
        ),
        turns=[Turn("so um what are the solid principles")],
    ),
    Scenario(
        name="midquestion_pause_merges",
        note=(
            "THE key case: 'explain the solid principles' alone already "
            "matches should_answer=True (starts with 'explain ') after just "
            "4 words -- but the interviewer paused 1.3s to keep talking. "
            "The soft-complete grace window (question_soft_wait_seconds, "
            "default 1.8s) should hold it and merge the continuation instead "
            "of answering the truncated fragment."
        ),
        turns=[
            Turn("explain the solid principles"),
            Turn("in software design", delay_before=1.3),
        ],
    ),
    Scenario(
        name="midquestion_pause_too_long",
        note=(
            "Same shape as above, but the pause (2.5s) is LONGER than "
            "question_soft_wait_seconds (1.8s default) -- shows the honest "
            "limit of the grace window: it answers the truncated fragment "
            "before the continuation arrives. If this fires often in real "
            "use, raise QUESTION_SOFT_WAIT_SECONDS in .env."
        ),
        turns=[
            Turn("explain the solid principles"),
            Turn("in software design", delay_before=2.5),
        ],
    ),
    Scenario(
        name="keyword_match_garbled_fragment",
        note=(
            "Not a question at all -- garbled STT output that happens to "
            "keep a recognizable tech term intact (real example from a live "
            "test: 'Pote, Java.' for a mangled real question). Explicit "
            "product decision: answer it anyway, on the term alone, marked "
            "low_confidence in the overlay -- accepted tradeoff for not "
            "missing real terse mentions, at the cost of sometimes "
            "answering a topic that was only mentioned in passing."
        ),
        turns=[Turn("Pote, Java.")],
    ),
    Scenario(
        name="keyword_match_fuzzy_term",
        note=(
            "Fuzzy-pattern keyword match, not the exact-token list -- 'talk "
            "about that' + 'normalization' doesn't match any question "
            "prefix/phrase, so this only answers because of the fuzzy "
            "normalization pattern, resolved to a clean label instead of "
            "raw regex match text."
        ),
        turns=[Turn("yeah normalization stuff")],
    ),
    Scenario(
        name="multi_sentence_scenario_question",
        note=(
            "Scenario-style question spread across 3 short sentences with "
            "natural inter-sentence gaps. Neither of the first two sentences "
            "alone looks like a question -- should keep buffering and only "
            "answer once combined with the third."
        ),
        turns=[
            Turn("The production server's disk partition is 100% full."),
            Turn("The app has stopped working.", delay_before=0.6),
            Turn("How would you troubleshoot?", delay_before=0.6),
        ],
    ),
    Scenario(
        name="grammar_mistake_disfluency",
        note=(
            "Restart mid-sentence + stutter + filler, the way people "
            "actually talk live. The phrase-match rule only needs 'how "
            "would you' to appear anywhere in the text, so this should still "
            "answer."
        ),
        turns=[Turn("how -- how would you, um, handle a situation where the app is is down")],
    ),
    Scenario(
        name="small_talk_no_answer",
        note="Should never answer -- should_answer=False, reason=small_talk, no answer_start at all.",
        turns=[Turn("thank you")],
    ),
    Scenario(
        name="noise_fragments_around_question",
        note=(
            "Verbatim from a real session log: mistranscribed noise landed "
            "either side of 'tell me about yourself', the gate saw one blob "
            "starting with 'Tt.', matched nothing, and the most important "
            "question of the interview was silently dropped. Should now "
            "answer, with the noise stripped out of what the LLM is asked."
        ),
        turns=[
            Turn("Tt."),
            Turn("Uday, tell me about yourself.", delay_before=0.4),
            Turn("Ct.js.", delay_before=0.4),
        ],
    ),
    Scenario(
        name="stale_noise_not_inherited",
        note=(
            "Noise, then a gap, then a real question. The question must be "
            "answered on its own -- the earlier fragments must have been "
            "discarded rather than buffered and glued onto it."
        ),
        turns=[
            Turn("Correct."),
            Turn("Hello.", delay_before=0.5),
            Turn("What is Django?", delay_before=2.0),
        ],
    ),
    Scenario(
        name="multi_part_question",
        note=(
            "One question asked in three parts, the way people actually ask "
            "them. Reported from a real interview: the copilot answered only "
            "'have you ever heard of it' and ignored 'what is it' and 'why "
            "use it'. All three parts must reach the LLM together."
        ),
        turns=[Turn("What is LoRA? Why use LoRA? Have you ever heard of LoRA?")],
    ),
    Scenario(
        name="multi_part_question_in_pieces",
        note=(
            "Same multi-part question, but the interviewer pauses between "
            "the parts so each arrives as its own transcript. The parts must "
            "be merged into one question rather than answered one at a time "
            "with the earlier answers cancelled."
        ),
        turns=[
            Turn("what is lora"),
            Turn("why do we use lora", delay_before=0.7),
            Turn("have you ever worked with it", delay_before=0.7),
        ],
    ),
    Scenario(
        name="long_scenario_question",
        note=(
            "A four-sentence scenario question. The three setup sentences "
            "carry the facts the answer has to reason from -- they must all "
            "reach the LLM, not just the final 'how would you debug this?'."
        ),
        turns=[
            Turn("We have a Django API in production."),
            Turn("Response times jumped from 200ms to 3 seconds.", delay_before=0.6),
            Turn("The database CPU is sitting at 90%.", delay_before=0.6),
            Turn("How would you debug this?", delay_before=0.6),
        ],
    ),
    Scenario(
        name="rag_scenario_setup_then_question",
        note=(
            "Verbatim from a real interview (logs/default_2026-08-19.jsonl): a "
            "three-sentence RAG scenario. Both setup sentences were being "
            "discarded before the question arrived, so the LLM answered 'how "
            "will you debug the problem' with no idea what the problem was. "
            "All three sentences must reach it as one question."
        ),
        turns=[
            Turn("Your Rack chatbot is answering correctly for 80% of queries."),
            Turn("But for the remaining 20%, it retrieves a relevant document.", delay_before=1.0),
            Turn("How will you debug the problem, step by step?", delay_before=1.0),
        ],
    ),
    Scenario(
        name="scenario_setup_quoting_an_example",
        note=(
            "Also from the real log. The setup sentence quotes an example "
            "question inside it ('short questions like what is product X'), "
            "which the gate read as a question being asked -- so it answered "
            "the setup and then answered the real question separately, "
            "producing two half-answers. An example is not a question."
        ),
        turns=[
            Turn("Your Rack system works well with short questions like what is product X."),
            Turn(
                "but performs badly on long questions containing multiple "
                "requirements. How would you handle this?",
                delay_before=1.2,
            ),
        ],
    ),
    Scenario(
        name="four_sentence_scenario",
        note=(
            "The longest shape an interviewer actually uses: premise, symptom, "
            "measurement, then the question. Every sentence carries a fact the "
            "answer needs."
        ),
        turns=[
            Turn("After adding 100,000 new documents to the index,"),
            Turn("the chatbot's accuracy suddenly decreased.", delay_before=0.8),
            Turn("Retrieval latency also went up by about 200 milliseconds.", delay_before=0.8),
            Turn("What could be causing this and how would you investigate?", delay_before=0.8),
        ],
    ),
    Scenario(
        name="question_mark_in_garbage",
        note=(
            "Speech-to-text puts a '?' on any rising intonation, including "
            "on mistranscribed noise ('Cora?' appeared six times in one "
            "session). A stray question mark must NOT promote garbage to a "
            "question -- expect no answer at all."
        ),
        turns=[Turn("Ct-4, P. Cora?")],
    ),
    Scenario(
        name="garbled_interview_intent",
        note=(
            "The intent is unmistakable but the words came through wrong. "
            "Should still be recognised as 'tell me about yourself' by "
            "meaning rather than by exact string, and answered for what was "
            "meant instead of for the literal transcription."
        ),
        turns=[Turn("so uday tell me about your self")],
    ),
    Scenario(
        name="bare_followup_after_history",
        note=(
            "First question gets answered normally (creating history). "
            "Then a bare 'why?' on its own -- should classify as "
            "reason=follow_up and get rewritten to reference the previous "
            "Q&A instead of being answered with no context."
        ),
        turns=[
            Turn("What is a race condition?"),
            Turn("why?", wait_for_settle=True),
        ],
    ),
    Scenario(
        name="fast_followup_interrupts_answer",
        note=(
            "The interviewer fires a bare 'why?' WHILE the first answer is "
            "still streaming -- fast enough that, before the fix, it would "
            "cancel the first answer before add_history() ever ran, leaving "
            "the follow-up with no context to resolve against. Should now "
            "show NO answer_CANCELLED for the first answer -- it finishes "
            "normally, THEN the follow-up answers, referencing it."
        ),
        turns=[
            Turn("What is a race condition?"),
            Turn("why?", delay_before=2.5),
        ],
        settle_timeout=12.0,
    ),
    Scenario(
        name="barge_in_new_question",
        note=(
            "Interviewer starts a genuinely NEW question while the copilot "
            "is still mid-answer to the first one. Should cancel the first "
            "answer cleanly (answer_cancelled) and answer only the second -- "
            "NOT stream both concurrently/interleaved."
        ),
        turns=[
            Turn("What is the difference between authentication and authorization?"),
            Turn("Actually, what is a race condition?", delay_before=1.0),
        ],
    ),
    Scenario(
        name="stray_word_during_answer",
        note=(
            "Known tradeoff, not a bug: ANY new speech while an answer is "
            "streaming cancels it, even a short filler word -- the pipeline "
            "can't tell 'meant to interrupt' from 'unrelated background "
            "remark'. Shown here so it's a known, visible behavior rather "
            "than a surprise."
        ),
        turns=[
            Turn("What is the difference between authentication and authorization?"),
            Turn("okay", delay_before=1.0),
        ],
    ),
]


class _StubLLMClient:
    """Replaces the real Groq/OpenAI/etc client in offline mode -- yields a
    short canned answer word-by-word with a small delay per word, so answers
    still visibly "stream" (long enough to interrupt mid-stream for the
    barge-in scenarios) without any network call or API cost."""

    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        question = _extract_question(messages)
        canned = (
            f'(stub answer) Sure, about "{question}" -- here is a short made-up '
            "answer just so we have something to stream and interrupt."
        )
        for word in canned.split(" "):
            await asyncio.sleep(0.09)
            yield word + " "


def _extract_question(messages: list[dict]) -> str:
    content = messages[-1]["content"]
    marker = "Interview question:\n"
    if marker in content:
        rest = content.split(marker, 1)[1]
        question = rest.split("\n\n", 1)[0].strip()
    else:
        question = "(question)"
    # A follow-up gets rewritten into a long instruction that embeds the
    # previous question AND its full answer (see _resolve_followup) --
    # bounded here so the stub's canned reply (which echoes this back)
    # stays a short, fast, bounded stream like a real LLM answer would be,
    # instead of ballooning into a multi-second stream of a 100+ word echo.
    if len(question) > 80:
        question = question[:77] + "..."
    return question


def _stub_build_llm_client(settings, *, provider=None, model=None, max_tokens=None, temperature=None):
    return _StubLLMClient()


class _RecorderSocket:
    """Duck-types just enough of a WebSocket (an async send_json) to receive
    everything broadcast through answer_hub for one session, and prints it
    live with an elapsed-time prefix -- the same events the real overlay
    would receive over /ws/answers."""

    def __init__(self, start_time: float) -> None:
        self._start = start_time
        self.events: list[tuple[float, dict]] = []
        self._answer_buf: list[str] = []

    async def send_json(self, payload: dict) -> None:
        elapsed = time.monotonic() - self._start
        self.events.append((elapsed, payload))
        self._print(elapsed, payload)

    def _print(self, elapsed: float, payload: dict) -> None:
        t = f"[{elapsed:5.2f}s]"
        message_type = payload.get("type")

        if message_type == "transcript":
            conf = payload.get("confidence")
            print(f"{t} heard: \"{payload.get('text')}\" (confidence={conf})")
        elif message_type == "question_gate":
            mark = "-> ANSWER" if payload.get("should_answer") else "-> hold/skip"
            print(f"{t} gate: reason={payload.get('reason'):<20} {mark}  text=\"{payload.get('text')}\"")
        elif message_type == "answer_start":
            self._answer_buf = []
            print(f"{t} answer_start: question=\"{payload.get('question')}\"")
        elif message_type == "answer_token":
            self._answer_buf.append(payload.get("token", ""))
        elif message_type == "answer_done":
            print(f"{t} answer_done: \"{''.join(self._answer_buf).strip()}\"")
        elif message_type == "answer_cancelled":
            print(f"{t} answer_CANCELLED (partial was: \"{''.join(self._answer_buf).strip()}\")")
        elif message_type == "error":
            print(f"{t} error: {payload.get('message')}")
        else:
            print(f"{t} {payload}")


async def _wait_until_quiet(session_id: str, timeout: float) -> None:
    """Polls the pipeline's internal per-session state until nothing is
    pending/streaming for this session, or the timeout elapses."""
    pipeline = get_question_pipeline()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending = pipeline._pending.get(session_id)
        if pending is None:
            return
        task_done = pending.task is None or pending.task.done()
        answer_done = pending.answer_task is None or pending.answer_task.done()
        if task_done and answer_done:
            return
        await asyncio.sleep(0.05)
    print("  (settle timeout reached -- moving on)")


async def run_scenario(scenario: Scenario, *, session_id: str) -> None:
    pipeline = get_question_pipeline()
    start = time.monotonic()
    recorder = _RecorderSocket(start)
    await gate_module.answer_hub.connect(session_id, recorder)

    print(f"\n=== {scenario.name} ===")
    print(f"note: {scenario.note}")

    for turn in scenario.turns:
        if turn.wait_for_settle:
            await _wait_until_quiet(session_id, scenario.settle_timeout)
        elif turn.delay_before:
            await asyncio.sleep(turn.delay_before)
        print(f"[{time.monotonic() - start:5.2f}s] (speaking) \"{turn.text}\"")
        await pipeline.submit_transcript(session_id=session_id, text=turn.text, confidence=1.0)

    await _wait_until_quiet(session_id, scenario.settle_timeout)
    await gate_module.answer_hub.disconnect(session_id, recorder)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real .env settings (real Groq calls, real fast-intent tier) instead of the offline stub.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated substrings -- only run scenarios whose name contains one of these.",
    )
    parser.add_argument("--list", action="store_true", help="List scenario names and notes, then exit.")
    args = parser.parse_args()

    if args.list:
        for scenario in SCENARIOS:
            print(f"{scenario.name}\n  {scenario.note}\n")
        return

    selected = SCENARIOS
    if args.only:
        needles = [needle.strip() for needle in args.only.split(",") if needle.strip()]
        selected = [s for s in SCENARIOS if any(n in s.name for n in needles)]
        if not selected:
            print(f"No scenario name matched --only={args.only!r}. Use --list to see names.")
            return

    if args.live:
        print("Running in --live mode: real provider/model from .env, real fast-intent classifier.\n")
    else:
        print(
            "Running offline: LLM calls are stubbed (no network/cost), "
            "fast-intent tier is disabled so results are deterministic.\n"
        )
        gate_module.build_llm_client = _stub_build_llm_client
        get_question_pipeline()._settings.fast_intent_enabled = False

    for index, scenario in enumerate(selected):
        await run_scenario(scenario, session_id=f"gate-test-{scenario.name}-{index}")

    print("\nAll scenarios done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
