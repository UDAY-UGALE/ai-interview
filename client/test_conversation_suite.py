"""Permanent regression suite for human-conversation handling.

Everything here was a MEASURED failure before the conversation work landed,
so each case is a specific defect with a specific fix, not a hypothetical.
The suite is deliberately offline and deterministic -- stubbed LLM, tier-2
classifier disabled -- because what it tests is the gate, the wait policy
and the merge/correction logic, all of which decide *when* and *what* to
ask. Answer quality is a separate concern with a separate harness.

    python client\\test_conversation_suite.py
    python client\\test_conversation_suite.py --only followup,correction
    python client\\test_conversation_suite.py --verbose

Exit code is non-zero when any case fails, so this can gate a commit.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.redis_client import InterviewSessionContext, get_session_store  # noqa: E402
from app.services import question_gate as gate_module  # noqa: E402
from app.services.question_gate import get_question_pipeline  # noqa: E402
from app.services.session_vocabulary import (  # noqa: E402
    build_session_vocabulary,
    session_term_set,
)
from app.services.transcript_normalizer import normalize_transcript  # noqa: E402


RESUME = """UDAY UGALE -- AI / Full-Stack Engineer, Mannlowe, Pune.
Built a RAG chatbot over product documentation with LangChain and a Chroma
vector store. FSM-driven onboarding agent on FastAPI with tool calling.
Backend: Python, FastAPI, Django REST Framework, PostgreSQL, Redis, Celery.
Frontend: React, Redux. Infra: Docker, GitHub Actions, AWS EC2/ECR.
ERPNext / Frappe customisation."""

JD = """AI Engineer. Python, FastAPI, RAG pipelines, vector databases,
LangChain, Docker, AWS. Nice to have: Kubernetes, Redis, PostgreSQL."""

# A session where "Rack" is a REAL term the candidate works with. Used by the
# negative normalization tests: the same input that becomes RAG above must
# stay untouched here, or the normalizer is unsafe.
RUBY_RESUME = """PRIYA SHARMA -- Ruby Backend Engineer.
Ruby on Rails, Rack middleware, Sinatra, Puma, Sidekiq, PostgreSQL, Redis.
Built custom Rack middleware for request tracing across a Rails monolith."""


# ---------------------------------------------------------------------------
# Pure-function checks: normalization and vocabulary
# ---------------------------------------------------------------------------

def _vocab(resume: str, jd: str = "") -> set[str]:
    """The NORMALIZER's evidence set -- session terms only.

    include_baseline=False on purpose, and it is the whole safety story: the
    generic technical list is fine for biasing a recognizer but is not
    evidence that THIS interview is about RAG. With it included, a session
    with no resume rewrote a genuine "What is Rack?" into RAG.
    """
    return session_term_set(
        build_session_vocabulary(
            resume_text=resume, job_description=jd, include_baseline=False
        )
    )


NORMALIZATION_CASES: list[tuple[str, str, str, str]] = [
    # (name, resume, heard, expected normalized)
    ("rag_from_rack", RESUME, "Can you give me an idea about Rack?",
     "Can you give me an idea about RAG?"),
    ("rag_in_pipeline", RESUME, "You need to build a Rack pipeline.",
     "You need to build a RAG pipeline."),
    ("redis_from_red_is", RESUME, "Where did you use red is in that project?",
     "Where did you use Redis in that project?"),
    ("docker_from_doctor", RESUME, "So you work with a doctor. Do you use it in production?",
     "So you work with a Docker. Do you use it in production?"),
    ("fastapi_from_fast_repair", RESUME, "You work with a fast repair. Tell me about it.",
     "You work with a FastAPI. Tell me about it."),
    ("django_from_jango", RESUME, "Tell me about your jango project.",
     "Tell me about your Django project."),
    ("http_from_h3b", RESUME, "Give me an idea about these H3B methods.",
     "Give me an idea about these HTTP methods."),
    ("github_from_get_hub", RESUME, "Do you use get hub actions?",
     "Do you use GitHub actions?"),
    ("kubernetes_from_kubernets", RESUME + "\nKubernetes", "A Kubernets pod is restarting.",
     "A Kubernetes pod is restarting."),
    # ---- NEGATIVE: these must NOT be rewritten --------------------------
    ("keep_rack_for_ruby", RUBY_RESUME, "What is Rack?", "What is Rack?"),
    ("keep_rack_middleware", RUBY_RESUME, "Explain your Rack middleware.",
     "Explain your Rack middleware."),
    ("no_vocab_no_change", "", "Can you give me an idea about Rack?",
     "Can you give me an idea about Rack?"),
    ("doctor_without_tech_context", "", "I saw a doctor yesterday.",
     "I saw a doctor yesterday."),
    ("correct_term_untouched", RESUME, "Can you give me an idea about RAG?",
     "Can you give me an idea about RAG?"),
]


def run_normalization_cases(verbose: bool) -> tuple[int, int]:
    passed = 0
    print("\n=== transcript normalization (session-gated) ===")
    for name, resume, heard, expected in NORMALIZATION_CASES:
        result = normalize_transcript(heard, session_terms=_vocab(resume))
        ok = result.normalized == expected
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")
        if not ok or verbose:
            print(f"         heard    : {heard!r}")
            print(f"         expected : {expected!r}")
            print(f"         got      : {result.normalized!r}")
            if result.rejected:
                print(f"         declined : {result.rejected}")
    return passed, len(NORMALIZATION_CASES)


VOCABULARY_CASES = [
    ("resume_terms_present", RESUME, ["FastAPI", "Chroma", "Celery", "ERPNext"]),
    ("jd_terms_present", JD, ["Kubernetes", "LangChain"]),
]


def run_vocabulary_cases(verbose: bool) -> tuple[int, int]:
    passed = 0
    print("\n=== session vocabulary extraction ===")
    for name, text, expected_terms in VOCABULARY_CASES:
        terms = _vocab(text)
        missing = [t for t in expected_terms if t.lower() not in terms]
        ok = not missing
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + ("" if ok else f"  missing={missing}"))
    # The resume must survive even when a long JD is also present -- this is
    # the failure that made the old Whisper prompt useless.
    combined = session_term_set(
        build_session_vocabulary(resume_text=RESUME, job_description=JD * 6)
    )
    ok = "chroma" in combined and "erpnext" in combined
    passed += ok
    print(f"  [{'PASS' if ok else 'FAIL'}] resume_survives_long_jd")
    return passed, len(VOCABULARY_CASES) + 1


# ---------------------------------------------------------------------------
# Pipeline checks: what actually gets asked, and how many times
# ---------------------------------------------------------------------------

@dataclass
class Utt:
    text: str
    speaking: float = 1.4
    stt: float = 0.30
    gap: float = 0.0
    settle: bool = False


@dataclass
class Case:
    name: str
    group: str
    utts: list[Utt]
    # How many answers the user should be LEFT with (superseded ones excluded).
    expect_final: int = 1
    # Substrings that must all appear in the last question actually asked.
    expect_contains: list[str] = field(default_factory=list)
    # Substrings that must NOT appear in it.
    expect_missing: list[str] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    note: str = ""


CASES: list[Case] = [
    # ---- PHASE 7 CASE A: complete follow-up, answer immediately ---------
    Case("followup_why", "followup", [Utt("Why would you use it?", 1.2)],
         expect_contains=["Why would you use it"],
         history=[("What is RAG?", "RAG is retrieval augmented generation.")],
         note="complete follow-up -- must fire with no continuation window"),
    Case("followup_disadvantages", "followup", [Utt("What are the disadvantages?", 1.4)],
         expect_contains=["disadvantages"],
         history=[("What is RAG?", "RAG is retrieval augmented generation.")]),
    Case("followup_have_you_used", "followup", [Utt("Have you used it?", 1.1)],
         expect_contains=["Have you used it"],
         history=[("What is RAG?", "RAG is retrieval augmented generation.")]),
    Case("followup_challenges", "followup", [Utt("What challenges did you face?", 1.6)],
         expect_contains=["challenges"],
         history=[("Tell me about your Django project.", "I built a REST API.")],
         note="must NOT be truncated to 'in your project?' by intent matching"),
    Case("followup_what_about_django", "followup", [Utt("What about Django?", 1.1)],
         expect_contains=["Django"],
         history=[("Why did you choose FastAPI?", "Async support.")]),
    Case("followup_example", "followup", [Utt("Can you give me an example?", 1.4)],
         expect_contains=["example"],
         history=[("What is an API?", "An API is a contract between services.")]),
    Case("followup_what_else", "followup", [Utt("What else?", 0.8)],
         expect_contains=["What else"],
         history=[("What is RAG?", "RAG is retrieval augmented generation.")]),
    Case("followup_why_period", "followup", [Utt("Why.", 0.6)],
         expect_contains=["Why"],
         history=[("What is a race condition?", "Two threads racing.")],
         note="Whisper emits '.' on falling intonation -- must still answer"),

    # ---- PHASE 7 CASE B: incomplete follow-up, wait then merge ----------
    Case("incomplete_challenges", "incomplete",
         [Utt("What challenges did you", 1.3), Utt("face in your project?", 1.5, gap=0.6)],
         expect_contains=["challenges", "project"],
         history=[("Tell me about your Django project.", "I built a REST API.")],
         note="PHASE 7 CASE B -- one merged question"),
    Case("incomplete_can_you_explain", "incomplete",
         [Utt("Can you explain?", 1.2), Utt("RAG.", 0.6, gap=0.6)],
         expect_contains=["explain", "RAG"],
         note="measured: 2 answers at EVERY pause length before the fix"),
    Case("incomplete_will_you_give", "incomplete",
         [Utt("Will you give me the...", 1.3),
          Utt("idea about your recent projects.", 1.9, gap=0.8)],
         expect_contains=["idea about your recent projects"],
         note="verbatim from a real session log"),
    Case("incomplete_difference_between", "incomplete",
         [Utt("What is the difference between", 1.7),
          Utt("FastAPI and Django.", 1.8, gap=0.6)],
         expect_contains=["difference between", "FastAPI"]),

    # ---- PHASE 7 CASE C: a genuinely new question is not merged --------
    Case("new_question_after_topic", "newquestion",
         [Utt("Tell me about your RAG project.", 1.8),
          Utt("What is Kubernetes?", 1.3, settle=True)],
         expect_final=2,
         expect_contains=["Kubernetes"],
         expect_missing=["RAG project"],
         note="must NOT be glued onto the previous question"),

    # ---- PHASE 10: corrections -----------------------------------------
    Case("correction_flask_django", "correction",
         [Utt("Tell me about your Flask project.", 1.9),
          Utt("Actually, I mean my Django project.", 2.2, gap=0.5)],
         expect_contains=["Django"],
         note="measured: answered Flask at every pause length, never Django"),
    Case("correction_mysql_postgres", "correction",
         [Utt("Which database did you use?", 1.6),
          Utt("MySQL. Sorry, I mean PostgreSQL.", 2.0, gap=0.5)],
         expect_contains=["PostgreSQL"]),
    Case("correction_http", "correction",
         [Utt("Give me an idea about these H3B methods.", 2.4),
          Utt("I said the HTTPS methods.", 1.4, gap=0.6)],
         expect_contains=["HTTPS"],
         note="verbatim from a real session; the correction was discarded"),
    Case("correction_not_x_but_y", "correction",
         [Utt("Tell me about Flask.", 1.2), Utt("Not Flask, Django.", 1.2, gap=0.5)],
         expect_contains=["ango"]),
    Case("correction_rephrase", "correction",
         [Utt("How do you handle load?", 1.5),
          Utt("Let me rephrase. How would you scale the API?", 2.4, gap=0.5)],
         expect_contains=["scale"]),
    Case("correction_actually_new_question", "correction",
         [Utt("What is RAG?", 1.2),
          Utt("Actually, what is a race condition?", 2.0, gap=0.6)],
         expect_contains=["race condition"],
         note="'actually' + a standalone question is a NEW question, not a "
              "correction of the old topic -- it must not answer about RAG"),
]


class _StubLLM:
    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        for word in ("stub", "answer."):
            await asyncio.sleep(0.05)
            yield word + " "


def _stub_build(settings, *, provider=None, model=None, max_tokens=None, temperature=None):
    return _StubLLM()


class _Sink:
    def __init__(self) -> None:
        self.asked: list[str] = []
        self.done = 0
        self.superseded = 0
        self.cancelled = 0
        self.gates: list[tuple[str, str]] = []

    async def send_json(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind == "answer_start":
            self.asked.append(payload.get("question", ""))
        elif kind == "answer_done":
            self.done += 1
        elif kind == "answer_superseded":
            self.superseded += 1
        elif kind == "answer_cancelled":
            self.cancelled += 1
        elif kind == "question_gate":
            self.gates.append((payload.get("reason", ""), payload.get("text", "")))

    @property
    def final(self) -> int:
        return self.done - self.superseded


async def _wait_quiet(session_id: str, timeout: float = 15.0) -> None:
    pipeline = get_question_pipeline()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending = pipeline._pending.get(session_id)
        if pending is not None and (
            (pending.task is None or pending.task.done())
            and (pending.answer_task is None or pending.answer_task.done())
        ):
            return
        await asyncio.sleep(0.05)


async def run_case(case: Case, verbose: bool) -> bool:
    pipeline = get_question_pipeline()
    session_id = f"suite-{case.name}"
    store = get_session_store()
    await store.set_context(
        session_id, InterviewSessionContext(resume_text=RESUME, job_description=JD)
    )
    for question, answer in case.history:
        await store.add_history(session_id, question, answer)
    pipeline.refresh_vocabulary(session_id)

    sink = _Sink()
    await gate_module.answer_hub.connect(session_id, sink)

    async def deliver(text: str, uid: int, delay: float) -> None:
        await asyncio.sleep(delay)
        await pipeline.submit_transcript(
            session_id=session_id, text=text, confidence=0.75,
            confidence_known=True, utterance_id=uid, utterance_final=True,
            spoken_at=time.monotonic(),
        )

    tasks = []
    for index, utt in enumerate(case.utts, start=1):
        if utt.settle:
            await _wait_quiet(session_id)
            await asyncio.sleep(0.4)
        elif utt.gap:
            await asyncio.sleep(utt.gap)
        await pipeline.set_speech_active(session_id=session_id, active=True)
        await asyncio.sleep(utt.speaking)
        await pipeline.set_speech_active(session_id=session_id, active=False)
        tasks.append(asyncio.create_task(deliver(utt.text, index, utt.stt)))

    await asyncio.gather(*tasks)
    await _wait_quiet(session_id)
    await asyncio.sleep(0.5)
    await _wait_quiet(session_id)
    await gate_module.answer_hub.disconnect(session_id, sink)

    last = sink.asked[-1] if sink.asked else ""
    problems: list[str] = []
    if sink.final != case.expect_final:
        problems.append(f"final answers {sink.final} != {case.expect_final}")
    for needle in case.expect_contains:
        if needle.lower() not in last.lower():
            problems.append(f"missing {needle!r}")
    for needle in case.expect_missing:
        if needle.lower() in last.lower():
            problems.append(f"unexpected {needle!r}")

    ok = not problems
    print(f"  [{'PASS' if ok else 'FAIL'}] {case.name}")
    if not ok or verbose:
        if case.note:
            print(f"         note     : {case.note}")
        print(f"         asked    : {sink.asked}")
        print(f"         final={sink.final} done={sink.done} "
              f"superseded={sink.superseded} cancelled={sink.cancelled}")
        print(f"         gates    : {[g[0] for g in sink.gates]}")
        for problem in problems:
            print(f"         PROBLEM  : {problem}")
    return ok


async def run_scenario_mode(verbose: bool) -> bool:
    """MODE B: Start -> speak a long question with pauses -> Stop.

    The whole point is that nothing is inferred, so the assertion is exact:
    one answer, containing every part of what was said.
    """
    pipeline = get_question_pipeline()
    session_id = "suite-scenario-mode"
    await get_session_store().set_context(
        session_id, InterviewSessionContext(resume_text=RESUME, job_description=JD)
    )
    pipeline.refresh_vocabulary(session_id)
    sink = _Sink()
    await gate_module.answer_hub.connect(session_id, sink)

    await pipeline.start_scenario(session_id=session_id)
    for text in (
        "Okay, so let's take a scenario.",
        "Your production server's disk partition is 100% full.",
        "The application has stopped responding.",
        "And the monitoring dashboard shows no alerts.",
        "How would you troubleshoot this?",
    ):
        await pipeline.submit_transcript(
            session_id=session_id, text=text, confidence=0.8, confidence_known=True
        )
        await asyncio.sleep(0.45)
    await pipeline.stop_scenario(session_id=session_id)
    await _wait_quiet(session_id)
    await gate_module.answer_hub.disconnect(session_id, sink)

    last = sink.asked[-1] if sink.asked else ""
    problems = []
    if sink.final != 1:
        problems.append(f"final answers {sink.final} != 1")
    for needle in ("disk partition", "stopped responding", "no alerts", "troubleshoot"):
        if needle.lower() not in last.lower():
            problems.append(f"setup lost: {needle!r}")

    ok = not problems
    print(f"  [{'PASS' if ok else 'FAIL'}] scenario_mode_single_answer")
    if not ok or verbose:
        print(f"         asked    : {sink.asked}")
        for problem in problems:
            print(f"         PROBLEM  : {problem}")
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="comma-separated groups or names")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gate_module.build_llm_client = _stub_build
    get_question_pipeline()._settings.fast_intent_enabled = False

    passed = total = 0
    p, t = run_normalization_cases(args.verbose)
    passed += p
    total += t
    p, t = run_vocabulary_cases(args.verbose)
    passed += p
    total += t

    selected = CASES
    if args.only:
        needles = [n.strip() for n in args.only.split(",") if n.strip()]
        selected = [c for c in CASES if c.group in needles or any(n in c.name for n in needles)]

    groups = sorted({c.group for c in selected})
    for group in groups:
        print(f"\n=== {group} ===")
        for case in [c for c in selected if c.group == group]:
            ok = await run_case(case, args.verbose)
            passed += ok
            total += 1

    if not args.only or "scenario" in (args.only or ""):
        print("\n=== scenario mode (MODE B) ===")
        ok = await run_scenario_mode(args.verbose)
        passed += ok
        total += 1

    print(f"\n{'=' * 60}\nTOTAL {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nStopped.")
