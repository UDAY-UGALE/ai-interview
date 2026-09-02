"""
Offline tests for the answer-depth layer in app/services/answer_depth.py, plus
a --live mode that checks real generated answers actually respect the level.

The bug this guards against: the system prompt rewards specificity, so a broad
question ("how would you use GenAI in fraud detection?") was being answered
with fine-tuned embeddings and agentic workflows in the opening sentence. That
is a candidate who cannot tell a broad question from a deep one, which reads
as over-engineering however correct the content is.

The rule is question depth -> answer depth, never question -> maximum detail.

Offline cases cover the classifier and the wiring into _build_answer_messages
(no network, no API key). --live sends the six worked examples to the real
provider and checks that level-1 answers stay clear of level-3 vocabulary
while level-3 and level-4 answers are free to use it.

Usage (from the project root, where .env lives):
    python client\\test_answer_depth.py
    python client\\test_answer_depth.py --list
    python client\\test_answer_depth.py --only url_shortener
    python client\\test_answer_depth.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.redis_client import ConversationTurn, InterviewSessionContext  # noqa: E402
from app.services.answer_depth import (  # noqa: E402
    ARCHITECTURE,
    IMPLEMENTATION,
    MECHANISM,
    OVERVIEW,
    build_depth_block,
    classify_answer_depth,
)
from app.services.candidate_context import ProjectContext  # noqa: E402
from app.services.llm import build_llm_client  # noqa: E402
from app.services.question_gate import _build_answer_messages  # noqa: E402


# ---------------------------------------------------------------------------
# Harness (same shape as client/test_project_context.py)
# ---------------------------------------------------------------------------

CASES: list[tuple[str, str, callable]] = []


def case(name: str, note: str):
    def register(function):
        CASES.append((name, note, function))
        return function

    return register


class Failure(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def turn(question: str, answer: str = "...") -> ConversationTurn:
    return ConversationTurn(question=question, answer=answer, created_at="")


CONTEXT = InterviewSessionContext(
    resume_text="Engineer. Python, FastAPI, PostgreSQL, Redis, Kafka, Docker, Kubernetes.",
    projects=(
        ProjectContext(
            name="Checkout platform",
            role="I owned the payment service.",
            architecture="FastAPI services, PostgreSQL per service, Kafka for events, Redis cache.",
            technologies="Python, FastAPI, PostgreSQL, Redis, Kafka",
        ),
    ),
)


# ---------------------------------------------------------------------------
# The six worked examples from the spec
# ---------------------------------------------------------------------------

WORKED_EXAMPLES: list[tuple[str, str, int]] = [
    ("genai_broad", "How would you use GenAI in fraud detection?", OVERVIEW),
    ("genai_implement", "How would you implement GenAI fraud detection?", IMPLEMENTATION),
    ("genai_novel", "How would you detect previously unseen fraud patterns?", MECHANISM),
    ("genai_architecture", "Design a scalable GenAI fraud detection architecture.", ARCHITECTURE),
    ("url_shortener", "How would you design a URL shortener?", IMPLEMENTATION),
    ("deploy_python", "How would you deploy a Python application?", OVERVIEW),
]


@case("worked_examples", "The six spec examples each land on the level the spec asks for.")
def _worked_examples() -> str:
    for name, question, expected in WORKED_EXAMPLES:
        got = classify_answer_depth(question)
        expect(
            got.level == expected,
            f"{name}: expected level {expected}, got {got.level} ({got.reason})",
        )
    return "all six examples classify as specified (1, 2, 3, 4, 2, 1)"


@case("design_needs_scale_for_level_4", "A design verb alone is level 2; scale words unlock level 4.")
def _design_needs_scale() -> str:
    plain = [
        "How would you design a URL shortener?",
        "Design a notification system.",
        "How would you build a rate limiter?",
    ]
    scaled = [
        "How would you design a scalable notification system?",
        "Design a distributed URL shortener.",
        "Design a highly available payment architecture.",
    ]
    for q in plain:
        expect(
            classify_answer_depth(q).level == IMPLEMENTATION,
            f"expected level 2 for {q!r}, got {classify_answer_depth(q).level}",
        )
    for q in scaled:
        expect(
            classify_answer_depth(q).level == ARCHITECTURE,
            f"expected level 4 for {q!r}, got {classify_answer_depth(q).level}",
        )
    return "3 plain design questions at level 2, 3 scaled ones at level 4"


@case("broad_questions_stay_broad", "Broad 'how would you use/approach X' questions stay at level 1.")
def _broad_stay_broad() -> str:
    broad = [
        "How would you use GenAI in fraud detection?",
        "How would you deploy a Python application?",
        "How do you approach testing?",
        "What is Redis?",
        "How would you use caching?",
        "Tell me about your experience with Kafka.",
    ]
    for q in broad:
        got = classify_answer_depth(q)
        expect(got.level == OVERVIEW, f"expected level 1 for {q!r}, got {got.level} ({got.reason})")
    return "6 broad questions all at level 1"


@case("followup_escalates", "A follow-up asking to go deeper raises the level.")
def _followup_escalates() -> str:
    history = [turn("How would you use GenAI in fraud detection?")]
    base = classify_answer_depth("How would you use GenAI in fraud detection?")
    expect(base.level == OVERVIEW, "baseline should be level 1")

    deeper = classify_answer_depth("Can you go deeper on that?", history)
    expect(deeper.level > base.level, f"expected escalation, got level {deeper.level}")

    exactly = classify_answer_depth("How exactly would that work?", history)
    expect(exactly.level > base.level, f"expected escalation, got level {exactly.level}")
    return f"level 1 -> {deeper.level} on 'go deeper', -> {exactly.level} on 'how exactly'"


@case("followup_does_not_reset", "A short follow-up inside a deep thread does not fall back to level 1.")
def _followup_no_reset() -> str:
    history = [turn("Design a scalable fraud detection architecture.")]
    got = classify_answer_depth("And the storage?", history)
    expect(
        got.level == ARCHITECTURE,
        f"expected the thread's level 4 to carry, got {got.level} ({got.reason})",
    )
    alone = classify_answer_depth("And the storage?")
    expect(alone.level == OVERVIEW, f"without history this should be level 1, got {alone.level}")
    return "level 4 carries into a bare follow-up; the same words alone are level 1"


@case("resolved_followup_uses_quoted_text", "The resolver's rewritten paragraph is classified on the quoted follow-up.")
def _resolved_followup() -> str:
    # What _resolve_followup actually produces. The surrounding paragraph
    # repeats the previous architecture question, which must not inflate the
    # level on its own.
    rewritten = (
        'The interviewer just asked a short follow-up: "why?" -- this is about what '
        'you JUST said, not a new topic. The previous question was "How would you '
        'use GenAI in fraud detection?" and you answered: "I would use it to help '
        'analysts investigate alerts." Directly justify, expand on, or clarify that '
        "specific answer with real additional reasoning or detail."
    )
    got = classify_answer_depth(rewritten)
    expect(
        got.level == OVERVIEW,
        f"quoted follow-up is 'why?', so level should stay 1, got {got.level} ({got.reason})",
    )
    return "classifies the quoted follow-up, not the surrounding paragraph"


@case("block_reaches_the_prompt", "The depth block reaches the user message, ahead of the question.")
def _block_in_prompt() -> str:
    question = "How would you use GenAI in fraud detection?"
    messages = _build_answer_messages(question, CONTEXT, [], char_budget=5200)
    user = messages[-1]["content"]
    expect("ANSWER DEPTH -- level 1 of 4 (overview)" in user, "depth header missing")
    expect(user.strip().endswith(question), "question must remain last in the user message")
    expect(
        user.index("ANSWER DEPTH") < user.index("CURRENT QUESTION"),
        "depth block must come before the question",
    )
    expect(len(messages) == 3, f"message shape changed: got {len(messages)} messages")
    return "block present, before the question, question still last, still 3 messages"


@case("levels_differ_in_prompt", "Different questions produce visibly different directives.")
def _levels_differ() -> str:
    blocks = {q: build_depth_block(q) for _n, q, _l in WORKED_EXAMPLES}
    expect(len(set(blocks.values())) == 4, "expected 4 distinct directives across the 6 examples")
    broad = blocks["How would you use GenAI in fraud detection?"]
    arch = blocks["Design a scalable GenAI fraud detection architecture."]
    expect("Name the STAGE rather than the product" in broad, "level 1 restraint rule missing")
    expect("NOT by naming a tool" in broad, "level 1 tool-name restraint missing")
    expect("full picture is fair game" in arch, "level 4 permission missing")
    return "4 distinct directives; level 1 restrains, level 4 permits"


@case("l1_reasoning_skeleton", "Level 1 asks for Approach -> Why -> Example -> Boundary, then 'In short'.")
def _l1_skeleton() -> str:
    block = build_depth_block("How would you use GenAI in fraud detection?")
    for needle, what in [
        ("**1.", "numbered section 1"),
        ("**2.", "numbered section 2"),
        ("**3.", "numbered section 3"),
        ("**4.", "numbered section 4"),
        ("the approach", "approach step"),
        ("why that works", "why step"),
        ("one concrete example", "example step"),
        ("the boundary", "boundary step"),
        ("**In short:**", "closing summary"),
    ]:
        expect(needle in block, f"level 1 skeleton missing {what} ({needle!r})")
    return "Approach -> Why -> Example -> Boundary -> In short, all present"


@case("l1_is_reasoning_not_technology", "Level 1 forbids technology-driven points and buzzword padding.")
def _l1_reasoning_not_tech() -> str:
    block = build_depth_block("How would you use GenAI in fraud detection?")
    expect("every header is a step in your reasoning, never a technology name" in block,
           "headers-are-reasoning rule missing")
    expect("NOT by naming a tool" in block, "sub-bullets-earn-by-reason rule missing")
    expect("name at most ONE product in the whole answer" in block,
           "one-product ceiling missing")
    expect("works alongside it rather than replacing it" in block,
           "complement-not-replace rule missing at level 1")
    return "reasoning-over-technology, one-product ceiling and complement rules present"


@case("format_defers_to_depth", "The system prompt hands shape control to the depth block.")
def _format_defers() -> str:
    system = _build_answer_messages("What is Redis?", CONTEXT, [], char_budget=5200)[0]["content"]
    expect("ANSWER DEPTH block in the user message sets the shape" in system,
           "FORMAT does not defer to the depth block")
    expect("UNLESS the ANSWER DEPTH block asks for them" in system,
           "the headings/bold/summary ban is still unconditional")
    return "FORMAT defers to ANSWER DEPTH for shape and for headings/summary"


@case("deep_levels_keep_plain_bullets", "Levels 3 and 4 stay plain bullets, not the L1/L2 skeleton.")
def _deep_levels_plain() -> str:
    for question, level in [
        ("How would you detect previously unseen fraud patterns?", MECHANISM),
        ("Design a scalable GenAI fraud detection architecture.", ARCHITECTURE),
    ]:
        got = classify_answer_depth(question)
        expect(got.level == level, f"{question!r} classified as {got.level}, expected {level}")
        block = build_depth_block(question)
        expect("no section headers and no summary line" in block,
               f"level {level} should ask for plain bullets")
        expect("**In short:**" not in block, f"level {level} must not ask for a summary line")
        expect("**1." not in block, f"level {level} must not use the L1/L2 skeleton")
    return "levels 3 and 4 ask for plain bullets with no headers or summary"


@case("accuracy_rule_present", "The unsupported-claim rule is in the system prompt.")
def _accuracy_rule() -> str:
    messages = _build_answer_messages("What is Redis?", CONTEXT, [], char_budget=5200)
    system = messages[0]["content"]
    expect("Credit a benefit only to the thing that actually delivers it" in system,
           "attribution rule missing")
    expect("works ALONGSIDE it" in system, "complement-not-replace rule missing")
    return "attribution and complement-not-replace rules both present"


@case("classifier_is_cheap", "Classification is fast enough for the hot path.")
def _cheap() -> str:
    import time

    questions = [q for _n, q, _l in WORKED_EXAMPLES]
    start = time.perf_counter()
    for _ in range(200):
        for q in questions:
            classify_answer_depth(q)
    per_call = (time.perf_counter() - start) / (200 * len(questions)) * 1_000_000
    expect(per_call < 500, f"{per_call:.0f}us per classification is too slow for the hot path")
    return f"{per_call:.0f}us per classification"


# ---------------------------------------------------------------------------
# --live: does the real model actually respect the level?
# ---------------------------------------------------------------------------

# Vocabulary that belongs at level 3/4 and should not lead a level-1 answer.
# Two tiers, because "don't jump to Kubernetes, service mesh, blue-green,
# Terraform, Helm" is a complaint about the PILE, not about any one word.
#
# ADVANCED has no place in a level-1 answer at all: naming a vector database
# for a question that asked what you would do is the over-engineering itself.
ADVANCED_VOCAB = [
    "embedding", "vector database", "vector store", "rag", "fine-tun",
    "feature store", "consistent hashing", "bloom filter", "sharding",
    "service mesh", "istio", "terraform", "helm", "blue-green",
]
# PLATFORM names are mainstream and are in this candidate's own context, so one
# of them in passing is a normal thing for an engineer to say. Two or more is
# the pile-on the spec is about.
PLATFORM_VOCAB = ["kafka", "kubernetes", "k8s"]
PLATFORM_BUDGET_AT_LEVEL_1 = 1

# The answer is spoken aloud. Past roughly 45 seconds an interviewer stops
# listening, so a level-1 answer that runs long has failed at its job even if
# every sentence in it is good. Generous ceilings -- these catch a regression
# into essay mode, not ordinary variation.
WORD_CEILING = {OVERVIEW: 200, IMPLEMENTATION: 260, MECHANISM: 260, ARCHITECTURE: 320}


def _mentions(text: str, term: str) -> bool:
    """Leading-boundary match, trailing inflections allowed.

    Plain substring search reported "rag" inside "storage" and "average" and
    marked clean answers as reaching for retrieval-augmented generation. A
    full word boundary on both ends fixes that but then loses "fine-tuning"
    for the stem "fine-tun", so only the START is anchored.
    """
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"[a-z]*", text) is not None


@dataclass
class LiveResult:
    name: str
    level: int
    words: int
    advanced: list[str]
    platforms: list[str]
    sections: int
    has_summary: bool
    answer: str


async def run_live(only: list[str] | None, provider: str | None, model: str | None) -> int:
    settings = get_settings()
    client = build_llm_client(settings, provider=provider, model=model)
    print(f"--live: {provider or settings.answer_provider} / "
          f"{model or settings.answer_model}\n")

    failures = 0
    results: list[LiveResult] = []
    for name, question, expected in WORKED_EXAMPLES:
        if only and not any(n in name for n in only):
            continue
        messages = _build_answer_messages(
            question, CONTEXT, [], char_budget=settings.candidate_context_char_budget
        )
        answer = ""
        async for token in client.stream_chat(messages):
            answer += token
        # Models emit non-breaking (U+2011) and en-dash hyphens, so "fine-tun"
        # silently missed "fine‑tune" until this normalisation went in.
        lowered = answer.lower()
        for dash in "‐‑‒–—−":
            lowered = lowered.replace(dash, "-")
        advanced = sorted({t for t in ADVANCED_VOCAB if _mentions(lowered, t)})
        platforms = sorted({t for t in PLATFORM_VOCAB if _mentions(lowered, t)})
        sections = len(re.findall(r"^\s*\*\*\d+\.", answer, flags=re.M))
        has_summary = bool(re.search(r"\*\*in short:?\*\*", lowered))
        results.append(
            LiveResult(name, expected, len(answer.split()), advanced, platforms,
                       sections, has_summary, answer.strip())
        )

    for r in results:
        verdict = "ok  "
        if r.words > WORD_CEILING[r.level]:
            verdict = "FAIL"
            detail = (f"too long to speak: {r.words} words "
                      f"(level-{r.level} ceiling {WORD_CEILING[r.level]})")
            failures += 1
        elif r.level in (OVERVIEW, IMPLEMENTATION) and r.sections < 3:
            verdict = "FAIL"
            detail = f"expected the numbered reasoning skeleton, got {r.sections} section(s)"
            failures += 1
        elif r.level == OVERVIEW and not r.has_summary:
            verdict = "FAIL"
            detail = "level-1 answer is missing the 'In short' summary"
            failures += 1
        elif r.level == OVERVIEW and r.advanced:
            verdict = "FAIL"
            detail = f"level-1 answer reached for advanced tech: {', '.join(r.advanced)}"
            failures += 1
        elif r.level == OVERVIEW and len(r.platforms) > PLATFORM_BUDGET_AT_LEVEL_1:
            verdict = "FAIL"
            detail = f"level-1 answer piled on platforms: {', '.join(r.platforms)}"
            failures += 1
        elif r.level == OVERVIEW:
            named = ", ".join(r.platforms) or "none"
            detail = (f"{r.sections} sections + summary; no advanced tech; "
                      f"platforms named: {named} (budget {PLATFORM_BUDGET_AT_LEVEL_1})")
        else:
            used = ", ".join(r.advanced + r.platforms) or "none"
            shape = f"{r.sections} sections" if r.sections else "plain bullets"
            detail = f"level {r.level}; {shape}; deeper terms used: {used}"
        print(f"[{verdict}] {r.name}  ({r.words} words)\n       {detail}")
        print("       " + r.answer.replace("\n", "\n       ")[:900] + "\n")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    parser.add_argument("--only", default=None, help="Comma-separated substrings of case names.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also generate the six worked examples with the real provider and check "
        "that level-1 answers stay out of level-3 vocabulary (real API calls).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override ANSWER_PROVIDER for --live (e.g. anthropic), so verification "
        "is not blocked by one vendor's daily token budget.",
    )
    parser.add_argument("--model", default=None, help="Override ANSWER_MODEL for --live.")
    args = parser.parse_args()

    if args.list:
        for name, note, _f in CASES:
            print(f"{name}\n  {note}\n")
        for name, question, level in WORKED_EXAMPLES:
            print(f"{name} (--live only)\n  asks: {question}  [expects level {level}]\n")
        return 0

    only = [n.strip() for n in args.only.split(",") if n.strip()] if args.only else None
    selected = [c for c in CASES if not only or any(n in c[0] for n in only)]

    print(f"Running {len(selected)} offline cases -- no network, no API key.\n")
    failures = 0
    for name, note, function in selected:
        try:
            detail = function()
        except Failure as exc:
            failures += 1
            print(f"[FAIL] {name}\n       {note}\n       {exc}\n")
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {name}\n       {note}\n       {type(exc).__name__}: {exc}\n")
        else:
            print(f"[ok  ] {name}\n       {note}\n       {detail}\n")

    if args.live:
        print("-" * 70)
        failures += asyncio.run(run_live(only, args.provider, args.model))

    print(f"{len(selected)} offline cases, {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
