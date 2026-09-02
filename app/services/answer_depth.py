"""How deep should this answer go?

The system prompt pushes hard for specificity -- name the component, state the
trade-off, never say "the retrieval configuration needs tuning" when you could
say "the vector index is flat". That is exactly right for a question that
asked how something works, and exactly wrong for one that asked what you would
do. Applied to "how would you use GenAI in fraud detection?", the same rules
produce fine-tuned embeddings and agentic workflows in the opening sentence --
technically defensible, and the wrong answer, because the interviewer asked a
broad question and got a candidate who cannot tell a broad question from a
deep one.

So this module answers one question before the LLM is called: **how much depth
did the interviewer actually ask for?** The answer becomes a short directive in
the user message, which is the only per-question slot in the prompt.

Four levels, and the ordering rule is that the HIGHEST matching level wins:

  1 overview        "how would you use X for Y"        -> what you'd do, why
  2 implementation  "how would you implement/design X" -> components, data flow
  3 mechanism       "how would you detect novel X"     -> the technique itself
  4 architecture    "design a SCALABLE X architecture" -> the full picture

Two design points that are not obvious:

**Level 4 needs two signals, not one.** "Design a URL shortener" and "design a
scalable notification architecture" both ask you to design something, but only
the second is asking for Kafka and sharding. A design verb alone lands at
level 2 (core components and flow); it takes an explicit scale or architecture
word on top of it to unlock level 4. Getting this wrong in the permissive
direction is the whole bug this module exists to fix.

**Depth ratchets forward through a thread, it does not reset.** An interviewer
who asks "how would you implement that?" after a broad answer has escalated,
and the next question after that should not fall back to level 1 just because
it was phrased simply. So a follow-up inherits the previous question's level,
and an explicit "go deeper" cue pushes it one further.

Pure functions, no I/O, no settings -- so the gate can call this on the hot
path between "they stopped talking" and "answer starts" without adding a
network hop. Measured at well under a millisecond.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


OVERVIEW = 1
IMPLEMENTATION = 2
MECHANISM = 3
ARCHITECTURE = 4

_LABELS = {
    OVERVIEW: "overview",
    IMPLEMENTATION: "implementation",
    MECHANISM: "mechanism",
    ARCHITECTURE: "architecture",
}


@dataclass(frozen=True, slots=True)
class AnswerDepth:
    level: int
    reason: str

    @property
    def label(self) -> str:
        return _LABELS[self.level]


# --------------------------------------------------------------------------
# Signals. Each is deliberately about the SHAPE of the request rather than its
# subject matter -- nothing here knows what fraud detection or a URL shortener
# is, which is what stops this from becoming a lookup table of questions.
# --------------------------------------------------------------------------

# Asking you to design or architect something.
_DESIGN = re.compile(
    r"\b(design|architect|architecture|blueprint|lay\s*out|structure)\b|"
    r"\bhow\s+would\s+you\s+(build|create|set\s*up)\b", re.I)

# Words that turn a design question into an ARCHITECTURE question. Without one
# of these, "design X" is asking for the core shape, not the distributed one.
_SCALE = re.compile(
    r"\b(scalab\w+|scale|scaling|high[\s-]?throughput|high[\s-]?volume|"
    r"high[\s-]?availab\w+|highly\s+available|fault[\s-]?toleran\w+|"
    r"resilien\w+|distributed|multi[\s-]?region|production[\s-]?grade|"
    r"enterprise|millions?|billions?|petabytes?|"
    r"end[\s-]?to[\s-]?end\s+architecture|system\s+design)\b", re.I)

# Asking how it would actually be put together.
_IMPLEMENT = re.compile(
    r"\b(implement\w*|integrat\w+|wire\s*(it|them|this)?\s*up|"
    r"put\s+(it|this|that)\s+together|code\s+(it|this|that)|"
    r"walk\s+me\s+through\s+(the\s+)?(implementation|build|code|steps)|"
    r"what\s+(would\s+)?the\s+(flow|pipeline|components?)\s+look\s+like)\b", re.I)

# Asking about a specific mechanism, edge case or failure -- the level where
# embeddings, clustering, consistency models and tuning genuinely belong.
_MECHANISM = re.compile(
    r"\b(how\s+does\s+.{0,40}\s*work|under\s+the\s+hood|internal\w*|"
    r"algorithm\w*|previously\s+unseen|unseen|novel|zero[\s-]?day|"
    r"never\s+seen\s+before|edge\s+cases?|race\s+condition|deadlock|"
    r"exactly[\s-]?once|idempoten\w+|consistency\s+model|"
    r"optimi[sz]\w+|fine[\s-]?tun\w+|tune|tuning|"
    r"deep\s+dive|in\s+depth|technically\s+how|"
    r"what\s+happens\s+(if|when)|why\s+(does|is|would)\s+.{0,40}(slow|fail|break))\b",
    re.I)

# An interviewer explicitly asking for more than they just got.
_ESCALATE = re.compile(
    r"\b(go\s+deeper|deeper|more\s+detail|in\s+more\s+detail|more\s+specific\w*|"
    r"how\s+exactly|exactly\s+how|specifically|elaborate|expand\s+on\s+that|"
    r"under\s+the\s+hood|walk\s+me\s+through|technically|"
    r"what\s+about\s+(at\s+)?scale)\b", re.I)

# The resolver rewrites a bare follow-up into a paragraph that quotes the
# original. Classify the QUOTED text -- the surrounding paragraph repeats the
# previous question and answer, which would inflate the level.
_RESOLVED_FOLLOWUP = re.compile(
    r'short follow-?up:\s*"([^"]{1,200})"', re.I)

_FOLLOWUP_WORD_LIMIT = 9


def _target_text(question: str) -> str:
    match = _RESOLVED_FOLLOWUP.search(question)
    return match.group(1) if match else question


def _classify_text(text: str) -> tuple[int, str]:
    """Highest matching level wins."""
    design = bool(_DESIGN.search(text))
    scale = bool(_SCALE.search(text))

    if design and scale:
        return ARCHITECTURE, "design request with an explicit scale/architecture ask"
    if _MECHANISM.search(text):
        return MECHANISM, "asks about a specific mechanism, edge case or failure"
    if _IMPLEMENT.search(text):
        return IMPLEMENTATION, "asks how it would be built"
    if design:
        return IMPLEMENTATION, "design request with no scale ask -- core shape, not infrastructure"
    if scale:
        return MECHANISM, "raises scale directly"
    return OVERVIEW, "broad question -- no implementation, mechanism or scale ask"


def classify_answer_depth(question: str, history=None) -> AnswerDepth:
    """How much depth this question is asking for.

    `history` is the session's recent turns (anything with a `.question`
    attribute); it is used only to stop depth collapsing back to level 1 in
    the middle of a thread that has already gone deep.
    """
    text = _target_text(question or "")
    level, reason = _classify_text(text)

    previous = None
    if history:
        try:
            previous = getattr(history[-1], "question", None)
        except (IndexError, TypeError):
            previous = None

    if previous:
        is_followup = (
            len(text.split()) <= _FOLLOWUP_WORD_LIMIT
            or bool(_RESOLVED_FOLLOWUP.search(question or ""))
            or bool(_ESCALATE.search(text))
        )
        if is_followup:
            prior_level, _ = _classify_text(previous)
            if _ESCALATE.search(text):
                # They asked for more than they just got.
                target = max(level, min(prior_level + 1, ARCHITECTURE))
                if target > level:
                    level, reason = target, "follow-up explicitly asking to go deeper"
            elif prior_level > level:
                # A thread that already went deep does not reset because the
                # next question happened to be short.
                level = prior_level
                reason = "follow-up inside a thread already at this depth"

    return AnswerDepth(level=level, reason=reason)


# --------------------------------------------------------------------------
# The directive. Kept short on purpose: every line added here is a line the
# model weighs against the VOICE rules, and a benchmark of longer depth
# instructions cost more in spoken-style than it bought in depth.
# --------------------------------------------------------------------------

_DIRECTIVES = {
    OVERVIEW: (
        "This is a broad question. Answer it the way a strong candidate talks: judgment "
        "first, technology last. What should land is \"I understand the problem, I know "
        "where this helps, and I know where I would not use it\" -- never \"I know a lot "
        "of technologies\".\n"
        "Use this shape EXACTLY, 3-4 sections, with one or two \"- \" sub-bullets under "
        "each -- one spoken sentence per sub-bullet, never a paragraph:\n"
        "**1. <the approach, as a short claim>**\n"
        "- what you would actually do\n"
        "**2. <why that works, or where it adds value>**\n"
        "- the reasoning, not a restatement of point 1\n"
        "**3. <one concrete example>**\n"
        "- specific enough that the interviewer can picture it\n"
        "**4. <the boundary>**\n"
        "- what this should NOT own, and what stays in charge of the decision\n"
        "**In short:** one sentence tying it together.\n"
        "Then, within that shape: keep the whole answer under 150 words, because it is "
        "spoken aloud; every header is a step in your reasoning, never a technology "
        "name; each sub-bullet earns its place with a reason or a trade-off, NOT by "
        "naming a tool; Name the STAGE rather than the product (\"run it on a cloud "
        "platform\", not the platform's name) and name at most ONE product in the whole "
        "answer; and where an established approach already covers most of the problem, "
        "say the newer thing works alongside it rather than replacing it."
    ),
    IMPLEMENTATION: (
        "They are asking how it is built. Give the pieces and the path data takes "
        "through them, in order, and say what each piece is responsible for. Stay at the "
        "level of services and flow; drop into a technique only where a choice needs "
        "justifying.\n"
        "Use this shape, 3-5 sections, each header a real step rather than a component "
        "name, and one or two \"- \" sub-bullets under each -- one spoken sentence per "
        "sub-bullet, never a paragraph:\n"
        "**1. <the approach>**\n"
        "- what you would build\n"
        "**2. <how the pieces fit / what flows where>**\n"
        "- the path the data takes\n"
        "**3. <the choice that needed making, and why>**\n"
        "- the reason, or the trade-off you accepted\n"
        "**4. <the boundary or what you would watch>**\n"
        "- what this does not own, or where it would strain\n"
        "**In short:** one sentence, when it adds something.\n"
        "Keep the whole answer under about 200 words -- it is still spoken aloud.\n"
        "Naming the real components is right at this level -- but a component with no "
        "stated job is noise, and a header is still a step in the reasoning, not a "
        "product name."
    ),
    MECHANISM: (
        "They are asking about the mechanism, so this is where the specific technique "
        "belongs. Name it, say why it fits this problem, and say what it costs or "
        "where it stops working. Depth is what is being asked for here -- but only on "
        "the thing they asked about. Plain bullets starting with \"- \", no section "
        "headers and no summary line; the technique carries the answer here."
    ),
    ARCHITECTURE: (
        "They asked for an architecture at scale, so the full picture is fair game: "
        "components, data flow, what stores what, how the parts talk, and what "
        "happens under load or when a part fails. Every component you name has to "
        "earn its place with the problem it solves -- one with no stated job is noise. "
        "Plain bullets starting with \"- \", no section headers and no summary line. "
        "Keep the whole answer under about 280 words -- even an architecture answer is "
        "spoken, not written."
    ),
}

_HEADER = "ANSWER DEPTH"


def depth_directive(depth: AnswerDepth) -> str:
    """The block appended to the user message, immediately before the question."""
    return (
        f"{_HEADER} -- level {depth.level} of 4 ({depth.label})\n"
        f"{_DIRECTIVES[depth.level]}"
    )


def build_depth_block(question: str, history=None) -> str:
    """Convenience: classify and render in one call."""
    return depth_directive(classify_answer_depth(question, history))
