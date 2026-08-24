"""What the candidate actually did, in a form the model can answer from.

A resume says "worked on a real-time AI application". An interviewer asks
"why Deepgram over Whisper?", "what broke in production?", "what did *you*
build versus what did the team build?" -- and the resume cannot answer any
of them. Left to itself the model fills that gap by inventing a plausible
engineer, which is the single worst failure this product has: an invented
detail is a detail the interviewer will ask a follow-up about.

So this module holds two things:

* `ProjectContext` -- one project the candidate described, field by field
  (role, architecture, decisions, trade-offs, metrics...). Free text per
  field, because a candidate's real answer to "what was the trade-off" is a
  sentence, not an enum.
* `build_candidate_context()` -- the assembly layer that turns the session's
  stored context into ONE labelled block for the prompt, inside a character
  budget.

Two design points that are not obvious:

**The budget is the point, not an afterthought.** Everything this builds is
re-sent on every question of the interview, and providers meter tokens per
minute. The existing resume/JD caps were sized against a measured 8,000 TPM
limit where an uncapped resume alone pushed a single question past the
budget -- which surfaces as answers stalling for 20+ seconds, not as an
error. Project context is bigger than a resume, so it gets allocated, not
appended: sections are filled in priority order with floors reserved for
the resume and JD so a large project blob can never starve them.

**Project selection is a seam.** `select_projects()` scores projects
against the question and orders them by relevance before the budget is
applied, so today (few projects, all fitting) it changes nothing but the
order, and tomorrow -- more projects than fit -- the same call site drops
the irrelevant ones. Swapping the lexical score for embeddings or a small
model touches this one function and nothing in the interview pipeline.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields


# ---------------------------------------------------------------------------
# Budgets. Characters, not tokens, because that is what we can cheaply
# measure -- roughly 4 characters per token for English prose.
# ---------------------------------------------------------------------------

# Everything below is re-sent on EVERY question, so this total is what
# decides how many questions a minute the provider will serve before it
# starts throttling. Overridable per deployment via
# CANDIDATE_CONTEXT_CHAR_BUDGET (see app/core/config.py) -- raise it if you
# are on a plan with real token headroom, because more grounded context is
# strictly better for answer quality; the cap exists for the rate limit,
# not for the model.
DEFAULT_CHAR_BUDGET = 5200

# Per-section ceilings. These are fairness limits inside the total above --
# no single section may take more than this even if the budget allows it.
RESUME_CHAR_LIMIT = 2000
JD_CHAR_LIMIT = 1500
NOTES_CHAR_LIMIT = 500
PROJECTS_CHAR_LIMIT = 3200
PER_PROJECT_CHAR_LIMIT = 1400
EXPERIENCE_NOTES_CHAR_LIMIT = 1600
INTERVIEW_STORIES_CHAR_LIMIT = 1200

# Reserved before anything is distributed, so a long project blob cannot
# push the resume or the JD out entirely. Only reserved for sections that
# actually have content.
_RESUME_FLOOR = 700
_JD_FLOOR = 350

# More projects than this and the ones past it are almost certainly not what
# the current question is about; the budget usually bites first anyway.
MAX_PROJECTS_IN_CONTEXT = 4


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """One project, as the candidate described it.

    Every field is optional free text. A candidate who pastes a paragraph
    into `description` and nothing else is a valid, useful project; forcing
    twelve fields to be filled would mean most people fill none.

    `source` records where it came from ("manual", or "upload:notes.pdf")
    -- kept because it is the difference between "the candidate typed this"
    and "this fell out of a PDF and may be garbled", which matters the day
    an answer quotes something odd.
    """

    name: str = ""
    description: str = ""
    role: str = ""
    responsibilities: str = ""
    architecture: str = ""
    technologies: str = ""
    challenges: str = ""
    solutions: str = ""
    decisions: str = ""
    tradeoffs: str = ""
    metrics: str = ""
    additional_notes: str = ""
    source: str = "manual"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object) -> "ProjectContext":
        """Tolerant construction from stored JSON or an API payload.

        Unknown keys are dropped and non-strings are coerced rather than
        raising: this runs when loading a session out of Redis, and a
        session that fails to load is an interview that cannot start.
        """
        if isinstance(data, ProjectContext):
            return data
        if not isinstance(data, dict):
            return cls()
        known = {field.name for field in fields(cls)}
        payload: dict[str, str] = {}
        for key, value in data.items():
            if key not in known or value is None:
                continue
            payload[key] = value if isinstance(value, str) else str(value)
        return cls(**payload)

    def is_empty(self) -> bool:
        """A project with a name and nothing else carries no information the
        model can answer from, so it is treated as empty."""
        return not any(
            getattr(self, field.name).strip()
            for field in fields(self)
            if field.name not in ("name", "source")
        )

    def label(self) -> str:
        return self.name.strip() or "Unnamed project"

    def searchable_text(self) -> str:
        """Everything except `source`, for relevance scoring."""
        return " ".join(
            getattr(self, field.name)
            for field in fields(self)
            if field.name != "source"
        )

    def to_prompt_block(self, *, index: int, char_budget: int = PER_PROJECT_CHAR_LIMIT) -> str:
        """Render as labelled lines.

        The labels are the whole point -- "My role:" is what lets the model
        tell what the candidate personally built from what their team built,
        which is the distinction it otherwise guesses at.

        Fields are emitted in the order below and the budget is spent in
        that order too, so if a project is trimmed it loses trailing notes
        rather than its architecture.
        """
        ordered = (
            ("What it is", self.description),
            ("My role", self.role),
            ("My responsibilities", self.responsibilities),
            ("Architecture", self.architecture),
            ("Technologies", self.technologies),
            ("Challenges", self.challenges),
            ("How I solved them", self.solutions),
            ("Technical decisions", self.decisions),
            ("Trade-offs", self.tradeoffs),
            ("Results and metrics", self.metrics),
            ("Notes", self.additional_notes),
        )

        header = f"Project {index}: {self.label()}"
        lines = [header]
        remaining = max(0, char_budget - len(header))
        for label, value in ordered:
            value = _collapse(value)
            if not value:
                continue
            if remaining <= len(label) + 8:
                break
            body = _trim(value, remaining - len(label) - 2)
            if not body:
                break
            line = f"{label}: {body}"
            lines.append(line)
            remaining -= len(line) + 1
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")

# Words that appear in most interview questions and therefore separate no
# project from any other. Deliberately short: an aggressive stop list starts
# removing real signal ("system", "data", "model" are all discriminating in
# a stack of projects).
_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for from with without about into over
    is are was were be been being do does did done can could would should will
    you your yours i me my mine we our ours they them their it its this that
    these those how what why when where which who whom tell explain describe
    walk give talk say said just also then than so as by not no yes any some
    """.split()
)


def _tokens(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]


def select_projects(
    projects,
    question: str = "",
    *,
    char_budget: int = PROJECTS_CHAR_LIMIT,
    max_projects: int = MAX_PROJECTS_IN_CONTEXT,
) -> list[ProjectContext]:
    """The projects worth sending for THIS question, most relevant first.

    Today this is a lexical overlap score, and with a handful of small
    projects everything fits and the only visible effect is ordering. That
    is intentional: the interview pipeline calls this one function, so
    replacing the score with embeddings or a retrieval step later is a
    change here and nowhere else.

    Ordering matters even when nothing is dropped -- if the budget bites
    mid-list it is the least relevant project that loses its tail, and a
    model reading top-down sees the project the question is about first.
    """
    usable = [p for p in projects or () if isinstance(p, ProjectContext) and not p.is_empty()]
    if not usable:
        return []

    question_terms = set(_tokens(question))
    if question_terms:
        scored = []
        for order, project in enumerate(usable):
            strong = set(_tokens(f"{project.name} {project.technologies}"))
            body = set(_tokens(project.searchable_text()))
            score = 3 * len(question_terms & strong) + len(question_terms & body)
            scored.append((-score, order, project))
        scored.sort()
        ranked = [project for _score, _order, project in scored]
    else:
        ranked = usable

    selected: list[ProjectContext] = []
    remaining = char_budget
    for project in ranked[:max_projects]:
        # Cheap size estimate before rendering: the block is never longer
        # than the text it is built from plus its labels.
        cost = min(PER_PROJECT_CHAR_LIMIT, len(project.searchable_text()) + 160)
        if selected and cost > remaining:
            break
        selected.append(project)
        remaining -= cost
    return selected


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Section:
    key: str
    heading: str
    text: str
    cap: int
    floor: int = 0


def build_candidate_context(
    context,
    *,
    question: str = "",
    char_budget: int | None = None,
) -> str:
    """The candidate half of the prompt: profile, then project context.

    `context` is anything with the InterviewSessionContext attributes --
    typed structurally rather than imported, so this module stays free of
    the session store and the store can import ProjectContext from here.

    Returns "" when there is nothing to say, so the caller can leave the
    message out entirely rather than sending an empty heading.
    """
    budget = max(1500, char_budget or DEFAULT_CHAR_BUDGET)

    projects = select_projects(
        getattr(context, "projects", ()) or (),
        question,
        char_budget=PROJECTS_CHAR_LIMIT,
    )
    project_text = "\n\n".join(
        project.to_prompt_block(index=index, char_budget=PER_PROJECT_CHAR_LIMIT)
        for index, project in enumerate(projects, start=1)
    )

    # Priority order, and it is the order in the spec for a reason: what the
    # candidate wrote about their own work outranks a resume bullet, which
    # outranks the JD, which outranks loose notes. When the budget bites,
    # it bites from the bottom.
    sections = [
        _Section("projects", "", project_text, PROJECTS_CHAR_LIMIT),
        _Section(
            "resume",
            "Resume:",
            _collapse_blocks(getattr(context, "resume_text", "")),
            RESUME_CHAR_LIMIT,
            _RESUME_FLOOR,
        ),
        _Section(
            "experience_notes",
            "Experience notes (candidate-provided):",
            _collapse_blocks(getattr(context, "experience_notes", "")),
            EXPERIENCE_NOTES_CHAR_LIMIT,
        ),
        _Section(
            "job_description",
            "Job description for the role being interviewed for:",
            _collapse_blocks(getattr(context, "job_description", "")),
            JD_CHAR_LIMIT,
            _JD_FLOOR,
        ),
        _Section(
            "interview_stories",
            "Interview stories (candidate-provided, for behavioural questions):",
            _collapse_blocks(getattr(context, "interview_stories", "")),
            INTERVIEW_STORIES_CHAR_LIMIT,
        ),
        _Section(
            "notes",
            "Additional notes:",
            _collapse_blocks(getattr(context, "notes", "")),
            NOTES_CHAR_LIMIT,
        ),
    ]

    # Headings are part of what gets sent, so they come out of the budget
    # rather than riding on top of it -- otherwise the number in the config
    # is not the number on the wire, which is the kind of drift that makes a
    # token budget useless.
    allocation = _allocate(sections, budget - _overhead(sections))
    rendered = {
        section.key: _trim(section.text, allocation[section.key])
        for section in sections
        if allocation.get(section.key)
    }
    if not rendered:
        return ""

    profile_parts = [
        f"{section.heading}\n{rendered[section.key]}"
        for section in sections
        if section.key in rendered and section.key != "projects"
    ]

    blocks: list[str] = []
    if profile_parts:
        blocks.append(_PROFILE_HEADING + "\n\n".join(profile_parts))
    if "projects" in rendered:
        blocks.append(_PROJECT_HEADING + rendered["projects"])
    blocks.append(_CONTEXT_BOUNDARY)
    return "\n\n".join(blocks)


# The closing line of the block, and it earns its ~70 tokens.
#
# The system prompt already forbids inventing experience, and a fast model
# asked "how did you deploy and scale this?" invents a deployment anyway --
# measured on openai/gpt-oss-120b: Docker, AWS ECS, an Application Load
# Balancer, Fargate, CloudWatch and Secrets Manager, none of them anywhere
# in the context. A rule 1,700 tokens earlier loses to the pull of an
# unanswerable question. Stating the boundary HERE, at the end of the facts
# themselves, is the difference between "these are some facts" and "these
# are ALL the facts" -- and it is a statement about the context rather than
# another behavioural rule, which is why it lives in this block instead of
# the prompt.
_CONTEXT_BOUNDARY = (
    "END OF CANDIDATE CONTEXT. Nothing outside this block is part of this "
    "candidate's experience. If the question asks about something not covered "
    "above -- how it was deployed, scaled, monitored or tested, or any tool not "
    "named above -- never say you used, ran, built with or chose a tool, service "
    "or platform that is not written above. Say plainly that it was not your "
    "part of the work, then answer in the conditional -- what you would do and "
    "why -- so it is unmistakably an approach rather than a claim."
)

_PROFILE_HEADING = "CANDIDATE PROFILE\n\n"
_PROJECT_HEADING = (
    "PROJECT CONTEXT (written by the candidate about their own work -- "
    "this is the authority on what they personally did)\n\n"
)


def _overhead(sections: list[_Section]) -> int:
    """What the block costs before a single character of content: the two
    top-level headings and the per-section labels, plus their separators."""
    present = [section for section in sections if section.text]
    if not present:
        return 0
    total = sum(len(section.heading) + 3 for section in present) + len(_CONTEXT_BOUNDARY) + 2
    if any(section.key != "projects" for section in present):
        total += len(_PROFILE_HEADING)
    if any(section.key == "projects" for section in present):
        total += len(_PROJECT_HEADING)
    return total


def _allocate(sections: list[_Section], budget: int) -> dict[str, int]:
    """Characters per section: floors first, then the surplus by priority.

    Floors are what stop a 3,000-character project blob from evicting the
    resume; priority is what makes the trimming predictable -- notes and
    stories lose their tails before anything the answer is grounded in
    does.
    """
    present = [section for section in sections if section.text]
    budget = max(0, budget)
    reserved = sum(min(section.floor, section.cap, len(section.text)) for section in present)
    surplus = max(0, budget - reserved)

    allocation: dict[str, int] = {}
    for section in present:
        floor = min(section.floor, section.cap, len(section.text))
        want = min(section.cap, len(section.text))
        extra = min(want - floor, surplus)
        surplus -= extra
        allocation[section.key] = floor + extra
    return allocation


def _collapse(text: str) -> str:
    """One field, on one line. Newlines inside a labelled field would break
    the label/value shape the model reads the block by."""
    return re.sub(r"\s+", " ", text or "").strip()


def _collapse_blocks(text: str) -> str:
    """Paragraph structure kept, incidental whitespace removed."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _trim(text: str, limit: int) -> str:
    """Truncate on a word boundary and say so.

    The marker earns its four characters: a resume cut mid-sentence
    otherwise reads to the model as a resume that ends there, and it will
    answer as though the rest of the career does not exist.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - 4)]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    if boundary > limit * 0.6:
        cut = cut[:boundary]
    return cut.rstrip() + " ..."


def vocabulary_text(context) -> str:
    """Candidate-written project text, for the session vocabulary.

    Project notes are where the candidate's actual stack is named -- often
    in more detail than the resume -- and every term the recognizer is told
    to expect is a term it stops mishearing. Same text, different consumer:
    this one is about *hearing* the question, not answering it.
    """
    parts = [
        project.searchable_text()
        for project in getattr(context, "projects", ()) or ()
        if isinstance(project, ProjectContext)
    ]
    for attribute in ("experience_notes", "interview_stories"):
        value = getattr(context, attribute, "") or ""
        if value:
            parts.append(value)
    return "\n".join(parts)
