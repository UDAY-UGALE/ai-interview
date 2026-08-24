"""Repair known mis-transcriptions of technical terms, safely.

The audit found this failure to be the single largest accuracy defect, and
also the least visible one: when speech-to-text turns RAG into "Rack", every
stage downstream behaves perfectly and the answer is still completely wrong.
Measured, with a resume and a job description that BOTH name RAG:

    heard:  "What is Rack?"
    answer: "Rack is a minimal Ruby interface that standardises how web
             servers communicate with Ruby web frameworks..."
    then:   "Why would you use it?"  ->  Nginx versus Puma

"Rack" is a real framework, so the model had no reason to doubt it. That is
what separates this class of error from the ones the LLM already recovers on
its own ("fast API" -> FastAPI, "deployment pack" -> deployment part): those
are phonetic nonsense, and nonsense is self-announcing.

The obvious fix -- a find-and-replace table -- is worse than the disease. A
Ruby candidate really can be asked "what is Rack?", a doctor really can be
discussed, and unconditionally rewriting either one produces exactly the
same silent wrongness in the opposite direction.

So a substitution that could plausibly destroy a real question needs
EVIDENCE FROM THIS SESSION:

  1. the canonical term must appear in the session vocabulary (resume, job
     description, notes, or something already discussed in this interview),
  2. and the heard form must NOT itself appear there,
  3. and an everyday English word additionally needs technical surroundings.

Rule 2 is what protects the Ruby interview: a resume listing Ruby, Rails and
Rack puts "rack" in the vocabulary, so the guard fires and "What is Rack?"
is left exactly as spoken.

Rule 1 is skipped for heard forms that are not words at all -- "H3B",
"Kubernets", "Vlesk". Nobody meant to say those, so there is no genuine
question to protect and requiring evidence would only leave a known error
uncorrected in sessions with no resume loaded. See `requires_evidence` on
each entry.

Every applied and every REJECTED candidate is logged with its reason, so the
behaviour is auditable from the session log rather than having to be trusted.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Substitution:
    """One applied repair, in the shape the session log wants."""

    heard: str
    canonical: str
    reason: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    original: str
    normalized: str
    substitutions: list[Substitution] = field(default_factory=list)
    # Candidates that MATCHED a known mis-transcription but were deliberately
    # not applied, with why. Recorded because "the normalizer correctly did
    # nothing" and "the normalizer never noticed" look identical otherwise,
    # and only one of them is good news.
    rejected: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.substitutions)


# (pattern, canonical term, note, requires_evidence).
#
# Every entry comes from a mis-transcription actually observed in the
# production session logs or reproduced in the audit's degraded-audio runs --
# this is not a list of things that might happen.
#
# `requires_evidence` is the safety dial, and it turns on ONE question: could
# the heard form be what the speaker actually said?
#
#   True  -- the heard form is a real word or phrase. "Rack" is a Ruby
#            framework, "doctor" is a doctor, "sequel" is a word. Repairing
#            these without session evidence would break a genuine question,
#            so the canonical term must appear in the resume/JD/history.
#   False -- the heard form is not a word in any language ("H3B",
#            "Kubernets", "Vlesk", "D'Ango"). Nobody meant to say it, so
#            there is nothing to break and no evidence is needed. Requiring
#            it here only means a real mis-transcription goes unrepaired in
#            sessions with no resume loaded.
_MISHEARD_TERMS: tuple[tuple[re.Pattern, str, str, bool], ...] = (
    # RAG -- 21 occurrences of "Rack" in the real corpus
    (re.compile(r"\brack\b", re.IGNORECASE), "RAG", "observed 21x in session logs", True),
    (re.compile(r"\brag\b", re.IGNORECASE), "RAG", "lower-cased acronym", True),
    (re.compile(r"\bwrack\b", re.IGNORECASE), "RAG", "homophone", True),
    (re.compile(r"\brite\b", re.IGNORECASE), "RAG", "observed under fast speech", True),
    (re.compile(r"\bdrag\s+chart\b", re.IGNORECASE), "RAG chat", "observed in session logs", True),
    (re.compile(r"\br\.?\s?a\.?\s?g\.?\b", re.IGNORECASE), "RAG", "spelled out", False),
    # Redis
    (re.compile(r"\bred\s+is\b", re.IGNORECASE), "Redis", "word-split", True),
    (re.compile(r"\breedy'?s\b", re.IGNORECASE), "Redis", "observed in degraded audio", False),
    (re.compile(r"\breddis\b", re.IGNORECASE), "Redis", "spelling variant", False),
    # Docker
    (re.compile(r"\bdoctors?\b", re.IGNORECASE), "Docker", "observed in session logs", True),
    (re.compile(r"\bdockar\b", re.IGNORECASE), "Docker", "spelling variant", False),
    # FastAPI
    (re.compile(r"\bfast\s+repair\b", re.IGNORECASE), "FastAPI", "observed in session logs", True),
    (re.compile(r"\bfast\s+api\b", re.IGNORECASE), "FastAPI", "word-split", True),
    (re.compile(r"\bfastapi\b", re.IGNORECASE), "FastAPI", "casing", True),
    # Django
    (re.compile(r"\bjango\b", re.IGNORECASE), "Django", "phonetic", False),
    (re.compile(r"\bgango\b", re.IGNORECASE), "Django", "observed in degraded audio", False),
    (re.compile(r"\bd'?ango\b", re.IGNORECASE), "Django", "observed in clean audio", False),
    # Whisper's consistent rendering of "Django" on clean US-English audio --
    # 3 of 3 occurrences in the provider comparison. Deepgram got all three
    # right, which is one of the clearer results in that comparison.
    (re.compile(r"\bdiango\b", re.IGNORECASE), "Django", "observed 3/3 in Whisper comparison", False),
    (re.compile(r"\bjiango\b", re.IGNORECASE), "Django", "phonetic", False),
    # LangChain / LangGraph split into two words by Whisper.
    (re.compile(r"\blang\s+chain\b", re.IGNORECASE), "LangChain", "word-split", True),
    (re.compile(r"\blang\s+graph\b", re.IGNORECASE), "LangGraph", "word-split", True),
    (re.compile(r"\bjengo\b", re.IGNORECASE), "Django", "phonetic", False),
    # Flask -- absent from the old recognizer vocabulary, broke in 8/10 conditions
    (re.compile(r"\bvlesk\b", re.IGNORECASE), "Flask", "observed in degraded audio", False),
    (re.compile(r"\bblask\b", re.IGNORECASE), "Flask", "observed in degraded audio", False),
    # Kubernetes
    (re.compile(r"\bkubernets\b", re.IGNORECASE), "Kubernetes", "observed 9/10 conditions", False),
    (re.compile(r"\bkibernets\b", re.IGNORECASE), "Kubernetes", "observed in degraded audio", False),
    (re.compile(r"\bcubernetes\b", re.IGNORECASE), "Kubernetes", "phonetic", False),
    (re.compile(r"\bkavir\s+nets\b", re.IGNORECASE), "Kubernetes", "observed in degraded audio", False),
    # HTTP
    (re.compile(r"\bh3b\b", re.IGNORECASE), "HTTP", "observed in session logs", False),
    (re.compile(r"\bh\.?\s?t\.?\s?t\.?\s?p\.?\b", re.IGNORECASE), "HTTP", "spelled out", False),
    # GitHub
    (re.compile(r"\bget\s+hub\b", re.IGNORECASE), "GitHub", "word-split", True),
    (re.compile(r"\bgit\s+hub\b", re.IGNORECASE), "GitHub", "word-split", True),
    # SQL
    (re.compile(r"\bsequel\b", re.IGNORECASE), "SQL", "phonetic", True),
    (re.compile(r"\bs\.?\s?q\.?\s?l\.?\b", re.IGNORECASE), "SQL", "spelled out", False),
    # React
    (re.compile(r"\brecture\b", re.IGNORECASE), "React", "observed in session logs", False),
    (re.compile(r"\breacty\b", re.IGNORECASE), "React", "observed in session logs", False),
    # Frappe / ERPNext -- this product's usual domain
    (re.compile(r"\bfrappy\b", re.IGNORECASE), "Frappe", "phonetic", False),
    (re.compile(r"\berp\s+next\b", re.IGNORECASE), "ERPNext", "word-split", True),
    # Celery
    (re.compile(r"\bsalary\b", re.IGNORECASE), "Celery", "phonetic", True),
    # PostgreSQL
    (re.compile(r"\bpost\s?gress?\b", re.IGNORECASE), "PostgreSQL", "phonetic", True),
)


# Words whose ordinary English meaning is common enough that repairing them
# needs more than the canonical term being present -- the heard form has to
# be sitting next to something technical, or we would rewrite "I saw a
# doctor" in the middle of a perfectly normal sentence.
#
# The measured failure ("So you work with a doctor. Do you remember right
# about doctors?") satisfies this easily: "work with" is the context. A
# genuine sentence about medicine does not.
_AMBIGUOUS_HEARD_FORMS = frozenset({"rack", "rag", "doctor", "doctors", "rite", "sequel", "salary", "drag chart"})

_TECHNICAL_CONTEXT = re.compile(
    r"\b(?:use[ds]?|using|work(?:ed|ing)?|built?|building|implement\w*|deploy\w*|"
    r"pipeline|architecture|chatbot|system|stack|framework|project|experience|"
    r"application|app|service|api|server|database|container|model|explain|"
    r"what\s+is|what'?s|tell\s+me|about|know|familiar)\b",
    re.IGNORECASE,
)


def _has_technical_context(text: str) -> bool:
    return bool(_TECHNICAL_CONTEXT.search(text))


def normalize_transcript(
    text: str,
    *,
    session_terms: set[str],
    log_only: bool = False,
) -> NormalizationResult:
    """Repair mis-transcribed technical terms that THIS session evidences.

    `session_terms` is the lower-cased vocabulary from
    `session_vocabulary.build_session_vocabulary()`. An empty set disables
    every substitution, which is the correct behaviour for a session with no
    resume and no history: with no evidence, there is nothing to be confident
    about and guessing is what this module exists to avoid.
    """
    if not text or not session_terms:
        return NormalizationResult(original=text, normalized=text)

    substitutions: list[Substitution] = []
    rejected: list[tuple[str, str, str]] = []
    result = text

    for pattern, canonical, note, requires_evidence in _MISHEARD_TERMS:
        match = pattern.search(result)
        if not match:
            continue

        heard = match.group(0)
        heard_key = heard.lower().strip()
        canonical_key = canonical.lower()

        # Already correct -- nothing to repair. Checked before the evidence
        # rules so an exact hit never shows up as a rejection.
        if heard_key == canonical_key:
            continue

        # RULE 1: this session must actually be about the canonical term.
        # Skipped only when the heard form is not a word anyone could have
        # meant ("H3B", "Kubernets") -- there is no genuine question to
        # protect, so demanding evidence would just leave a known error in
        # place for any session with no resume loaded.
        if requires_evidence and canonical_key not in session_terms:
            rejected.append((heard, canonical, "canonical term not in session vocabulary"))
            continue

        # RULE 2: the heard form must not be a real term for this session.
        # This is what keeps a Ruby interview's "Rack" intact, and it applies
        # even to the no-evidence entries.
        if heard_key in session_terms:
            rejected.append((heard, canonical, "heard form is itself a session term"))
            continue

        # RULE 3: an everyday English word needs technical surroundings.
        if heard_key in _AMBIGUOUS_HEARD_FORMS and not _has_technical_context(result):
            rejected.append((heard, canonical, "ambiguous word with no technical context"))
            continue

        substitutions.append(
            Substitution(
                heard=heard,
                canonical=canonical,
                reason=note,
                start=match.start(),
                end=match.end(),
            )
        )
        if not log_only:
            result = result[: match.start()] + canonical + result[match.end() :]

    if substitutions:
        logger.info(
            "Transcript normalization%s: %r -> %r  [%s]",
            " (log-only)" if log_only else "",
            text,
            result,
            "; ".join(f"{s.heard!r}->{s.canonical!r} ({s.reason})" for s in substitutions),
        )
    if rejected:
        logger.debug(
            "Transcript normalization declined %d candidate(s) in %r: %s",
            len(rejected),
            text,
            "; ".join(f"{h!r}->{c!r} ({why})" for h, c, why in rejected),
        )

    return NormalizationResult(
        original=text,
        normalized=text if log_only else result,
        substitutions=substitutions,
        rejected=rejected,
    )
