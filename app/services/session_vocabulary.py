"""The terms THIS interview is actually going to use.

Speech-to-text gets dramatically better at a word when it is told the word
might be coming. The previous strategy for that was to paste the resume and
the job description into the recognizer's prompt as prose, and it did not
survive contact with a real session: the prompt has a hard ~850 UTF-8 byte
cap, the fixed technical vocabulary already spends 433 of them, and the
resume is appended LAST. Measured on a normal session -- a 500-character JD
and a 300-character resume -- the total was 1,114 bytes and the truncation
removed the resume entirely. The candidate's own project names, frameworks
and tools, which are exactly the words most likely to be spoken and most
likely to be misheard, were getting no biasing at all.

A term LIST fits the same budget several times over, so this module turns
resume + JD + conversation history + the generic technical list into a
compact, deduplicated, priority-ordered vocabulary. It has two consumers:

* the recognizer -- Deepgram keyterms, or a much denser Whisper prompt
* `transcript_normalizer`, which will only repair a misheard technical term
  into a canonical one when THIS session provides evidence for it

Ordering is the whole point. Session-specific terms come first, so when a
budget bites it is the generic list that loses entries, never the resume.
"""

from __future__ import annotations

import re


# Terms worth biasing towards in any technical interview, whatever the
# candidate's own stack is. This is the same vocabulary the Whisper prompt
# carried, broadened with the terms the audit found missing -- Flask was the
# clearest case: absent from the list, and mis-transcribed in 8 of 10
# degraded-audio conditions ("Vlesk", "Blask", "Blast", "Black") while RAG
# and Docker, both present, survived all 10.
BASELINE_TERMS: tuple[str, ...] = (
    # languages / runtimes
    "Python", "JavaScript", "TypeScript", "Node.js", "Java", "Golang", "Rust",
    "C++", "C#", "Kotlin", "Swift", "SQL", "Bash", "Linux",
    # web frameworks
    "FastAPI", "Django", "Django REST Framework", "Flask", "Express",
    "React", "Next.js", "Angular", "Vue", "Redux", "Tailwind", "Spring Boot",
    # AI / ML
    "RAG", "Retrieval-Augmented Generation", "LLM", "LangChain", "LangGraph",
    "GPT-4", "GPT-5", "Claude", "Groq", "embeddings", "vector database",
    "fine-tuning", "LoRA", "prompt engineering", "hallucination",
    "Hugging Face", "transformer", "BERT", "agentic AI", "tool calling",
    # data stores
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Elasticsearch",
    "ChromaDB", "Pinecone", "Qdrant", "Weaviate", "Milvus", "FAISS",
    # infra
    "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform", "Jenkins",
    "GitHub", "GitHub Actions", "CI/CD", "Nginx", "Celery", "Kafka",
    "RabbitMQ", "microservices", "serverless",
    # protocols / API
    "HTTP", "HTTPS", "REST API", "GraphQL", "WebSocket", "gRPC", "JSON",
    "OAuth", "JWT", "authentication", "authorization",
    # engineering practice
    "unit testing", "pytest", "TDD", "Agile", "Scrum", "code review",
    "indexing", "caching", "sharding", "load balancing", "scalability",
    # domain-specific to this product's usual candidates
    "ERPNext", "Frappe",
)


# A token worth treating as a technical term when it appears in a resume or
# job description. Deliberately shape-based rather than dictionary-based, so
# a stack this code has never heard of still gets picked up.
_TERM_PATTERNS: tuple[re.Pattern, ...] = (
    # Dotted / versioned product names: Node.js, Next.js, .NET, Vue.js
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:\.[a-z]{2,4})+\b"),
    # CamelCase and internal capitals: FastAPI, PostgreSQL, LangGraph, ChromaDB
    re.compile(r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+\b"),
    # Acronyms, with or without digits: RAG, LLM, AWS, S3, GPT-4, CI/CD
    re.compile(r"\b[A-Z]{2,}(?:[-/][A-Z0-9]{1,})*\b"),
    # Hyphenated technical compounds: fine-tuning, retrieval-augmented
    re.compile(r"\b[a-z]{3,}-[a-z]{3,}(?:-[a-z]{3,})?\b"),
    # Mixed-case product names where the capitals come first: ERPNext,
    # GitHub, PostgreSQL, IPython. Pattern 2 cannot match these because it
    # requires lower-case letters before the internal capital.
    re.compile(r"\b[A-Z]{2,}[a-z][A-Za-z0-9]*\b"),
    # Capitalised single words that are not sentence-initial noise.
    #
    # The exclusion is "period followed by a SPACE", not "period followed by
    # any whitespace": a resume is mostly short lines, so a term on its own
    # line sits right after ".\n" and the whitespace form silently dropped
    # it. Measured: a resume ending in a "Kubernetes" line produced a
    # vocabulary with no Kubernetes in it, which in turn meant the
    # normalizer refused to repair "Kubernets" for that session.
    re.compile(r"(?<![.!?] )(?<!^)\b[A-Z][a-z]{2,}\b"),
)


# Shapes that match the patterns above but are never a technical term. Kept
# small on purpose: a false POSITIVE here costs almost nothing (a useless
# keyterm), while a false NEGATIVE costs the session its own vocabulary.
_STOP_TERMS = frozenset(
    {
        "the", "and", "for", "with", "from", "this", "that", "have", "has",
        "was", "were", "our", "your", "their", "using", "used", "use",
        "built", "build", "developed", "designed", "created", "worked",
        "experience", "years", "year", "team", "project", "projects",
        "company", "role", "work", "working", "responsible", "including",
        "must", "nice", "hiring", "engineer", "developer", "senior", "junior",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "summary", "skills", "education", "objective", "profile", "contact",
        "present", "current", "currently", "also", "well", "good", "strong",
    }
)

# Never let a one- or two-character fragment become a keyterm; it matches
# far too much ordinary speech to be a useful bias.
_MIN_TERM_LENGTH = 2


def _clean(term: str) -> str:
    return term.strip(" \t\r\n.,;:!?()[]{}\"'").strip()


def _is_plausible_term(term: str) -> bool:
    if len(term) < _MIN_TERM_LENGTH:
        return False
    if term.lower() in _STOP_TERMS:
        return False
    # Pure numbers, dates, and version strings on their own are not terms.
    if not re.search(r"[A-Za-z]", term):
        return False
    return True


def extract_terms(text: str, *, limit: int = 60) -> list[str]:
    """Pull the technical/domain terms out of one document.

    Order of appearance is preserved, because a resume leads with what the
    candidate most wants to be asked about -- which is also what they are
    most likely to be asked about.
    """
    if not text:
        return []

    seen: dict[str, str] = {}
    for pattern in _TERM_PATTERNS:
        for match in pattern.finditer(text):
            term = _clean(match.group(0))
            if not _is_plausible_term(term):
                continue
            key = term.lower()
            if key not in seen:
                seen[key] = term
            if len(seen) >= limit * 3:
                break

    # Catch-all: any term from the known technical vocabulary that appears
    # VERBATIM in this document counts as a session term, whatever shape it
    # has. Shape-based extraction is good at finding words it has never seen
    # and occasionally misses ones it has -- "Node.js", "CI/CD" and "C++"
    # all defeat at least one of the patterns above. This costs one scan and
    # removes a whole class of "the resume says it but the vocabulary does
    # not have it" bug, which matters because the normalizer treats the
    # vocabulary as evidence.
    lowered = text.lower()
    for term in BASELINE_TERMS:
        key = term.lower()
        if key in seen:
            continue
        if re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", lowered):
            seen[key] = term

    ordered = sorted(seen.values(), key=lambda t: lowered.find(t.lower()))
    return ordered[:limit]


def build_session_vocabulary(
    *,
    resume_text: str = "",
    job_description: str = "",
    notes: str = "",
    project_context: str = "",
    history_questions: list[str] | None = None,
    include_baseline: bool = True,
    max_terms: int = 120,
) -> list[str]:
    """The full, priority-ordered term list for one session.

    Priority is project context > resume > JD > notes > what has already
    been discussed > generic. When `max_terms` bites it truncates the tail,
    so the generic list is what gets dropped and the candidate's own stack
    never does -- the exact failure this module exists to fix.

    `project_context` outranks even the resume, because it is the same
    stack described at more length and in the words the candidate would use
    out loud: a resume line says "real-time AI application", the project
    notes behind it name Deepgram, the VAD and the question gate. Those are
    the words the interviewer will say back, and the ones a recognizer with
    no hint will mangle.
    """
    buckets: list[list[str]] = [
        extract_terms(project_context, limit=60),
        extract_terms(resume_text, limit=60),
        extract_terms(job_description, limit=45),
        extract_terms(notes, limit=15),
        extract_terms(" ".join(history_questions or []), limit=20),
    ]
    if include_baseline:
        buckets.append(list(BASELINE_TERMS))

    ordered: list[str] = []
    seen: set[str] = set()
    for bucket in buckets:
        for term in bucket:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(term)
            if len(ordered) >= max_terms:
                return ordered
    return ordered


def session_term_set(vocabulary: list[str]) -> set[str]:
    """Lower-cased lookup set, used by the normalizer to decide whether this
    session actually provides evidence for a canonical term."""
    return {term.lower() for term in vocabulary}


def as_whisper_prompt(vocabulary: list[str], *, max_bytes: int = 850) -> str:
    """Render the vocabulary as a Whisper `prompt`.

    Same byte cap as before, but spent on terms rather than prose, so the
    session's own words are the ones that fit. Truncation still happens at
    the tail -- the difference is that the tail is now the generic list.
    """
    header = "Technical interview. Correct terms include: "
    budget = max_bytes - len(header.encode("utf-8")) - 1
    parts: list[str] = []
    used = 0
    for term in vocabulary:
        chunk = (", " if parts else "") + term
        size = len(chunk.encode("utf-8"))
        if used + size > budget:
            break
        parts.append(term)
        used += size
    return header + ", ".join(parts) + "." if parts else header.strip()
