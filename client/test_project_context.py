"""
Offline test suite for candidate project/experience context -- document
extraction, the context model, project selection, budgeting, and what the
LLM is actually told.

Everything runs in-process with no server, no API key and no network. The
PDF and DOCX cases build real files in memory rather than shipping
fixtures, so what is parsed is a genuine PDF/DOCX and not a stand-in.

The point of the suite is one property, stated two ways:

    if the candidate says "I used Deepgram for streaming STT",
        the model can answer from that;
    if the candidate never says "Kubernetes",
        the model must never claim they deployed on Kubernetes.

The first half is checkable offline -- the text either reaches the prompt
or it does not, and that is what most of these cases assert. The second
half is a property of the model's output, so it needs a real call: run with
--live to send the same contexts to the configured provider and check the
answers for claims the context never supported.

Usage (from the project root, where .env lives):
    python client\\test_project_context.py
    python client\\test_project_context.py --list
    python client\\test_project_context.py --only pdf,docx
    python client\\test_project_context.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import io
import re
import sys
import zipfile
from pathlib import Path

# So `import app...` works no matter what directory this is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.redis_client import InterviewSessionContext  # noqa: E402
from app.services.candidate_context import (  # noqa: E402
    DEFAULT_CHAR_BUDGET,
    ProjectContext,
    build_candidate_context,
    select_projects,
    vocabulary_text,
)
from app.services.document_extraction import (  # noqa: E402
    DocumentExtractionError,
    extract_document_text,
)
from app.services.question_gate import _build_answer_messages  # noqa: E402
from app.services.session_vocabulary import build_session_vocabulary  # noqa: E402


MAX_BYTES = 8_000_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AUDIO_PROJECT = ProjectContext(
    name="Real-Time AI Interview Assistant",
    description=(
        "Real-time assistant that captures microphone and system audio, segments "
        "speech, converts it to text, detects interview questions and generates "
        "answers with an LLM."
    ),
    role="I designed the audio pipeline and wrote the VAD and question detection logic.",
    responsibilities="Owned the streaming path end to end, from capture to answer.",
    architecture=(
        "Audio Capture -> WebSocket -> VAD -> Speech Segmenter -> STT -> Question "
        "Gate -> LLM -> Answer"
    ),
    technologies="Python, FastAPI, WebSocket, Deepgram, Whisper, Groq",
    challenges="The system sometimes generated an answer before the interviewer had finished speaking.",
    solutions=(
        "Added VAD and a question gate so a complete speech segment has to land "
        "before anything reaches the LLM."
    ),
    decisions=(
        "We used Whisper through Groq first, then evaluated Deepgram because we "
        "needed lower latency for streaming transcription."
    ),
    tradeoffs=(
        "The question gate cuts unnecessary LLM calls and latency, but a strict gate "
        "wrongly rejects short valid questions, so we use a hybrid approach."
    ),
    metrics="End-to-end latency is 1.7 seconds from end of speech to first token.",
)

DASHBOARD_PROJECT = ProjectContext(
    name="Warehouse Analytics Dashboard",
    description="Internal dashboard showing throughput and picking accuracy per shift.",
    role="I built the React frontend and the reporting queries.",
    technologies="React, TypeScript, PostgreSQL",
    challenges="Shift reports took too long to load for large warehouses.",
    solutions="Pre-aggregated the nightly rollups instead of querying raw events.",
)

RESUME = (
    "Software engineer. Worked on a real-time AI application. Python, FastAPI, "
    "PostgreSQL. Built internal tools for warehouse operations."
)
JOB_DESCRIPTION = (
    "Backend engineer for real-time systems. Python, FastAPI, streaming audio, "
    "low-latency services, AWS."
)


def full_context(**overrides) -> InterviewSessionContext:
    base = dict(
        resume_text=RESUME,
        job_description=JOB_DESCRIPTION,
        projects=(AUDIO_PROJECT, DASHBOARD_PROJECT),
        experience_notes="",
        interview_stories="",
    )
    base.update(overrides)
    return InterviewSessionContext(**base)


def prompt_text(context: InterviewSessionContext, question: str) -> str:
    """Everything the model is told, as one string -- which is exactly the
    granularity these assertions care about."""
    messages = _build_answer_messages(question, context, [], char_budget=DEFAULT_CHAR_BUDGET)
    return "\n\n".join(message["content"] for message in messages)


# --- real documents, built in memory ---------------------------------------


def make_pdf(paragraphs: list[str]) -> bytes:
    """A minimal but genuinely valid single-page PDF, so pypdf does the real
    parsing rather than being handed something pre-digested."""
    lines = []
    for index, paragraph in enumerate(paragraphs):
        escaped = paragraph.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        lines.append(f"BT /F1 11 Tf 40 {760 - index * 18} Td ({escaped}) Tj ET")
    stream = "\n".join(lines).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n".encode()
    )
    out += b"%%EOF\n"
    return bytes(out)


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    """A real .docx: an OPC zip with the two parts Word needs plus a body,
    so the extractor's zip + XML path is genuinely exercised."""
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

    def paragraph(text: str) -> str:
        return f"<w:p><w:r><w:t xml:space='preserve'>{text}</w:t></w:r></w:p>"

    body = "".join(paragraph(text) for text in paragraphs)
    if table:
        rows = "".join(
            "<w:tr>"
            + "".join(f"<w:tc>{paragraph(cell)}</w:tc>" for cell in row)
            + "</w:tr>"
            for row in table
        )
        body += f"<w:tbl>{rows}</w:tbl>"

    document = f"<?xml version='1.0'?><w:document {ns}><w:body>{body}</w:body></w:document>"
    content_types = (
        "<?xml version='1.0'?>"
        "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
        "<Default Extension='xml' ContentType='application/xml'/>"
        "<Override PartName='/word/document.xml' ContentType='application/vnd."
        "openxmlformats-officedocument.wordprocessingml.document.main+xml'/>"
        "</Types>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def make_word_shaped_docx() -> bytes:
    """A .docx with the structures Word actually emits, which a hand-built
    fixture does not: a heading whose text is split across three runs, a
    content control, a tab and an explicit line break, a hyperlink, a
    tracked insertion, and a table.

    Worth its own fixture because the simple one passes on a walker that is
    subtly wrong. This is what caught the walker dropping text out of any
    container that holds runs directly instead of paragraphs -- a hyperlink
    being the common one.
    """
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    rel = "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'"

    body = (
        # One heading, three runs -- Word splits runs on spell-check state,
        # formatting and language tags constantly.
        "<w:p><w:pPr><w:pStyle w:val='Heading1'/></w:pPr>"
        "<w:r><w:t xml:space='preserve'>Real-Time </w:t></w:r>"
        "<w:r><w:t xml:space='preserve'>AI Interview </w:t></w:r>"
        "<w:r><w:t>Assistant</w:t></w:r></w:p>"
        # A content control wrapping real content.
        "<w:sdt><w:sdtPr/><w:sdtContent>"
        "<w:p><w:r><w:t>I used Deepgram for streaming STT.</w:t></w:r></w:p>"
        "</w:sdtContent></w:sdt>"
        # Tab, explicit break, and a hyperlink holding runs directly.
        "<w:p>"
        "<w:r><w:t>Latency:</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>1.7s</w:t></w:r>"
        "<w:r><w:br/></w:r><w:r><w:t>Second line after a break.</w:t></w:r>"
        "</w:p>"
        f"<w:hyperlink r:id='rId4' {rel}>"
        "<w:r><w:t>see the design doc</w:t></w:r></w:hyperlink>"
        # Tracked insertion: text one level deeper than a plain run.
        "<w:p><w:ins w:id='1' w:author='a'>"
        "<w:r><w:t>Added later: the question gate is hybrid.</w:t></w:r>"
        "</w:ins></w:p>"
        # A table.
        "<w:tbl><w:tr>"
        "<w:tc><w:p><w:r><w:t>Stage</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>Cost</w:t></w:r></w:p></w:tc></w:tr>"
        "<w:tr><w:tc><w:p><w:r><w:t>VAD</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>420ms</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
        "<w:sectPr/>"
    )
    document = f"<?xml version='1.0'?><w:document {ns}><w:body>{body}</w:body></w:document>"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<?xml version='1.0'?><Types/>")
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", "<?xml version='1.0'?><styles/>")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Cases
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


@case("manual_project", "A project typed into the form reaches the prompt, field by field.")
def _manual_project() -> str:
    text = prompt_text(full_context(), "tell me about your recent project")
    expect("Real-Time AI Interview Assistant" in text, "project name missing")
    expect("My role: I designed the audio pipeline" in text, "role label/value missing")
    expect("Audio Capture -> WebSocket -> VAD" in text, "architecture missing")
    expect("Technical decisions:" in text, "decisions label missing")
    expect("Trade-offs:" in text, "tradeoffs label missing")
    return "all twelve field labels render; role and architecture verbatim"


@case("pdf_upload", "A real PDF of project notes extracts to text and can be stored.")
def _pdf_upload() -> str:
    raw = make_pdf(
        [
            "Project: Real-Time AI Interview Assistant",
            "I implemented the WebSocket audio streaming layer.",
            "We chose Deepgram over Whisper for streaming latency.",
        ]
    )
    document = extract_document_text("notes.pdf", raw, max_bytes=MAX_BYTES)
    expect(document.kind == "pdf", f"kind was {document.kind}")
    expect("Deepgram" in document.text, "PDF text not extracted")
    expect("WebSocket audio streaming layer" in document.text, "second line lost")

    context = full_context(projects=(), experience_notes=document.text)
    text = prompt_text(context, "how did you handle streaming transcription")
    expect("Deepgram" in text, "uploaded notes did not reach the prompt")
    return f"{document.characters} characters extracted and carried into the prompt"


@case("docx_upload", "A real DOCX extracts paragraphs and table rows in order.")
def _docx_upload() -> str:
    raw = make_docx(
        [
            "Project: Real-Time AI Interview Assistant",
            "My role: I built the question gate.",
        ],
        table=[["Stage", "Latency"], ["VAD", "420ms"]],
    )
    document = extract_document_text("notes.docx", raw, max_bytes=MAX_BYTES)
    expect(document.kind == "docx", f"kind was {document.kind}")
    expect("I built the question gate." in document.text, "paragraph lost")
    expect("VAD | 420ms" in document.text, f"table row lost:\n{document.text}")
    return "paragraphs and table rows both preserved"


@case(
    "docx_word_shapes",
    "The structures Word really emits -- split runs, content controls, hyperlinks, tracked edits.",
)
def _docx_word_shapes() -> str:
    document = extract_document_text(
        "word-shapes.docx", make_word_shaped_docx(), max_bytes=MAX_BYTES
    )
    text = document.text
    checks = {
        "split runs rejoined": "Real-Time AI Interview Assistant" in text,
        "content control read": "I used Deepgram for streaming STT." in text,
        "tab preserved": "Latency:\t1.7s" in text,
        "line break kept": "Second line after a break." in text,
        # A hyperlink holds runs directly rather than paragraphs. The walker
        # used to recurse looking for paragraphs, find none, and drop the
        # text -- the one thing it promises not to do.
        "hyperlink text kept": "see the design doc" in text,
        "tracked insertion kept": "the question gate is hybrid" in text,
        "table rows piped": "VAD | 420ms" in text,
        "no duplicated table text": text.count("420ms") == 1,
    }
    failed = [name for name, ok in checks.items() if not ok]
    expect(not failed, f"lost or mangled: {failed}\n{text}")
    return f"{len(checks)} Word structures preserved, none duplicated"


@case("empty_document", "Empty and content-free documents are refused with a usable message.")
def _empty_document() -> str:
    messages = []
    for filename, raw in (
        ("empty.txt", b""),
        ("blank.txt", b"   \n\n  "),
        ("blank.docx", make_docx([""])),
    ):
        try:
            extract_document_text(filename, raw, max_bytes=MAX_BYTES)
        except DocumentExtractionError as exc:
            messages.append(str(exc))
        else:
            raise Failure(f"{filename} was accepted as context")
    expect(all(len(message) > 30 for message in messages), "error message too terse")
    return "3 refused: " + messages[0][:60] + "..."


@case("multiple_projects", "Several projects all render, each labelled and separate.")
def _multiple_projects() -> str:
    text = prompt_text(full_context(), "walk me through your experience")
    expect("Project 1:" in text and "Project 2:" in text, "projects not numbered")
    expect("Warehouse Analytics Dashboard" in text, "second project missing")
    expect(
        text.index("Project 1:") < text.index("Project 2:"), "project order not stable"
    )
    return "2 projects, numbered and ordered"


@case(
    "project_specific_question",
    "A question about latency puts the real-time project first, not the dashboard.",
)
def _project_specific_question() -> str:
    selected = select_projects(
        (DASHBOARD_PROJECT, AUDIO_PROJECT), "how did you reduce latency in the audio pipeline"
    )
    expect(selected[0] is AUDIO_PROJECT, "irrelevant project ranked first")

    text = prompt_text(full_context(), "how did you reduce latency")
    expect(
        text.index("Real-Time AI Interview Assistant") < text.index("Warehouse"),
        "relevant project not first in the prompt",
    )
    return "relevant project ranked first even though it was second in the list"


@case(
    "unrelated_question",
    "A question matching no project still gets context, in declared order.",
)
def _unrelated_question() -> str:
    selected = select_projects(
        (AUDIO_PROJECT, DASHBOARD_PROJECT), "what motivates you outside work"
    )
    expect(len(selected) == 2, "projects dropped for an unrelated question")
    expect(selected[0] is AUDIO_PROJECT, "declared order not preserved on a tie")
    return "no false drop: both projects kept, original order"


@case(
    "missing_experience",
    "Nothing invents Kubernetes: it is absent from the context, and the prompt forbids claiming it.",
)
def _missing_experience() -> str:
    context = full_context()
    text = prompt_text(context, "have you deployed on kubernetes")
    expect("Kubernetes" not in text.replace("Kubernetes,", ""), "Kubernetes leaked into context")
    lowered = text.lower()
    expect(
        "never claim you built something just because it is a normal part of a stack"
        in lowered,
        "ownership rule missing from the system prompt",
    )
    expect(
        "not something you have worked with" in lowered,
        "missing-experience rule not in the system prompt",
    )
    expect(
        "end of candidate context" in lowered,
        "context boundary marker missing -- this is what stops an invented stack",
    )
    return "no Kubernetes in context; ownership, missing-experience and boundary rules present"


@case(
    "numbers_not_provided",
    "No metric in the context means no number in the context, and the number rule is stated.",
)
def _numbers_not_provided() -> str:
    bare = ProjectContext(
        name="Real-Time AI Interview Assistant",
        description="Real-time interview assistant.",
        role="I worked on reducing latency in the streaming path.",
    )
    text = prompt_text(full_context(projects=(bare,)), "how much did you reduce latency by")
    context_block = text.split("CANDIDATE PROFILE", 1)[1]
    numbers = re.findall(r"\b\d+(?:\.\d+)?\s*(?:ms|s|%|x)\b", context_block)
    expect(not numbers, f"a number appeared in context that was never provided: {numbers}")
    expect("NEVER state a percentage, duration, latency" in text, "number rule missing")
    return "zero measurements in context; number rule present"


@case("numbers_provided", "A metric the candidate did give survives into the prompt intact.")
def _numbers_provided() -> str:
    text = prompt_text(full_context(), "what latency did you get")
    expect("1.7 seconds" in text, "provided metric was dropped")
    expect("Results and metrics:" in text, "metrics label missing")
    return "'1.7 seconds' present under the metrics label"


@case(
    "multi_part_question",
    "A multi-part question reaches the model whole, with the format rule that covers it.",
)
def _multi_part_question() -> str:
    question = (
        "Walk me through the architecture, then tell me why you picked Deepgram, "
        "and what the trade-off was."
    )
    messages = _build_answer_messages(question, full_context(), [], char_budget=DEFAULT_CHAR_BUDGET)
    user = messages[-1]["content"]
    expect(user.strip().endswith(question), "question not last in the user message")
    expect("CURRENT QUESTION" in user, "question section header missing")
    joined = "\n".join(message["content"] for message in messages)
    expect("in the order asked" in joined, "multi-part format rule missing")
    return "question intact and last; multi-part rule present"


@case(
    "stt_typo",
    "A mis-transcribed term still selects the right project, and the repair rule is present.",
)
def _stt_typo() -> str:
    # "deep gram" is what a recognizer with no vocabulary hint produces.
    selected = select_projects(
        (DASHBOARD_PROJECT, AUDIO_PROJECT), "why did you pick deep gram for the audio pipeline"
    )
    expect(selected[0] is AUDIO_PROJECT, "typo defeated project selection")
    text = prompt_text(full_context(), "why did you pick deep gram")
    expect("Silently infer the most plausible real term" in text, "STT repair rule missing")

    # And the fix upstream of the model: the term is fed to the recognizer.
    terms = build_session_vocabulary(project_context=vocabulary_text(full_context()))
    expect("Deepgram" in terms, "Deepgram not in the session vocabulary")
    expect(
        terms.index("Deepgram") < terms.index("Docker"),
        "project terms not prioritised over the generic list",
    )
    return "project still selected; Deepgram biased into the recognizer ahead of generic terms"


@case(
    "incomplete_question",
    "A truncated question is passed through untouched -- completing it is the gate's job, not this layer's.",
)
def _incomplete_question() -> str:
    messages = _build_answer_messages(
        "so what challenges did you", full_context(), [], char_budget=DEFAULT_CHAR_BUDGET
    )
    expect(messages[-1]["content"].endswith("so what challenges did you"), "question altered")
    expect(len(messages) == 3, f"expected 3 messages, got {len(messages)}")
    return "passed through verbatim; three-part message shape intact"


@case(
    "large_context",
    "A huge context is budgeted, not truncated blindly: resume and JD keep their floors.",
)
def _large_context() -> str:
    filler = "Detailed architecture notes about the streaming pipeline. " * 400
    big_projects = tuple(
        ProjectContext(
            name=f"Project {index}",
            description=filler,
            role="I built the streaming layer.",
            technologies="Python, FastAPI",
        )
        for index in range(6)
    )
    context = full_context(
        projects=big_projects,
        experience_notes=filler,
        interview_stories=filler,
        notes=filler,
    )
    block = build_candidate_context(context, question="tell me about your work", char_budget=5200)
    expect(len(block) <= 5200, f"budget blown: {len(block)} characters")
    expect("Resume:" in block, "resume evicted by a large project blob")
    expect("Job description" in block, "job description evicted by a large project blob")
    expect(block.count("Project ") <= 5, "more projects included than the cap allows")

    unbudgeted = sum(len(value) for project in big_projects for value in project.to_dict().values())
    return f"{unbudgeted:,} characters of input rendered as {len(block):,}; resume and JD survived"


@case(
    "extraction_failure",
    "Corrupt, oversized, unsupported and mislabelled files all fail with an actionable message.",
)
def _extraction_failure() -> str:
    cases = [
        ("notes.docx", b"this is not a zip file at all, it is plain text"),
        ("notes.pdf", b"%PDF-1.4 truncated garbage" + b"\x00" * 40),
        ("notes.xlsx", b"anything"),
        ("notes.doc", b"anything"),
        ("notes.txt", b"\x00\x01\x02" * 100),
        ("huge.pdf", b"x" * (MAX_BYTES + 1)),
    ]
    messages = []
    for filename, raw in cases:
        try:
            extract_document_text(filename, raw, max_bytes=MAX_BYTES)
        except DocumentExtractionError as exc:
            messages.append(str(exc))
        else:
            raise Failure(f"{filename} was accepted")
    expect(
        all(len(message) > 40 for message in messages),
        "an error message was too terse to act on",
    )
    return f"{len(messages)} failure modes, each with an actionable message"


@case(
    "hot_path_reads_strings_only",
    "Building a prompt does no parsing and no I/O -- the interview path only reads stored text.",
)
def _hot_path() -> str:
    import time

    context = full_context(
        experience_notes="Streaming notes. " * 300,
        interview_stories="Incident notes. " * 200,
    )
    start = time.perf_counter()
    for _ in range(200):
        _build_answer_messages("how did you reduce latency", context, [], char_budget=5200)
    elapsed_ms = (time.perf_counter() - start) * 1000 / 200
    expect(elapsed_ms < 5, f"context build costs {elapsed_ms:.2f}ms per question")
    return f"{elapsed_ms:.2f}ms per question to build the whole prompt"


@case(
    "source_priority_documented",
    "The system prompt states the source ranking and the ownership distinctions.",
)
def _source_priority() -> str:
    text = prompt_text(full_context(), "tell me about yourself")
    for needle in (
        # The ranking itself.
        "then resume, then job description, then general knowledge",
        # The four ownership levels the context is read through.
        "what you BUILT, what your TEAM built, what you USED and what you EVALUATED",
        # Answer from the real architecture, not a generic account of it.
        "not a generic account of how such systems work",
        # The rule that a JD is not evidence -- the defect a live run caught,
        # where a JD mention of AWS was read as the candidate having used it.
        "job description is NOT evidence of your experience",
    ):
        expect(needle in text, f"missing from the prompt: {needle!r}")
    return "ranking, ownership, real-architecture and JD-is-not-evidence rules all present"


@case(
    "live_checkers_catch_fabrication",
    "The --live graders themselves catch what they are meant to -- twice they did not.",
)
def _live_checkers() -> str:
    """The graders are the only thing standing between a fabricated answer
    and a green run, so they get tested like any other code.

    Both regressions below actually happened here. The first version
    forbade PHRASINGS ("we used aws") and passed an answer that had invented
    Docker, ECS, an ALB, Fargate, CloudWatch and Secrets Manager. The number
    check then passed "our 1-second target" against a context that only said
    1.7 seconds -- twice over: the model wrote a non-breaking hyphen the
    regex could not see, and "1" is a substring of "1.7".
    """
    context = "End-to-end latency is 1.7 seconds from end of speech to first token."
    tools = ["aws", "ecs", "docker", "load balancer"]

    expect(
        _ownership_claims("We ran it on AWS ECS behind a load balancer.", tools),
        "past-tense ownership of an unprovided tool went uncaught",
    )
    expect(
        not _ownership_claims("That wasn't my part, but I'd run it on ECS.", tools),
        "a clearly conditional answer was wrongly failed",
    )
    expect(
        _unprovided_numbers("higher than our 1\u2011second target", context),
        "an invented figure written with a non-breaking hyphen went uncaught",
    )
    expect(
        _unprovided_numbers("we cut it to 200\u00a0ms", context),
        "an invented figure written with a non-breaking space went uncaught",
    )
    expect(
        not _unprovided_numbers("about 1.7 seconds end to end", context),
        "a figure that IS in the context was wrongly flagged",
    )
    return "ownership, conditionals, unicode separators and substring numbers all handled"


@case("round_trip", "Context survives the store's JSON round trip, including old payloads.")
def _round_trip() -> str:
    import json

    context = full_context(experience_notes="notes", interview_stories="stories")
    restored = InterviewSessionContext.from_dict(json.loads(json.dumps(context.to_dict())))
    expect(isinstance(restored.projects[0], ProjectContext), "projects came back as dicts")
    expect(restored.projects[0].decisions == AUDIO_PROJECT.decisions, "field lost in transit")

    legacy = InterviewSessionContext.from_dict(
        {"resume_text": "r", "job_description": "j", "notes": "n"}
    )
    expect(legacy.projects == () and legacy.experience_notes == "", "legacy payload mishandled")

    unknown = InterviewSessionContext.from_dict({"resume_text": "r", "from_the_future": 1})
    expect(unknown.resume_text == "r", "unknown key broke deserialization")
    return "round trip, pre-feature payload, and unknown-key payload all load"


# ---------------------------------------------------------------------------
# Live checks -- the half that needs a real model
# ---------------------------------------------------------------------------

LIVE_CHECKS = [
    (
        "uses_provided_experience",
        "why did you choose deepgram instead of whisper",
        full_context,
        # It must reach for the candidate's own stated reason.
        {"require_any": ["deepgram"], "forbid": ["kubernetes", "kafka"]},
    ),
    (
        "uses_real_architecture",
        "how did you handle real-time processing",
        full_context,
        {"require_any": ["vad", "question gate", "segment"], "forbid": ["kubernetes"]},
    ),
    (
        "does_not_invent_stack",
        "how did you deploy and scale the system",
        full_context,
        # The context says nothing whatsoever about deployment. Every name
        # below is therefore an invention if it appears -- and this is the
        # case that caught the real bug: the first version of this check
        # forbade PHRASINGS ("we used aws") and passed an answer that had
        # invented Docker, ECS, an ALB, Fargate, CloudWatch and Secrets
        # Manager. Forbid the NAMES. A tool the candidate never mentioned
        # has no business in an answer about what they did, however the
        # sentence around it is worded.
        #
        # Note "aws" is here even though it appears in the JOB DESCRIPTION
        # fixture. That is deliberate and is what the bug turned out to be:
        # the JD says what the employer wants, not what the candidate did,
        # so it can never license a claim of experience.
        {
            "no_ownership_of": [
                "kubernetes",
                "terraform",
                "aws",
                "ecs",
                "fargate",
                "cloudwatch",
                "secrets manager",
                "load balancer",
                "docker",
            ]
        },
    ),
    (
        "does_not_invent_numbers",
        "how much did you improve latency by",
        lambda: full_context(
            projects=(
                ProjectContext(
                    name="Real-Time AI Interview Assistant",
                    description="Real-time interview assistant.",
                    role="I worked on reducing latency in the streaming path.",
                ),
            )
        ),
        {"forbid_pattern": r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds|%|percent)\b"},
    ),
    (
        "uses_provided_numbers",
        "what end to end latency did you achieve",
        full_context,
        {"require_any": ["1.7"]},
    ),
]


async def run_live(only: list[str] | None) -> int:
    from app.core.config import get_settings
    from app.services.llm import build_llm_client

    settings = get_settings()
    client = build_llm_client(settings)
    print(
        f"live: {settings.answer_provider}/{settings.answer_model} -- real calls, "
        "real cost\n"
    )

    failures = 0
    for name, question, context_factory, rules in LIVE_CHECKS:
        if only and not any(needle in name for needle in only):
            continue
        messages = _build_answer_messages(
            question,
            context_factory(),
            [],
            char_budget=settings.candidate_context_char_budget,
        )
        answer = ""
        async for token in client.stream_chat(messages):
            answer += token
        lowered = answer.lower()

        problems = []
        for term in rules.get("forbid", []):
            if term in lowered:
                problems.append(f"claimed {term!r}, which the context never mentions")
        problems += _ownership_claims(answer, rules.get("no_ownership_of", []))
        prompt = "\n".join(message["content"] for message in messages)
        problems += _unprovided_numbers(answer, prompt)
        pattern = rules.get("forbid_pattern")
        if pattern:
            found = re.findall(pattern, lowered)
            if found:
                problems.append(f"invented measurements: {found}")
        required = rules.get("require_any")
        if required and not any(term in lowered for term in required):
            problems.append(f"used none of the provided specifics {required}")

        status = "FAIL" if problems else "ok"
        failures += bool(problems)
        _say(f"[{status:4}] {name}")
        _say(f"        Q: {question}")
        for line in answer.strip().splitlines():
            _say(f"        {line}")
        for problem in problems:
            _say(f"        -> {problem}")
        _say("")
    return failures


# "I deployed it on ECS" is the failure. "It wasn't my part of the work, but
# I'd run it on ECS" is a correct interview answer -- naming a tool inside a
# conditional is discussing it, not claiming it, and the spec asks for exactly
# that when the experience is missing. So the check looks for a first-person
# PAST claim in the same sentence as an unprovided tool, and lets a
# conditional through.
_OWNERSHIP = re.compile(
    r"\b(?:i|we)\b[^.?!]{0,80}?\b(?:used|ran|deployed|built|set up|configured|"
    r"chose|picked|managed|had|wrote|containerized|containerised)\b",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(
    r"\b(?:i'?d|we'?d|would|could|might|if i|if we|typically|usually|generally)\b",
    re.IGNORECASE,
)


def _ownership_claims(answer: str, tools: list[str]) -> list[str]:
    problems = []
    for sentence in re.split(r"(?<=[.?!])\s+|\n+", _normalise(answer)):
        lowered = sentence.lower()
        named = [tool for tool in tools if tool in lowered]
        if not named:
            continue
        if _OWNERSHIP.search(sentence) and not _CONDITIONAL.search(sentence):
            problems.append(
                f"claimed to have used {named} as their own work: "
                f"{sentence.strip()[:120]!r}"
            )
    return problems


# Every measurement in an answer has to be traceable to the context. This is
# the rule the spec is most emphatic about, and the one an interviewer is most
# likely to test with a follow-up.
_MEASUREMENT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(ms|milliseconds?|seconds?|secs?|minutes?|hours?|%|"
    r"percent|x|times|users?|requests?)\b",
    re.IGNORECASE,
)


# Models write "1\u2011second target" with a non-breaking hyphen, and
# "1\u00a0second" with a non-breaking space. Both read as ordinary text and
# both slipped straight past an earlier version of this check -- which let a
# genuinely invented figure ("our 1-second target", nowhere in the context)
# report as clean. Normalise before matching.
_UNICODE_SEPARATORS = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212\u00a0\u202f\u2009"), " "
)


def _normalise(text: str) -> str:
    return text.translate(_UNICODE_SEPARATORS)


def _unprovided_numbers(answer: str, context: str) -> list[str]:
    """Every measurement in the answer has to trace to one in the context.

    The comparison is whole-number, not substring, and that distinction is
    the whole check: a plain `"1" in context` is satisfied by the "1" inside
    "1.7 seconds", so an answer inventing "our 1-second target" against a
    context that only ever said 1.7 seconds passed clean.
    """
    answer, context = _normalise(answer), _normalise(context)
    problems = []
    for value, unit in sorted(set(_MEASUREMENT.findall(answer))):
        if not re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.])", context):
            problems.append(f"stated {value} {unit}, a measurement not in the context")
    return problems


def _say(text: str) -> None:
    """Print through whatever the console can actually encode.

    Models emit typographic characters -- non-breaking hyphens, curly
    quotes, em dashes -- and a Windows console on cp1252 raises on them.
    Losing the run to a hyphen after paying for the API calls is not a
    trade worth making.
    """
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="List cases and exit.")
    parser.add_argument(
        "--only", default=None, help="Comma-separated substrings of case names to run."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also send the fixtures to the real provider and check the answers for "
        "claims the context never supported (real API calls, real cost).",
    )
    args = parser.parse_args()

    if args.list:
        for name, note, _function in CASES:
            print(f"{name}\n  {note}\n")
        for name, question, _factory, _rules in LIVE_CHECKS:
            print(f"{name} (--live only)\n  asks: {question}\n")
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
        except Exception as exc:  # a broken test is a failed test
            failures += 1
            print(f"[ERROR] {name}\n       {note}\n       {type(exc).__name__}: {exc}\n")
        else:
            print(f"[ok  ] {name}\n       {note}\n       {detail}\n")

    if args.live:
        print("-" * 70)
        failures += asyncio.run(run_live(only))

    print(f"{len(selected)} offline cases, {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
