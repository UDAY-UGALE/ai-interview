from dataclasses import replace

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.redis_client import InterviewSessionContext, get_session_store
from app.services.candidate_context import ProjectContext, build_candidate_context
from app.services.document_extraction import (
    DocumentExtractionError,
    extract_document_text,
)
from app.services.llm import MODEL_CATALOG
from app.services.question_gate import get_question_pipeline


router = APIRouter()

# Which context fields an uploaded document (or a POST /session field) may
# target. resume/job_description are the originals; the rest are the
# project-and-experience context added for grounding answers in what the
# candidate actually built.
TEXT_FIELDS: tuple[str, ...] = (
    "resume",
    "job_description",
    "experience_notes",
    "interview_stories",
    "notes",
)

# The upload API calls it "resume"; the context field is "resume_text".
_FIELD_ATTRIBUTES = {
    "resume": "resume_text",
    "job_description": "job_description",
    "experience_notes": "experience_notes",
    "interview_stories": "interview_stories",
    "notes": "notes",
}


class ProjectPayload(BaseModel):
    """One project the candidate described. Every field optional -- someone
    who types a paragraph into `description` and nothing else has given us
    something useful, and a form that demands twelve fields gets filled by
    nobody."""

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

    def to_context(self) -> ProjectContext:
        return ProjectContext(**self.model_dump())


class SessionContextRequest(BaseModel):
    session_id: str = "default"
    # All fields are optional and PARTIAL-MERGE into the existing session
    # context -- e.g. sending just answer_provider/answer_model (from the
    # frontend model picker) will not wipe out a resume/JD you set earlier.
    # Omit a field (or leave it unset) to keep its current value; sending it
    # updates it, including to an empty string if you want to clear it.
    resume_text: str | None = None
    job_description: str | None = None
    notes: str | None = None
    # Candidate-provided project and experience context. `projects` is sent
    # as the WHOLE list when sent at all -- the overlay's project dialog
    # holds the list and saves it, so replace-on-write is what "I deleted a
    # project" has to mean. Send [] to clear.
    projects: list[ProjectPayload] | None = None
    experience_notes: str | None = None
    interview_stories: str | None = None
    answer_provider: str | None = None
    answer_model: str | None = None


def _merged(
    existing: InterviewSessionContext, payload: SessionContextRequest
) -> InterviewSessionContext:
    """Partial merge: only the fields actually sent are changed.

    Written as a loop over what was sent rather than a field-by-field
    constructor so that adding a context field is one line in the model and
    nothing here.
    """
    updates: dict[str, object] = {
        key: value
        for key, value in payload.model_dump(exclude_unset=True).items()
        if key not in ("session_id", "projects") and value is not None
    }
    if payload.projects is not None:
        updates["projects"] = tuple(project.to_context() for project in payload.projects)
    return replace(existing, **updates)


async def _store_context(session_id: str, context: InterviewSessionContext) -> None:
    store = get_session_store()
    await store.set_context(session_id, context)
    # The vocabulary is derived from the resume/JD/project context and is
    # cached per session because it is consulted on every transcript.
    # Dropping it here is what makes a mid-session upload actually take
    # effect.
    get_question_pipeline().refresh_vocabulary(session_id)


@router.post("/session")
async def set_session_context(payload: SessionContextRequest) -> dict[str, str]:
    store = get_session_store()
    existing = await store.get_context(payload.session_id)
    await _store_context(payload.session_id, _merged(existing, payload))
    return {"status": "ok", "session_id": payload.session_id}


@router.get("/session/{session_id}")
async def get_session_context(session_id: str) -> dict:
    context = await get_session_store().get_context(session_id)
    history = await get_session_store().get_history(session_id)
    return {
        "session_id": session_id,
        "context": context.to_dict(),
        "history": [turn.to_dict() for turn in history],
    }


@router.get("/session/{session_id}/context-preview")
async def preview_candidate_context(session_id: str, question: str = "") -> dict:
    """Exactly what the model will be told about the candidate, and how big
    it is.

    The whole feature is "the answer should come from my real work", and
    the only way to know whether it will is to see what actually gets sent
    after budgeting and project selection. Pass `question` to see which
    projects are selected for that question.
    """
    settings = get_settings()
    context = await get_session_store().get_context(session_id)
    block = build_candidate_context(
        context,
        question=question,
        char_budget=settings.candidate_context_char_budget,
    )
    return {
        "session_id": session_id,
        "question": question,
        "characters": len(block),
        "approximate_tokens": len(block) // 4,
        "character_budget": settings.candidate_context_char_budget,
        "context": block,
    }


@router.get("/models")
async def list_models() -> dict:
    """Curated provider/model catalog for a frontend picker. Any model
    string still works even if it isn't listed here."""
    return {"catalog": MODEL_CATALOG}


@router.post("/session/extract")
async def extract_document(file: UploadFile = File(...)) -> dict:
    """Extract text from a PDF/DOCX/TXT and hand it straight back, storing
    nothing.

    This is the upload path for the project-context dialog, and it does not
    write to the session on purpose: the dialog shows the user the text
    that will become their interview context, lets them edit it, remove a
    document, and add several -- and then saves the composed result through
    POST /session. Storing each file server-side instead would mean a
    server-side document registry to make "Remove" work, for no gain.

    /session/upload is the other half: same extractor, but it merges into
    one field directly, which is what the overlay's one-click resume/JD
    buttons want.
    """
    settings = get_settings()
    raw = await file.read()
    try:
        document = extract_document_text(
            file.filename or "document", raw, max_bytes=settings.context_upload_max_bytes
        )
    except DocumentExtractionError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "status": "ok",
        "filename": document.filename,
        "kind": document.kind,
        "characters_extracted": document.characters,
        "truncated": document.truncated,
        "text": document.text,
    }


@router.post("/session/upload")
async def upload_session_document(
    session_id: str = Form(...),
    field: str = Form(...),
    file: UploadFile = File(...),
    append: bool = Form(False),
) -> dict:
    """Upload a PDF, DOCX or TXT and merge its text into one context field.

    `field` is one of resume, job_description, experience_notes,
    interview_stories, notes. The extracted text is what actually gets sent
    to the LLM as context -- the same as if you had pasted it in as plain
    text via POST /session, just sourced from a document instead.

    `append=true` adds to what is already in that field instead of
    replacing it, which is how several sets of project notes accumulate
    into experience_notes. Re-uploading text the field already contains is
    refused rather than duplicated: a doubled document is a doubled token
    bill on every question of the interview.
    """
    if field not in TEXT_FIELDS:
        raise HTTPException(400, f"field must be one of: {', '.join(TEXT_FIELDS)}.")

    settings = get_settings()
    raw = await file.read()
    try:
        document = extract_document_text(
            file.filename or "document", raw, max_bytes=settings.context_upload_max_bytes
        )
    except DocumentExtractionError as exc:
        raise HTTPException(400, str(exc)) from exc

    attribute = _FIELD_ATTRIBUTES[field]
    store = get_session_store()
    existing = await store.get_context(session_id)
    current = getattr(existing, attribute) or ""

    if append and document.text.strip() and document.text.strip() in current:
        raise HTTPException(
            409,
            f"{document.filename} looks like it has already been added to "
            f"{field} -- its text is already there. Remove it first if you want "
            "to re-add it.",
        )

    merged = f"{current}\n\n{document.text}".strip() if append and current else document.text
    await _store_context(session_id, replace(existing, **{attribute: merged}))

    return {
        "status": "ok",
        "session_id": session_id,
        "field": field,
        "filename": document.filename,
        "kind": document.kind,
        "characters_extracted": document.characters,
        "truncated": document.truncated,
        "preview": document.text[:200],
    }


class AskRequest(BaseModel):
    session_id: str = "default"
    question: str


@router.post("/ask")
async def ask_directly(payload: AskRequest) -> dict[str, str]:
    """Manual override for when speech-to-text misheard the question (bad
    network, unclear audio, etc). Skips STT/the auto-gate entirely and
    answers exactly the text you send -- cancelling any in-flight automatic
    answer for this session first. Meant to be wired to an editable
    question field in the overlay."""
    await get_question_pipeline().ask_directly(
        session_id=payload.session_id, text=payload.question
    )
    return {"status": "ok", "session_id": payload.session_id}


class ScenarioRequest(BaseModel):
    session_id: str = "default"


@router.post("/scenario/start")
async def scenario_start(payload: ScenarioRequest) -> dict[str, str]:
    """MODE B: start capturing one long question under user control.

    Everything heard between this call and /scenario/stop is accumulated as
    ONE question and nothing is answered in the meantime. This exists
    because automatic question completion is a good default and a bad
    universal rule: a scenario question is several sentences of setup, a
    thinking pause, then the ask, and the automatic path was measured
    answering the setup on its own and then answering the real question
    without the setup facts. Here the Stop press defines the end of the
    question, so there is nothing to infer.
    """
    await get_question_pipeline().start_scenario(session_id=payload.session_id)
    return {"status": "listening", "session_id": payload.session_id}


@router.post("/scenario/stop")
async def scenario_stop(payload: ScenarioRequest) -> dict[str, str]:
    """End MODE B capture and answer the whole captured question, once."""
    await get_question_pipeline().stop_scenario(session_id=payload.session_id)
    return {"status": "ok", "session_id": payload.session_id}


@router.get("/session/{session_id}/vocabulary")
async def session_vocabulary(session_id: str) -> dict:
    """What terms this session will bias the recognizer toward, and what the
    transcript normalizer is therefore willing to repair.

    Exposed because both behaviours are otherwise invisible: whether "Rack"
    becomes RAG depends entirely on whether RAG is in here.
    """
    terms, _lookup = await get_question_pipeline().session_vocabulary(session_id)
    return {"session_id": session_id, "term_count": len(terms), "terms": terms}
