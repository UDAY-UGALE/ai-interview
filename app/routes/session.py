import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.redis_client import InterviewSessionContext, get_session_store
from app.services.llm import MODEL_CATALOG
from app.services.question_gate import get_question_pipeline


router = APIRouter()


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
    answer_provider: str | None = None
    answer_model: str | None = None


@router.post("/session")
async def set_session_context(payload: SessionContextRequest) -> dict[str, str]:
    store = get_session_store()
    existing = await store.get_context(payload.session_id)

    context = InterviewSessionContext(
        resume_text=(
            payload.resume_text if payload.resume_text is not None else existing.resume_text
        ),
        job_description=(
            payload.job_description
            if payload.job_description is not None
            else existing.job_description
        ),
        notes=payload.notes if payload.notes is not None else existing.notes,
        answer_provider=(
            payload.answer_provider
            if payload.answer_provider is not None
            else existing.answer_provider
        ),
        answer_model=(
            payload.answer_model if payload.answer_model is not None else existing.answer_model
        ),
    )
    await store.set_context(payload.session_id, context)
    # The resume/JD are what the session vocabulary is derived from, and that
    # vocabulary is cached per session because it is consulted on every
    # transcript. Dropping it here is what makes a mid-session resume upload
    # actually take effect.
    get_question_pipeline().refresh_vocabulary(payload.session_id)
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


@router.get("/models")
async def list_models() -> dict:
    """Curated provider/model catalog for a frontend picker. Any model
    string still works even if it isn't listed here."""
    return {"catalog": MODEL_CATALOG}


@router.post("/session/upload")
async def upload_session_document(
    session_id: str = Form(...),
    field: str = Form(...),  # "resume" or "job_description"
    file: UploadFile = File(...),
) -> dict:
    """Upload a PDF resume or job description -- extracts the text and
    merges it into the session context the same way POST /session does
    (so it doesn't touch anything else already set, like notes or the
    model choice). The extracted text is what actually gets sent to the
    LLM as context on every question -- same as if you'd pasted it in as
    plain text via POST /session, just sourced from a PDF instead."""
    if field not in ("resume", "job_description"):
        raise HTTPException(400, "field must be 'resume' or 'job_description'.")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported right now.")

    raw = await file.read()
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(raw))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except ImportError as exc:
        raise HTTPException(500, "Install pypdf (pip install pypdf) to enable PDF upload.") from exc
    except Exception as exc:
        raise HTTPException(400, f"Could not read PDF: {exc}") from exc

    if not text:
        raise HTTPException(
            400,
            "No extractable text found in that PDF -- it might be a scanned/image PDF, "
            "which needs OCR (not supported here). Try a text-based PDF, or paste the text "
            "directly via POST /session instead.",
        )

    store = get_session_store()
    existing = await store.get_context(session_id)
    context = InterviewSessionContext(
        resume_text=text if field == "resume" else existing.resume_text,
        job_description=text if field == "job_description" else existing.job_description,
        notes=existing.notes,
        answer_provider=existing.answer_provider,
        answer_model=existing.answer_model,
    )
    await store.set_context(session_id, context)
    get_question_pipeline().refresh_vocabulary(session_id)

    return {
        "status": "ok",
        "session_id": session_id,
        "field": field,
        "characters_extracted": len(text),
        "preview": text[:200],
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
