from fastapi import APIRouter
from pydantic import BaseModel

from app.services.question_gate import get_question_pipeline


router = APIRouter()


class AnalyzeScreenRequest(BaseModel):
    session_id: str = "default"
    # Base64-encoded screenshot bytes (no "data:image/png;base64," prefix --
    # just the raw base64), captured client-side by the overlay.
    image_base64: str
    media_type: str = "image/png"
    # Optional guiding question ("what does this error mean?", "help me
    # answer this"). If omitted, a generic "what's on my screen" prompt is
    # used instead.
    question: str | None = None


@router.post("/analyze-screen")
async def analyze_screen(payload: AnalyzeScreenRequest) -> dict[str, str]:
    """Takes a screenshot (base64) plus an optional guiding question,
    cancels any in-flight automatic answer for the session, and streams a
    vision-LLM analysis back through /ws/answers -- the same event stream
    used for a normal spoken question, so the overlay renders it without
    any special-casing. Returns immediately; the actual answer streams over
    the websocket."""
    await get_question_pipeline().analyze_screen(
        session_id=payload.session_id,
        image_base64=payload.image_base64,
        media_type=payload.media_type,
        question=payload.question,
    )
    return {"status": "ok", "session_id": payload.session_id}
