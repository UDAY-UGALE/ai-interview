from fastapi import FastAPI

from app.core.config import get_settings
from app.routes.answers_ws import router as answers_ws_router
from app.routes.audio_ws import router as audio_ws_router
from app.routes.screen import router as screen_router
from app.routes.session import router as session_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(audio_ws_router)
    app.include_router(answers_ws_router)
    app.include_router(session_router)
    app.include_router(screen_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
