from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.auth import require_token
from app.core.config import Settings, get_settings
from app.routes.answers_ws import router as answers_ws_router
from app.routes.audio_ws import router as audio_ws_router
from app.routes.screen import router as screen_router
from app.routes.session import router as session_router


def _configure_cors(app: FastAPI, settings: Settings) -> None:
    """Installs CORS only if origins were configured.

    The desktop client is not a browser: it sends no Origin header and CORS
    is irrelevant to it. This exists for a web page (the site, or a future
    browser-based console) calling the API, and stays off by default so the
    common case ships no middleware it doesn't use.
    """
    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    if not origins:
        return

    if "*" in origins and settings.app_auth_token:
        # allow_credentials is False here, so "*" would not leak cookies --
        # but a token-protected backend that accepts every origin is a
        # CSRF-shaped hole waiting for someone to add credentials later.
        # Fail loudly at startup instead of quietly allowing it.
        raise ValueError(
            "CORS_ALLOW_ORIGINS='*' with APP_AUTH_TOKEN set is refused. "
            "List the origins that should be allowed."
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,  # auth is a bearer token, never a cookie
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Auth-Token"],
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    _configure_cors(app, settings)

    app.include_router(audio_ws_router)
    app.include_router(answers_ws_router)
    # The websocket routers check the token themselves (a handshake has no
    # place to raise an HTTPException from); the HTTP ones take it as a
    # dependency.
    app.include_router(session_router, dependencies=[Depends(require_token)])
    app.include_router(screen_router, dependencies=[Depends(require_token)])

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe. Deliberately unauthenticated and dependency-free:
        every platform's health check hits it anonymously (Render health
        check, ALB target group, Container Apps probe), and the pre-interview
        warm-up ping uses it to wake a sleeping free instance."""
        return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, str | bool]:
        """Which build is running and roughly how it is configured.

        Answers "did my deploy actually go out, and is the token on?" without
        a shell on the box. Names configuration, never values -- no key, no
        token, not even a prefix.
        """
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "stt_provider": settings.stt_provider,
            "answer_provider": settings.answer_provider,
            "session_store": settings.session_store_backend,
            "auth_required": bool(settings.app_auth_token),
        }

    return app


app = create_app()
