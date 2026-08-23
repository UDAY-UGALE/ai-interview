import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.core.config import get_settings


logger = logging.getLogger(__name__)

# Default only. The real path comes from settings at call time, because a
# module constant computed from __file__ assumes the process owns its own
# source tree -- true on a laptop, false in a container with a read-only
# image and a mounted volume somewhere else entirely.
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def _log_dir() -> Path:
    configured = get_settings().session_log_dir
    return Path(configured).expanduser() if configured else _DEFAULT_LOG_DIR

# "answer_token" is skipped -- the full answer text already arrives as one
# piece in "answer_done", so logging every individual streamed token would
# just bloat the file without adding anything useful for a post-interview
# review.
_SKIP_LOG_TYPES = {"answer_token"}

# session_id arrives from a query parameter / request body and is used to
# build a filename, so it cannot be trusted to stay inside the log directory
# ("../../x" would not). Anything outside this set is replaced rather than
# rejected, so an odd session id still gets logged somewhere sane instead of
# losing the record.
_UNSAFE_SESSION_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_log_name(session_id: str) -> str:
    cleaned = _UNSAFE_SESSION_CHARS.sub("_", session_id).strip("._")[:64]
    return cleaned or "default"


class AnswerHub:
    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections[session_id].add(websocket)

    async def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections[session_id].discard(websocket)
            if not self._connections[session_id]:
                del self._connections[session_id]

    async def broadcast_json(self, session_id: str, payload: dict) -> None:
        self._log_event(session_id, payload)

        async with self._lock:
            connections = list(self._connections.get(session_id, set()))

        stale: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                stale.append(websocket)

        if stale:
            async with self._lock:
                for websocket in stale:
                    self._connections[session_id].discard(websocket)
                if not self._connections[session_id]:
                    self._connections.pop(session_id, None)

    def _log_event(self, session_id: str, payload: dict) -> None:
        """Append every meaningful event to a permanent, per-session,
        per-day log file. This is the only DURABLE record of an interview --
        the overlay and terminal are both transient. It captures every heard
        transcript, every gate decision (including silent skips, with the
        reason why), every answer, and every error, so the whole session can
        be reviewed after the fact instead of relying on having watched it
        live."""
        event_type = payload.get("type")
        if event_type in _SKIP_LOG_TYPES:
            return

        settings = get_settings()
        if not settings.session_log_enabled:
            return

        try:
            log_dir = _log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_path = log_dir / f"{_safe_log_name(session_id)}_{date_str}.jsonl"
            entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to write session log entry")


answer_hub = AnswerHub()
