import json
import logging
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.candidate_context import ProjectContext


try:
    from redis.asyncio import Redis
except ImportError:
    Redis = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InterviewSessionContext:
    """Everything one session knows about the candidate.

    resume_text / job_description / notes are the original three. The rest
    are the candidate's own account of their work -- structured projects,
    free-form experience notes, and behavioural stories -- which is what
    lets an answer be about what they actually built instead of a plausible
    invention. See app/services/candidate_context.py for how they are
    rendered into the prompt.
    """

    resume_text: str = ""
    job_description: str = ""
    notes: str = ""
    # Structured, one entry per project: role, architecture, decisions,
    # trade-offs, metrics. Entered in the overlay's project dialog.
    projects: tuple[ProjectContext, ...] = ()
    # The paste-or-upload path: project notes, technical documentation,
    # achievements, skills -- whatever did not fit the form.
    experience_notes: str = ""
    # Difficult bug, production incident, disagreement, failure and lesson
    # -- kept separate because behavioural questions retrieve from a
    # different place than technical ones.
    interview_stories: str = ""
    answer_provider: str | None = None
    answer_model: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InterviewSessionContext":
        """Rebuild from stored JSON, tolerantly.

        Two things this has to survive, both real: a context written before
        these fields existed (missing keys -> defaults), and nested project
        dicts that json gives back as plain dicts rather than dataclasses.
        A session that fails to deserialize is an interview that cannot
        start, so unknown keys are dropped rather than raising.
        """
        known = {field.name for field in fields(cls)}
        payload = {key: value for key, value in (data or {}).items() if key in known}
        raw_projects = payload.pop("projects", None) or ()
        payload["projects"] = tuple(ProjectContext.from_dict(item) for item in raw_projects)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    question: str
    answer: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class SessionStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._redis: Redis | None = None
        self._redis_failed = False
        self._contexts: dict[str, InterviewSessionContext] = {}
        self._history: dict[str, list[ConversationTurn]] = {}

    async def set_context(self, session_id: str, context: InterviewSessionContext) -> None:
        self._contexts[session_id] = context
        redis = await self._get_redis()
        if redis:
            await redis.set(self._context_key(session_id), json.dumps(context.to_dict()))

    async def get_context(self, session_id: str) -> InterviewSessionContext:
        redis = await self._get_redis()
        if redis:
            raw = await redis.get(self._context_key(session_id))
            if raw:
                try:
                    return InterviewSessionContext.from_dict(json.loads(raw))
                except Exception:
                    # A context we cannot parse must not take the session
                    # down with it -- fall through to the in-process copy,
                    # which for the writing instance is the same object.
                    logger.exception("Could not decode stored session context")
        return self._contexts.get(session_id, InterviewSessionContext())

    async def add_history(self, session_id: str, question: str, answer: str) -> None:
        turn = ConversationTurn(
            question=question,
            answer=answer,
            created_at=datetime.now(UTC).isoformat(),
        )
        history = self._history.setdefault(session_id, [])
        history.append(turn)
        del history[:-8]

        redis = await self._get_redis()
        if redis:
            key = self._history_key(session_id)
            await redis.rpush(key, json.dumps(turn.to_dict()))
            await redis.ltrim(key, -8, -1)

    async def get_history(self, session_id: str) -> list[ConversationTurn]:
        redis = await self._get_redis()
        if redis:
            rows = await redis.lrange(self._history_key(session_id), 0, -1)
            return [ConversationTurn(**json.loads(row)) for row in rows]
        return list(self._history.get(session_id, []))

    async def _get_redis(self) -> Redis | None:
        if self._settings.session_store_backend != "redis" or self._redis_failed:
            return None
        if Redis is None:
            logger.warning("redis package is not installed; using in-memory session state")
            self._redis_failed = True
            return None
        if self._redis is None:
            self._redis = Redis.from_url(self._settings.redis_url, decode_responses=True)
            try:
                await self._redis.ping()
            except Exception:
                logger.warning("Redis is unavailable; using in-memory session state")
                self._redis_failed = True
                self._redis = None
        return self._redis

    @staticmethod
    def _context_key(session_id: str) -> str:
        return f"interview-copilot:session:{session_id}:context"

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"interview-copilot:session:{session_id}:history"


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore(get_settings())
