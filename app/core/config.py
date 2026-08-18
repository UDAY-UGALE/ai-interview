from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


AnswerProvider = Literal["groq", "openai", "anthropic", "deepseek"]
VadBackend = Literal["energy", "webrtc"]
SessionStoreBackend = Literal["memory", "redis"]


class Settings(BaseSettings):
    app_name: str = "InterviewCopilot"

    groq_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    deepseek_api_key: str | None = None

    # "groq" needs no local resources -- the server just makes API calls, so
    # it's the right choice for anything you deploy (cloud VM, shared server,
    # etc). "faster_whisper" runs Whisper fully locally via CTranslate2 --
    # zero network latency, but the machine running the backend must have
    # the RAM/CPU/storage to hold the model (roughly 1-2GB), and it competes
    # for that machine's own resources under concurrent use. Treat it as a
    # local/dev-machine-only option, not something to flip on for a shared
    # deployment unless you've actually sized the server for it.
    stt_provider: Literal["groq", "faster_whisper"] = "groq"
    # Reverted to the standard turbo model -- distil-whisper was faster but
    # measurably worse at actually hearing questions correctly, and getting
    # the question right matters more than shaving a few hundred ms.
    stt_model: str = "whisper-large-v3-turbo"
    stt_language: str | None = "en"
    stt_timeout_seconds: int = Field(
        default=12,
        gt=0,
        description=(
            "Raised from 8 -- STT timeouts were showing up in bursts whenever the "
            "'Analyze Screen' screenshot upload competed for uplink bandwidth with "
            "live audio going to Groq's STT API (worse on a call where you're also "
            "screen-sharing). The screenshot itself is now downscaled+JPEG'd to be "
            "far smaller (see overlay_app.py _grab_screen_base64), which is the "
            "real fix; this is just extra slack for ordinary network jitter."
        ),
    )
    stt_race_providers: bool = Field(
        default=False,
        description=(
            "If true and a fallback STT provider is configured, call Groq and the "
            "fallback CONCURRENTLY and use whichever returns first, instead of only "
            "trying the fallback after Groq fails/times out. Cuts worst-case latency "
            "on a shaky connection at the cost of always paying for two STT calls."
        ),
    )
    stt_confidence_threshold: float = Field(
        default=0.55,
        ge=0,
        le=1,
        description=(
            "Below this rough confidence score (derived from Whisper's segment "
            "avg_logprob/no_speech_prob), a transcript is flagged as low_confidence "
            "so the overlay can show a warning instead of silently trusting it."
        ),
    )
    stt_no_speech_threshold: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description=(
            "Above this avg no_speech_prob, a transcript is dropped entirely rather "
            "than answered -- Whisper is known to hallucinate plausible-sounding text "
            "on silence/background noise instead of returning nothing."
        ),
    )
    stt_max_concurrent_segments: int = Field(
        default=3,
        gt=0,
        description=(
            "A long, continuous utterance gets force-cut into multiple VAD segments "
            "(segment_max_seconds) even without a pause. Each segment's STT call is "
            "kicked off as soon as it's captured instead of waiting for the PREVIOUS "
            "segment's call to finish first, up to this many overlapping in flight -- "
            "otherwise a long interviewer question transcribes strictly one segment at "
            "a time, and the latency stacks with every extra segment. Results are still "
            "delivered to the gate in original spoken order regardless of which "
            "segment's API call happens to finish first."
        ),
    )
    stt_prompt: str = (
        "Technical interview about software engineering. Correct terms include: "
        "React 19, React Server Components, Next.js 15, TypeScript, JavaScript, Node.js, "
        "Python, FastAPI, Django, LangChain, LangGraph, RAG, LLM, GPT-4, GPT-5, Claude, "
        "Groq, Redis, PostgreSQL, MongoDB, ChromaDB, vector database, embeddings, Docker, "
        "Kubernetes, AWS, GCP, Azure, CI/CD, REST API, GraphQL, WebSocket, gRPC, Kafka, "
        "microservices, agentic AI, ERPNext, Frappe."
    )

    # Only used when stt_provider=faster_whisper (local-only, see note above).
    faster_whisper_model: str = "large-v3-turbo"
    faster_whisper_device: Literal["cpu", "cuda"] = "cpu"
    faster_whisper_compute_type: str = "int8"
    faster_whisper_cpu_threads: int = Field(
        default=0, ge=0, description="0 = let CTranslate2 pick automatically."
    )

    answer_provider: AnswerProvider = "groq"
    # llama-3.3-70b-versatile was deprecated and removed by Groq on
    # 2026-08-16 (llama-3.1-8b-instant, used below for fast_intent_model,
    # was removed the same day) -- both now 404 with "model_not_found".
    # openai/gpt-oss-120b is Groq's own recommended replacement.
    answer_model: str = "openai/gpt-oss-120b"
    # Hard ceiling backing up the prompt's own line-limit rules -- a prompt
    # instruction alone isn't 100% reliable, so this caps it physically too.
    # Raised from 150 -- open-ended background/project questions ('tell me
    # about a project you worked on') are explicitly allowed up to ~8-9
    # lines in the prompt (more than a quick technical Q&A), and 150 tokens
    # wasn't enough room for that, cutting those answers off mid-thought.
    # ~220 covers that case without inviting long essay-style answers back
    # in for ordinary quick Q&A, which the prompt itself keeps short.
    answer_max_tokens: int = Field(default=220, gt=0)
    # Slightly lower than before -- 0.6 gave good human-sounding variety but
    # also let word choice drift toward fancier vocabulary sometimes. 0.5 is
    # a middle ground: still varied phrasing, less likely to reach for an
    # unnecessarily complex word.
    answer_temperature: float = Field(default=0.5, ge=0, le=2)

    # Used only by the "Analyze Screen" button -- separate from
    # answer_provider/answer_model because that pair is picked from the
    # overlay's text-chat model dropdown and may point at a model with no
    # vision support at all (e.g. groq/llama-3.3-70b-versatile). Screen
    # analysis always uses this fixed provider/model instead, so it works
    # regardless of what's selected for regular Q&A.
    vision_provider: AnswerProvider = "openai"
    vision_model: str = "gpt-4o"
    # Deliberately much higher than answer_max_tokens (150) -- that limit is
    # tuned for short spoken interview answers. Screen analysis often needs
    # to actually solve something (fix code, work through an error, answer a
    # multi-part question), and if VISION_MODEL is a reasoning model (e.g.
    # Groq's qwen/qwen3.6-27b), its internal thinking tokens count against
    # this same budget even though they're hidden from the output -- so it
    # needs real headroom or the final answer gets cut off mid-way.
    vision_max_tokens: int = Field(default=1800, gt=0)
    vision_temperature: float = Field(default=0.4, ge=0, le=2)

    audio_sample_rate: int = 16000
    audio_frame_ms: int = 20
    vad_backend: VadBackend = "energy"
    vad_mode: int = Field(default=2, ge=0, le=3)
    vad_energy_threshold: int = Field(default=500, gt=0)
    segment_min_seconds: float = Field(default=0.8, gt=0)
    segment_max_seconds: float = Field(default=4.0, gt=0)
    segment_end_silence_ms: int = Field(default=300, gt=0)
    # Raised from 350 -- a real interviewer's ordinary micro-pauses (a breath,
    # a short "uh" between clauses) were long enough to clear the old
    # debounce and get scored as "done" before they'd actually finished
    # speaking.
    question_debounce_ms: int = Field(default=500, gt=0)
    question_max_wait_seconds: float = Field(
        default=6.0,
        gt=0,
        description=(
            "If the transcript still looks unfinished (mid-sentence pause) after the "
            "debounce window, keep waiting for more speech up to this many seconds "
            "total before answering with whatever was said."
        ),
    )
    question_soft_wait_seconds: float = Field(
        default=1.8,
        gt=0,
        description=(
            "A short transcript (<= question_soft_wait_word_limit words) that the gate "
            "already scores as answerable is still held back for up to this many "
            "seconds (instead of firing immediately) in case the interviewer was just "
            "pausing to think before continuing -- a real interviewer often goes quiet "
            "for a few seconds mid-question, and a short fragment ending on a real word "
            "looks 'complete' to the gate even when it isn't. Not gated on '?' -- "
            "live speech-to-text frequently omits it even on genuine questions, so "
            "punctuation isn't a reliable 'this is really finished' signal. Longer "
            "transcripts and reason='follow_up' (deliberately short, e.g. bare 'why?') "
            "skip this and answer immediately -- this grace period only matters for the "
            "short/ambiguous case."
        ),
    )
    question_soft_wait_word_limit: int = Field(default=7, gt=0)

    # Second-tier fallback for the gate: the rule gate (regex/word-list based)
    # is a zero-latency first pass that catches the obvious cases (a "?", a
    # clear question-word prefix, a known follow-up phrase). Anything it
    # can't confidently classify either way used to just get silently dropped
    # (reason="no_question_signal") -- which is exactly what caused missed
    # answers for phrasings we hadn't hand-coded a pattern for ("df vs du",
    # scenario questions, etc), and would keep happening for every new
    # phrasing we haven't seen yet. This tier calls a small, fast LLM ONLY
    # for that ambiguous bucket (never for the obvious cases the rule gate
    # already resolves), to make a real semantic ANSWER/WAIT/IGNORE judgment
    # instead of a hardcoded pattern match. Adds one extra fast LLM call, but
    # only for the minority of utterances the rule gate can't already decide.
    fast_intent_enabled: bool = True
    # llama-3.1-8b-instant was deprecated and removed by Groq on 2026-08-16
    # (404 model_not_found) -- openai/gpt-oss-20b is Groq's recommended
    # replacement for it.
    fast_intent_model: str = "openai/gpt-oss-20b"

    session_store_backend: SessionStoreBackend = "memory"
    redis_url: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
