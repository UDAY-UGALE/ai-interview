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
    # "nvidia" covers both Riva/ASR-NIM deployments -- see nvidia_stt_mode.
    stt_provider: Literal["groq", "openai", "faster_whisper", "nvidia"] = "groq"
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
            "If true and a fallback STT provider is configured, call the primary and "
            "the fallback CONCURRENTLY and use whichever returns first, instead of "
            "only trying the fallback after the primary fails/times out. Cuts "
            "worst-case latency on a shaky connection, but DOUBLES the request rate "
            "against both providers -- a fast way to hit a per-minute rate limit."
        ),
    )
    stt_fallback_enabled: bool = Field(
        default=True,
        description=(
            "Retry a failed transcription against OpenAI Whisper when OPENAI_API_KEY "
            "is set. Only fires on an actual failure, so it costs nothing while the "
            "primary provider is healthy."
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
            "on silence/background noise instead of returning nothing. Only applied "
            "when the provider actually reported the number."
        ),
    )
    stt_drop_confidence_threshold: float = Field(
        default=0.35,
        ge=0,
        le=1,
        description=(
            "Below this confidence a transcript is DROPPED rather than merely flagged. "
            "Distinct from stt_confidence_threshold, which only marks a transcript as "
            "uncertain in the overlay: this is the level at which the text is judged "
            "too unreliable to treat as something the interviewer said at all. Only "
            "applied when the provider reported a real confidence score."
        ),
    )
    stt_max_concurrent_segments: int = Field(
        default=3,
        gt=0,
        description=(
            "A long, continuous utterance gets force-cut into multiple VAD segments "
            "(segment_max_seconds) even without a pause. Each segment's STT call is "
            "kicked off as soon as it's captured instead of waiting for the PREVIOUS "
            "segment's call to finish, up to this many overlapping in flight. Results "
            "are delivered to the gate in original spoken order, and each one is "
            "delivered the moment IT is ready -- never held back waiting for later "
            "segments to exist."
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

    # ---- NVIDIA STT (stt_provider=nvidia) --------------------------------
    # "riva" talks gRPC to a Riva ASR server or ASR NIM container -- that is
    # the normal way to run NVIDIA's own models (Parakeet, Conformer),
    # whether self-hosted or as a hosted NVCF function. "nim" talks HTTP to
    # an OpenAI-compatible /v1/audio/transcriptions endpoint, which recent
    # NIM builds expose and which needs no extra client library.
    nvidia_stt_mode: Literal["riva", "nim"] = "riva"
    nvidia_api_key: str | None = None
    # Riva model name. Empty means "let the server pick its default", which
    # is what a single-model NIM container wants.
    nvidia_stt_model: str = ""
    nvidia_stt_language: str = "en-US"
    nvidia_riva_server: str = "localhost:50051"
    # Set when calling an NVIDIA-hosted NVCF function instead of your own
    # server (server=grpc.nvcf.nvidia.com:443, plus the function id).
    nvidia_riva_function_id: str | None = None
    nvidia_riva_use_ssl: bool = False
    # Base URL for nvidia_stt_mode=nim, e.g. http://localhost:9000/v1
    nvidia_stt_base_url: str | None = None

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
    # This budget is shared with the model's HIDDEN reasoning tokens, which
    # is easy to miss because they never appear in the output. Measured on
    # openai/gpt-oss-120b with a five-sentence scenario question: at 150 the
    # model used the whole budget thinking and returned an empty answer, and
    # at 220 it returned a fragment. The prompt is what keeps answers short
    # (5-6 lines for ordinary Q&A); this only has to be large enough that
    # thinking cannot crowd the answer out. Lower it only if you have
    # switched to a non-reasoning model.
    answer_max_tokens: int = Field(default=500, gt=0)
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
    vad_energy_threshold: int = Field(
        default=500,
        gt=0,
        description=(
            "FLOOR for the speech trigger level, not the trigger level itself (see "
            "vad_adaptive_threshold). Raise it if very quiet background noise still "
            "opens segments; lower it if genuinely quiet speech is missed."
        ),
    )
    vad_adaptive_threshold: bool = Field(
        default=True,
        description=(
            "Track the ambient noise floor and require speech to stand a margin above "
            "THAT, instead of trusting one fixed number. System-loopback audio from a "
            "video call carries continuous codec comfort noise whose level moves "
            "around; against a fixed threshold that either triggers constantly (which "
            "is what produced a transcription request every couple of seconds, and "
            "with it the 429s and the invented transcripts) or misses quiet speech."
        ),
    )
    vad_onset_frames: int = Field(
        default=3,
        gt=0,
        description=(
            "Consecutive voiced frames needed to open a segment. One loud frame is a "
            "click or a keystroke, not speech."
        ),
    )
    vad_preroll_ms: int = Field(
        default=300,
        ge=0,
        description=(
            "Audio kept from just BEFORE the trigger, which is where the first "
            "consonant of the first word lives. Too little and questions arrive with "
            "their opening word clipped ('ell me about yourself')."
        ),
    )
    segment_min_seconds: float = Field(default=0.4, gt=0)
    segment_max_seconds: float = Field(
        default=8.0,
        gt=0,
        description=(
            "Hard cut for one continuous utterance. Raised from 4s: a cut mid-sentence "
            "costs a transcription request, damages the words either side of it, and "
            "makes one question arrive as several transcripts. Segments now carry an "
            "utterance id so the pieces are reassembled correctly, but fewer cuts is "
            "still better."
        ),
    )
    segment_end_silence_ms: int = Field(
        default=420,
        gt=0,
        description=(
            "Silence that ends an utterance. This is the single biggest fixed cost in "
            "the whole latency budget -- it is paid on every question -- so it should "
            "be just long enough not to trip on a mid-sentence breath."
        ),
    )
    segment_min_speech_ms: int = Field(
        default=320,
        gt=0,
        description=(
            "Minimum VOICED audio a segment must contain to be worth transcribing. "
            "The most effective filter in the pipeline: without it, every click and "
            "noise blip became a transcription request, and a request made up mostly "
            "of silence comes back as invented text ('Thank you.', 'Tch.', '.') at "
            "full apparent confidence."
        ),
    )
    segment_carryover_ms: int = Field(
        default=200,
        ge=0,
        description=(
            "Audio replayed from the end of a force-cut segment into the start of the "
            "next one, so a word split across the cut survives intact in one of them."
        ),
    )
    # Short, because it is no longer doing the work it used to. The VAD now
    # reports whether an utterance actually ended, so this is just a small
    # coalescing window for transcripts of the same utterance arriving back
    # to back -- not a guess about whether the interviewer is done talking.
    question_debounce_ms: int = Field(default=180, gt=0)
    question_max_wait_seconds: float = Field(
        default=4.0,
        gt=0,
        description=(
            "Absolute cap on how long one buffer can be held before it is acted on, "
            "however unfinished it still looks. A safety net, not a normal path."
        ),
    )
    question_soft_wait_seconds: float = Field(
        default=1.5,
        gt=0,
        description=(
            "Extra grace given ONLY to an answerable transcript that trails off "
            "mid-sentence ('what is the difference between'), measured from the last "
            "fragment rather than from the start of the buffer. A transcript that "
            "looks finished is no longer delayed at all -- holding every short "
            "question 'just in case' put ~1.8s in front of exactly the questions that "
            "were already unambiguous."
        ),
    )
    question_soft_wait_word_limit: int = Field(
        default=7,
        gt=0,
        description=(
            "Word count past which an answerable transcript is assumed to be a whole "
            "question even without terminal punctuation. Only consulted for providers "
            "that return unpunctuated text."
        ),
    )
    utterance_merge_gap_seconds: float = Field(
        default=4.0,
        gt=0,
        description=(
            "How long a not-yet-answerable buffer waits for a continuation before "
            "being discarded, counted from the moment the interviewer stops talking. "
            "This is what stops unrelated speech from accumulating: fragments that "
            "nothing follows are dropped instead of surviving to be glued onto the "
            "next real question. "
            "It must comfortably exceed the transcription round trip, because the "
            "words of the sentence that just ended are still in flight when the "
            "window opens -- at 2s a provider having a slow moment (>2s) would see "
            "the setup of a scenario question discarded a fraction of a second "
            "before it arrived. Raising it costs no answer latency at all: an "
            "answerable buffer is acted on immediately regardless, so this only "
            "delays discarding something that was never going to be answered. Keep "
            "question_max_wait_seconds above this value, since that cap is checked "
            "first."
        ),
    )

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
    fast_intent_timeout_seconds: float = Field(
        default=1.2,
        gt=0,
        description=(
            "Hard cap on the tier-2 classifier call. It sits directly between the "
            "interviewer finishing and the answer starting, so a slow response has to "
            "degrade to 'no signal' rather than hold the answer back."
        ),
    )

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
