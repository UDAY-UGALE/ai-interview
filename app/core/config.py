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
    stt_provider: Literal["groq", "openai", "faster_whisper", "nvidia", "deepgram"] = "groq"
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

    # ---- Deepgram (stt_provider=deepgram) --------------------------------
    # Deepgram is wired behind the same STTService protocol as every other
    # backend, so nothing outside app/services/stt has to know it exists.
    # Two independent things live here:
    #
    #   * the BATCH path (transcribe_pcm16), which is a drop-in replacement
    #     for the Whisper call the VAD-segment pipeline already makes. This
    #     is what STT_PROVIDER=deepgram switches on, and it changes nothing
    #     else about the pipeline.
    #   * the STREAMING path (deepgram_streaming=true), which additionally
    #     opens a live socket and uses Deepgram's own endpointing. It is OFF
    #     by default and stays off until the Whisper-vs-Deepgram comparison
    #     in client/test_stt_comparison.py has actually been run -- there is
    #     no measured evidence for it yet.
    deepgram_api_key: str | None = None
    # nova-3 is the current general model and the one this integration was
    # written against; nova-2 and enhanced are accepted for A/B testing.
    deepgram_model: str = "nova-3"
    deepgram_language: str = "en"
    # Deepgram's own accent/dialect hint. "en" lets the model decide, which
    # is what we want for Indian English -- there is no en-IN model, and
    # pinning en-US measurably hurt nothing but is not the default here
    # because it removes the model's freedom to adapt.
    deepgram_smart_format: bool = Field(
        default=True,
        description=(
            "Deepgram's punctuation/number/acronym formatting. ON because the "
            "question gate depends heavily on terminal punctuation ('?' vs '.') to "
            "decide whether a question is finished -- see _looks_finished."
        ),
    )
    deepgram_punctuate: bool = True
    deepgram_timeout_seconds: float = Field(default=12.0, gt=0)
    deepgram_keyterm_limit: int = Field(
        default=100,
        gt=0,
        description=(
            "How many session vocabulary terms are sent as Deepgram keyterms. "
            "Unlike Whisper's ~850-BYTE prompt (which silently truncated the whole "
            "resume away once a JD was loaded -- see session_vocabulary.py), "
            "keyterms are a term LIST, so the budget is counted in terms rather "
            "than characters and session-specific terms are never crowded out by "
            "generic prose."
        ),
    )

    deepgram_streaming: bool = Field(
        default=False,
        description=(
            "Open a live Deepgram socket alongside the existing VAD instead of only "
            "transcribing closed segments. OFF by default: it is implemented and "
            "wired, but switching the live path to it before the measured comparison "
            "exists would be exactly the unevidenced migration this work is meant to "
            "avoid. Requires DEEPGRAM_API_KEY."
        ),
    )
    deepgram_interim_results: bool = Field(
        default=True,
        description=(
            "Emit interim (non-final) hypotheses on the streaming path. They are "
            "broadcast to the overlay as low-confidence 'heard' text ONLY -- they "
            "never reach the question gate, because acting on a hypothesis that can "
            "still change is how a half-question gets answered."
        ),
    )
    deepgram_endpointing_ms: int = Field(
        default=400,
        ge=0,
        description=(
            "Deepgram's own end-of-speech detection, in ms. Deliberately close to "
            "segment_end_silence_ms (420) so the streaming path and the VAD path "
            "agree about when a question ended and the latency budget is unchanged."
        ),
    )
    deepgram_utterance_end_ms: int = Field(
        default=1000,
        ge=0,
        description=(
            "How long Deepgram waits before declaring UtteranceEnd. Used only as the "
            "'nothing more is coming' signal, the same role take_closed_utterances() "
            "plays on the VAD path."
        ),
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
    # (see the word budget below); this only has to be large enough that
    # thinking cannot crowd the answer out. Lower it only if you have
    # switched to a non-reasoning model.
    #
    # The prompt's own budget is 40 words for a single-part question and 90
    # across 2-4 bullets for anything larger -- roughly 30 seconds spoken,
    # which is about as long as an interviewer listens before the answer
    # stops landing.
    answer_max_tokens: int = Field(default=500, gt=0)
    # 0.5 was tuned for human-sounding variety, but variety and technical
    # precision pull in opposite directions: the same sampling that varies
    # phrasing also lets the model reach past the exact term for a vaguer
    # near-synonym, which is most of what "the answer sounds generic" is.
    # 0.3 keeps enough variation that answers don't read as canned, while
    # making the specific -- the component name, the number, the failure
    # mode -- the likeliest next token rather than one of several.
    answer_temperature: float = Field(default=0.3, ge=0, le=2)

    # Used only by the "Analyze Screen" button -- separate from
    # answer_provider/answer_model because that pair is picked from the
    # overlay's text-chat model dropdown and may point at a model with no
    # vision support at all (e.g. groq/llama-3.3-70b-versatile). Screen
    # analysis always uses this fixed provider/model instead, so it works
    # regardless of what's selected for regular Q&A.
    vision_provider: AnswerProvider = "openai"
    vision_model: str = "gpt-4o"
    # Deliberately much higher than answer_max_tokens -- that limit is
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
        default=150,
        gt=0,
        description=(
            "FLOOR for the speech trigger level, not the trigger level itself (see "
            "vad_adaptive_threshold). Raise it if very quiet background noise still "
            "opens segments; lower it if genuinely quiet speech is missed. "
            "Lowered from 500. The trigger is max(this, noise*2.5), so on a QUIET "
            "line -- measured noise floor around 30 RMS -- the adaptive half was "
            "fully bypassed and the trigger sat at 500 regardless. Any speech below "
            "500 RMS (-36 dBFS) then produced no segment, no STT request, no "
            "transcript and no log line at all: silent, total deafness, which is "
            "what 'it did not hear me' looks like from outside. Measured on the "
            "33-stream corpus, recall 95.7% -> 100%. This value is ONLY safe "
            "together with vad_calibration_ms: lowering it alone took false "
            "triggers from 50% to 60-70%, because the two defects are independent."
        ),
    )
    vad_calibration_ms: int = Field(
        default=1000,
        ge=0,
        description=(
            "How long the detector MEASURES the line at the start of a connection "
            "before it may report speech at all.\n"
            "The noise floor is estimated only from non-speech frames, so a detector "
            "that latches on its first frames never gets to measure: on a line "
            "noisier than the seeded estimate every frame then reads as speech, "
            "indefinitely, and the 30s escape hatch is far too slow to help. "
            "Measured on a noise-only corpus that was 5 of 10 streams producing "
            "transcription requests -- the source of the invented transcripts "
            "('Thank you.' x49, 'Ct.js.' x18) -- and 2 of 10 with this in place. "
            "Costs nothing in practice: the client connects long before anyone "
            "speaks, and 1000ms was the measured knee (2000ms starts missing speech "
            "that is already in progress when the socket opens). Set 0 to disable."
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
    question_settled_debounce_ms: int = Field(
        default=30,
        gt=0,
        description=(
            "The wait used INSTEAD of question_debounce_ms on the first evaluation "
            "of a buffer the VAD has already settled -- utterance closed, nobody "
            "talking -- i.e. when there is nothing left to coalesce with.\n"
            "The decision loop sleeps BEFORE it evaluates, so the full debounce was "
            "paid on every answer including ones that were already unambiguous. It "
            "is measurable in the session logs: since_last_fragment_ms is 185ms at "
            "p50 and 193ms at p90 over 241 real answers -- essentially the debounce "
            "and nothing else. This keeps a small window so two transcripts landing "
            "in the same instant still merge, and returns the rest. Anything the "
            "gate does not answer immediately falls back to the full debounce for "
            "every later cycle, so multi-part questions still coalesce normally. "
            "Also bounds barge-in responsiveness, which was debounce-limited."
        ),
    )
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

    # --- Session vocabulary -------------------------------------------
    # Replaces the old "paste the resume and JD into the recognizer prompt"
    # strategy. That was measured sending 1,114 bytes into an 850-byte cap,
    # so with any job description loaded the RESUME HALF WAS SILENTLY CUT --
    # i.e. the session's own project and tool names, the terms most likely
    # to be spoken and most likely to be misheard, got no biasing at all.
    # A compact TERM LIST fits the same budget many times over.
    session_vocabulary_enabled: bool = True
    session_vocabulary_max_terms: int = Field(
        default=120,
        gt=0,
        description=(
            "Cap on extracted terms. Session-specific terms (resume, JD, "
            "conversation history) are always kept ahead of the generic technical "
            "list when the cap bites."
        ),
    )

    # --- Transcript normalization -------------------------------------
    transcript_normalization_enabled: bool = Field(
        default=True,
        description=(
            "Repair known mis-transcriptions of technical terms (Rack->RAG, "
            "red is->Redis, doctor->Docker) BEFORE the question gate sees them.\n"
            "This is deliberately NOT a global find-and-replace. A substitution "
            "only happens when the canonical term is evidenced by this session "
            "(resume, JD, or conversation history) AND the heard form is not itself "
            "in that vocabulary -- so an interview about Ruby, where 'Rack' is a "
            "real framework the candidate lists, leaves 'Rack' alone. Every "
            "substitution is logged with its evidence."
        ),
    )
    transcript_normalization_log_only: bool = Field(
        default=False,
        description=(
            "Detect and log substitutions without applying them. Use this to "
            "measure what the normalizer WOULD do against a real interview before "
            "letting it change what reaches the LLM."
        ),
    )

    # --- Question completion ------------------------------------------
    continuation_window_ms: int = Field(
        default=350,
        ge=0,
        description=(
            "Extra grace given ONLY to a buffer whose SPEECH is unfinished -- it "
            "trails off ('can you explain...'), ends on an auxiliary+pronoun "
            "('what challenges did you'), or carries a correction marker.\n"
            "This is NOT a global delay and must never become one: a question that "
            "already looks finished is released with no wait at all, which is what "
            "protects the measured 182ms p50 from last spoken word to LLM call. It "
            "REPLACES the old behaviour for these buffers, which was to sit on them "
            "for question_soft_wait_seconds (1.5s) and then answer half a question."
        ),
    )
    supersede_window_seconds: float = Field(
        default=4.0,
        gt=0,
        description=(
            "How long after a question is answered a continuation can still arrive "
            "and MERGE with it instead of becoming a second answer.\n"
            "This is what fixes the measured 'one question, two answers' case: "
            "'Can you explain?' [pause] 'RAG.' fired twice at every pause length "
            "from 0.2s to 4.0s. A complete question still fires immediately -- the "
            "merge happens retroactively, cancelling the first answer, so latency "
            "is unchanged and correctness is recovered."
        ),
    )
    correction_handling_enabled: bool = Field(
        default=True,
        description=(
            "Treat 'actually', 'sorry, I mean', 'I said ...' as REPLACING the "
            "question just asked, cancelling its answer if one is streaming. "
            "Measured before this existed: 'Tell me about your Flask project. "
            "Actually, I mean my Django project.' answered Flask at every pause "
            "length tested and never answered Django."
        ),
    )

    session_store_backend: SessionStoreBackend = "memory"
    redis_url: str = "redis://localhost:6379/0"

    # --- Deployment / portability -------------------------------------
    # Everything below is infrastructure, not behaviour: it is what lets the
    # same image run on a laptop, Render, ECS or Container Apps without a
    # code change. Nothing here is provider-specific.

    app_version: str = Field(
        default="0.1.0",
        description=(
            "Reported by GET /version so you can tell which build is actually "
            "running after a deploy. Set it from CI (or the platform's commit SHA "
            "variable) if you want it to track releases automatically."
        ),
    )

    cors_allow_origins: str = Field(
        default="",
        description=(
            "Comma-separated origins allowed to call the HTTP API from a browser, "
            "e.g. 'https://interviewcopilot.app'. Empty (the default) installs no "
            "CORS middleware at all, which is correct for the desktop client -- it "
            "is not a browser and sends no Origin header. Only needed if a web page "
            "ever talks to this backend. '*' is rejected when an auth token is set."
        ),
    )

    session_log_enabled: bool = Field(
        default=True,
        description=(
            "Write per-session JSONL transcripts. Turn OFF on any deployment with a "
            "read-only or ephemeral filesystem where the files would be lost anyway "
            "(Render free, Fargate, Container Apps) -- the writes are pure overhead "
            "there, and on a read-only mount they are a caught exception per event."
        ),
    )
    session_log_dir: str | None = Field(
        default=None,
        description=(
            "Where those transcripts go. None means <repo>/logs, which is what you "
            "want locally. On a container set it to a mounted volume (or leave "
            "session_log_enabled off). Keeping it configurable is what stops the "
            "app from assuming it owns its own source tree."
        ),
    )

    app_auth_token: str | None = Field(
        default=None,
        description=(
            "Shared secret every client must present once the backend is reachable "
            "from the internet -- the audio/answer websockets spend real API credit, "
            "so an unauthenticated public URL is an open tab on your bill. Unset "
            "(the default) means no check at all, which is what you want when the "
            "server is bound to 127.0.0.1 on your own machine."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
