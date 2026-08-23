# Architecture

What the system actually does, why the client has to be desktop software,
and what changes when it moves between clouds.

## 1. The lifecycle, traced through the code

Every step below is a real file, in the order control actually flows.

```
 ┌─ CLIENT PC (Windows) ────────────────────────────────────────────────┐
 │                                                                      │
 │  client/test_loopback_stream.py                                      │
 │    soundcard WASAPI *loopback* on the default OUTPUT device --       │
 │    i.e. what the speakers are playing, which during a call is the    │
 │    other person's voice. Resampled to 16kHz mono, converted to       │
 │    PCM16, sliced into 20ms frames.                                   │
 │      │                                                               │
 └──────┼───────────────────────────────────────────────────────────────┘
        │  binary websocket frames, ~32 KB/s
        ▼  wss://…/ws/audio?session_id=…&token=…
 ┌─ BACKEND ────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  app/routes/audio_ws.py                                              │
 │    ├─ app/core/auth.py            token checked BEFORE accept()      │
 │    ├─ app/services/vad.py         SpeechSegmenter: adaptive-threshold│
 │    │    energy VAD tracking the call's own noise floor. Emits        │
 │    │    AudioSegments, and groups them into *utterances* so a pause  │
 │    │    mid-sentence doesn't split one question into two.            │
 │    │                                                                 │
 │    ├─ two decoupled stages, on purpose:                              │
 │    │    _transcription_dispatcher  starts an STT call the moment a   │
 │    │      segment exists, bounded by a semaphore                     │
 │    │    _result_consumer           delivers each result the moment   │
 │    │      THAT result is ready, in spoken order                      │
 │    │                                                                 │
 │    ├─ app/services/stt/registry.py                                   │
 │    │    build_stt_service() picks the provider; composite.py adds    │
 │    │    fallback or racing. Routes never name a vendor.              │
 │    │      → Groq whisper-large-v3-turbo (default)                    │
 │    │                                                                 │
 │    └─ app/services/transcript_quality.py                             │
 │         drops transcripts that are noise rather than speech          │
 │              │                                                       │
 │              ▼                                                       │
 │  app/services/question_gate.py — QuestionAnswerPipeline              │
 │    submit_transcript()                                               │
 │      ├─ ~180ms debounce, coalescing transcripts that belong together │
 │      ├─ _clean_transcript, _split_clauses                            │
 │      ├─ _classify_question   rule tier: question marks, interview    │
 │      │    intent patterns, terse/garbled detection, trailing-off     │
 │      ├─ optional langgraph StateGraph (falls back to direct calls)   │
 │      ├─ _run_fast_intent_classifier  tier 2: small LLM, hard 1.2s    │
 │      │    cap -- it sits between "they stopped talking" and "answer  │
 │      │    starts", so a slow reply must degrade, not block           │
 │      ├─ _should_keep_waiting  holds only when something concrete     │
 │      │    says more is coming (open utterance, or text trailing off) │
 │      └─ _resolve_followup    resolves "and what about that?" against │
 │           session history                                            │
 │              │                                                       │
 │              ▼                                                       │
 │  app/services/llm.py                                                 │
 │    StreamingLLMClient Protocol + per-vendor adapters                 │
 │    (Groq / OpenAI / Anthropic / DeepSeek). build_llm_client()        │
 │    resolves provider+model from session context, then env default.   │
 │              │                                                       │
 │              ▼  token by token                                       │
 │  app/services/answer_hub.py                                          │
 │    broadcast_json(session_id, …) → every socket on that session      │
 │    optional JSONL transcript to disk (session_log_enabled)           │
 └──────┼───────────────────────────────────────────────────────────────┘
        │  wss://…/ws/answers?session_id=…&token=…
        ▼
 ┌─ CLIENT PC ──────────────────────────────────────────────────────────┐
 │  client/overlay_app.py                                               │
 │    frameless, always-on-top, Qt.Tool (no taskbar, no alt-tab)        │
 │    SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)            │
 │    progressive reveal as tokens arrive; Alt+←/→ browses history      │
 └──────────────────────────────────────────────────────────────────────┘
```

Side channels, all HTTP, all on the same session id:

| Route | Purpose |
| --- | --- |
| `POST /session` | partial-merge résumé / JD / notes / model choice |
| `POST /session/upload` | PDF résumé or JD, text extracted with pypdf |
| `POST /ask` | type a question when STT misheard; skips STT and the gate |
| `POST /analyze-screen` | base64 screenshot + optional question → vision model, streamed back through `/ws/answers` so the overlay needs no special case |
| `GET /session/{id}` | current context + history |
| `GET /models` | catalog for the overlay's model picker |
| `GET /health` | liveness, unauthenticated |
| `GET /version` | build + configuration shape, no secrets |

### State

`app/core/redis_client.py` holds session context (résumé, JD, notes, model
choice) and recent conversation turns behind a `SessionStore` with two
backends: in-process dict, or Redis. Redis failures degrade to memory rather
than erroring.

**This is the one thing that constrains scaling.** With the memory backend,
the session store *and* the answer hub live inside one process. A client's
audio socket and its answer socket must therefore land on the same process:
one instance, one worker. Redis fixes the store but not the hub — a second
instance also needs the hub to fan out over pub/sub. Until then, scale
vertically.

## 2. Why the client cannot be a browser

This was evaluated against the actual requirements, not preference. Three
blockers, each verified in the code:

**Screen-capture exclusion.** [`overlay_app.py`](client/overlay_app.py)
calls `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)`. There is no
web equivalent — not in any browser, not behind a flag. A web page cannot
influence how the compositor treats the window it happens to be rendered in.

**System-audio capture.** The pipeline needs the *other* participant's voice,
which reaches the machine as speaker output from a separate desktop
application (Zoom, Teams, Meet). The browser's closest primitive is
`getDisplayMedia({audio: true})`, which is Chromium-on-Windows only, captures
*tab* audio rather than system audio, requires picking a source through a
permission dialog on every session, and shows a persistent sharing indicator.
It cannot reach another application's audio at all.

**Floating over other applications.** The overlay sits above the video call
window. A browser tab is confined to the browser's own window; even a PWA in
standalone mode cannot pin itself above arbitrary applications, and
`Qt.Tool`'s absence from the taskbar and alt-tab list has no web analogue.

Screenshot capture for `/analyze-screen` is a fourth, weaker point: possible
in a browser but only through a per-use permission prompt.

**Decision: hybrid.** Desktop client (the existing PySide6 app, unchanged) +
a static website for distribution and documentation. No browser client is
planned, because the browser cannot deliver the product's defining behaviour.

## 3. Deployment shape

```
                      PRODUCT
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
     WEBSITE          CLIENT           BACKEND
   (static HTML)   (Windows .exe)     (container)
        │                │                │
        │                │                ├── Render      (now, free)
        │                │                ├── AWS         (later)
        │                │                └── Azure       (alternative)
        │                │
        │                ├── loopback audio capture
        │                ├── overlay + capture exclusion
        │                └── screenshot capture
        │
        └── landing / download / docs / legal
```

The website is a separate deployment on purpose: it must keep serving the
landing page and the download link while the backend is asleep, broken, or
redeploying. It makes no request to the backend.

## 4. What makes it portable

Portability here is not a layer — it is the absence of provider assumptions.

| Concern | How it stays portable |
| --- | --- |
| Runtime | One Dockerfile. `python:3.12-slim`, no base image from any cloud vendor. |
| Port | Binds `0.0.0.0:${PORT:-8000}`. Render injects `PORT`; ECS and Container Apps use the default; local uses 8000. |
| Configuration | Every setting is an env var read by `app/core/config.py`. No config file is required at runtime. |
| Secrets | Env vars only. Never in the image, never in git, never sent to a client. |
| Filesystem | The only write is session logs, behind `SESSION_LOG_ENABLED` / `SESSION_LOG_DIR`. The app runs correctly on a read-only filesystem with logging off. |
| State | Behind `SessionStore`. Memory today; Redis is a config change, not a rewrite. |
| Health | `GET /health` — unauthenticated, dependency-free, works as a Render health check, an ALB target-group check, or a Container Apps probe unchanged. |
| Vendors | STT and LLM are behind `registry.py` and `llm.py`. Swapping Groq for Bedrock or Azure OpenAI is a new adapter, not a refactor. |
| Client | Server address resolved at runtime by `client/config.py`. One build works against localhost, Render, and AWS. |

The only provider-specific file in the repository is `render.yaml`, and it
is a deployment manifest — it declares nothing the application reads.

## 5. Cloud mapping

Same image everywhere. What changes is the manifest around it.

| Concern | Render (now) | AWS (later) | Azure (alternative) |
| --- | --- | --- | --- |
| Compute | Web Service, Docker, free | ECS Fargate, or App Runner | Container Apps |
| Image registry | built by Render from the repo | ECR | ACR |
| TLS + domain | automatic | ALB + ACM, or CloudFront | built in |
| Websockets | supported | ALB (set idle timeout > interview length) | supported |
| Secrets | dashboard env vars | Secrets Manager / SSM Parameter Store | Key Vault |
| Website | Static Site | S3 + CloudFront | Static Web Apps |
| Client downloads | not on the backend | S3 + CloudFront | Blob Storage + CDN |
| Logs | stdout | CloudWatch Logs | Log Analytics |
| Session store when scaling out | Key Value | ElastiCache | Azure Cache for Redis |

Two things to watch on any provider:

- **Websocket idle timeout.** An interview holds one connection open for an
  hour. AWS ALB defaults to 60 seconds of idle timeout; the Dockerfile's
  `--ws-ping-interval 20` keeps traffic flowing, but raise the timeout too.
- **Request-duration limits.** Anything that caps request duration (API
  Gateway's 29s, some serverless products) is incompatible with `/ws/audio`.
  Use a container product, not a function product.

## 6. Latency

Measured budget, backend on localhost:

| Stage | Cost |
| --- | --- |
| capture gulp | ~100ms |
| end-of-speech detection (`SEGMENT_END_SILENCE_MS`) | 420ms |
| transcription round trip | ~400–800ms |
| transcript coalescing (`QUESTION_DEBOUNCE_MS`) | 180ms |
| LLM time-to-first-token | ~300–900ms |
| **total** | **~1.4–2.2s** |

Deploying adds the client↔backend round trip **twice** — audio up, tokens
back down — plus TLS. Roughly +150–200ms from India to Singapore, +40–60ms
to Mumbai. The provider calls are unaffected: they always crossed the
internet, they just leave from the server now.

### Why VAD stays on the server

An obvious-looking optimisation is to move voice detection into the client
and upload only speech. It is not worth it here:

- **No latency win.** Audio still has to reach the server for STT. Detecting
  the end of speech client-side saves the transit time of the final frames —
  tens of milliseconds against a 420ms detection window.
- **Real coupling.** `SpeechSegmenter` does not just detect speech; it groups
  segments into utterances, and `question_gate` uses open-utterance state to
  decide whether to hold an answer back. Splitting the two puts that
  state on the wrong side of the network.
- **A second implementation to keep correct.** The adaptive threshold was
  tuned against real call audio. Reimplementing it client-side means two
  behaviours to keep in sync, and the offline test suite could no longer
  exercise the whole path in one process.

The genuine win would be bandwidth (~115 MB/hour), which nothing currently
constrains. Revisit if clients start running on metered connections.
