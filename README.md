# InterviewCopilot

A real-time interview copilot: it listens to the interviewer through system
audio, works out when a question has been asked, and streams an answer to an
overlay.

The pipeline, end to end:

- capture system (loopback) or microphone audio locally
- stream 20ms PCM frames to FastAPI over `/ws/audio`
- segment speech with an adaptive-threshold VAD that also groups segments
  into **utterances**
- transcribe each segment through a pluggable STT provider (Groq Whisper,
  OpenAI, a local Whisper, or an **NVIDIA** model)
- filter out transcripts that are noise rather than speech
- decide whether a real question was asked, and which part of what was heard
  the question actually is
- stream an LLM answer over `/ws/answers`

Answer providers are selectable through environment variables: `groq`,
`openai`, `anthropic`, or `deepseek`.

## Latency budget

From "interviewer stops talking" to "answer starts", with default tuning:

| stage | cost |
| --- | --- |
| capture gulp (`--record-chunk-ms`) | ~100ms |
| end-of-speech detection (`SEGMENT_END_SILENCE_MS`) | 420ms |
| transcription round trip | ~400-800ms |
| transcript coalescing (`QUESTION_DEBOUNCE_MS`) | 180ms |
| LLM time-to-first-token | ~300-900ms |
| **total** | **~1.4-2.2s** |

Measured live for a five-sentence scenario question: first word at 0.62s
after the LLM call starts, complete five-bullet answer in 1.11s.

`python client\test_pipeline_e2e.py` measures this end to end offline, with
no API keys and no microphone.

Two things to know before tuning any of it down further:

- `SEGMENT_END_SILENCE_MS` is the price of knowing the interviewer actually
  finished. Cut it too far and questions get answered half-spoken.
- A transcript is only held back when something concrete says more is
  coming: the VAD reporting an open utterance, or the text trailing off
  mid-sentence. Anything that looks finished is answered immediately.

### If answers suddenly take 20+ seconds

That is almost always the provider's **tokens-per-minute** limit, not the
pipeline. The SDK absorbs the 429 and waits for the window to reset, so it
looks like a slow answer rather than an error. Check it directly:

```powershell
python -c "import asyncio,os;from groq import AsyncGroq;c=AsyncGroq(api_key=os.environ['GROQ_API_KEY']);r=asyncio.run(c.chat.completions.with_raw_response.create(model='openai/gpt-oss-120b',messages=[{'role':'user','content':'hi'}],max_tokens=5));print({k:v for k,v in r.headers.items() if 'ratelimit' in k})"
```

Every question re-sends the system prompt, the resume/JD, and recent
history, so the per-question cost is what decides how many questions a
minute you get. Those are capped deliberately (`_RESUME_CHAR_LIMIT` and
friends in `question_gate.py`, and `ANSWER_MAX_TOKENS`) to keep one question
around 3k tokens. On an 8,000 TPM free tier that is roughly two to three
questions a minute -- enough for a normal interview, not for rapid-fire
questioning. Raising `ANSWER_MAX_TOKENS` or uncapping the resume/JD text
trades directly against that.

### Anthropic (Claude)

`ANSWER_PROVIDER=anthropic` with `claude-haiku-4-5` is the fast option and
what the overlay's model picker defaults to. Two things differ from the Groq
path:

- Haiku has no hidden reasoning tokens, so `ANSWER_MAX_TOKENS` is purely
  visible output. 500 comfortably fits a five-bullet answer (~250 tokens);
  at the old 150 it truncated mid-answer.
- Rate limits are far higher (millions of tokens/minute vs. thousands), so
  the token budgeting below matters much less here.

Measured on `claude-haiku-4-5`: first word 0.7-1.4s, complete five-bullet
scenario answer in ~3s.

Model ids are complete as written -- never append a date suffix
(`claude-haiku-4-5`, not `claude-haiku-4-5-20251001`).

### The answer's field comes from the resume

Nothing in the answer prompt assumes software engineering. The candidate's
field is read from the resume and job description, and answers are pitched
at a practitioner in *that* field -- its vocabulary, its metrics, the
trade-offs people in it actually argue about.

The same question diverges accordingly. Asked "how would you handle a
difficult client situation?", a backend-engineer resume produces "I'd check
the database query plan and Redis hit rate before scaling infrastructure";
an enterprise-sales resume produces "I'd pull the last three Outreach
touchpoints... budget freeze, competing priority, procurement gate". Neither
was hardcoded.

Depth is enforced rather than hoped for: every bullet has to carry a named
component/tool/method, a number or threshold, a specific failure mode, or a
real trade-off with its cost. Phrases that sound like answers but say
nothing -- "monitor it", "add logging", "follow best practices",
"communicate with stakeholders", "do a root cause analysis" -- are banned as
standalone points in every field, with the test being: if a point could
appear verbatim in an answer to a completely different question, it is
filler.

So the single highest-leverage thing you control is the resume text you
upload. It sets the field, the vocabulary, and the specifics available.

### Answers that hedge

If answers open with "I haven't worked with that directly", the fix is the
resume/JD, not the prompt. The model answers from the context it is given:
with Kafka in the resume it opens with "I built an event pipeline using
Kafka... three topics, consumer groups, idempotent consumers"; with the same
question and a resume that doesn't mention it, it hedges. Anything you have
genuinely worked with belongs in the resume text you upload -- that is what
turns a hedged answer into a specific one. The prompt keeps the
acknowledgement short and spends the answer on technical substance, but it
will not claim experience the resume doesn't support.

### Reasoning models

`openai/gpt-oss-*` and `qwen/*` spend hidden reasoning tokens out of the
same `ANSWER_MAX_TOKENS` budget as the visible reply, and hiding the
reasoning does not stop it being generated. At 150 tokens a five-sentence
scenario question returned an **empty** answer -- the model spent the whole
budget thinking. `reasoning_effort="low"` is set for these models in
`llm.py` for exactly this reason: it cuts time-to-first-word sharply and
leaves the budget for the actual answer.

## Speech-to-text providers

The STT layer is a replaceable component. Everything outside
`app/services/stt/` talks to one `transcribe_pcm16` method and never names a
vendor, so switching is a `.env` change:

```text
STT_PROVIDER=groq            # Groq Whisper (default)
STT_PROVIDER=openai          # OpenAI Whisper
STT_PROVIDER=faster_whisper  # local, no network -- needs faster-whisper
STT_PROVIDER=nvidia          # NVIDIA -- see below
```

### NVIDIA

`STT_PROVIDER=nvidia` supports both shapes NVIDIA's models ship in:

```text
# gRPC to a Riva ASR server or ASR NIM container (Parakeet, Conformer).
# Needs: pip install nvidia-riva-client
NVIDIA_STT_MODE=riva
NVIDIA_RIVA_SERVER=localhost:50051
NVIDIA_STT_MODEL=          # blank = the server's default model
NVIDIA_STT_LANGUAGE=en-US

# ...or NVIDIA-hosted NVCF functions instead of your own server:
NVIDIA_RIVA_SERVER=grpc.nvcf.nvidia.com:443
NVIDIA_RIVA_USE_SSL=true
NVIDIA_RIVA_FUNCTION_ID=<function id>
NVIDIA_API_KEY=nvapi-...

# ...or HTTP, for a NIM exposing /v1/audio/transcriptions:
NVIDIA_STT_MODE=nim
NVIDIA_STT_BASE_URL=http://localhost:9000/v1
NVIDIA_STT_MODEL=<model name>
```

The vocabulary bias that Whisper receives as a prompt is passed to Riva as
word boosting instead, so framework names stay just as protected against
mishearing.

To add another backend: write a class with one `transcribe_pcm16` method
returning a `TranscriptionResult`, and add one line to
`app/services/stt/registry.py`. Set `confidence_known=False` if the backend
reports no confidence score -- the filters treat "unknown" very differently
from "certain", and getting that wrong is what previously let hallucinated
text through at full confidence.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

```text
GROQ_API_KEY=your-groq-key
```

The answer-generation settings are:

```text
ANSWER_PROVIDER=groq
ANSWER_MODEL=llama-3.3-70b-versatile
ANSWER_MAX_TOKENS=350
ANSWER_TEMPERATURE=0.2
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

To use OpenAI:

```text
ANSWER_PROVIDER=openai
ANSWER_MODEL=your-openai-model-id
OPENAI_API_KEY=your-openai-key
```

To use Claude through Anthropic:

```text
ANSWER_PROVIDER=anthropic
ANSWER_MODEL=your-claude-model-id
ANTHROPIC_API_KEY=your-anthropic-key
```

## Run The Backend

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Stream Mic Audio

In a second terminal with the virtual environment active:

```powershell
python client\test_mic_stream.py
```

Talk into your microphone. When VAD detects a speech segment, you should see transcript lines in the client:

```text
[transcript] tell me about yourself (2.14s, 650ms STT)
```

You should also see `Transcript: ...` in the backend logs.

## Stream Answers

In a third terminal with the virtual environment active:

```powershell
python client\test_answer_stream.py
```

Ask an interview-style question into the mic, for example:

```text
What is FastAPI?
```

The answer terminal should show:

```text
[heard] What is FastAPI?
[question] What is FastAPI?
[answer:groq/llama-3.3-70b-versatile] ...
[done]
```

The mic client only sends audio and prints transcripts. The answer client listens to `/ws/answers`.

## Session Context

Set resume/JD context before an interview:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/session `
  -ContentType "application/json" `
  -Body '{
    "session_id": "default",
    "resume_text": "Paste resume text here",
    "job_description": "Paste job description here",
    "notes": "Any extra positioning notes"
  }'
```

By default session state is in memory:

```text
SESSION_STORE_BACKEND=memory
```

To use Redis later:

```text
SESSION_STORE_BACKEND=redis
REDIS_URL=redis://localhost:6379/0
```

## Audio Device Help

List input devices:

```powershell
python client\test_mic_stream.py --list-devices
```

Use a specific device:

```powershell
python client\test_mic_stream.py --device 1
```

The backend and client both assume 16 kHz mono signed 16-bit PCM audio. Keep `AUDIO_SAMPLE_RATE`, `AUDIO_FRAME_MS`, `--samplerate`, and `--chunk-ms` aligned.

## VAD And Segmentation Tuning

Defaults to a pure-Python energy VAD so installation works cleanly on
Windows/Python 3.14 without Visual C++ Build Tools:

```text
VAD_BACKEND=energy
VAD_ENERGY_THRESHOLD=500
VAD_ADAPTIVE_THRESHOLD=true
```

`VAD_ENERGY_THRESHOLD` is a **floor** for the trigger level, not the trigger
level itself. With `VAD_ADAPTIVE_THRESHOLD=true` the detector tracks the
actual noise floor of the call and requires speech to stand a margin above
it. This matters for system-loopback audio specifically: a video call
carries continuous codec comfort noise whose level moves around, and against
one fixed number that either triggers constantly or misses quiet speech.

If speech is missed, lower the floor to `300`. If quiet background noise
still opens segments, raise it to `800`.

`VAD_BACKEND=webrtc` also works, but requires `webrtcvad-wheels`, which may
need Microsoft C++ Build Tools on Python versions without prebuilt wheels.

### Which audio becomes a transcription request

```text
SEGMENT_MIN_SPEECH_MS=320
SEGMENT_END_SILENCE_MS=420
SEGMENT_MAX_SECONDS=8.0
SEGMENT_CARRYOVER_MS=200
```

`SEGMENT_MIN_SPEECH_MS` is the most important of these. A segment must
contain at least this much genuinely voiced audio or it is discarded without
a transcription request at all. Every recognizer returns *text* rather than
nothing when handed near-silence, so a request made mostly of silence comes
back as invented words -- and each one costs a request against your rate
limit. Raise it if clicks and keyboard noise still produce transcripts;
lower it if very short real answers ("yes", "why") are being missed.

A question longer than `SEGMENT_MAX_SECONDS` is cut into several segments,
which are transcribed concurrently. They keep one shared utterance id, so
the pipeline still treats them as one question, and `SEGMENT_CARRYOVER_MS`
of audio is replayed across each cut so a word split by it survives.

### When a question gets answered

```text
QUESTION_DEBOUNCE_MS=180
QUESTION_SOFT_WAIT_SECONDS=1.5
UTTERANCE_MERGE_GAP_SECONDS=1.2
QUESTION_MAX_WAIT_SECONDS=4.0
```

- `QUESTION_DEBOUNCE_MS` only coalesces transcripts arriving back to back.
  It is no longer used to guess whether the interviewer has finished -- the
  VAD reports that directly.
- `QUESTION_SOFT_WAIT_SECONDS` applies **only** to an answerable transcript
  that does not look finished (trails off mid-sentence, or arrives without
  terminal punctuation from a provider that emits none). A question that
  looks complete is never delayed.
- `UTTERANCE_MERGE_GAP_SECONDS` is how long speech that is not yet a
  question waits for a continuation before being discarded, measured from
  the moment the interviewer stops talking. It is what stops unrelated
  fragments from accumulating and being glued onto a later question.

The gap timer is only half the story, and the smaller half. The VAD reports
whether the interviewer is speaking **right now**, and a buffered
part-question is never discarded while they are — because the delay between
two transcripts of one spoken question is mostly the time taken to *say* the
next sentence, which no timer can tell apart from having finished. Measured
on a real interview, sentences of one scenario question arrived 2.7s, 4.0s
and 6.9s apart; against a 2s timer alone every setup sentence was thrown
away before the question landed, and the model was asked "how would you
debug the problem" with no idea what the problem was.

So `UTTERANCE_MERGE_GAP_SECONDS` only has to cover a *silent* thinking pause
plus one transcription round trip — not how long a sentence takes to speak.
Raise it if an interviewer who pauses to think mid-scenario still gets
split.

That round trip is the part worth being careful about. The window opens when
the interviewer stops talking, but the words they just said are still inside
the recognizer for another second or so. At 2s a provider having a slow
moment lost the setup of a scenario question a fraction of a second before
it arrived; the default is now 4s, which holds at a simulated 3s
transcription latency. Widening it costs no answer latency — an answerable
buffer is acted on immediately either way, so this only delays discarding
something that was never going to be answered. Keep
`QUESTION_MAX_WAIT_SECONDS` above it, since that cap is checked first.

## Testing without a call

Two offline harnesses, both free to run -- no API keys, no microphone, no
backend server:

```powershell
python client\test_question_gate_scenarios.py   # gating, merging, barge-in
python client\test_pipeline_e2e.py              # audio -> answer, with timings
```

`test_pipeline_e2e.py` feeds synthetic audio through the real VAD,
transcription worker and question gate with a fake STT backend, and reports
both the end-of-speech-to-answer latency and how many transcription requests
each scenario cost. `--stt-latency` simulates a slower provider.

`test_question_gate_scenarios.py --live` uses real `.env` settings and makes
real API calls, for a final check once the offline run looks right.

## Phase Boundaries

Included now:

- `/ws/audio`
- `/ws/answers`
- `/session`
- microphone test client
- answer stream test client
- utterance-aware VAD segmentation with an adaptive noise floor
- pluggable STT: Groq Whisper, OpenAI, local faster-whisper, NVIDIA (Riva/NIM)
- transcript quality filtering (noise, hallucinations, low confidence)
- clause-level question gating with fuzzy interview-intent matching
- selectable answer providers: Groq, OpenAI, Anthropic, DeepSeek
- overlay UI (`client/overlay_app.py`)
- system audio loopback capture (`client/test_loopback_stream.py`)
- screen analysis through a vision model

Not included yet:

- PostgreSQL persistence
