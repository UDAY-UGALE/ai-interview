# InterviewCopilot

MVP for a real-time interview copilot:

- capture microphone audio locally
- stream PCM audio chunks to FastAPI over `/ws/audio`
- use a pure-Python energy VAD to segment speech
- send each speech segment to Groq Whisper
- print live transcripts in the backend logs and test client
- combine nearby transcript fragments so natural pauses do not immediately trigger answers
- detect interview questions and stream LLM answers over `/ws/answers`

The answer pipeline supports selectable providers through environment variables: `groq`, `openai`, or `anthropic`.

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

## VAD Tuning

Phase 1 defaults to a pure-Python energy VAD so installation works cleanly on Windows/Python 3.14 without Visual C++ Build Tools:

```text
VAD_BACKEND=energy
VAD_ENERGY_THRESHOLD=500
```

If speech is not detected, lower `VAD_ENERGY_THRESHOLD` to `300`. If background noise triggers transcripts, raise it to `800` or `1000`.

The code also supports `VAD_BACKEND=webrtc`, but that requires installing `webrtcvad-wheels`, which may need Microsoft C++ Build Tools on Python versions without prebuilt wheels.

If an interviewer pauses mid-question, these settings control when speech and question text are finalized:

```text
SEGMENT_END_SILENCE_MS=1000
QUESTION_DEBOUNCE_MS=1800
```

Raise `QUESTION_DEBOUNCE_MS` to `2500` if answers start too early during pauses. Lower it to `1000` if answers feel slow.

## Phase Boundaries

Included now:

- `/ws/audio`
- `/ws/answers`
- `/session`
- microphone test client
- answer stream test client
- VAD segmentation
- pure-Python energy VAD segmentation
- Groq Whisper STT
- selectable answer providers: Groq, OpenAI, Anthropic
- LangGraph-backed question gate, with a direct fallback if LangGraph is unavailable

Not included yet:

- overlay UI
- system audio loopback
- PostgreSQL persistence
