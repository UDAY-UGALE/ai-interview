"""
End-to-end audio -> answer test, offline.

Drives the REAL pipeline -- SpeechSegmenter, the transcription worker in
app/routes/audio_ws.py, the transcript quality filter, and
QuestionAnswerPipeline -- with synthetic audio and a fake STT backend, and
measures the two things that actually matter in a live interview:

  * how long after the interviewer stops talking the answer starts
  * how many transcription requests the pipeline made to get there

No microphone, no backend server, no API keys, no cost. The fake STT
returns whatever transcript the scenario says that audio "contains", after a
configurable round-trip delay, so the timings below are real pipeline
timings with a realistic network stand-in.

Usage (from the project root):
    python client\\test_pipeline_e2e.py
    python client\\test_pipeline_e2e.py --stt-latency 0.8
"""

from __future__ import annotations

import argparse
import asyncio
import math
import struct
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.routes import audio_ws as audio_ws_module  # noqa: E402
from app.services import question_gate as gate_module  # noqa: E402
from app.services.question_gate import get_question_pipeline  # noqa: E402
from app.services.stt.base import TranscriptionResult  # noqa: E402
from app.services.vad import SpeechSegmenter  # noqa: E402


SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2


# ---------------------------------------------------------------- audio ----


def speech(seconds: float, amplitude: int = 7000) -> bytes:
    """A voiced-sounding burst: a couple of formant-ish tones plus a little
    noise, so it reads as speech to an energy detector the way a flat tone
    would not."""
    count = int(SAMPLE_RATE * seconds)
    out = bytearray()
    for i in range(count):
        t = i / SAMPLE_RATE
        # Amplitude envelope wobbles like syllables do.
        envelope = 0.6 + 0.4 * math.sin(2 * math.pi * 4.0 * t)
        value = (
            math.sin(2 * math.pi * 210 * t) * 0.6 + math.sin(2 * math.pi * 900 * t) * 0.4
        )
        out += struct.pack("<h", int(max(-1.0, min(1.0, value * envelope)) * amplitude))
    return bytes(out)


def silence(seconds: float, floor: int = 40) -> bytes:
    """Not digital zero -- a real call carries codec comfort noise, which is
    exactly what a fixed-threshold detector kept mistaking for speech."""
    count = int(SAMPLE_RATE * seconds)
    out = bytearray()
    for i in range(count):
        out += struct.pack("<h", int(floor * math.sin(2 * math.pi * 60 * i / SAMPLE_RATE)))
    return bytes(out)


def blip(seconds: float = 0.08, amplitude: int = 9000) -> bytes:
    """A click/keystroke/door -- loud, but far too short to be speech."""
    return speech(seconds, amplitude=amplitude)


# ------------------------------------------------------------- scenarios ----


@dataclass
class Step:
    audio: bytes
    # What a real STT engine would return for THIS piece of audio. None means
    # "nothing was said" -- silence and blips get whatever `hallucination`
    # says, to model a model that invents text on non-speech.
    says: str | None = None


@dataclass
class E2EScenario:
    name: str
    note: str
    steps: list[Step]
    expect_answer: bool
    expect_question_contains: str = ""
    # What the STT backend invents when handed audio with no speech in it.
    hallucination: str = "Thank you."
    tail_silence: float = 2.5
    events: list[tuple[float, dict]] = field(default_factory=list)


def build_scenarios() -> list[E2EScenario]:
    return [
        E2EScenario(
            name="short_question",
            note="One short question, then silence. The baseline latency case.",
            steps=[Step(speech(1.4), "What is Django?")],
            expect_answer=True,
            expect_question_contains="Django",
        ),
        E2EScenario(
            name="long_multi_segment_question",
            note=(
                "A long continuous question that gets force-cut into several "
                "segments. Must arrive as ONE question, not several -- and the "
                "answer must not wait for extra audio that never comes."
            ),
            steps=[
                Step(speech(7.0), "Tell me about yourself and explain one project"),
                Step(speech(5.0), "you worked on and what challenges you faced."),
            ],
            expect_answer=True,
            expect_question_contains="challenges",
        ),
        E2EScenario(
            name="silence_and_noise_only",
            note=(
                "Nobody says anything for 20 seconds -- just room tone and a few "
                "clicks. This is the case that produced a transcription request "
                "every couple of seconds and filled the logs with invented text. "
                "Expect ZERO requests and no answer."
            ),
            steps=[
                Step(silence(4.0)),
                Step(blip()),
                Step(silence(3.0)),
                Step(blip()),
                Step(silence(4.0)),
                Step(blip(0.05)),
                Step(silence(4.0)),
            ],
            expect_answer=False,
        ),
        E2EScenario(
            name="noise_then_real_question",
            note=(
                "Clicks and room tone, then a real question. The question must "
                "be answered, and must NOT arrive with the noise glued to it."
            ),
            steps=[
                Step(blip()),
                Step(silence(1.5)),
                Step(blip()),
                Step(silence(2.0)),
                Step(speech(1.6), "Uday, tell me about yourself."),
            ],
            expect_answer=True,
            expect_question_contains="yourself",
        ),
        E2EScenario(
            name="question_then_stray_noise",
            note=(
                "A question immediately followed by a stray noise blip. The "
                "answer must survive, and the noise must not be answered."
            ),
            steps=[
                Step(speech(1.5), "What is a REST API?"),
                Step(silence(0.8)),
                Step(blip()),
                Step(silence(2.0)),
            ],
            expect_answer=True,
            expect_question_contains="REST",
        ),
    ]


# ------------------------------------------------------------- harness -----


class FakeSTT:
    """Stands in for Groq/NVIDIA/whatever. Returns the transcript the
    scenario says that audio contains, after a realistic delay."""

    name = "fake"

    def __init__(self, latency: float) -> None:
        self._latency = latency
        self.requests = 0
        self.plan: list[tuple[bytes, str]] = []
        self.hallucination = "Thank you."

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        self.requests += 1
        await asyncio.sleep(self._latency)
        text, confident = self._lookup(pcm)
        return TranscriptionResult(
            text=text,
            confidence=0.88 if confident else 0.22,
            confidence_known=True,
            no_speech_prob=0.05 if confident else 0.8,
            provider="fake",
        )

    def _lookup(self, pcm: bytes) -> tuple[str, bool]:
        """Attribute this segment to whichever scripted utterances its audio
        overlaps, the way a real recognizer would only produce words for the
        parts that actually contained speech."""
        spoken = [text for audio, text in self.plan if text and _overlaps(pcm, audio)]
        if spoken:
            return " ".join(spoken), True
        return self.hallucination, False


def _overlaps(segment_pcm: bytes, source_pcm: bytes) -> bool:
    """Did this segment capture part of that source clip? Compared on a
    sampled fingerprint so it stays cheap for multi-second clips."""
    if len(source_pcm) < FRAME_BYTES * 8:
        return False
    probes = [
        source_pcm[offset : offset + FRAME_BYTES]
        for offset in range(0, len(source_pcm) - FRAME_BYTES, FRAME_BYTES * 10)
    ]
    hits = sum(1 for probe in probes if probe in segment_pcm)
    return hits >= 2


class _Recorder:
    def __init__(self, scenario: E2EScenario, start: float) -> None:
        self.scenario = scenario
        self.start = start

    async def send_json(self, payload: dict) -> None:
        self.scenario.events.append((time.monotonic() - self.start, payload))


class _StubLLM:
    async def stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        await asyncio.sleep(0.25)  # stand-in for time-to-first-token
        yield "(stub answer)"


def _stub_build_llm_client(settings, **kwargs):
    return _StubLLM()


async def run_scenario(scenario: E2EScenario, *, stt_latency: float) -> dict:
    settings = get_settings()
    session_id = f"e2e-{scenario.name}"
    pipeline = get_question_pipeline()
    start = time.monotonic()
    recorder = _Recorder(scenario, start)
    await gate_module.answer_hub.connect(session_id, recorder)

    stt = FakeSTT(stt_latency)
    stt.plan = [(step.audio, step.says or "") for step in scenario.steps]
    stt.hallucination = scenario.hallucination

    segmenter = SpeechSegmenter(
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_MS,
        vad_backend="energy",
        vad_mode=2,
        energy_threshold=settings.vad_energy_threshold,
        min_segment_seconds=settings.segment_min_seconds,
        max_segment_seconds=settings.segment_max_seconds,
        end_silence_ms=settings.segment_end_silence_ms,
        min_speech_ms=settings.segment_min_speech_ms,
        onset_frames=settings.vad_onset_frames,
        preroll_ms=settings.vad_preroll_ms,
        carryover_ms=settings.segment_carryover_ms,
        adaptive_threshold=settings.vad_adaptive_threshold,
    )

    segment_queue: asyncio.Queue = asyncio.Queue()
    result_queue: asyncio.Queue = asyncio.Queue()
    dispatcher = asyncio.create_task(
        audio_ws_module._transcription_dispatcher(
            stt=stt,
            segment_queue=segment_queue,
            result_queue=result_queue,
            max_concurrent_segments=settings.stt_max_concurrent_segments,
        )
    )
    consumer = asyncio.create_task(
        audio_ws_module._result_consumer(
            websocket=_NullSocket(),
            result_queue=result_queue,
            session_id=session_id,
            settings=settings,
        )
    )

    # Feed audio at real time, 20ms per frame, exactly like the capture client.
    stream = b"".join(step.audio for step in scenario.steps) + silence(scenario.tail_silence)
    speech_end_offset = _last_speech_offset(scenario)
    speech_end_at: float | None = None

    for index in range(0, len(stream) - FRAME_BYTES + 1, FRAME_BYTES):
        frame = stream[index : index + FRAME_BYTES]
        if speech_end_at is None and index >= speech_end_offset:
            speech_end_at = time.monotonic()
        for segment in segmenter.accept(frame):
            await segment_queue.put(segment)
        await asyncio.sleep(FRAME_MS / 1000)

    # Let anything still in flight finish.
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        pending = pipeline._pending.get(session_id)
        idle = pending is None or (
            (pending.task is None or pending.task.done())
            and (pending.answer_task is None or pending.answer_task.done())
        )
        if idle and segment_queue.empty() and result_queue.empty():
            break
        await asyncio.sleep(0.05)

    await segment_queue.put(None)
    await dispatcher
    await result_queue.put(None)
    await consumer
    await gate_module.answer_hub.disconnect(session_id, recorder)

    answer_start = next(
        (t for t, e in scenario.events if e.get("type") == "answer_start"), None
    )
    question = next(
        (e.get("question", "") for _t, e in scenario.events if e.get("type") == "answer_start"),
        "",
    )
    return {
        "requests": stt.requests,
        "dropped_segments": segmenter.dropped_segments,
        "answer_start": answer_start,
        "speech_end_at": (speech_end_at or start) - start,
        "question": question,
    }


def _last_speech_offset(scenario: E2EScenario) -> int:
    offset = 0
    last_end = 0
    for step in scenario.steps:
        offset += len(step.audio)
        if step.says:
            last_end = offset
    return last_end


class _NullSocket:
    async def send_json(self, payload: dict) -> None:
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stt-latency",
        type=float,
        default=0.5,
        help="Simulated STT round trip in seconds (default 0.5).",
    )
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    gate_module.build_llm_client = _stub_build_llm_client
    get_question_pipeline()._settings.fast_intent_enabled = False

    scenarios = build_scenarios()
    if args.only:
        needles = [n.strip() for n in args.only.split(",") if n.strip()]
        scenarios = [s for s in scenarios if any(n in s.name for n in needles)]

    print(
        f"Simulated STT round trip: {args.stt_latency}s. "
        "Latency below is measured from the last frame of real speech.\n"
    )
    failures = 0
    for scenario in scenarios:
        result = await run_scenario(scenario, stt_latency=args.stt_latency)
        answered = result["answer_start"] is not None
        ok = answered == scenario.expect_answer
        if ok and scenario.expect_question_contains:
            ok = scenario.expect_question_contains.lower() in result["question"].lower()
        failures += not ok

        print(f"=== {scenario.name} ===")
        print(f"    {scenario.note}")
        print(
            f"    STT requests: {result['requests']}   "
            f"segments dropped before STT: {result['dropped_segments']}"
        )
        if answered:
            latency = result["answer_start"] - result["speech_end_at"]
            print(f"    speech ended -> answer started: {latency:.2f}s")
            print(f"    question answered: {result['question']!r}")
        else:
            print("    no answer (as expected)" if not scenario.expect_answer else "    NO ANSWER")
        print(f"    {'PASS' if ok else 'FAIL'}\n")

    print("All expectations met." if not failures else f"{failures} scenario(s) FAILED.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
