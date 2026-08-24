"""Whisper vs Deepgram Nova-3, on the same audio, through the same pipeline.

The point of this harness is to make the migration decision on EVIDENCE.
Nova-3 may well be better for Indian-English technical speech -- that is the
hypothesis -- but "may well be" is not a reason to switch the recognizer a
live interview depends on, and the audit that motivated this work found
several places where an assumption about STT turned out to be wrong (most
notably: the RAG -> "Rack" error is accent-conditioned, NOT caused by audio
quality, so it survived every noise/bandwidth degradation tested).

Both providers are driven through the application's OWN segmenter, quality
filter and question gate, so what is compared is the end of the pipeline the
user actually experiences, not a raw WER number.

    # 1. render the corpus (Windows SAPI; any 16 kHz mono WAVs will do)
    python client\\test_stt_comparison.py --generate

    # 2. compare (needs GROQ_API_KEY and DEEPGRAM_API_KEY in .env)
    python client\\test_stt_comparison.py
    python client\\test_stt_comparison.py --providers groq
    python client\\test_stt_comparison.py --degraded

Audio can also be supplied directly: drop 16 kHz mono WAV files into
client/stt_corpus/ named <case-id>.wav and add the expected text to
EXPECTED below (or a sidecar <case-id>.txt).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.question_gate import _classify_question  # noqa: E402
from app.services.session_vocabulary import (  # noqa: E402
    as_whisper_prompt,
    build_session_vocabulary,
    session_term_set,
)
from app.services.transcript_normalizer import normalize_transcript  # noqa: E402
from app.services.transcript_quality import assess_transcript  # noqa: E402
from app.services.vad import SpeechSegmenter  # noqa: E402


CORPUS = Path(__file__).resolve().parent / "stt_corpus"
SETTINGS = get_settings()
SR = SETTINGS.audio_sample_rate
FRAME_BYTES = int(SR * SETTINGS.audio_frame_ms / 1000) * 2

RESUME = """UDAY UGALE -- AI / Full-Stack Engineer, Mannlowe, Pune.
RAG chatbot with LangChain and a Chroma vector store. FSM onboarding agent on
FastAPI. Python, Django REST Framework, PostgreSQL, Redis, Celery, Flask.
React, Redux. Docker, Kubernetes, GitHub Actions, AWS EC2/ECR. SQL. HTTP.
ERPNext / Frappe."""
JD = """AI Engineer: Python, FastAPI, RAG pipelines, vector databases,
LangChain, Docker, AWS, Kubernetes, Redis, PostgreSQL, GitHub Actions."""


# The technical terms the audit found breaking, as whole spoken questions --
# a bare word list would not exercise the gate, and the gate is half of what
# is being compared.
CASES: dict[str, str] = {
    "rag_what_is": "What is RAG?",
    "rag_pipeline": "Can you explain the architecture of your RAG pipeline?",
    "rag_spelled": "Can you give me an idea about R A G?",
    "redis_where": "Where did you use Redis in that project?",
    "docker_prod": "What is Docker and have you used it in production?",
    "fastapi_why": "Why did you choose FastAPI over Flask?",
    "django_project": "Tell me about your Django project.",
    "kubernetes_pod": "A Kubernetes pod is continuously restarting. How will you troubleshoot it?",
    "http_methods": "Can you explain the HTTP methods?",
    "github_actions": "Do you use GitHub Actions for CI CD?",
    "sql_optimise": "How do you optimize a slow SQL query?",
    "postgres_vs_mongo": "Why did you pick PostgreSQL over MongoDB?",
    "langchain_diff": "How do LangChain and LangGraph differ?",
    "celery_tasks": "How do you handle background tasks with Celery?",
    "flask_vs_django": "What is the difference between Flask and Django?",
    # long / scenario
    "scenario_long": (
        "Okay, so let's take a scenario. Your production server's disk partition is "
        "one hundred percent full, the application has stopped responding, and the "
        "monitoring dashboard shows no alerts. How would you troubleshoot this?"
    ),
    "multipart": "What is RAG? Why would you use it? And have you built one yourself?",
    # pauses inside a question (rendered as two files, joined with silence)
    "pause_a1": "Can you explain",
    "pause_a2": "RAG?",
    "pause_b1": "What challenges did you",
    "pause_b2": "face in your project?",
    # self-correction
    "correction_django": "Tell me about your Flask project. Actually, I mean my Django project.",
    # informal / fast
    "informal_howrag": "How RAG works?",
}

PAUSE_PAIRS = [("pause_a", 0.6), ("pause_b", 0.6)]

DEGRADATIONS = {
    "clean": dict(snr_db=None, telephone=False, gain=1.0),
    "snr12": dict(snr_db=12, telephone=False, gain=1.0),
    "snr6": dict(snr_db=6, telephone=False, gain=1.0),
    "telephone": dict(snr_db=None, telephone=True, gain=1.0),
    "tel+snr6": dict(snr_db=6, telephone=True, gain=1.0),
    "quiet+snr12": dict(snr_db=12, telephone=False, gain=0.30),
}


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def generate_corpus() -> None:
    """Render the corpus with Windows SAPI.

    A stated limitation, because it changes how the results must be read:
    SAPI ships US-English voices only, so this measures the EASY case. The
    errors that matter most in production are accent-conditioned, and the
    honest way to compare on those is to replay real captured audio -- drop
    16 kHz mono WAVs into client/stt_corpus/ and this harness will use them.
    """
    if sys.platform != "win32":
        print("--generate uses Windows SAPI. On other platforms, put 16 kHz mono "
              "WAV files in client/stt_corpus/ named <case-id>.wav instead.")
        return
    CORPUS.mkdir(exist_ok=True)
    script = """
Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($line in (Get-Content $env:SPEC_PATH -Encoding UTF8)) {
    if ($line.Trim() -eq "") { continue }
    $parts = $line -split "`t"
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SetOutputToWaveFile((Join-Path $env:OUT_DIR ($parts[0] + ".wav")), $fmt)
    $s.Speak($parts[1]); $s.SetOutputToNull(); $s.Dispose()
}
"""
    spec = CORPUS / "_spec.tsv"
    spec.write_text(
        "\n".join(f"{k}\t{v}" for k, v in CASES.items()), encoding="utf-8"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        env={**dict(__import__("os").environ), "SPEC_PATH": str(spec), "OUT_DIR": str(CORPUS)},
        check=True,
    )
    print(f"rendered {len(CASES)} files into {CORPUS}")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SR or handle.getnchannels() != 1:
            raise SystemExit(
                f"{path.name}: need 16 kHz mono, got {handle.getframerate()} Hz "
                f"{handle.getnchannels()}ch"
            )
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def _noise(seconds: float, rms: float = 28.0) -> np.ndarray:
    return np.random.default_rng(42).normal(0, rms, int(seconds * SR)).astype(np.int16)


def degrade(signal: np.ndarray, *, snr_db, telephone: bool, gain: float) -> np.ndarray:
    out = signal.astype(np.float64)
    if telephone:
        spectrum = np.fft.rfft(out)
        freqs = np.fft.rfftfreq(len(out), 1 / SR)
        spectrum[(freqs < 300) | (freqs > 3400)] *= 0.06
        out = np.fft.irfft(spectrum, len(out))
    out *= gain
    if snr_db is not None:
        power = np.mean(out ** 2) or 1.0
        out = out + np.random.default_rng(3).normal(
            0, np.sqrt(power / (10 ** (snr_db / 10))), len(out)
        )
    return np.clip(out, -32768, 32767).astype(np.int16)


def build_stream(parts: list) -> np.ndarray:
    """Lead-in and tail of comfort noise, so the VAD's 1 s calibration window
    and 420 ms end-silence detector behave as they do on a live call."""
    chunks = [_noise(1.6)]
    for part in parts:
        chunks.append(_noise(float(part)) if isinstance(part, (int, float)) else part)
    chunks.append(_noise(1.2))
    return np.concatenate(chunks)


def segment(stream: np.ndarray) -> list:
    segmenter = SpeechSegmenter(
        sample_rate=SR, frame_ms=SETTINGS.audio_frame_ms,
        vad_backend=SETTINGS.vad_backend, vad_mode=SETTINGS.vad_mode,
        energy_threshold=SETTINGS.vad_energy_threshold,
        min_segment_seconds=SETTINGS.segment_min_seconds,
        max_segment_seconds=SETTINGS.segment_max_seconds,
        end_silence_ms=SETTINGS.segment_end_silence_ms,
        min_speech_ms=SETTINGS.segment_min_speech_ms,
        onset_frames=SETTINGS.vad_onset_frames, preroll_ms=SETTINGS.vad_preroll_ms,
        carryover_ms=SETTINGS.segment_carryover_ms,
        adaptive_threshold=SETTINGS.vad_adaptive_threshold,
        calibration_ms=SETTINGS.vad_calibration_ms,
    )
    pcm = stream.tobytes()
    segments = []
    for offset in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        segments.extend(segmenter.accept(pcm[offset : offset + FRAME_BYTES]))
    tail = segmenter.flush()
    if tail:
        segments.append(tail)
    return segments


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

TERMS_UNDER_TEST = [
    "RAG", "Redis", "Docker", "FastAPI", "Django", "Kubernetes", "HTTP",
    "GitHub", "SQL", "PostgreSQL", "MongoDB", "LangChain", "LangGraph",
    "Celery", "Flask",
]


def word_error_rate(expected: str, heard: str) -> float:
    import difflib

    a = expected.lower().replace("?", "").replace(".", "").replace(",", "").split()
    b = heard.lower().replace("?", "").replace(".", "").replace(",", "").split()
    if not a:
        return 0.0
    matcher = difflib.SequenceMatcher(None, a, b)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return max(0.0, (len(a) - matched) / len(a))


def technical_terms(expected: str, heard: str) -> tuple[int, int]:
    """(terms correct, terms expected) for the vocabulary under test."""
    lowered = heard.lower()
    wanted = [t for t in TERMS_UNDER_TEST if t.lower() in expected.lower()]
    correct = sum(1 for t in wanted if t.lower() in lowered)
    return correct, len(wanted)


@dataclass
class Result:
    case: str
    provider: str
    condition: str
    expected: str
    raw: str = ""
    normalized: str = ""
    segments: int = 0
    latency_ms: list[int] = field(default_factory=list)
    confidence: float = 0.0
    confidence_known: bool = False
    dropped: list[str] = field(default_factory=list)
    gate_reason: str = ""
    gate_answers: bool = False
    error: str = ""

    @property
    def wer(self) -> float:
        return word_error_rate(self.expected, self.normalized or self.raw)


async def transcribe(service, segments, *, terms: set[str]) -> tuple[str, list[int], float, bool, list[str]]:
    texts, latencies, drops = [], [], []
    confidences, known_any = [], False
    for segment_obj in segments:
        start = time.perf_counter()
        result = await service.transcribe_pcm16(
            segment_obj.pcm, sample_rate=segment_obj.sample_rate
        )
        latencies.append(int((time.perf_counter() - start) * 1000))
        text = result.text.strip()
        if not text:
            continue
        confidences.append(result.confidence)
        known_any = known_any or result.confidence_known
        verdict = assess_transcript(
            text, confidence=result.confidence,
            confidence_known=result.confidence_known,
            speech_seconds=segment_obj.speech_seconds,
            min_confidence=SETTINGS.stt_drop_confidence_threshold,
        )
        if verdict.keep:
            texts.append(text)
        else:
            drops.append(f"{verdict.reason}:{text}")
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return " ".join(texts), latencies, mean_conf, known_any, drops


def build_services(providers: list[str], vocabulary: list[str]) -> dict:
    services = {}
    if "groq" in providers:
        from app.services.stt.whisper_api import GroqSTTService

        if SETTINGS.groq_api_key:
            services["groq-whisper"] = GroqSTTService.from_settings(
                SETTINGS, prompt=as_whisper_prompt(vocabulary)
            )
        else:
            print("  (skipping Whisper: GROQ_API_KEY is not set)")
    if "deepgram" in providers:
        from app.services.stt.deepgram import DeepgramSTTService

        if SETTINGS.deepgram_api_key:
            services["deepgram-nova3"] = DeepgramSTTService.from_settings(
                SETTINGS, keyterms=vocabulary
            )
        else:
            print("  (skipping Deepgram: DEEPGRAM_API_KEY is not set -- add it to .env "
                  "to produce the comparison this decision needs)")
    return services


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="render the corpus with SAPI")
    parser.add_argument("--providers", default="groq,deepgram")
    parser.add_argument("--degraded", action="store_true", help="also run the noisy conditions")
    parser.add_argument("--out", default="stt_comparison.json")
    args = parser.parse_args()

    if args.generate:
        generate_corpus()
        return 0
    if not CORPUS.exists() or not any(CORPUS.glob("*.wav")):
        print(f"No audio in {CORPUS}. Run with --generate first, or drop 16 kHz "
              f"mono WAVs there named <case-id>.wav")
        return 1

    vocabulary = build_session_vocabulary(resume_text=RESUME, job_description=JD)
    evidence = session_term_set(
        build_session_vocabulary(
            resume_text=RESUME, job_description=JD, include_baseline=False
        )
    )
    services = build_services([p.strip() for p in args.providers.split(",")], vocabulary)
    if not services:
        print("No provider is configured; nothing to compare.")
        return 1

    conditions = DEGRADATIONS if args.degraded else {"clean": DEGRADATIONS["clean"]}
    results: list[Result] = []

    singles = {k: v for k, v in CASES.items() if not k.startswith("pause_")}
    for case, expected in singles.items():
        path = CORPUS / f"{case}.wav"
        if not path.exists():
            continue
        base = read_wav(path)
        for condition, kwargs in conditions.items():
            segments = segment(build_stream([degrade(base, **kwargs)]))
            for name, service in services.items():
                record = Result(case=case, provider=name, condition=condition,
                                expected=expected, segments=len(segments))
                try:
                    raw, lat, conf, known, drops = await transcribe(
                        service, segments, terms=evidence
                    )
                except Exception as exc:  # provider outage, auth, quota
                    record.error = f"{type(exc).__name__}: {exc}"
                    results.append(record)
                    continue
                record.raw, record.latency_ms = raw, lat
                record.confidence, record.confidence_known = conf, known
                record.dropped = drops
                record.normalized = normalize_transcript(
                    raw, session_terms=evidence
                ).normalized
                verdict = _classify_question(record.normalized)
                record.gate_reason, record.gate_answers = verdict.reason, verdict.should_answer
                results.append(record)

    # paused questions: two files joined by real silence
    for prefix, gap in PAUSE_PAIRS:
        first, second = CORPUS / f"{prefix}1.wav", CORPUS / f"{prefix}2.wav"
        if not (first.exists() and second.exists()):
            continue
        expected = f"{CASES[prefix + '1']} {CASES[prefix + '2']}"
        segments = segment(build_stream([read_wav(first), gap, read_wav(second)]))
        for name, service in services.items():
            record = Result(case=f"{prefix}_paused", provider=name, condition="clean",
                            expected=expected, segments=len(segments))
            try:
                raw, lat, conf, known, drops = await transcribe(service, segments, terms=evidence)
            except Exception as exc:
                record.error = f"{type(exc).__name__}: {exc}"
                results.append(record)
                continue
            record.raw, record.latency_ms = raw, lat
            record.confidence, record.confidence_known = conf, known
            record.dropped = drops
            record.normalized = normalize_transcript(raw, session_terms=evidence).normalized
            verdict = _classify_question(record.normalized)
            record.gate_reason, record.gate_answers = verdict.reason, verdict.should_answer
            results.append(record)

    # ---- report -----------------------------------------------------------
    print(f"\n{'=' * 78}\nPER-CASE TRANSCRIPTS\n{'=' * 78}")
    for case in sorted({r.case for r in results}):
        rows = [r for r in results if r.case == case and r.condition == "clean"]
        if not rows:
            continue
        print(f"\n{case}: {rows[0].expected!r}")
        for row in rows:
            if row.error:
                print(f"   {row.provider:16} ERROR {row.error}")
                continue
            correct, wanted = technical_terms(row.expected, row.normalized)
            print(f"   {row.provider:16} {row.normalized!r}")
            print(f"   {'':16} wer={row.wer:.2f} terms={correct}/{wanted} "
                  f"conf={row.confidence:.2f}{'' if row.confidence_known else ' (unscored)'} "
                  f"segs={row.segments} stt={row.latency_ms} gate={row.gate_reason}"
                  + (f" DROPPED={row.dropped}" if row.dropped else ""))

    print(f"\n{'=' * 78}\nSUMMARY BY PROVIDER\n{'=' * 78}")
    print(f"{'provider':18} {'n':>4} {'WER':>7} {'terms':>9} {'conf':>6} {'p50 ms':>7} "
          f"{'p90 ms':>7} {'gated':>6} {'errs':>5}")
    for name in sorted({r.provider for r in results}):
        rows = [r for r in results if r.provider == name and not r.error]
        errs = sum(1 for r in results if r.provider == name and r.error)
        if not rows:
            print(f"{name:18} {'-':>4}  no successful transcriptions ({errs} errors)")
            continue
        wers = [r.wer for r in rows]
        term_correct = sum(technical_terms(r.expected, r.normalized)[0] for r in rows)
        term_total = sum(technical_terms(r.expected, r.normalized)[1] for r in rows)
        lat = sorted(ms for r in rows for ms in r.latency_ms)
        gated = sum(1 for r in rows if r.gate_answers)
        p50 = lat[len(lat) // 2] if lat else 0
        p90 = lat[int(len(lat) * 0.9)] if lat else 0
        print(f"{name:18} {len(rows):>4} {sum(wers)/len(wers):>7.3f} "
              f"{term_correct:>4}/{term_total:<4} {sum(r.confidence for r in rows)/len(rows):>6.2f} "
              f"{p50:>7} {p90:>7} {gated:>6} {errs:>5}")

    if args.degraded:
        print(f"\n{'=' * 78}\nBY CONDITION (WER)\n{'=' * 78}")
        print(f"{'condition':16}" + "".join(f"{n:>18}" for n in sorted({r.provider for r in results})))
        for condition in conditions:
            line = f"{condition:16}"
            for name in sorted({r.provider for r in results}):
                rows = [r for r in results if r.provider == name and r.condition == condition and not r.error]
                line += f"{(sum(r.wer for r in rows)/len(rows) if rows else float('nan')):>18.3f}"
            print(line)

    providers = {r.provider for r in results}
    if len(providers) < 2:
        print(f"\nOnly {providers or 'no providers'} ran, so this is a BASELINE, not a "
              f"comparison. No migration decision can be justified from it.")

    Path(args.out).write_text(
        json.dumps([r.__dict__ for r in results], indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
