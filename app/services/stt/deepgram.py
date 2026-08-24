"""Deepgram Nova-3, behind the same contract as every other backend.

Two independent paths live here, because the pipeline has two shapes of
need and conflating them would force a rewrite of working code:

`DeepgramSTTService` is the BATCH path. It implements `transcribe_pcm16`
and nothing else, so it is a drop-in for the Whisper call the existing
VAD-segment pipeline already makes -- `STT_PROVIDER=deepgram` changes which
recognizer runs and changes nothing else about segmentation, filtering,
gating or answering.

`DeepgramStreamingSession` is the LIVE path. It holds a websocket open and
receives interim and final transcripts plus Deepgram's own endpointing, so
the recognizer's view of "they stopped talking" is available alongside the
VAD's. It is OFF by default and deliberately does not replace the VAD: the
energy segmenter is doing real work that the audit measured as sound (noise
floor tracking, the 320ms voiced minimum that suppresses hallucinated text),
and throwing it away to chase a provider feature would be exactly the
unevidenced migration this integration is meant to avoid.

Both talk HTTP/WebSocket directly rather than through the `deepgram-sdk`
package. The SDK is a large dependency whose async surface has changed
shape several times across major versions; the two endpoints used here are
stable, documented, and about forty lines of client code. `httpx` and
`websockets` are already installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from app.core.config import Settings
from app.services.stt.base import (
    MissingSTTConfigError,
    TranscriptionResult,
    pcm16_to_wav,
)


try:
    import websockets
except ImportError:  # pragma: no cover - exercised only on a broken install
    websockets = None


logger = logging.getLogger(__name__)

_LISTEN_URL = "https://api.deepgram.com/v1/listen"
_STREAM_URL = "wss://api.deepgram.com/v1/listen"


def _common_params(settings: Settings, keyterms: list[str] | None) -> list[tuple[str, str]]:
    """Query parameters shared by the batch and streaming endpoints.

    `keyterm` is the replacement for Whisper's prompt-stuffing. It is
    repeated once per term rather than being one blob of prose, which is why
    the resume can no longer be silently truncated away by a job description
    -- see app/services/session_vocabulary.py for the measurement that
    motivated this.
    """
    params: list[tuple[str, str]] = [
        ("model", settings.deepgram_model),
        ("language", settings.deepgram_language),
        ("punctuate", "true" if settings.deepgram_punctuate else "false"),
        ("smart_format", "true" if settings.deepgram_smart_format else "false"),
    ]
    for term in (keyterms or [])[: settings.deepgram_keyterm_limit]:
        # Deepgram rejects empty/oversized keyterms; both are cheap to avoid.
        cleaned = term.strip()
        if cleaned and len(cleaned) <= 60:
            params.append(("keyterm", cleaned))
    return params


def _confidence_from_alternative(alternative: dict) -> tuple[float, bool]:
    """Deepgram reports a real 0..1 confidence per alternative.

    Returned alongside a `known` flag for the same reason the Whisper path
    does it: a provider that reports nothing falls back to 1.0, and treating
    that as certainty is what let every hallucinated fragment through the
    quality filter at full apparent confidence.
    """
    raw = alternative.get("confidence")
    if raw is None:
        words = alternative.get("words") or []
        scored = [w.get("confidence") for w in words if isinstance(w, dict) and w.get("confidence") is not None]
        if not scored:
            return 1.0, False
        return max(0.0, min(1.0, sum(scored) / len(scored))), True
    try:
        return max(0.0, min(1.0, float(raw))), True
    except (TypeError, ValueError):
        return 1.0, False


class DeepgramSTTService:
    """Batch transcription of one closed VAD segment."""

    name = "deepgram"

    def __init__(
        self,
        *,
        api_key: str,
        settings: Settings,
        keyterms: list[str] | None = None,
    ) -> None:
        if not api_key:
            raise MissingSTTConfigError(
                "Set DEEPGRAM_API_KEY to use STT_PROVIDER=deepgram."
            )
        self._api_key = api_key
        self._settings = settings
        self._keyterms = keyterms or []
        # keepalive_expiry, not the default 5 seconds. Questions in an
        # interview arrive 10-60 seconds apart, so with the default every
        # single request pays a fresh TLS handshake. Measured directly:
        # back-to-back calls came back with a p10 of 352ms and a warm p50 of
        # 610ms, while the same calls spaced 7 seconds apart measured a p50
        # of 2,170ms -- a ~1.5s penalty caused entirely by the connection
        # being allowed to lapse between questions, which is exactly the
        # usage pattern this runs in.
        self._client = httpx.AsyncClient(
            timeout=settings.deepgram_timeout_seconds,
            limits=httpx.Limits(
                max_keepalive_connections=4,
                max_connections=8,
                keepalive_expiry=300.0,
            ),
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, *, keyterms: list[str] | None = None
    ) -> "DeepgramSTTService":
        return cls(
            api_key=settings.deepgram_api_key or "",
            settings=settings,
            keyterms=keyterms,
        )

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        # Sent as a WAV container rather than raw PCM: Deepgram accepts both,
        # but the container carries the sample rate, which removes an entire
        # class of "it transcribed at the wrong rate and produced plausible
        # nonsense" bug that raw PCM leaves available.
        audio = pcm16_to_wav(pcm, sample_rate=sample_rate)
        url = f"{_LISTEN_URL}?{urlencode(_common_params(self._settings, self._keyterms))}"

        response = await self._client.post(
            url,
            content=audio,
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "audio/wav",
            },
        )
        response.raise_for_status()
        payload = response.json()

        channels = (payload.get("results") or {}).get("channels") or []
        if not channels:
            return TranscriptionResult(text="", provider=self.name, model=self._settings.deepgram_model)
        alternatives = channels[0].get("alternatives") or []
        if not alternatives:
            return TranscriptionResult(text="", provider=self.name, model=self._settings.deepgram_model)

        best = alternatives[0]
        confidence, known = _confidence_from_alternative(best)
        return TranscriptionResult(
            text=(best.get("transcript") or "").strip(),
            confidence=confidence,
            confidence_known=known,
            # Deepgram has no Whisper-style no_speech_prob. Leaving it at 0.0
            # is correct rather than merely convenient: audio_ws only applies
            # that filter when `confidence_known` says the provider actually
            # reported the number, so a missing signal is never mistaken for
            # a confident "this is speech".
            no_speech_prob=0.0,
            provider=self.name,
            model=self._settings.deepgram_model,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


@dataclass(slots=True)
class StreamingTranscript:
    """One result off the live socket.

    Carries strictly more than the batch result, because the streaming path
    knows things a closed segment cannot: whether this hypothesis can still
    change, and why the recognizer thinks the utterance ended.
    """

    text: str
    is_final: bool
    confidence: float
    confidence_known: bool
    # "endpoint" (Deepgram's silence detector fired), "utterance_end", or
    # "speech_final" -- carried into the session log so a slow or split
    # question can be traced to the recognizer's own decision rather than
    # guessed at.
    endpoint_reason: str = ""
    latency_ms: int = 0
    provider: str = "deepgram"


class DeepgramStreamingSession:
    """A live Deepgram socket for one audio websocket.

    Deliberately a passive component: it accepts the same 20ms PCM frames
    the VAD already receives and hands transcripts to a callback. It makes
    no decisions about segmentation, so it can run ALONGSIDE the existing
    segmenter and be compared against it on real audio before anything is
    switched over.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        sample_rate: int,
        keyterms: list[str] | None = None,
        on_transcript: Callable[[StreamingTranscript], object] | None = None,
    ) -> None:
        if websockets is None:
            raise MissingSTTConfigError(
                "DEEPGRAM_STREAMING=true needs the `websockets` package installed."
            )
        if not settings.deepgram_api_key:
            raise MissingSTTConfigError(
                "Set DEEPGRAM_API_KEY to use DEEPGRAM_STREAMING=true."
            )
        self._settings = settings
        self._sample_rate = sample_rate
        self._keyterms = keyterms or []
        self._on_transcript = on_transcript
        self._socket = None
        self._reader: asyncio.Task | None = None
        self._started_at: float = 0.0

    def _url(self) -> str:
        params = _common_params(self._settings, self._keyterms)
        params.extend(
            [
                ("encoding", "linear16"),
                ("sample_rate", str(self._sample_rate)),
                ("channels", "1"),
                ("interim_results", "true" if self._settings.deepgram_interim_results else "false"),
                ("endpointing", str(self._settings.deepgram_endpointing_ms)),
                ("utterance_end_ms", str(self._settings.deepgram_utterance_end_ms)),
                # Tells Deepgram to mark the last result of an utterance, which
                # is the streaming equivalent of the VAD's is_final segment.
                ("vad_events", "true"),
            ]
        )
        return f"{_STREAM_URL}?{urlencode(params)}"

    async def start(self) -> None:
        self._socket = await websockets.connect(
            self._url(),
            additional_headers={"Authorization": f"Token {self._settings.deepgram_api_key}"},
            max_size=None,
        )
        self._reader = asyncio.create_task(self._read_loop())
        logger.info(
            "Deepgram streaming session open (model=%s, %d keyterms)",
            self._settings.deepgram_model,
            len(self._keyterms),
        )

    async def send_audio(self, pcm: bytes) -> None:
        if self._socket is None:
            return
        try:
            await self._socket.send(pcm)
        except Exception:
            # A dead streaming socket must never take the audio websocket
            # down with it: the VAD + batch path is still running and is
            # still the thing the pipeline acts on.
            logger.warning("Deepgram streaming send failed; continuing without it", exc_info=True)
            await self.aclose()

    async def _read_loop(self) -> None:
        assert self._socket is not None
        try:
            async for raw in self._socket:
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    continue

                kind = payload.get("type")
                if kind == "UtteranceEnd":
                    self._emit(
                        StreamingTranscript(
                            text="",
                            is_final=True,
                            confidence=1.0,
                            confidence_known=False,
                            endpoint_reason="utterance_end",
                        )
                    )
                    continue
                if kind and kind != "Results":
                    continue

                channel = payload.get("channel") or {}
                alternatives = channel.get("alternatives") or []
                if not alternatives:
                    continue
                best = alternatives[0]
                text = (best.get("transcript") or "").strip()
                if not text:
                    continue

                confidence, known = _confidence_from_alternative(best)
                is_final = bool(payload.get("is_final"))
                speech_final = bool(payload.get("speech_final"))
                self._emit(
                    StreamingTranscript(
                        text=text,
                        is_final=is_final or speech_final,
                        confidence=confidence,
                        confidence_known=known,
                        endpoint_reason=(
                            "speech_final" if speech_final else "is_final" if is_final else "interim"
                        ),
                        latency_ms=int((payload.get("duration") or 0.0) * 1000),
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Deepgram streaming read loop ended", exc_info=True)

    def _emit(self, transcript: StreamingTranscript) -> None:
        if self._on_transcript is None:
            return
        try:
            result = self._on_transcript(transcript)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        except Exception:
            logger.exception("Deepgram streaming transcript handler failed")

    async def aclose(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                await socket.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            try:
                await socket.close()
            except Exception:
                pass
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None


async def stream_transcripts(
    audio: AsyncIterator[bytes],
    *,
    settings: Settings,
    sample_rate: int,
    keyterms: list[str] | None = None,
) -> AsyncIterator[StreamingTranscript]:
    """Convenience wrapper used by the comparison harness, not by the app."""
    queue: asyncio.Queue[StreamingTranscript | None] = asyncio.Queue()
    session = DeepgramStreamingSession(
        settings=settings,
        sample_rate=sample_rate,
        keyterms=keyterms,
        on_transcript=queue.put_nowait,
    )
    await session.start()

    async def _pump() -> None:
        try:
            async for chunk in audio:
                await session.send_audio(chunk)
        finally:
            await asyncio.sleep(1.0)  # let trailing finals arrive
            await queue.put(None)

    pump = asyncio.create_task(_pump())
    try:
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item
    finally:
        pump.cancel()
        await session.aclose()
