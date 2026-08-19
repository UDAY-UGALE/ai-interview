"""Wrappers that combine two STT services into one."""

from __future__ import annotations

import asyncio
import logging

from app.services.stt.base import TranscriptionResult


logger = logging.getLogger(__name__)


class FallbackSTTService:
    """Wraps a primary STT service with a secondary one. If the primary call
    fails (a network blip, a rate limit, a provider outage), automatically
    retries the same audio against the fallback provider instead of losing
    that segment."""

    def __init__(self, primary, fallback) -> None:
        self._primary = primary
        self._fallback = fallback
        self.name = f"{getattr(primary, 'name', 'primary')}+{getattr(fallback, 'name', 'fallback')}"

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        try:
            return await self._primary.transcribe_pcm16(pcm, sample_rate=sample_rate)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._fallback is None:
                raise
            logger.warning("Primary STT provider failed; retrying with fallback", exc_info=True)
            return await self._fallback.transcribe_pcm16(pcm, sample_rate=sample_rate)


class RacingSTTService:
    """Calls the primary and fallback providers CONCURRENTLY and returns
    whichever finishes first (successfully).

    Trades an extra STT call every time for a lower worst-case latency when
    one provider is having a slow moment. Note the cost side carefully: this
    doubles the request rate against both providers, which is a real way to
    hit a per-minute rate limit -- leave it off unless the latency win is
    actually needed.
    """

    def __init__(self, primary, fallback) -> None:
        self._primary = primary
        self._fallback = fallback
        self.name = f"{getattr(primary, 'name', 'primary')}|{getattr(fallback, 'name', 'fallback')}"

    async def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> TranscriptionResult:
        tasks = [
            asyncio.create_task(self._primary.transcribe_pcm16(pcm, sample_rate=sample_rate)),
            asyncio.create_task(self._fallback.transcribe_pcm16(pcm, sample_rate=sample_rate)),
        ]
        pending = set(tasks)
        last_error: Exception | None = None

        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exc = task.exception()
                    if exc is None:
                        return task.result()
                    last_error = exc
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

        raise last_error or RuntimeError("Both STT providers failed")
