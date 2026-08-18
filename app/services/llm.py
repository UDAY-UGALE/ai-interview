from collections.abc import AsyncIterator
from typing import Protocol

from groq import AsyncGroq

from app.core.config import Settings


try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


ChatMessage = dict[str, str]

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Curated catalog for the frontend model picker. Any model string still
# works even if it's not listed here -- this is just what's offered as
# quick-pick options.
MODEL_CATALOG: dict[str, list[str]] = {
    "groq": [
        # llama-3.3-70b-versatile and llama-3.1-8b-instant were deprecated
        # and removed by Groq on 2026-08-16 (404 model_not_found) -- these
        # are Groq's own recommended replacements.
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
    ],
    "anthropic": [
        "claude-3-5-sonnet-latest",
        "claude-haiku-4-5-20251001",
        "claude-3-opus-latest",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
}


class MissingLLMConfigError(RuntimeError):
    pass


def client_supports_vision(client: "StreamingLLMClient") -> bool:
    """Vision support isn't part of the StreamingLLMClient Protocol (not
    every provider/model can take an image), so callers that need it check
    for the stream_vision_chat method at runtime instead."""
    return hasattr(client, "stream_vision_chat")


def _groq_reasoning_kwargs(model: str) -> dict:
    """Some Groq models (Qwen3.x, GPT-OSS) are reasoning models that emit
    their internal <think>...</think> chain-of-thought as part of the
    output by default -- that's what was leaking into answers as raw
    thinking traces instead of a clean final answer. These params tell Groq
    to keep reasoning ON (still needed for correctness on coding/math) but
    hide it from the streamed output, so only the final answer comes
    through. Deliberately a no-op (returns {}) for non-reasoning models
    (e.g. llama-3.3-70b-versatile) since passing unknown params to those
    would error."""
    if model.startswith("qwen/"):
        return {"reasoning_effort": "default", "reasoning_format": "hidden"}
    if model.startswith("openai/gpt-oss"):
        return {"include_reasoning": False}
    return {}


class StreamingLLMClient(Protocol):
    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        ...


# Cached per (provider, model, max_tokens, temperature) so the underlying
# SDK client -- and its HTTP connection pool -- is reused across questions
# instead of rebuilt from scratch every time. Building a fresh client per
# question means paying a full new-connection/TLS-handshake cost on every
# single answer, which adds up and gets worse as a conversation goes on;
# reusing one keeps the connection warm. max_tokens/temperature are part of
# the key too since screen analysis intentionally uses a higher max_tokens
# than the short spoken interview answers do, even when pointed at the same
# provider/model.
_client_cache: dict[tuple[str, str, int, float], StreamingLLMClient] = {}


def build_llm_client(
    settings: Settings,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> StreamingLLMClient:
    """Build (or reuse a cached) streaming LLM client. `provider`/`model` let
    a caller override the global .env defaults -- e.g. a per-session choice
    made from the frontend model picker, or the fixed vision provider/model
    used by screen analysis -- without mutating global settings. Likewise
    `max_tokens`/`temperature` let a caller (e.g. screen analysis, which
    needs more room than a short spoken answer) override those per-call."""
    resolved_provider = (provider or settings.answer_provider).lower()
    resolved_model = model or settings.answer_model
    resolved_max_tokens = max_tokens if max_tokens is not None else settings.answer_max_tokens
    resolved_temperature = (
        temperature if temperature is not None else settings.answer_temperature
    )

    cache_key = (resolved_provider, resolved_model, resolved_max_tokens, resolved_temperature)
    cached = _client_cache.get(cache_key)
    if cached is not None:
        return cached

    client = _build_new_client(
        settings, resolved_provider, resolved_model, resolved_max_tokens, resolved_temperature
    )
    _client_cache[cache_key] = client
    return client


def _build_new_client(
    settings: Settings,
    resolved_provider: str,
    resolved_model: str,
    max_tokens: int,
    temperature: float,
) -> StreamingLLMClient:
    if resolved_provider == "groq":
        if not settings.groq_api_key:
            raise MissingLLMConfigError("Set GROQ_API_KEY to use provider=groq.")
        return GroqLLMClient(settings, model=resolved_model, max_tokens=max_tokens, temperature=temperature)

    if resolved_provider == "openai":
        if AsyncOpenAI is None:
            raise MissingLLMConfigError("Install openai to use provider=openai.")
        if not settings.openai_api_key:
            raise MissingLLMConfigError("Set OPENAI_API_KEY to use provider=openai.")
        return OpenAILLMClient(settings, model=resolved_model, max_tokens=max_tokens, temperature=temperature)

    if resolved_provider == "anthropic":
        if AsyncAnthropic is None:
            raise MissingLLMConfigError("Install anthropic to use provider=anthropic.")
        if not settings.anthropic_api_key:
            raise MissingLLMConfigError("Set ANTHROPIC_API_KEY to use provider=anthropic.")
        return AnthropicLLMClient(settings, model=resolved_model, max_tokens=max_tokens, temperature=temperature)

    if resolved_provider == "deepseek":
        if AsyncOpenAI is None:
            raise MissingLLMConfigError(
                "Install openai (used as the DeepSeek client) to use provider=deepseek."
            )
        if not settings.deepseek_api_key:
            raise MissingLLMConfigError("Set DEEPSEEK_API_KEY to use provider=deepseek.")
        return DeepSeekLLMClient(settings, model=resolved_model, max_tokens=max_tokens, temperature=temperature)

    raise MissingLLMConfigError(f"Unsupported provider={resolved_provider}.")


class GroqLLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = model or settings.answer_model
        self._max_tokens = max_tokens if max_tokens is not None else settings.answer_max_tokens
        self._temperature = (
            temperature if temperature is not None else settings.answer_temperature
        )

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
            **_groq_reasoning_kwargs(self._model),
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    async def stream_vision_chat(
        self, *, system_prompt: str, user_text: str, image_base64: str, media_type: str
    ) -> AsyncIterator[str]:
        # NOTE: Groq's vision-capable model lineup has churned a lot in 2026
        # (Llama 4 Scout/Maverick were both deprecated). As of this writing
        # the current one is qwen/qwen3.6-27b -- see
        # https://console.groq.com/docs/vision for whatever's current if
        # this model string stops working.
        data_url = f"data:{media_type};base64,{image_base64}"
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
            **_groq_reasoning_kwargs(self._model),
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token


class OpenAILLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.answer_model
        self._max_tokens = max_tokens if max_tokens is not None else settings.answer_max_tokens
        self._temperature = (
            temperature if temperature is not None else settings.answer_temperature
        )

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

    async def stream_vision_chat(
        self, *, system_prompt: str, user_text: str, image_base64: str, media_type: str
    ) -> AsyncIterator[str]:
        data_url = f"data:{media_type};base64,{image_base64}"
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token


class DeepSeekLLMClient:
    """DeepSeek's API is OpenAI-compatible, so we reuse the OpenAI SDK
    pointed at DeepSeek's base URL instead of writing a new HTTP client."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=DEEPSEEK_BASE_URL)
        self._model = model or "deepseek-chat"
        self._max_tokens = max_tokens if max_tokens is not None else settings.answer_max_tokens
        self._temperature = (
            temperature if temperature is not None else settings.answer_temperature
        )

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token


class AnthropicLLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = model or settings.answer_model
        self._max_tokens = max_tokens if max_tokens is not None else settings.answer_max_tokens
        self._temperature = (
            temperature if temperature is not None else settings.answer_temperature
        )

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        system_prompt = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        claude_messages = [
            message for message in messages if message["role"] in {"user", "assistant"}
        ]
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=claude_messages,
        ) as stream:
            async for token in stream.text_stream:
                if token:
                    yield token

    async def stream_vision_chat(
        self, *, system_prompt: str, user_text: str, image_base64: str, media_type: str
    ) -> AsyncIterator[str]:
        claude_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=claude_messages,
        ) as stream:
            async for token in stream.text_stream:
                if token:
                    yield token
