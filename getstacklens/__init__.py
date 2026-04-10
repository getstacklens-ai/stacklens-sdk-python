"""
StackLens Python SDK
====================

Observability and governance for your AI stack.

Quickstart::

    import getstacklens

    getstacklens.configure(api_key="sl-xxxx")
    getstacklens.trace("my-llm-call", model="gpt-4o", provider="openai",
                    input_tokens=150, output_tokens=200)

Full example with context manager::

    with getstacklens.start_trace("agent-run") as span:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        span.record_llm(
            model="gpt-4o",
            provider="openai",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            completion=response.choices[0].message.content,
        )

Docs: https://getstacklens.ai/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import AsyncIterator, Iterator

from .exceptions import (
    ApiError,
    AuthError,
    ConfigurationError,
    NetworkError,
    StackLensError,
)
from .prompts import AsyncPromptsClient, PromptsClient
from .tracer import AsyncTracer, Span, Tracer

__version__ = "0.0.1"

__all__ = [
    "configure",
    "trace",
    "start_trace",
    "prompts",
    "atrace",
    "astart_trace",
    "aprompts",
    "Tracer",
    "AsyncTracer",
    "Span",
    "PromptsClient",
    "AsyncPromptsClient",
    "StackLensError",
    "ConfigurationError",
    "AuthError",
    "ApiError",
    "NetworkError",
]

_DEFAULT_ENDPOINT = "https://api.getstacklens.ai"

_tracer: Tracer | None = None
_prompts_client: PromptsClient | None = None
_async_tracer: AsyncTracer | None = None
_async_prompts_client: AsyncPromptsClient | None = None


def configure(api_key: str, endpoint: str = _DEFAULT_ENDPOINT) -> None:
    """
    Configure the StackLens SDK.

    Call this once at application startup before any tracing or prompt calls.

    Args:
        api_key:  Your StackLens API key (starts with ``sl-``).
                  Generate one from the StackLens dashboard under Settings → API Keys.
        endpoint: Override the API base URL for self-hosted deployments.
                  Defaults to ``https://api.getstacklens.ai``.

    Example::

        import getstacklens
        getstacklens.configure(api_key="sl-xxxx")

        # Self-hosted:
        getstacklens.configure(api_key="sl-xxxx", endpoint="https://api.your-domain.com")
    """
    global _tracer, _prompts_client, _async_tracer, _async_prompts_client
    _tracer = Tracer(api_key=api_key, endpoint=endpoint)
    _prompts_client = PromptsClient(api_key=api_key, endpoint=endpoint)
    _async_tracer = AsyncTracer(api_key=api_key, endpoint=endpoint)
    _async_prompts_client = AsyncPromptsClient(api_key=api_key, endpoint=endpoint)


def _require_tracer() -> Tracer:
    if _tracer is None:
        raise ConfigurationError(
            "StackLens is not configured. "
            "Call getstacklens.configure(api_key='sl-...') before tracing."
        )
    return _tracer


def trace(
    name: str,
    *,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int | None = None,
    cost_usd: float = 0.0,
    attributes: dict[str, str] | None = None,
    tags: list[str] | None = None,
    status: str = "ok",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> str:
    """
    Record a single LLM call and send it to StackLens. Returns the trace ID.

    This is the simplest tracing path — one call, no context managers needed.

    Args:
        name:         A descriptive name for this operation (e.g. ``"chat-completion"``).
        model:        The model name (e.g. ``"gpt-4o"``, ``"claude-3-5-sonnet"``).
        provider:     The provider name (e.g. ``"openai"``, ``"anthropic"``, ``"gemini"``).
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Total tokens (computed from input + output if omitted).
        cost_usd:     Estimated cost in USD.
        attributes:   Arbitrary key-value pairs attached to the span.
        tags:         String tags for filtering in the dashboard.
        status:       ``'ok'`` (default) or ``'error'``.
        start_time:   When the LLM call started. Record before the call for accurate latency.
        end_time:     When the LLM call ended. Defaults to now if omitted.

    Returns:
        The trace ID string.

    Example::

        start = datetime.now(timezone.utc)
        response = client.chat.completions.create(...)
        getstacklens.trace(
            "my-llm-call",
            model="gpt-4o",
            provider="openai",
            input_tokens=150,
            output_tokens=200,
            start_time=start,
        )
    """
    return _require_tracer().record(
        name,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        attributes=attributes,
        tags=tags,
        status=status,
        start_time=start_time,
        end_time=end_time,
    )


@contextmanager
def start_trace(name: str) -> Iterator[Span]:
    """
    Context manager for tracing a multi-step or agent operation.

    The span is sent to StackLens when the context exits. If an exception
    is raised, the span status is automatically set to ``'error'``.

    Args:
        name: A descriptive name for this trace (e.g. ``"agent-run"``).

    Yields:
        :class:`~getstacklens.tracer.Span` — call :meth:`~getstacklens.tracer.Span.record_llm`
        on it to attach LLM metadata.

    Example::

        with getstacklens.start_trace("agent-run") as span:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )
            span.record_llm(
                model="gpt-4o",
                provider="openai",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                completion=response.choices[0].message.content,
            )
    """
    with _require_tracer().start_trace(name) as span:
        yield span


def _require_async_tracer() -> AsyncTracer:
    if _async_tracer is None:
        raise ConfigurationError(
            "StackLens is not configured. "
            "Call getstacklens.configure(api_key='sl-...') before tracing."
        )
    return _async_tracer


async def atrace(
    name: str,
    *,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int | None = None,
    cost_usd: float = 0.0,
    attributes: dict[str, str] | None = None,
    tags: list[str] | None = None,
    status: str = "ok",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> str:
    """
    Async version of :func:`trace`. Record a single LLM call. Returns the trace ID.

    Use this in asyncio / FastAPI applications instead of the sync :func:`trace`.

    Example::

        trace_id = await getstacklens.atrace(
            "chat-completion",
            model="gpt-4o",
            provider="openai",
            input_tokens=150,
            output_tokens=200,
        )
    """
    return await _require_async_tracer().arecord(
        name,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        attributes=attributes,
        tags=tags,
        status=status,
        start_time=start_time,
        end_time=end_time,
    )


@asynccontextmanager
async def astart_trace(name: str) -> AsyncIterator[Span]:
    """
    Async context manager for tracing a multi-step or agent operation.

    Use this in asyncio / FastAPI applications instead of the sync :func:`start_trace`.

    Example::

        async with getstacklens.astart_trace("agent-run") as span:
            response = await async_client.chat.completions.create(...)
            span.record_llm(
                model="gpt-4o",
                provider="openai",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
    """
    async with _require_async_tracer().astart_trace(name) as span:
        yield span


class _PromptsNamespace:
    """Access FlowOps versioned prompts. Use ``getstacklens.prompts.get()``."""

    def get(self, name: str, *, env: str = "production") -> str:
        """
        Fetch the active prompt for the given name and environment.

        Args:
            name: The prompt name as configured in FlowOps.
            env:  The environment — ``'dev'``, ``'staging'``, or ``'production'``
                  (default).

        Returns:
            The prompt content string.

        Example::

            system_prompt = getstacklens.prompts.get("support-system-prompt")
            user_prompt = getstacklens.prompts.get("onboarding-email", env="staging")
        """
        if _prompts_client is None:
            raise ConfigurationError(
                "StackLens is not configured. "
                "Call getstacklens.configure(api_key='sl-...') first."
            )
        return _prompts_client.get(name, env=env)


prompts = _PromptsNamespace()


class _AsyncPromptsNamespace:
    """Async access to FlowOps versioned prompts. Use ``await getstacklens.aprompts.get()``."""

    async def get(self, name: str, *, env: str = "production") -> str:
        """
        Async version of :meth:`_PromptsNamespace.get`.

        Args:
            name: The prompt name as configured in FlowOps.
            env:  The environment — ``'dev'``, ``'staging'``, or ``'production'``
                  (default).

        Returns:
            The prompt content string.

        Example::

            system_prompt = await getstacklens.aprompts.get("support-system-prompt")
        """
        if _async_prompts_client is None:
            raise ConfigurationError(
                "StackLens is not configured. "
                "Call getstacklens.configure(api_key='sl-...') first."
            )
        return await _async_prompts_client.get(name, env=env)


aprompts = _AsyncPromptsNamespace()
