"""StackTrace tracing client."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Iterator

from ._client import _AsyncHttpClient, _HttpClient

_TRACES_PATH = "/api/v1/stacktrace/v1/traces"


class Span:
    """
    A single span within a trace.

    Create spans via :func:`getstacklens.start_trace` rather than instantiating directly.
    """

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
    ) -> None:
        self._name = name
        self._trace_id = trace_id
        self._span_id = span_id
        self._parent_span_id = parent_span_id
        self._start = datetime.now(timezone.utc)
        self._llm: dict | None = None
        self._attributes: dict[str, str] = {}
        self._tags: list[str] = []
        self._status = "ok"

    def set_attribute(self, key: str, value: str) -> "Span":
        """Attach a key-value attribute to this span."""
        self._attributes[str(key)] = str(value)
        return self

    def add_tag(self, *tags: str) -> "Span":
        """Add one or more string tags to this span."""
        self._tags.extend(tags)
        return self

    def set_status(self, status: str) -> "Span":
        """Set span status: ``'ok'`` (default) or ``'error'``."""
        self._status = status
        return self

    def record_llm(
        self,
        *,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int | None = None,
        cost_usd: float = 0.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        prompt: str | None = None,
        completion: str | None = None,
        is_streaming: bool = False,
        finish_reason: str | None = None,
    ) -> "Span":
        """
        Record LLM call metadata for this span.

        Example::

            with getstacklens.start_trace("chat") as span:
                response = openai_client.chat.completions.create(...)
                span.record_llm(
                    model="gpt-4o",
                    provider="openai",
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )
        """
        self._llm = {
            "model": model,
            "provider": provider,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens if total_tokens is not None else input_tokens + output_tokens,
            "estimatedCostUsd": cost_usd,
            "temperature": temperature,
            "maxTokens": max_tokens,
            "promptContent": prompt,
            "completionContent": completion,
            "isStreaming": is_streaming,
            "finishReason": finish_reason,
        }
        return self

    def _to_payload(self) -> dict:
        end = datetime.now(timezone.utc)
        return {
            "traceId": self._trace_id,
            "spanId": self._span_id,
            "parentSpanId": self._parent_span_id,
            "name": self._name,
            "kind": "llm" if self._llm else "internal",
            "status": self._status,
            "startTime": self._start.isoformat(),
            "endTime": end.isoformat(),
            "attributes": self._attributes,
            "tags": self._tags,
            "llmSpan": self._llm,
        }


class Tracer:
    """
    StackTrace client.

    Prefer using the module-level :func:`getstacklens.trace` and
    :func:`getstacklens.start_trace` helpers over instantiating this directly.
    """

    def __init__(self, api_key: str, endpoint: str) -> None:
        self._http = _HttpClient(api_key=api_key, endpoint=endpoint)

    def record(
        self,
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
        Send a single LLM span. Returns the trace ID.

        For accurate latency, record ``start_time`` before the LLM call and
        ``end_time`` after::

            start = datetime.now(timezone.utc)
            response = client.chat.completions.create(...)
            tracer.record("chat", model="gpt-4o", provider="openai",
                          input_tokens=..., output_tokens=...,
                          start_time=start, end_time=datetime.now(timezone.utc))

        If omitted, both timestamps are set to the moment ``record()`` is called
        and the span will show 0 ms duration.
        """
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        recorded_at = datetime.now(timezone.utc)
        payload = {
            "traceId": trace_id,
            "spanId": span_id,
            "parentSpanId": None,
            "name": name,
            "kind": "llm",
            "status": status,
            "startTime": (start_time or recorded_at).isoformat(),
            "endTime": (end_time or recorded_at).isoformat(),
            "attributes": attributes or {},
            "tags": tags or [],
            "llmSpan": {
                "model": model,
                "provider": provider,
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": total_tokens if total_tokens is not None else input_tokens + output_tokens,
                "estimatedCostUsd": cost_usd,
                "temperature": None,
                "maxTokens": None,
                "promptContent": None,
                "completionContent": None,
                "isStreaming": False,
                "finishReason": None,
            },
        }
        self._http.post(_TRACES_PATH, payload)
        return trace_id

    @contextmanager
    def start_trace(self, name: str) -> Iterator[Span]:
        """
        Context manager for tracing a multi-step or agent operation.

        Flushes the span to StackLens when the context exits (including on error).

        Example::

            with tracer.start_trace("my-agent-run") as span:
                response = client.chat.completions.create(...)
                span.record_llm(model="gpt-4o", provider="openai",
                                input_tokens=150, output_tokens=200)
        """
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        span = Span(name=name, trace_id=trace_id, span_id=span_id, parent_span_id=None)
        try:
            yield span
        except Exception:
            span.set_status("error")
            raise
        finally:
            self._http.post(_TRACES_PATH, span._to_payload())


class AsyncTracer:
    """
    Async StackTrace client for use with asyncio / FastAPI / async frameworks.

    Prefer the module-level :func:`getstacklens.atrace` and
    :func:`getstacklens.astart_trace` helpers over instantiating this directly.
    """

    def __init__(self, api_key: str, endpoint: str) -> None:
        self._http = _AsyncHttpClient(api_key=api_key, endpoint=endpoint)

    async def arecord(
        self,
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
        """Async version of :meth:`Tracer.record`. Returns the trace ID."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        recorded_at = datetime.now(timezone.utc)
        payload = {
            "traceId": trace_id,
            "spanId": span_id,
            "parentSpanId": None,
            "name": name,
            "kind": "llm",
            "status": status,
            "startTime": (start_time or recorded_at).isoformat(),
            "endTime": (end_time or recorded_at).isoformat(),
            "attributes": attributes or {},
            "tags": tags or [],
            "llmSpan": {
                "model": model,
                "provider": provider,
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": total_tokens if total_tokens is not None else input_tokens + output_tokens,
                "estimatedCostUsd": cost_usd,
                "temperature": None,
                "maxTokens": None,
                "promptContent": None,
                "completionContent": None,
                "isStreaming": False,
                "finishReason": None,
            },
        }
        await self._http.post(_TRACES_PATH, payload)
        return trace_id

    @asynccontextmanager
    async def astart_trace(self, name: str) -> AsyncIterator[Span]:
        """
        Async context manager for tracing a multi-step or agent operation.

        Flushes the span to StackLens when the context exits (including on error).

        Example::

            async with tracer.astart_trace("my-agent-run") as span:
                response = await async_openai_client.chat.completions.create(...)
                span.record_llm(model="gpt-4o", provider="openai",
                                input_tokens=150, output_tokens=200)
        """
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        span = Span(name=name, trace_id=trace_id, span_id=span_id, parent_span_id=None)
        try:
            yield span
        except Exception:
            span.set_status("error")
            raise
        finally:
            await self._http.post(_TRACES_PATH, span._to_payload())
