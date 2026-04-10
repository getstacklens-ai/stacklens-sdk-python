"""Tests for getstacklens.trace() and getstacklens.start_trace()."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
import respx

import getstacklens
from getstacklens.exceptions import ConfigurationError

from .conftest import TRACES_URL


@respx.mock
def test_trace_returns_trace_id():
    respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    trace_id = getstacklens.trace(
        "chat-completion",
        model="gpt-4o",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
    )
    assert isinstance(trace_id, str)
    assert len(trace_id) == 36  # UUID format


@respx.mock
def test_trace_sends_correct_payload():
    route = respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    getstacklens.trace(
        "my-call",
        model="claude-3-5-sonnet",
        provider="anthropic",
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.005,
        attributes={"user_id": "u_42"},
        tags=["production"],
        status="ok",
    )
    body = route.calls.last.request.content
    import json

    payload = json.loads(body)
    assert payload["name"] == "my-call"
    assert payload["kind"] == "llm"
    assert payload["status"] == "ok"
    assert payload["llmSpan"]["model"] == "claude-3-5-sonnet"
    assert payload["llmSpan"]["provider"] == "anthropic"
    assert payload["llmSpan"]["inputTokens"] == 200
    assert payload["llmSpan"]["outputTokens"] == 100
    assert payload["llmSpan"]["totalTokens"] == 300
    assert payload["llmSpan"]["estimatedCostUsd"] == 0.005
    assert payload["attributes"] == {"user_id": "u_42"}
    assert payload["tags"] == ["production"]


@respx.mock
def test_trace_with_start_time():
    route = respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    getstacklens.trace(
        "timed-call",
        model="gpt-4o",
        provider="openai",
        input_tokens=10,
        output_tokens=10,
        start_time=start,
    )
    import json

    payload = json.loads(route.calls.last.request.content)
    assert payload["startTime"] == "2024-01-01T12:00:00+00:00"


@respx.mock
def test_trace_total_tokens_computed_when_omitted():
    route = respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    getstacklens.trace(
        "call",
        model="gpt-4o",
        provider="openai",
        input_tokens=30,
        output_tokens=70,
    )
    import json

    payload = json.loads(route.calls.last.request.content)
    assert payload["llmSpan"]["totalTokens"] == 100


@respx.mock
def test_start_trace_context_manager_happy_path():
    route = respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    with getstacklens.start_trace("agent-run") as span:
        span.record_llm(
            model="gpt-4o",
            provider="openai",
            input_tokens=50,
            output_tokens=25,
            completion="Hello!",
        )
        span.set_attribute("user_id", "u_1")
        span.add_tag("support")
    assert route.called
    import json

    payload = json.loads(route.calls.last.request.content)
    assert payload["name"] == "agent-run"
    assert payload["status"] == "ok"
    assert payload["llmSpan"]["completionContent"] == "Hello!"
    assert payload["attributes"] == {"user_id": "u_1"}
    assert payload["tags"] == ["support"]


@respx.mock
def test_start_trace_sets_error_status_on_exception():
    route = respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        with getstacklens.start_trace("failing-agent") as span:
            raise ValueError("something went wrong")
    import json

    payload = json.loads(route.calls.last.request.content)
    assert payload["status"] == "error"


def test_trace_raises_if_not_configured():
    getstacklens._tracer = None
    with pytest.raises(ConfigurationError):
        getstacklens.trace(
            "x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1
        )


def test_start_trace_raises_if_not_configured():
    getstacklens._tracer = None
    with pytest.raises(ConfigurationError):
        with getstacklens.start_trace("x"):
            pass
