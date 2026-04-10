"""Tests for async API: getstacklens.atrace(), getstacklens.astart_trace(), getstacklens.aprompts."""

from __future__ import annotations

import httpx
import pytest
import respx

import getstacklens
from getstacklens.exceptions import AuthError, ConfigurationError, NetworkError

from .conftest import PROMPTS_URL, TRACES_URL


@respx.mock
async def test_atrace_returns_trace_id():
    respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    trace_id = await getstacklens.atrace(
        "async-call",
        model="gpt-4o",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
    )
    assert isinstance(trace_id, str)
    assert len(trace_id) == 36


@respx.mock
async def test_atrace_sends_correct_payload():
    route = respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    await getstacklens.atrace(
        "my-async-call",
        model="gpt-4o",
        provider="openai",
        input_tokens=20,
        output_tokens=10,
        attributes={"env": "test"},
        tags=["async"],
    )
    import json

    payload = json.loads(route.calls.last.request.content)
    assert payload["name"] == "my-async-call"
    assert payload["llmSpan"]["inputTokens"] == 20
    assert payload["attributes"] == {"env": "test"}
    assert payload["tags"] == ["async"]


@respx.mock
async def test_astart_trace_happy_path():
    route = respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    async with getstacklens.astart_trace("async-agent") as span:
        span.record_llm(
            model="gpt-4o",
            provider="openai",
            input_tokens=10,
            output_tokens=5,
        )
    assert route.called
    import json

    payload = json.loads(route.calls.last.request.content)
    assert payload["name"] == "async-agent"
    assert payload["status"] == "ok"


@respx.mock
async def test_astart_trace_sets_error_on_exception():
    route = respx.post(TRACES_URL).mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(RuntimeError):
        async with getstacklens.astart_trace("failing-async") as _:
            raise RuntimeError("async failure")
    import json

    payload = json.loads(route.calls.last.request.content)
    assert payload["status"] == "error"


@respx.mock
async def test_aprompts_get_returns_content():
    respx.get(f"{PROMPTS_URL}/async-prompt").mock(
        return_value=httpx.Response(200, json={"content": "Async system prompt."})
    )
    result = await getstacklens.aprompts.get("async-prompt")
    assert result == "Async system prompt."


@respx.mock
async def test_aprompts_get_passes_env():
    route = respx.get(f"{PROMPTS_URL}/sys").mock(
        return_value=httpx.Response(200, json={"content": "dev prompt"})
    )
    await getstacklens.aprompts.get("sys", env="dev")
    assert route.calls.last.request.url.params["env"] == "dev"


@respx.mock
async def test_atrace_401_raises_auth_error():
    respx.post(TRACES_URL).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        await getstacklens.atrace(
            "x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1
        )


@respx.mock
async def test_atrace_network_error_raises_network_error(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    respx.post(TRACES_URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(NetworkError):
        await getstacklens.atrace(
            "x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1
        )


@respx.mock
async def test_atrace_retry_on_5xx(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(500, json={"error": "server error"})
        return httpx.Response(200, json={})

    respx.post(TRACES_URL).mock(side_effect=side_effect)
    await getstacklens.atrace(
        "x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1
    )
    assert call_count == 2


async def test_atrace_raises_if_not_configured():
    getstacklens._async_tracer = None
    with pytest.raises(ConfigurationError):
        await getstacklens.atrace(
            "x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1
        )


async def test_aprompts_raises_if_not_configured():
    getstacklens._async_prompts_client = None
    with pytest.raises(ConfigurationError):
        await getstacklens.aprompts.get("x")
