"""Tests for error handling and retry logic."""
from __future__ import annotations

import httpx
import pytest
import respx

import getstacklens
from getstacklens.exceptions import ApiError, AuthError, NetworkError

from .conftest import TRACES_URL


@respx.mock
def test_401_raises_auth_error():
    respx.post(TRACES_URL).mock(return_value=httpx.Response(401))
    with pytest.raises(AuthError):
        getstacklens.trace("x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1)


@respx.mock
def test_403_raises_auth_error():
    respx.post(TRACES_URL).mock(return_value=httpx.Response(403))
    with pytest.raises(AuthError):
        getstacklens.trace("x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1)


@respx.mock
def test_404_raises_api_error():
    respx.post(TRACES_URL).mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    with pytest.raises(ApiError) as exc_info:
        getstacklens.trace("x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1)
    assert exc_info.value.status_code == 404


@respx.mock
def test_api_error_includes_status_code():
    respx.post(TRACES_URL).mock(
        return_value=httpx.Response(422, json={"error": "validation failed"})
    )
    with pytest.raises(ApiError) as exc_info:
        getstacklens.trace("x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1)
    assert exc_info.value.status_code == 422
    assert "422" in str(exc_info.value)


@respx.mock
def test_retry_on_5xx_succeeds_on_second_attempt(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(500, json={"error": "internal server error"})
        return httpx.Response(200, json={})

    respx.post(TRACES_URL).mock(side_effect=side_effect)
    getstacklens.trace("x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1)
    assert call_count == 2


@respx.mock
def test_retry_on_5xx_raises_after_all_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    respx.post(TRACES_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
    with pytest.raises(ApiError) as exc_info:
        getstacklens.trace("x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1)
    assert exc_info.value.status_code == 500


@respx.mock
def test_network_error_raises_network_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    respx.post(TRACES_URL).mock(side_effect=httpx.ConnectError("connection refused"))
    with pytest.raises(NetworkError) as exc_info:
        getstacklens.trace("x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1)
    assert "getgetstacklens" in str(exc_info.value)


@respx.mock
def test_timeout_raises_network_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    respx.post(TRACES_URL).mock(side_effect=httpx.TimeoutException("timed out"))
    with pytest.raises(NetworkError):
        getstacklens.trace("x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1)


@respx.mock
def test_no_retry_on_4xx(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, json={"error": "bad request"})

    respx.post(TRACES_URL).mock(side_effect=side_effect)
    with pytest.raises(ApiError):
        getstacklens.trace("x", model="gpt-4o", provider="openai", input_tokens=1, output_tokens=1)
    assert call_count == 1  # no retry on client errors
