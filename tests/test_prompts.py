"""Tests for getstacklens.prompts.get()."""

from __future__ import annotations

import httpx
import pytest
import respx

import getstacklens
from getstacklens.exceptions import ConfigurationError

from .conftest import PROMPTS_URL


@respx.mock
def test_prompts_get_returns_content():
    respx.get(f"{PROMPTS_URL}/my-prompt").mock(
        return_value=httpx.Response(
            200, json={"content": "You are a helpful assistant."}
        )
    )
    result = getstacklens.prompts.get("my-prompt")
    assert result == "You are a helpful assistant."


@respx.mock
def test_prompts_get_passes_env_param():
    route = respx.get(f"{PROMPTS_URL}/sys-prompt").mock(
        return_value=httpx.Response(200, json={"content": "staging prompt"})
    )
    getstacklens.prompts.get("sys-prompt", env="staging")
    assert route.calls.last.request.url.params["env"] == "staging"


@respx.mock
def test_prompts_get_defaults_to_production():
    route = respx.get(f"{PROMPTS_URL}/sys-prompt").mock(
        return_value=httpx.Response(200, json={"content": "prod prompt"})
    )
    getstacklens.prompts.get("sys-prompt")
    assert route.calls.last.request.url.params["env"] == "production"


def test_prompts_raises_if_not_configured():
    getstacklens._prompts_client = None
    with pytest.raises(ConfigurationError):
        getstacklens.prompts.get("my-prompt")
