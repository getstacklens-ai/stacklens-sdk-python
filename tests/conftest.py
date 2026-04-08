"""Shared test fixtures."""
from __future__ import annotations

import pytest

import getstacklens

API_KEY = "sl-test-key"
ENDPOINT = "https://api.getgetstacklens.ai"
TRACES_URL = f"{ENDPOINT}/api/v1/stacktrace/v1/traces"
PROMPTS_URL = f"{ENDPOINT}/api/v1/flowops/v1/prompts/by-name"


@pytest.fixture(autouse=True)
def configure_sdk():
    """Configure the SDK with a test key before each test; reset global state after."""
    getstacklens.configure(api_key=API_KEY, endpoint=ENDPOINT)
    yield
    getstacklens._tracer = None
    getstacklens._prompts_client = None
    getstacklens._async_tracer = None
    getstacklens._async_prompts_client = None
