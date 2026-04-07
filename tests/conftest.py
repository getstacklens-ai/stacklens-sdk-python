"""Shared test fixtures."""
from __future__ import annotations

import pytest

import stacklens

API_KEY = "sl-test-key"
ENDPOINT = "https://api.getstacklens.ai"
TRACES_URL = f"{ENDPOINT}/api/v1/stacktrace/v1/traces"
PROMPTS_URL = f"{ENDPOINT}/api/v1/flowops/v1/prompts/by-name"


@pytest.fixture(autouse=True)
def configure_sdk():
    """Configure the SDK with a test key before each test; reset global state after."""
    stacklens.configure(api_key=API_KEY, endpoint=ENDPOINT)
    yield
    stacklens._tracer = None
    stacklens._prompts_client = None
    stacklens._async_tracer = None
    stacklens._async_prompts_client = None
