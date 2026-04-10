# stacklens-sdk-python

Python SDK for [StackLens](https://getstacklens.ai) — observability and governance for your AI stack.

Trace LLM calls, fetch versioned prompts, and enforce AI governance policies — in three lines of Python.

## Installation

```bash
pip install getstacklens
```

Requires Python 3.9+.

## Quickstart

```python
import getstacklens

getstacklens.configure(api_key="sl-xxxx")
getstacklens.trace("my-llm-call", model="gpt-4o", provider="openai", input_tokens=150, output_tokens=200)
```

Get your API key from the [StackLens dashboard](https://app.getstacklens.ai) under **Settings → API Keys**.

## Tracing LLM calls

### Simple trace (one line)

For accurate latency, record `start_time` before the call and pass it in:

```python
from datetime import datetime, timezone
import getstacklens

getstacklens.configure(api_key="sl-xxxx")

start = datetime.now(timezone.utc)
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarise this document."}],
)
getstacklens.trace(
    "chat-completion",
    model="gpt-4o",
    provider="openai",
    input_tokens=response.usage.prompt_tokens,
    output_tokens=response.usage.completion_tokens,
    start_time=start,
)
```

### Context manager (recommended for agent workflows)

```python
import openai
import getstacklens

getstacklens.configure(api_key="sl-xxxx")
client = openai.OpenAI()

with getstacklens.start_trace("customer-support-agent") as span:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "How do I reset my password?"}],
    )
    span.record_llm(
        model="gpt-4o",
        provider="openai",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        completion=response.choices[0].message.content,
    )
    span.set_attribute("user_id", "u_123")
    span.add_tag("support", "production")
```

If an exception is raised inside the context, the span status is automatically set to `error`.

## Fetching versioned prompts (FlowOps)

Manage prompts in the StackLens dashboard, then fetch them at runtime — no deploys needed.

```python
import getstacklens

getstacklens.configure(api_key="sl-xxxx")

# Fetch the active prompt for the production environment
system_prompt = getstacklens.prompts.get("support-system-prompt", env="production")

# Use in an LLM call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
)
```

Available environments: `"dev"`, `"staging"`, `"production"` (default).

## Self-hosted deployments

Point the SDK at your own StackLens instance:

```python
getstacklens.configure(
    api_key="sl-xxxx",
    endpoint="https://api.your-domain.com",
)
```

See the [self-hosting guide](https://getstacklens.ai/docs/self-hosting) for setup instructions.

## Supported providers

Works with any LLM provider — pass the model and provider name you use:

| Provider | `provider` value |
|---|---|
| OpenAI | `"openai"` |
| Anthropic | `"anthropic"` |
| Google Gemini | `"gemini"` |
| Azure OpenAI | `"azure-openai"` |
| AWS Bedrock | `"bedrock"` |
| Any other | any string |

## Links

- [Documentation](https://getstacklens.ai/docs)
- [StackLens Platform](https://getstacklens.ai)
- [Dashboard](https://app.getstacklens.ai)
- [GitHub](https://github.com/getstacklens-ai/stacklens-sdk-python)
- [Report an issue](https://github.com/getstacklens-ai/stacklens-sdk-python/issues)
