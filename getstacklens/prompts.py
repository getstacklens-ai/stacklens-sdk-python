"""FlowOps prompt client."""

from __future__ import annotations

from ._client import _AsyncHttpClient, _HttpClient

_PROMPTS_PATH = "/api/v1/flowops/v1/prompts/by-name"


class PromptsClient:
    """
    FlowOps client for fetching versioned prompts at runtime.

    Prefer using the module-level ``getstacklens.prompts.get()`` helper over
    instantiating this directly.
    """

    def __init__(self, api_key: str, endpoint: str) -> None:
        self._http = _HttpClient(api_key=api_key, endpoint=endpoint)

    def get(self, name: str, *, env: str = "production") -> str:
        """
        Fetch the active prompt content for the given name and environment.

        Args:
            name: The prompt name as configured in FlowOps.
            env:  The environment — ``'dev'``, ``'staging'``, or ``'production'``
                  (default).

        Returns:
            The prompt content string.

        Example::

            system_prompt = getstacklens.prompts.get("support-system-prompt", env="production")
        """
        resp = self._http.get(f"{_PROMPTS_PATH}/{name}", params={"env": env})
        return resp["content"]


class AsyncPromptsClient:
    """
    Async FlowOps client for fetching versioned prompts at runtime.

    Prefer using the module-level ``getstacklens.aprompts.get()`` helper over
    instantiating this directly.
    """

    def __init__(self, api_key: str, endpoint: str) -> None:
        self._http = _AsyncHttpClient(api_key=api_key, endpoint=endpoint)

    async def get(self, name: str, *, env: str = "production") -> str:
        """
        Async version of :meth:`PromptsClient.get`.

        Example::

            system_prompt = await getstacklens.aprompts.get("support-system-prompt")
        """
        resp = await self._http.get(f"{_PROMPTS_PATH}/{name}", params={"env": env})
        return resp["content"]
