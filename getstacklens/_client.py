"""Internal HTTP client. Not part of the public API."""

from __future__ import annotations

import asyncio
import atexit
import time

import httpx

from .exceptions import ApiError, AuthError, NetworkError

_DEFAULT_TIMEOUT = 10.0
_RETRY_ATTEMPTS = 2
_RETRY_DELAY = 0.5  # seconds; only applied on 5xx and network errors


def _parse_response(resp: httpx.Response) -> dict:
    """Raise the appropriate SDK exception or return the parsed JSON body."""
    if resp.status_code == 401:
        raise AuthError("Invalid API key.")
    if resp.status_code == 403:
        raise AuthError("API key does not have the required scope.")
    if not resp.is_success:
        try:
            msg = resp.json().get("error", resp.text)
        except Exception:
            msg = resp.text
        raise ApiError(resp.status_code, msg)
    return resp.json()


class _HttpClient:
    def __init__(self, api_key: str, endpoint: str) -> None:
        self._base = endpoint.rstrip("/")
        self._http = httpx.Client(
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
            timeout=_DEFAULT_TIMEOUT,
        )
        atexit.register(self.close)

    def post(self, path: str, payload: dict) -> dict:
        return self._request("POST", path, json=payload)

    def get(self, path: str, params: dict | None = None) -> dict:
        return self._request("GET", path, params=params)

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self._base}{path}"
        last_exc: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = self._http.request(method, url, **kwargs)
                # Retry on server errors (5xx) but not on client errors (4xx)
                if resp.status_code >= 500 and attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_DELAY)
                    continue
                return _parse_response(resp)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_DELAY)
            except httpx.NetworkError as exc:
                last_exc = exc
                if attempt < _RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_DELAY)
        raise NetworkError(
            f"Could not reach GetStackLens API at {self._base}: {last_exc}"
        ) from last_exc

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass


class _AsyncHttpClient:
    def __init__(self, api_key: str, endpoint: str) -> None:
        self._base = endpoint.rstrip("/")
        self._http = httpx.AsyncClient(
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
            timeout=_DEFAULT_TIMEOUT,
        )

    async def post(self, path: str, payload: dict) -> dict:
        return await self._request("POST", path, json=payload)

    async def get(self, path: str, params: dict | None = None) -> dict:
        return await self._request("GET", path, params=params)

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self._base}{path}"
        last_exc: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                resp = await self._http.request(method, url, **kwargs)
                if resp.status_code >= 500 and attempt < _RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
                return _parse_response(resp)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < _RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(_RETRY_DELAY)
            except httpx.NetworkError as exc:
                last_exc = exc
                if attempt < _RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(_RETRY_DELAY)
        raise NetworkError(
            f"Could not reach GetStackLens API at {self._base}: {last_exc}"
        ) from last_exc

    async def close(self) -> None:
        try:
            await self._http.aclose()
        except Exception:
            pass
