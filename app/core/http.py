from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.errors import UpstreamError


@dataclass
class CircuitState:
    failures: int = 0
    open_until: float = 0.0


class ResilientHTTPClient:
    def __init__(
        self,
        timeout: float = 10.0,
        retries: int = 3,
        backoff_base: float = 0.4,
        breaker_threshold: int = 4,
        breaker_cooldown: int = 60,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.backoff_base = backoff_base
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown = breaker_cooldown
        self._client = httpx.AsyncClient(timeout=timeout)
        self._state: dict[str, CircuitState] = {}

    async def close(self) -> None:
        await self._client.aclose()

    def _is_open(self, host: str) -> bool:
        state = self._state.setdefault(host, CircuitState())
        return state.open_until > time.time()

    def _record_failure(self, host: str) -> None:
        state = self._state.setdefault(host, CircuitState())
        state.failures += 1
        if state.failures >= self.breaker_threshold:
            state.open_until = time.time() + self.breaker_cooldown

    def _record_success(self, host: str) -> None:
        self._state[host] = CircuitState()

    async def _request_json(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        host = httpx.URL(url).host or "unknown"
        if self._is_open(host):
            raise UpstreamError(f"Circuit open for {host}")

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = await self._client.request(method, url, params=params, headers=headers, json=json)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise UpstreamError(f"Transient status {response.status_code}")
                response.raise_for_status()
                self._record_success(host)
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._record_failure(host)
                if attempt >= self.retries:
                    break
                await asyncio.sleep(self.backoff_base * (2**attempt))

        raise UpstreamError(f"Failed to fetch {url}: {last_error}")

    async def get_json(
        self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> Any:
        return await self._request_json("GET", url, params=params, headers=headers)

    async def post_json(
        self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> Any:
        return await self._request_json("POST", url, headers=headers, json=payload)
