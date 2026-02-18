from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.adapters.symbols import normalize_symbol
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


class DerivativesAdapter:
    def __init__(self, http: ResilientHTTPClient, cache: RedisCache, futures_base: str) -> None:
        self.http = http
        self.cache = cache
        self.futures_base = futures_base

    async def get_funding_and_oi(self, symbol: str) -> dict:
        meta = normalize_symbol(symbol)
        key = f"funding:{meta.base}"
        cached = await self.cache.get_json(key)
        if cached:
            return cached

        payload = {
            "symbol": meta.base,
            "funding_rate": None,
            "open_interest": None,
            "source": "unavailable",
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        premium_task = asyncio.wait_for(
            self.http.get_json(
                f"{self.futures_base}/fapi/v1/premiumIndex",
                params={"symbol": meta.pair},
            ),
            timeout=5.0,
        )
        oi_task = asyncio.wait_for(
            self.http.get_json(
                f"{self.futures_base}/fapi/v1/openInterest",
                params={"symbol": meta.pair},
            ),
            timeout=5.0,
        )
        premium, oi = await asyncio.gather(premium_task, oi_task, return_exceptions=True)

        if not isinstance(premium, Exception):
            payload["funding_rate"] = (
                float(premium.get("lastFundingRate")) if premium.get("lastFundingRate") is not None else None
            )
            payload["source"] = "binance_futures"

        if not isinstance(oi, Exception):
            payload["open_interest"] = float(oi.get("openInterest")) if oi.get("openInterest") is not None else None
            payload["source"] = "binance_futures"

        await self.cache.set_json(key, payload, ttl=60)
        return payload
