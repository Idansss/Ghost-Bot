from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
import respx

from app.adapters.prices import PriceAdapter


class _DummyCache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    async def get_json(self, key: str):
        return self.store.get(key)

    async def set_json(self, key: str, value, ttl: int):
        self.store[key] = value


@dataclass
class _DummyHTTP:
    async def get_json(self, url: str, params=None, headers=None):
        raise RuntimeError("not stubbed")


@pytest.mark.asyncio
async def test_get_prices_uses_cache_and_test_mode_overrides() -> None:
    cache = _DummyCache()
    http = _DummyHTTP()

    # Cached BTC price
    cache.store["price:BTC"] = {
        "symbol": "BTC",
        "price": 123.0,
        "source": "cache",
        "ts": datetime.now(UTC).isoformat(),
    }

    adapter = PriceAdapter(
        http=http,  # type: ignore[arg-type]
        cache=cache,  # type: ignore[arg-type]
        binance_base="https://api.binance.com",
        coingecko_base="https://api.coingecko.com/api/v3",
        test_mode=True,
        mock_prices="ETH:456",
        market_router=None,
    )

    out = await adapter.get_prices(["btc", "eth"])
    assert out["BTC"]["price"] == 123.0
    assert out["ETH"]["price"] == 456.0


@pytest.mark.asyncio
@respx.mock
async def test_get_prices_binance_bulk_endpoint_filters_pairs() -> None:
    cache = _DummyCache()

    # Use real resilient client signature surface via a tiny stub that respx can catch through httpx,
    # but our adapter calls `self.http.get_json`, so we emulate that with a wrapper over httpx.
    import httpx

    class _HTTP:
        def __init__(self):
            self.client = httpx.AsyncClient()

        async def get_json(self, url: str, params=None, headers=None):
            r = await self.client.get(url, params=params, headers=headers)
            r.raise_for_status()
            return r.json()

    http = _HTTP()
    try:

        respx.get("https://api.binance.com/api/v3/ticker/price").respond(
            200,
            json=[
                {"symbol": "BTCUSDT", "price": "100.0"},
                {"symbol": "ETHUSDT", "price": "200.0"},
                {"symbol": "SOLUSDT", "price": "300.0"},
            ],
        )

        adapter = PriceAdapter(
            http=http,  # type: ignore[arg-type]
            cache=cache,  # type: ignore[arg-type]
            binance_base="https://api.binance.com",
            coingecko_base="https://api.coingecko.com/api/v3",
            test_mode=False,
            mock_prices="",
            market_router=None,
        )

        out = await adapter.get_prices(["BTC", "SOL"])
        assert out["BTC"]["price"] == 100.0
        assert out["SOL"]["price"] == 300.0
        assert "ETH" not in out
    finally:
        await http.client.aclose()

