from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.symbols import coingecko_id_for, normalize_symbol
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


class PriceAdapter:
    def __init__(self, http: ResilientHTTPClient, cache: RedisCache, binance_base: str, coingecko_base: str, test_mode: bool, mock_prices: str) -> None:
        self.http = http
        self.cache = cache
        self.binance_base = binance_base
        self.coingecko_base = coingecko_base
        self.test_mode = test_mode
        self.mock_prices_map = self._parse_mock_prices(mock_prices)

    async def set_mock_price(self, symbol: str, price: float) -> None:
        base = normalize_symbol(symbol).base
        self.mock_prices_map[base] = float(price)
        await self.cache.set_json(
            f"price:{base}",
            {
                "symbol": base,
                "price": float(price),
                "source": "test_mode_override",
                "ts": datetime.now(timezone.utc).isoformat(),
            },
            ttl=30,
        )

    def _parse_mock_prices(self, raw: str) -> dict[str, float]:
        out: dict[str, float] = {}
        if not raw:
            return out
        for item in raw.split(","):
            if ":" not in item:
                continue
            k, v = item.split(":", 1)
            try:
                out[k.strip().upper()] = float(v)
            except ValueError:
                continue
        return out

    async def get_price(self, symbol: str) -> dict:
        meta = normalize_symbol(symbol)
        key = f"price:{meta.base}"

        cached = await self.cache.get_json(key)
        if cached:
            return cached

        if self.test_mode and meta.base in self.mock_prices_map:
            payload = {
                "symbol": meta.base,
                "price": self.mock_prices_map[meta.base],
                "source": "test_mode",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            await self.cache.set_json(key, payload, ttl=5)
            return payload

        try:
            data = await self.http.get_json(f"{self.binance_base}/api/v3/ticker/price", params={"symbol": meta.pair})
            payload = {
                "symbol": meta.base,
                "price": float(data["price"]),
                "source": "binance",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            await self.cache.set_json(key, payload, ttl=15)
            return payload
        except Exception:  # noqa: BLE001
            pass

        cg_id = coingecko_id_for(meta.base)
        if not cg_id:
            search = await self.http.get_json(f"{self.coingecko_base}/search", params={"query": meta.base})
            coins = search.get("coins", [])
            for item in coins:
                if item.get("symbol", "").upper() == meta.base:
                    cg_id = item.get("id")
                    break

        if cg_id:
            data = await self.http.get_json(
                f"{self.coingecko_base}/simple/price",
                params={"ids": cg_id, "vs_currencies": "usd"},
            )
            if cg_id in data and "usd" in data[cg_id]:
                payload = {
                    "symbol": meta.base,
                    "price": float(data[cg_id]["usd"]),
                    "source": "coingecko",
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                await self.cache.set_json(key, payload, ttl=20)
                return payload

        raise RuntimeError(f"Price unavailable for {meta.base}")
