from __future__ import annotations

from datetime import UTC, datetime

from app.adapters.market_router import MarketDataRouter
from app.adapters.symbols import coingecko_id_for, normalize_symbol
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


class PriceAdapter:
    def __init__(
        self,
        http: ResilientHTTPClient,
        cache: RedisCache,
        binance_base: str,
        coingecko_base: str,
        test_mode: bool,
        mock_prices: str,
        market_router: MarketDataRouter | None = None,
    ) -> None:
        self.http = http
        self.cache = cache
        self.binance_base = binance_base
        self.coingecko_base = coingecko_base
        self.test_mode = test_mode
        self.mock_prices_map = self._parse_mock_prices(mock_prices)
        self.market_router = market_router

    async def set_mock_price(self, symbol: str, price: float) -> None:
        base = normalize_symbol(symbol).base
        self.mock_prices_map[base] = float(price)
        await self.cache.set_json(
            f"price:{base}",
            {
                "symbol": base,
                "price": float(price),
                "source": "test_mode_override",
                "ts": datetime.now(UTC).isoformat(),
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
                "ts": datetime.now(UTC).isoformat(),
            }
            await self.cache.set_json(key, payload, ttl=5)
            return payload

        if self.market_router:
            try:
                routed = await self.market_router.get_price(meta.base)
                payload = {
                    "symbol": routed["symbol"],
                    "price": float(routed["price"]),
                    "source": routed["source"],
                    "source_line": routed.get("source_line"),
                    "exchange": routed.get("exchange"),
                    "market_kind": routed.get("market_kind"),
                    "instrument_id": routed.get("instrument_id"),
                    "ts": routed.get("updated_at") or datetime.now(UTC).isoformat(),
                }
                await self.cache.set_json(key, payload, ttl=15)
                return payload
            except Exception:
                pass
        else:
            try:
                data = await self.http.get_json(f"{self.binance_base}/api/v3/ticker/price", params={"symbol": meta.pair})
                payload = {
                    "symbol": meta.base,
                    "price": float(data["price"]),
                    "source": "binance_spot",
                    "source_line": f"Data source: Binance Spot ({meta.pair}) | Updated: 0s ago",
                    "exchange": "binance",
                    "market_kind": "spot",
                    "instrument_id": meta.pair,
                    "ts": datetime.now(UTC).isoformat(),
                }
                await self.cache.set_json(key, payload, ttl=15)
                return payload
            except Exception:
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
                    "source_line": f"Data source: CoinGecko Spot ({meta.base}) | Updated: 0s ago",
                    "exchange": "coingecko",
                    "market_kind": "spot",
                    "instrument_id": cg_id,
                    "ts": datetime.now(UTC).isoformat(),
                }
                await self.cache.set_json(key, payload, ttl=20)
                return payload

        raise RuntimeError(f"Price unavailable for {meta.base}")

    async def get_prices(self, symbols: list[str]) -> dict[str, dict]:
        """Best-effort bulk prices keyed by base symbol.

        Uses cache where possible, then a Binance bulk endpoint fallback.
        """
        bases = []
        for s in symbols:
            base = normalize_symbol(s).base
            if base and base not in bases:
                bases.append(base)

        out: dict[str, dict] = {}
        # cache-first
        for base in list(bases):
            cached = await self.cache.get_json(f"price:{base}")
            if cached:
                out[base] = cached

        missing = [b for b in bases if b not in out]
        if not missing:
            return out

        # test mode overrides
        if self.test_mode:
            for base in list(missing):
                if base in self.mock_prices_map:
                    payload = {
                        "symbol": base,
                        "price": float(self.mock_prices_map[base]),
                        "source": "test_mode",
                        "ts": datetime.now(UTC).isoformat(),
                    }
                    await self.cache.set_json(f"price:{base}", payload, ttl=5)
                    out[base] = payload
            missing = [b for b in missing if b not in out]
            if not missing:
                return out

        # try market router per-symbol (keeps best-source behavior)
        if self.market_router:
            for base in list(missing):
                try:
                    routed = await self.market_router.get_price(base)
                    payload = {
                        "symbol": routed["symbol"],
                        "price": float(routed["price"]),
                        "source": routed["source"],
                        "source_line": routed.get("source_line"),
                        "exchange": routed.get("exchange"),
                        "market_kind": routed.get("market_kind"),
                        "instrument_id": routed.get("instrument_id"),
                        "ts": routed.get("updated_at") or datetime.now(UTC).isoformat(),
                    }
                    await self.cache.set_json(f"price:{base}", payload, ttl=15)
                    out[base] = payload
                except Exception:
                    continue
            missing = [b for b in missing if b not in out]
            if not missing:
                return out

        # Binance bulk endpoint returns all prices; filter locally.
        try:
            all_rows = await self.http.get_json(f"{self.binance_base}/api/v3/ticker/price")
            if isinstance(all_rows, list):
                want_pairs = {normalize_symbol(b).pair: b for b in missing}
                for row in all_rows:
                    sym = str(row.get("symbol") or "")
                    if sym in want_pairs:
                        base = want_pairs[sym]
                        payload = {
                            "symbol": base,
                            "price": float(row["price"]),
                            "source": "binance_spot_bulk",
                            "source_line": f"Data source: Binance Spot ({sym}) | Updated: 0s ago",
                            "exchange": "binance",
                            "market_kind": "spot",
                            "instrument_id": sym,
                            "ts": datetime.now(UTC).isoformat(),
                        }
                        await self.cache.set_json(f"price:{base}", payload, ttl=15)
                        out[base] = payload
        except Exception:
            pass

        return out
