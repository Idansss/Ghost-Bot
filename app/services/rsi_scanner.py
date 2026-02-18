from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd

from app.adapters.ohlcv import OHLCVAdapter
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient
from app.core.ta import rsi


class RSIScannerService:
    def __init__(
        self,
        http: ResilientHTTPClient,
        cache: RedisCache,
        ohlcv_adapter: OHLCVAdapter,
        binance_base: str,
    ) -> None:
        self.http = http
        self.cache = cache
        self.ohlcv_adapter = ohlcv_adapter
        self.binance_base = binance_base

    async def _top_usdt_symbols(self, universe_size: int = 80) -> list[str]:
        cache_key = f"rsi:universe:{universe_size}"
        cached = await self.cache.get_json(cache_key)
        if cached:
            return [str(x) for x in cached]

        rows = await self.http.get_json(f"{self.binance_base}/api/v3/ticker/24hr")
        scored: list[tuple[str, float]] = []
        for row in rows:
            symbol = str(row.get("symbol", ""))
            if not symbol.endswith("USDT"):
                continue
            try:
                quote_vol = float(row.get("quoteVolume", 0) or 0)
            except Exception:  # noqa: BLE001
                quote_vol = 0.0
            if quote_vol <= 0:
                continue
            scored.append((symbol[:-4], quote_vol))

        scored.sort(key=lambda x: x[1], reverse=True)
        symbols = [s for s, _ in scored[: max(20, universe_size)]]
        await self.cache.set_json(cache_key, symbols, ttl=180)
        return symbols

    async def _symbol_rsi(self, symbol: str, timeframe: str, rsi_length: int) -> dict | None:
        try:
            candles = await self.ohlcv_adapter.get_ohlcv(symbol, timeframe=timeframe, limit=max(180, rsi_length * 8))
        except Exception:  # noqa: BLE001
            return None
        if len(candles) < rsi_length + 5:
            return None
        df = pd.DataFrame(candles)
        if df.empty or "close" not in df:
            return None
        series = rsi(df["close"], rsi_length).dropna()
        if series.empty:
            return None
        value = float(series.iloc[-1])
        return {"symbol": symbol.upper(), "rsi": round(value, 2)}

    def _bucket(self, value: float, mode: str) -> str:
        if mode == "oversold":
            if value <= 20:
                return "extreme oversold"
            if value <= 30:
                return "oversold"
            return "neutral"
        if value >= 80:
            return "extreme overbought"
        if value >= 70:
            return "overbought"
        return "neutral"

    async def scan(
        self,
        timeframe: str = "1h",
        mode: str = "oversold",
        limit: int = 10,
        rsi_length: int = 14,
        symbol: str | None = None,
    ) -> dict:
        tf = timeframe or "1h"
        mode_norm = "overbought" if mode.lower() == "overbought" else "oversold"
        cap = max(1, min(limit, 20))

        if symbol:
            row = await self._symbol_rsi(symbol.upper(), tf, rsi_length)
            items = [row] if row else []
        else:
            universe = await self._top_usdt_symbols(universe_size=100)
            sem = asyncio.Semaphore(12)

            async def _one(sym: str) -> dict | None:
                async with sem:
                    return await self._symbol_rsi(sym, tf, rsi_length)

            raw = await asyncio.gather(*[_one(sym) for sym in universe], return_exceptions=False)
            items = [x for x in raw if x]

        if mode_norm == "oversold":
            items.sort(key=lambda x: x["rsi"])
        else:
            items.sort(key=lambda x: x["rsi"], reverse=True)

        ranked = []
        for row in items[:cap]:
            ranked.append(
                {
                    "symbol": row["symbol"],
                    "rsi": row["rsi"],
                    "note": self._bucket(float(row["rsi"]), mode_norm),
                }
            )

        return {
            "summary": f"RSI scan ({mode_norm}) on {tf} using RSI({rsi_length}).",
            "timeframe": tf,
            "mode": mode_norm,
            "rsi_length": rsi_length,
            "items": ranked,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
