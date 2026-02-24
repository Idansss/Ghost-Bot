from __future__ import annotations

import asyncio
from statistics import mean

import pandas as pd

from app.adapters.ohlcv import OHLCVAdapter
from app.adapters.prices import PriceAdapter
from app.core.ta import ema, rsi


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _trend_label(close_series: pd.Series) -> str:
    if close_series.empty:
        return "unknown"
    ema20 = ema(close_series, 20).dropna()
    ema50 = ema(close_series, 50).dropna()
    close = close_series.dropna()
    if ema20.empty or ema50.empty or close.empty:
        return "unknown"
    last_close = float(close.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    if last_close >= last_ema20 >= last_ema50:
        return "bullish"
    if last_close <= last_ema20 <= last_ema50:
        return "bearish"
    return "choppy"


def _volume_ratio(volumes: list[float]) -> float | None:
    if len(volumes) < 22:
        return None
    last = float(volumes[-1])
    baseline = mean(float(v) for v in volumes[-21:-1])
    if baseline <= 0:
        return None
    return last / baseline


def _sentiment_from_rsi(rsi_4h: float | None) -> str:
    if rsi_4h is None:
        return "neutral"
    if rsi_4h >= 70:
        return "greedy"
    if rsi_4h >= 60:
        return "risk-on"
    if rsi_4h <= 30:
        return "fearful"
    if rsi_4h <= 40:
        return "risk-off"
    return "neutral"


class MarketContextService:
    def __init__(self, price_adapter: PriceAdapter, ohlcv_adapter: OHLCVAdapter) -> None:
        self.price_adapter = price_adapter
        self.ohlcv_adapter = ohlcv_adapter

    async def _symbol_snapshot(self, symbol: str) -> dict:
        price_task = self.price_adapter.get_price(symbol)
        ohlcv_1h_task = self.ohlcv_adapter.get_ohlcv(symbol, "1h", 160)
        ohlcv_4h_task = self.ohlcv_adapter.get_ohlcv(symbol, "4h", 160)
        price, candles_1h, candles_4h = await asyncio.gather(price_task, ohlcv_1h_task, ohlcv_4h_task)

        df_1h = pd.DataFrame(candles_1h or [])
        df_4h = pd.DataFrame(candles_4h or [])

        close_1h = df_1h.get("close", pd.Series(dtype=float)).astype(float) if not df_1h.empty else pd.Series(dtype=float)
        close_4h = df_4h.get("close", pd.Series(dtype=float)).astype(float) if not df_4h.empty else pd.Series(dtype=float)
        volume_1h = df_1h.get("volume", pd.Series(dtype=float)).astype(float) if not df_1h.empty else pd.Series(dtype=float)

        rsi_1h_val = _safe_float(rsi(close_1h, 14).dropna().iloc[-1], None) if not close_1h.empty else None
        rsi_4h_val = _safe_float(rsi(close_4h, 14).dropna().iloc[-1], None) if not close_4h.empty else None

        return {
            "symbol": symbol.upper(),
            "price": _safe_float(price.get("price"), 0.0),
            "trend_1h": _trend_label(close_1h),
            "trend_4h": _trend_label(close_4h),
            "volume_ratio_1h": _volume_ratio(volume_1h.tolist()) if not volume_1h.empty else None,
            "rsi_1h": rsi_1h_val,
            "rsi_4h": rsi_4h_val,
        }

    async def _btc_dominance(self) -> float | None:
        http = getattr(self.price_adapter, "http", None)
        base = str(getattr(self.price_adapter, "coingecko_base", "")).rstrip("/")
        if not http or not base:
            return None
        try:
            payload = await http.get_json(f"{base}/global")
            return _safe_float(payload.get("data", {}).get("market_cap_percentage", {}).get("btc"), None)
        except Exception:  # noqa: BLE001
            return None

    async def get_market_context(self) -> dict:
        """
        Fetch quick market overview:
        - BTC, ETH, SOL price + 1h/4h trend + volume ratio
        - BTC dominance
        - Overall sentiment (derived from BTC RSI)
        Returns a dict to be injected into analysis prompts.
        """
        btc_task = self._symbol_snapshot("BTC")
        eth_task = self._symbol_snapshot("ETH")
        sol_task = self._symbol_snapshot("SOL")
        dom_task = self._btc_dominance()

        btc = {}
        eth = {}
        sol = {}
        dominance = None
        results = await asyncio.gather(btc_task, eth_task, sol_task, dom_task, return_exceptions=True)
        if not isinstance(results[0], Exception):
            btc = results[0]
        if not isinstance(results[1], Exception):
            eth = results[1]
        if not isinstance(results[2], Exception):
            sol = results[2]
        if not isinstance(results[3], Exception):
            dominance = results[3]

        btc_price = _safe_float(btc.get("price"), 0.0) or 0.0
        eth_price = _safe_float(eth.get("price"), 0.0) or 0.0
        eth_btc_ratio = (eth_price / btc_price) if btc_price > 0 else None
        sentiment = _sentiment_from_rsi(_safe_float(btc.get("rsi_4h"), None))

        return {
            "btc": btc,
            "eth": eth,
            "sol": sol,
            "btc_dominance": dominance,
            "eth_btc_ratio": eth_btc_ratio,
            "sentiment": sentiment,
        }


def format_market_context(context: dict) -> str:
    ctx = context or {}
    btc = ctx.get("btc", {}) if isinstance(ctx.get("btc"), dict) else {}
    eth = ctx.get("eth", {}) if isinstance(ctx.get("eth"), dict) else {}
    sol = ctx.get("sol", {}) if isinstance(ctx.get("sol"), dict) else {}
    dom = _safe_float(ctx.get("btc_dominance"), None)
    ratio = _safe_float(ctx.get("eth_btc_ratio"), None)
    sentiment = str(ctx.get("sentiment") or "neutral")

    btc_price = _safe_float(btc.get("price"), 0.0) or 0.0
    eth_price = _safe_float(eth.get("price"), 0.0) or 0.0
    sol_price = _safe_float(sol.get("price"), 0.0) or 0.0
    vol_ratio = _safe_float(btc.get("volume_ratio_1h"), None)
    btc_rsi_4h = _safe_float(btc.get("rsi_4h"), None)

    bits = [
        (
            f"BTC ${btc_price:,.2f}, 1h {btc.get('trend_1h', 'unknown')}, "
            f"4h {btc.get('trend_4h', 'unknown')}"
            + (f", volume {vol_ratio:.2f}x" if vol_ratio is not None else "")
        ),
        f"ETH ${eth_price:,.2f}, 1h {eth.get('trend_1h', 'unknown')}, 4h {eth.get('trend_4h', 'unknown')}",
    ]
    if sol_price > 0:
        bits.append(f"SOL ${sol_price:,.2f}, 1h {sol.get('trend_1h', 'unknown')}, 4h {sol.get('trend_4h', 'unknown')}")
    if dom is not None:
        bits.append(f"BTC.D {dom:.2f}%")
    if ratio is not None:
        bits.append(f"ETH/BTC {ratio:.5f}")
    if btc_rsi_4h is not None:
        bits.append(f"Sentiment {sentiment} (BTC RSI4h {btc_rsi_4h:.1f})")
    else:
        bits.append(f"Sentiment {sentiment}")
    return " | ".join(bits)
