from __future__ import annotations

import math

import pandas as pd

from app.adapters.ohlcv import OHLCVAdapter
from app.core.ta import atr, pivot_levels


class SetupReviewService:
    def __init__(self, ohlcv_adapter: OHLCVAdapter) -> None:
        self.ohlcv_adapter = ohlcv_adapter

    def _safe_last(self, series: pd.Series, default: float) -> float:
        cleaned = series.dropna()
        if cleaned.empty:
            return default
        value = float(cleaned.iloc[-1])
        if math.isnan(value):
            return default
        return value

    async def review(
        self,
        symbol: str,
        timeframe: str,
        entry: float,
        stop: float,
        targets: list[float],
        direction: str | None = None,
        amount_usd: float | None = None,
        leverage: float | None = None,
    ) -> dict:
        symbol = symbol.upper()
        tf = timeframe or "1h"

        candles = await self.ohlcv_adapter.get_ohlcv(symbol, timeframe=tf, limit=260)
        df = pd.DataFrame(candles)
        if df.empty:
            raise RuntimeError("No candle data available for setup review")

        close_last = float(df["close"].iloc[-1])
        if direction is None:
            direction = "long" if (sum(targets) / max(len(targets), 1)) > entry else "short"

        risk = abs(entry - stop)
        if risk <= 0:
            raise RuntimeError("Invalid setup: entry and stop cannot be the same.")

        if direction == "long":
            rewards = [max(t - entry, 0.0) for t in targets]
        else:
            rewards = [max(entry - t, 0.0) for t in targets]

        rr_values = [r / risk for r in rewards if r > 0]
        rr_first = rr_values[0] if rr_values else 0.0
        rr_best = max(rr_values) if rr_values else 0.0

        atr_val = self._safe_last(atr(df, 14), max(close_last * 0.01, 1e-9))
        stop_atr = risk / max(atr_val, 1e-9)

        support, resistance = pivot_levels(df, lookback=80)

        if direction == "long":
            near_level = entry <= support * 1.03
            entry_context = (
                f"Entry is near support ({support:.4f})."
                if near_level
                else f"Entry is above support ({support:.4f}); possible chase risk."
            )
        else:
            near_level = entry >= resistance * 0.97
            entry_context = (
                f"Entry is near resistance ({resistance:.4f})."
                if near_level
                else f"Entry is below resistance ({resistance:.4f}); may be late."
            )

        if stop_atr < 0.8:
            stop_note = f"Stop looks tight ({stop_atr:.2f} ATR) and can get hunted by noise."
        elif stop_atr > 3.5:
            stop_note = f"Stop is wide ({stop_atr:.2f} ATR); safer but capital inefficient."
        else:
            stop_note = f"Stop distance is reasonable ({stop_atr:.2f} ATR)."

        score = 0
        score += 1 if rr_first >= 1.5 else 0
        score += 1 if rr_best >= 3.0 else 0
        score += 1 if 0.8 <= stop_atr <= 3.5 else 0
        score += 1 if near_level else 0

        if score >= 3:
            verdict = "good"
        elif score == 2:
            verdict = "ok"
        else:
            verdict = "weak"

        if direction == "long":
            suggested_stop = min(stop, support - 0.35 * atr_val, entry - 1.2 * atr_val)
            suggested_tp1 = max(targets[0], entry + 1.5 * risk)
            suggested_tp2 = max(max(targets), entry + 2.8 * risk, resistance)
        else:
            suggested_stop = max(stop, resistance + 0.35 * atr_val, entry + 1.2 * atr_val)
            suggested_tp1 = min(targets[0], entry - 1.5 * risk)
            suggested_tp2 = min(min(targets), entry - 2.8 * risk, support)

        position = None
        size_note = "R-multiples shown. Add `amount` and `leverage` for dollar PnL estimates."
        margin = float(amount_usd) if amount_usd is not None else None
        lev = float(leverage) if leverage is not None else None
        if margin is not None and margin <= 0:
            margin = None
        if lev is not None and lev <= 0:
            lev = None

        if margin is not None and lev is not None:
            notional = margin * lev
            qty = notional / max(entry, 1e-9)

            def _pnl(exit_price: float) -> float:
                if direction == "long":
                    return (exit_price - entry) * qty
                return (entry - exit_price) * qty

            stop_pnl = _pnl(stop)
            tp_rows = [{"tp": float(tp), "pnl_usd": round(_pnl(float(tp)), 2)} for tp in targets]
            position = {
                "margin_usd": round(margin, 2),
                "leverage": round(lev, 3),
                "notional_usd": round(notional, 2),
                "qty": round(qty, 8),
                "stop_pnl_usd": round(stop_pnl, 2),
                "tp_pnls": tp_rows,
            }
            size_note = "PnL estimates assume fixed position size and no fees/funding/slippage."
        elif margin is not None and lev is None:
            size_note = "Margin captured. Add leverage (e.g. `10x`) and I will estimate dollar PnL."

        return {
            "symbol": symbol,
            "timeframe": tf,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "targets": targets,
            "verdict": verdict,
            "rr_first": round(rr_first, 2),
            "rr_best": round(rr_best, 2),
            "atr": round(float(atr_val), 6),
            "stop_atr": round(stop_atr, 2),
            "support": support,
            "resistance": resistance,
            "entry_context": entry_context,
            "stop_note": stop_note,
            "suggested": {
                "entry": round(entry, 6),
                "stop": round(suggested_stop, 6),
                "tp1": round(suggested_tp1, 6),
                "tp2": round(suggested_tp2, 6),
            },
            "position": position,
            "size_note": size_note,
            "risk_line": "Use this as a risk-planning map, then execute only if structure still holds.",
        }
