from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from app.adapters.ohlcv import OHLCVAdapter


class CyclesService:
    HALVING_DATE = date(2024, 4, 20)

    def __init__(self, ohlcv_adapter: OHLCVAdapter) -> None:
        self.ohlcv_adapter = ohlcv_adapter

    async def cycle_check(self) -> dict:
        candles = await self.ohlcv_adapter.get_ohlcv("BTC", timeframe="1d", limit=400)
        df = pd.DataFrame(candles)
        close = df["close"]

        now = datetime.now(timezone.utc).date()
        days_since_halving = (now - self.HALVING_DATE).days

        high_1y = float(close.tail(365).max()) if len(close) >= 30 else float(close.max())
        current = float(close.iloc[-1])
        drawdown = (current - high_1y) / high_1y * 100 if high_1y else 0

        if days_since_halving < 120:
            phase = "Early post-halving expansion"
            bias = "Constructive but volatile"
            confidence = 0.56
        elif days_since_halving < 320:
            phase = "Mid-cycle trend phase"
            bias = "Risk-on unless macro shock"
            confidence = 0.68
        else:
            phase = "Late-cycle distribution risk zone"
            bias = "Selective risk, tighten invalidations"
            confidence = 0.52

        if drawdown < -25:
            phase = "Deep correction / reset phase"
            bias = "Neutral-to-cautious"
            confidence = 0.6

        invalidation = "A sustained BTC break below 200D trend and weak breadth would invalidate bullish bias."

        return {
            "summary": f"Cycle phase: {phase}. Macro bias: {bias}.",
            "bullets": [
                f"Days since halving ({self.HALVING_DATE.isoformat()}): {days_since_halving}",
                f"BTC vs 1y high: {drawdown:.2f}%",
                invalidation,
            ],
            "confidence": confidence,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
