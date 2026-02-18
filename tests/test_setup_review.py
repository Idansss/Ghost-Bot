from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.setup_review import SetupReviewService


class DummyOHLCV:
    def __init__(self, candles: list[dict]) -> None:
        self._candles = candles

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 260) -> list[dict]:
        return self._candles


def _make_candles() -> list[dict]:
    base = 100.0
    candles = []
    ts = 1_700_000_000_000
    for i in range(260):
        close = base + np.sin(i / 10) * 3 + i * 0.02
        high = close + 1.2
        low = close - 1.1
        open_ = close - 0.2
        candles.append(
            {
                "ts": ts + i * 3600_000,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": 1000.0,
            }
        )
    return candles


@pytest.mark.asyncio
async def test_setup_review_returns_expected_keys() -> None:
    service = SetupReviewService(DummyOHLCV(_make_candles()))
    out = await service.review(
        symbol="BTC",
        timeframe="1h",
        entry=100.0,
        stop=96.0,
        targets=[106.0, 112.0],
        direction="long",
    )
    assert out["symbol"] == "BTC"
    assert out["verdict"] in {"good", "ok", "weak"}
    assert "suggested" in out
    assert out["rr_first"] > 0


@pytest.mark.asyncio
async def test_setup_review_short_direction() -> None:
    service = SetupReviewService(DummyOHLCV(_make_candles()))
    out = await service.review(
        symbol="ETH",
        timeframe="1h",
        entry=100.0,
        stop=104.0,
        targets=[95.0, 90.0],
        direction="short",
    )
    assert out["direction"] == "short"
    assert out["rr_best"] > 0


@pytest.mark.asyncio
async def test_setup_review_invalid_risk_raises() -> None:
    service = SetupReviewService(DummyOHLCV(_make_candles()))
    with pytest.raises(RuntimeError):
        await service.review(
            symbol="SOL",
            timeframe="1h",
            entry=100.0,
            stop=100.0,
            targets=[110.0],
            direction="long",
        )


@pytest.mark.asyncio
async def test_setup_review_position_sizing_outputs_pnl() -> None:
    service = SetupReviewService(DummyOHLCV(_make_candles()))
    out = await service.review(
        symbol="BTC",
        timeframe="1h",
        entry=100.0,
        stop=95.0,
        targets=[110.0, 120.0],
        direction="long",
        amount_usd=100.0,
        leverage=10.0,
    )
    position = out.get("position")
    assert position is not None
    assert position["notional_usd"] == 1000.0
    assert position["qty"] == 10.0
    assert position["stop_pnl_usd"] == -50.0
    assert position["tp_pnls"][0]["pnl_usd"] == 100.0
