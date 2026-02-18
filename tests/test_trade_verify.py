from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services.trade_verify import TradeVerifyService


class DummyOHLCV:
    def __init__(self, candles: list[dict]) -> None:
        self._candles = candles

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> list[dict]:
        return self._candles


def _candle(ts: int, o: float, h: float, l: float, c: float) -> dict:
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 100.0}


@pytest.mark.asyncio
async def test_trade_verify_win_path() -> None:
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candles = [
        _candle(start, 100, 101, 99, 100),
        _candle(start + 3600_000, 100, 106, 99.5, 105),
    ]
    service = TradeVerifyService(DummyOHLCV(candles))
    result = await service.verify("BTC", "1h", datetime(2025, 1, 1, tzinfo=timezone.utc), 100, 95, [105], mode="ambiguous")
    assert result["result"] == "win"
    assert result["r_multiple"] > 0


@pytest.mark.asyncio
async def test_trade_verify_same_candle_ambiguous() -> None:
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candles = [_candle(start, 100, 106, 94, 101)]
    service = TradeVerifyService(DummyOHLCV(candles))
    result = await service.verify("BTC", "1h", datetime(2025, 1, 1, tzinfo=timezone.utc), 100, 95, [105], mode="ambiguous")
    assert result["result"] == "ambiguous"


@pytest.mark.asyncio
async def test_trade_verify_same_candle_conservative() -> None:
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candles = [_candle(start, 100, 106, 94, 101)]
    service = TradeVerifyService(DummyOHLCV(candles))
    result = await service.verify("BTC", "1h", datetime(2025, 1, 1, tzinfo=timezone.utc), 100, 95, [105], mode="conservative")
    assert result["result"] == "loss"
    assert result["r_multiple"] == -1.0


@pytest.mark.asyncio
async def test_trade_verify_not_filled() -> None:
    start = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    candles = [_candle(start, 110, 111, 109, 110.5)]
    service = TradeVerifyService(DummyOHLCV(candles))
    result = await service.verify("BTC", "1h", datetime(2025, 1, 1, tzinfo=timezone.utc), 100, 95, [105], mode="ambiguous")
    assert result["result"] == "not_filled"
