"""Tests for multi-exchange fallback in MarketDataRouter."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.errors import UpstreamError


@pytest.mark.asyncio
async def test_alert_price_falls_back_to_price_adapter() -> None:
    """If the direct exchange adapter fails, AlertsService should fall back to PriceAdapter."""
    from app.services.alerts import AlertsService

    alert = MagicMock()
    alert.id = 1
    alert.symbol = "SOL"
    alert.source_exchange = "bybit"
    alert.instrument_id = "SOLUSDT"
    alert.market_kind = "spot"

    # Direct exchange adapter raises UpstreamError
    direct_adapter = AsyncMock()
    direct_adapter.get_price = AsyncMock(side_effect=UpstreamError("bybit down"))

    market_router = MagicMock()
    market_router.adapters = {"bybit": direct_adapter}

    # Fallback price adapter returns a valid price
    price_adapter = AsyncMock()
    price_adapter.get_price = AsyncMock(return_value={"price": 150.0, "exchange": "binance", "instrument_id": "SOLUSDT", "market_kind": "spot"})

    from app.core.cache import RedisCache
    cache = AsyncMock(spec=RedisCache)

    svc = AlertsService(
        db_factory=None,
        cache=cache,
        price_adapter=price_adapter,
        market_router=market_router,
        alerts_limit_per_day=10,
        cooldown_minutes=30,
    )

    price, source_info = await svc._get_alert_price(alert)

    assert price == 150.0
    assert source_info["exchange"] == "binance"
    price_adapter.get_price.assert_called_once_with("SOL")


@pytest.mark.asyncio
async def test_alert_price_returns_none_when_all_sources_fail() -> None:
    """If all sources fail, _get_alert_price should return (None, {}) without raising."""
    from app.core.cache import RedisCache
    from app.services.alerts import AlertsService

    alert = MagicMock()
    alert.id = 1
    alert.symbol = "PEPE"
    alert.source_exchange = None
    alert.instrument_id = None
    alert.market_kind = "spot"

    price_adapter = AsyncMock()
    price_adapter.get_price = AsyncMock(side_effect=UpstreamError("all down"))

    cache = AsyncMock(spec=RedisCache)

    svc = AlertsService(
        db_factory=None,
        cache=cache,
        price_adapter=price_adapter,
        market_router=None,
        alerts_limit_per_day=10,
        cooldown_minutes=30,
    )

    price, source_info = await svc._get_alert_price(alert)

    assert price is None
    assert source_info == {}
