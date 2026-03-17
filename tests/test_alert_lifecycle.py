"""Integration tests for the alert lifecycle: create, trigger, retry on notify failure, idempotency."""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.enums import AlertStatus
from app.services.alerts import AlertsService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(id: int = 1, symbol: str = "BTC", condition: str = "above", target: float = 100_000.0) -> MagicMock:
    alert = MagicMock()
    alert.id = id
    alert.symbol = symbol
    alert.condition = condition
    alert.target_price = target
    alert.status = AlertStatus.ACTIVE
    alert.cooldown_until = None
    alert.source_exchange = None
    alert.instrument_id = None
    alert.market_kind = "spot"
    alert.conditions_json = None
    alert.user_id = 1
    return alert


def _make_user(chat_id: int = 42) -> MagicMock:
    user = MagicMock()
    user.telegram_chat_id = chat_id
    user.id = 1
    return user


@asynccontextmanager
async def _session_ctx(alerts, user):
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=_execute_factory(alerts, user))
    session.commit = AsyncMock()
    yield session


def _execute_factory(alerts, user):
    """Return a side_effect function that routes queries to the right result."""
    call_count = [0]

    async def execute(query, *args, **kwargs):
        result = MagicMock()
        call_count[0] += 1
        # First call = fetch active alerts
        if call_count[0] == 1:
            result.scalars.return_value.all.return_value = alerts
        # Subsequent calls = user lookups
        else:
            result.scalar_one_or_none.return_value = user
        return result

    return execute


def _make_service(alerts, user) -> AlertsService:
    cache = AsyncMock()
    # Return a prev price below the default target (100k) so crossings are detected
    cache.get_json = AsyncMock(return_value={"price": 99_000.0})
    cache.set_json = AsyncMock()
    cache.set_if_absent = AsyncMock(return_value=True)
    cache.incr_with_expiry = AsyncMock(return_value=1)

    price_adapter = AsyncMock()
    price_adapter.get_price = AsyncMock(return_value={"price": 101_000.0, "exchange": "binance", "instrument_id": "BTCUSDT", "market_kind": "spot"})

    db_factory = lambda: _session_ctx(alerts, user)  # noqa: E731

    return AlertsService(
        db_factory=db_factory,
        cache=cache,
        price_adapter=price_adapter,
        market_router=None,
        alerts_limit_per_day=10,
        cooldown_minutes=30,
        max_deviation_pct=30.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alert_triggered_on_condition_met() -> None:
    """Alert should fire and be marked TRIGGERED when price crosses target."""
    alert = _make_alert(target=100_000.0, condition="above")
    user = _make_user()
    svc = _make_service([alert], user)

    notifier = AsyncMock()
    count = await svc.process_alerts(notifier)

    assert count == 1
    assert alert.status == AlertStatus.TRIGGERED
    notifier.assert_called_once()


@pytest.mark.asyncio
async def test_alert_not_triggered_below_target() -> None:
    """Alert above 110k should NOT fire when price is 101k."""
    alert = _make_alert(target=110_000.0, condition="above")
    user = _make_user()
    svc = _make_service([alert], user)

    notifier = AsyncMock()
    count = await svc.process_alerts(notifier)

    assert count == 0
    assert alert.status == AlertStatus.ACTIVE
    notifier.assert_not_called()


@pytest.mark.asyncio
async def test_alert_stays_active_when_notify_fails() -> None:
    """If Telegram delivery fails, alert must NOT be marked triggered (retry next cycle)."""
    alert = _make_alert(target=100_000.0, condition="above")
    user = _make_user()
    svc = _make_service([alert], user)

    async def failing_notifier(*args, **kwargs):
        raise RuntimeError("Telegram down")

    count = await svc.process_alerts(failing_notifier)

    assert count == 0
    assert alert.status == AlertStatus.ACTIVE


@pytest.mark.asyncio
async def test_idempotency_key_returns_existing_alert() -> None:
    """create_alert with a duplicate idempotency_key returns the existing alert."""
    existing = _make_alert(id=99)
    existing.idempotency_key = "idem-abc"

    cache = AsyncMock()
    cache.incr_with_expiry = AsyncMock(return_value=1)
    cache.get_json = AsyncMock(return_value=None)

    price_adapter = AsyncMock()
    price_adapter.get_price = AsyncMock(return_value={"price": 99_000.0})

    @asynccontextmanager
    async def db():
        session = AsyncMock()
        # First execute returns existing alert for idempotency check
        result = MagicMock()
        result.scalar_one_or_none.return_value = existing
        session.execute = AsyncMock(return_value=result)
        yield session

    svc = AlertsService(
        db_factory=db,
        cache=cache,
        price_adapter=price_adapter,
        market_router=None,
        alerts_limit_per_day=10,
        cooldown_minutes=30,
    )

    returned = await svc.create_alert(42, "BTC", "above", 100_000.0, idempotency_key="idem-abc")
    assert returned.id == 99
