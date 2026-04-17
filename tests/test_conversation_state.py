from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import conversation_state


class _Cache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.get_json = AsyncMock(side_effect=self._get_json)
        self.set_json = AsyncMock(side_effect=self._set_json)
        self.delete = AsyncMock(side_effect=self._delete)

    async def _get_json(self, key: str):
        return self.store.get(key)

    async def _set_json(self, key: str, value, ttl: int | None = None) -> None:
        self.store[key] = value

    async def _delete(self, key: str) -> None:
        self.store.pop(key, None)


class _TradeCheckRecord:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _SessionContext:
    def __init__(self) -> None:
        self.add = Mock(side_effect=self._add)
        self.commit = AsyncMock()
        self.added = None

    def _add(self, row) -> None:
        self.added = row

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.mark.asyncio
async def test_pending_alert_round_trip() -> None:
    cache = _Cache()

    await conversation_state.set_pending_alert(cache, 42, "sol")
    payload = await conversation_state.get_pending_alert(cache, 42)
    await conversation_state.clear_pending_alert(cache, 42)

    assert payload == "SOL"
    assert await conversation_state.get_pending_alert(cache, 42) is None


@pytest.mark.asyncio
async def test_tradecheck_and_command_wizard_round_trip() -> None:
    cache = _Cache()

    await conversation_state.set_tradecheck_wizard(cache, 42, {"step": "symbol"})
    await conversation_state.set_command_wizard(cache, 42, {"step": "dispatch_text", "prefix": "chart "})

    tradecheck_payload = await conversation_state.get_tradecheck_wizard(cache, 42)
    command_payload = await conversation_state.get_command_wizard(cache, 42)

    assert tradecheck_payload == {"step": "symbol"}
    assert command_payload == {"step": "dispatch_text", "prefix": "chart "}

    await conversation_state.clear_tradecheck_wizard(cache, 42)
    await conversation_state.clear_command_wizard(cache, 42)

    assert await conversation_state.get_tradecheck_wizard(cache, 42) is None
    assert await conversation_state.get_command_wizard(cache, 42) is None


@pytest.mark.asyncio
async def test_save_trade_check_persists_normalized_row() -> None:
    session = _SessionContext()
    ensure_user = AsyncMock(return_value=SimpleNamespace(id=7))

    await conversation_state.save_trade_check(
        ensure_user=ensure_user,
        session_factory=lambda: session,
        tradecheck_model=_TradeCheckRecord,
        chat_id=42,
        data={
            "symbol": "BTC",
            "timeframe": "1h",
            "timestamp": datetime(2026, 4, 16, 12, 0, 0),
            "entry": "100",
            "stop": "95",
            "targets": ["110", 120],
            "mode": "ambiguous",
        },
        result={"summary": "ok"},
    )

    ensure_user.assert_awaited_once_with(42)
    session.add.assert_called_once()
    session.commit.assert_awaited_once()
    assert session.added.user_id == 7
    assert session.added.symbol == "BTC"
    assert session.added.timeframe == "1h"
    assert session.added.entry == 100.0
    assert session.added.stop == 95.0
    assert session.added.targets_json == [110.0, 120.0]
    assert session.added.mode == "ambiguous"
    assert session.added.result_json == {"summary": "ok"}
