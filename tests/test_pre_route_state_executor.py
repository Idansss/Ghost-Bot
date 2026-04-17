from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import pre_route_state_executor


def _message(text: str = "btc") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        answer=AsyncMock(),
        chat=SimpleNamespace(id=42),
        from_user=SimpleNamespace(id=7, username="ghost"),
    )


def _hub() -> SimpleNamespace:
    return SimpleNamespace(
        giveaway_service=SimpleNamespace(
            start_giveaway=AsyncMock(return_value={"id": 5, "prize": "Prize", "end_time": "later"}),
        ),
        alerts_service=SimpleNamespace(
            delete_alerts_by_symbol=AsyncMock(return_value=2),
            create_alert=AsyncMock(),
        ),
        trade_verify_service=SimpleNamespace(
            verify=AsyncMock(return_value={"source_line": "Bybit BTCUSDT", "summary": "ok"}),
        ),
        user_service=SimpleNamespace(
            get_settings=AsyncMock(return_value={"tone": "ghost"}),
        ),
    )


def _deps(**overrides) -> pre_route_state_executor.PreRouteStateDependencies:
    defaults = {
        "hub": _hub(),
        "as_int": lambda value, default: int(value) if value is not None else default,
        "dispatch_command_text": AsyncMock(return_value=True),
        "get_pending_feedback_suggestion": AsyncMock(return_value=None),
        "clear_pending_feedback_suggestion": AsyncMock(),
        "log_feedback_event": AsyncMock(),
        "notify_admins_negative_feedback": AsyncMock(),
        "get_cmd_wizard": AsyncMock(return_value=None),
        "clear_cmd_wizard": AsyncMock(),
        "get_wizard": AsyncMock(return_value=None),
        "set_wizard": AsyncMock(),
        "clear_wizard": AsyncMock(),
        "parse_timestamp": lambda text: datetime(2026, 4, 15, 12, 0, 0) if text else None,
        "save_trade_check": AsyncMock(),
        "remember_source_context": AsyncMock(),
        "get_pending_alert": AsyncMock(return_value=None),
        "clear_pending_alert": AsyncMock(),
        "trade_verification_template": lambda result, settings: f"trade:{result['summary']}:{settings['tone']}",
        "giveaway_menu": lambda **kwargs: {"menu": kwargs},
        "alert_created_menu": lambda symbol: {"symbol": symbol},
    }
    defaults.update(overrides)
    return pre_route_state_executor.PreRouteStateDependencies(**defaults)


@pytest.mark.asyncio
async def test_pending_feedback_logs_and_short_circuits() -> None:
    message = _message("make it shorter")
    clear_pending = AsyncMock()
    log_feedback_event = AsyncMock()
    notify_admins = AsyncMock()
    deps = _deps(
        get_pending_feedback_suggestion=AsyncMock(
            return_value={"message_id": "13", "reason": "long", "reply_preview": "too long"}
        ),
        clear_pending_feedback_suggestion=clear_pending,
        log_feedback_event=log_feedback_event,
        notify_admins_negative_feedback=notify_admins,
    )

    handled = await pre_route_state_executor.handle_pre_route_state(
        message=message,
        text="make it shorter",
        chat_id=42,
        deps=deps,
    )

    assert handled is True
    clear_pending.assert_awaited_once_with(42)
    log_feedback_event.assert_awaited_once()
    notify_admins.assert_awaited_once()
    message.answer.assert_awaited_once_with("Thanks \u2014 I've passed that on personally. We'll use it to improve.")


@pytest.mark.asyncio
async def test_dispatch_text_wizard_clears_state_and_dispatches() -> None:
    message = _message("btc 1h")
    clear_cmd_wizard = AsyncMock()
    dispatch_command_text = AsyncMock(return_value=True)
    deps = _deps(
        get_cmd_wizard=AsyncMock(return_value={"step": "dispatch_text", "prefix": "chart "}),
        clear_cmd_wizard=clear_cmd_wizard,
        dispatch_command_text=dispatch_command_text,
    )

    handled = await pre_route_state_executor.handle_pre_route_state(
        message=message,
        text="btc 1h",
        chat_id=42,
        deps=deps,
    )

    assert handled is True
    clear_cmd_wizard.assert_awaited_once_with(42)
    dispatch_command_text.assert_awaited_once_with(message, "chart btc 1h")


@pytest.mark.asyncio
async def test_tradecheck_levels_wizard_verifies_and_clears() -> None:
    message = _message("entry 100 stop 95 targets 110 120")
    hub = _hub()
    save_trade_check = AsyncMock()
    remember_source_context = AsyncMock()
    clear_wizard = AsyncMock()
    deps = _deps(
        hub=hub,
        get_wizard=AsyncMock(
            return_value={
                "step": "levels",
                "data": {"symbol": "BTC", "timeframe": "1h", "timestamp": "2026-04-15T12:00:00"},
            }
        ),
        save_trade_check=save_trade_check,
        remember_source_context=remember_source_context,
        clear_wizard=clear_wizard,
    )

    handled = await pre_route_state_executor.handle_pre_route_state(
        message=message,
        text="entry 100 stop 95 targets 110 120",
        chat_id=42,
        deps=deps,
    )

    assert handled is True
    hub.trade_verify_service.verify.assert_awaited_once()
    save_trade_check.assert_awaited_once()
    remember_source_context.assert_awaited_once_with(
        42,
        source_line="Bybit BTCUSDT",
        symbol="BTC",
        context="trade check",
    )
    clear_wizard.assert_awaited_once_with(42)
    message.answer.assert_awaited_once_with("trade:ok:ghost")


@pytest.mark.asyncio
async def test_pending_alert_price_creates_alert_and_tracks_source() -> None:
    message = _message("set 145")
    alert = SimpleNamespace(
        source_exchange="Bybit",
        market_kind="spot",
        instrument_id="SOLUSDT",
    )
    hub = _hub()
    hub.alerts_service.create_alert = AsyncMock(return_value=alert)
    clear_pending_alert = AsyncMock()
    remember_source_context = AsyncMock()
    deps = _deps(
        hub=hub,
        get_pending_alert=AsyncMock(return_value="SOL"),
        clear_pending_alert=clear_pending_alert,
        remember_source_context=remember_source_context,
    )

    handled = await pre_route_state_executor.handle_pre_route_state(
        message=message,
        text="set 145",
        chat_id=42,
        deps=deps,
    )

    assert handled is True
    hub.alerts_service.create_alert.assert_awaited_once_with(42, "SOL", "cross", 145.0, source="button")
    clear_pending_alert.assert_awaited_once_with(42)
    remember_source_context.assert_awaited_once_with(
        42,
        exchange="Bybit",
        market_kind="spot",
        instrument_id="SOLUSDT",
        symbol="SOL",
        context="alert",
    )
    message.answer.assert_awaited_once()
    assert message.answer.await_args.kwargs["reply_markup"] == {"symbol": "SOL"}


@pytest.mark.asyncio
async def test_no_state_returns_false() -> None:
    message = _message("just chat")
    deps = _deps()

    handled = await pre_route_state_executor.handle_pre_route_state(
        message=message,
        text="just chat",
        chat_id=42,
        deps=deps,
    )

    assert handled is False
    message.answer.assert_not_awaited()
