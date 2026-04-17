from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import command_menu_flow


def _callback(data: str) -> SimpleNamespace:
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        answer=AsyncMock(),
    )
    return SimpleNamespace(
        data=data,
        message=message,
        from_user=SimpleNamespace(id=7),
        answer=AsyncMock(),
        bot=SimpleNamespace(),
    )


def _hub() -> SimpleNamespace:
    return SimpleNamespace(
        giveaway_service=SimpleNamespace(is_admin=Mock(return_value=True)),
        rsi_scanner_service=SimpleNamespace(scan=AsyncMock(return_value={"kind": "scan"})),
        alerts_service=SimpleNamespace(list_alerts=AsyncMock(return_value=[])),
    )


async def _run_with_typing_lock(bot, chat_id: int, runner) -> None:
    await runner()


def _deps(**overrides) -> command_menu_flow.CommandMenuFlowDependencies:
    defaults = {
        "hub": _hub(),
        "dispatch_command_text": AsyncMock(return_value=True),
        "run_with_typing_lock": AsyncMock(side_effect=_run_with_typing_lock),
        "set_cmd_wizard": AsyncMock(),
        "set_wizard": AsyncMock(),
        "as_int": lambda value, default: int(value) if value is not None else default,
        "alpha_quick_menu": lambda: {"menu": "alpha"},
        "watch_quick_menu": lambda: {"menu": "watch"},
        "chart_quick_menu": lambda: {"menu": "chart"},
        "heatmap_quick_menu": lambda: {"menu": "heatmap"},
        "rsi_quick_menu": lambda: {"menu": "rsi"},
        "ema_quick_menu": lambda: {"menu": "ema"},
        "news_quick_menu": lambda: {"menu": "news"},
        "alert_quick_menu": lambda: {"menu": "alert"},
        "findpair_quick_menu": lambda: {"menu": "findpair"},
        "setup_quick_menu": lambda: {"menu": "setup"},
        "scan_quick_menu": lambda: {"menu": "scan"},
        "giveaway_menu": Mock(return_value={"menu": "giveaway"}),
        "rsi_scan_template": lambda payload: f"scan:{payload['kind']}",
        "logger": SimpleNamespace(warning=Mock()),
    }
    defaults.update(overrides)
    return command_menu_flow.CommandMenuFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_menu_action_builds_giveaway_menu_from_admin_status() -> None:
    callback = _callback("cmd:menu:giveaway")
    hub = _hub()
    giveaway_menu = Mock(return_value={"menu": "gw"})
    deps = _deps(hub=hub, giveaway_menu=giveaway_menu)

    await command_menu_flow.handle_command_menu_callback(callback=callback, deps=deps)

    hub.giveaway_service.is_admin.assert_called_once_with(7)
    giveaway_menu.assert_called_once_with(is_admin=True)
    callback.message.answer.assert_awaited_once_with("Pick giveaway action.", reply_markup={"menu": "gw"})
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_custom_watch_sets_dispatch_wizard_and_prompts() -> None:
    callback = _callback("cmd:watch:custom")
    set_cmd_wizard = AsyncMock()
    deps = _deps(set_cmd_wizard=set_cmd_wizard)

    await command_menu_flow.handle_command_menu_callback(callback=callback, deps=deps)

    set_cmd_wizard.assert_awaited_once_with(42, {"step": "dispatch_text", "prefix": "watch "})
    callback.message.answer.assert_awaited_once_with("Send symbol and optional tf, e.g. `BTC 1h`.")
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_chart_preset_dispatches_with_typing_lock() -> None:
    callback = _callback("cmd:chart:BTC:4h")
    dispatch_command_text = AsyncMock(return_value=True)
    run_with_typing_lock = AsyncMock(side_effect=_run_with_typing_lock)
    deps = _deps(
        dispatch_command_text=dispatch_command_text,
        run_with_typing_lock=run_with_typing_lock,
    )

    await command_menu_flow.handle_command_menu_callback(callback=callback, deps=deps)

    run_with_typing_lock.assert_awaited_once()
    dispatch_command_text.assert_awaited_once_with(callback.message, "chart BTC 4h")
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_rsi_preset_scans_with_clamped_parameters() -> None:
    callback = _callback("cmd:rsi:1h:overbought:25:55")
    hub = _hub()
    hub.rsi_scanner_service.scan = AsyncMock(return_value={"kind": "done"})
    deps = _deps(hub=hub, rsi_scan_template=lambda payload: f"scan:{payload['kind']}")

    await command_menu_flow.handle_command_menu_callback(callback=callback, deps=deps)

    hub.rsi_scanner_service.scan.assert_awaited_once_with(
        timeframe="1h",
        mode="overbought",
        limit=20,
        rsi_length=50,
        symbol=None,
    )
    callback.message.answer.assert_awaited_once_with("scan:done")
    callback.answer.assert_awaited_once_with("Scanning...")


@pytest.mark.asyncio
async def test_alert_clear_prompts_for_confirmation_when_alerts_exist() -> None:
    callback = _callback("cmd:alert:clear")
    hub = _hub()
    hub.alerts_service.list_alerts = AsyncMock(return_value=[1, 2, 3])
    deps = _deps(hub=hub)

    await command_menu_flow.handle_command_menu_callback(callback=callback, deps=deps)

    hub.alerts_service.list_alerts.assert_awaited_once_with(42)
    callback.message.answer.assert_awaited_once()
    prompt = callback.message.answer.await_args.args[0]
    reply_markup = callback.message.answer.await_args.kwargs["reply_markup"]
    assert "Clear all <b>3</b> alerts?" in prompt
    assert reply_markup.inline_keyboard[0][0].callback_data == "confirm:clear_alerts:3"
    assert reply_markup.inline_keyboard[1][0].callback_data == "confirm:clear_alerts:no"
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_setup_wizard_sets_tradecheck_state() -> None:
    callback = _callback("cmd:setup:wizard")
    set_wizard = AsyncMock()
    deps = _deps(set_wizard=set_wizard)

    await command_menu_flow.handle_command_menu_callback(callback=callback, deps=deps)

    set_wizard.assert_awaited_once_with(42, {"step": "symbol", "data": {}})
    callback.message.answer.assert_awaited_once_with("Tradecheck wizard: send symbol (e.g., ETH).")
    callback.answer.assert_awaited_once_with()
