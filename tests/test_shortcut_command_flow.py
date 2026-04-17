from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import shortcut_command_flow


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=42),
        message_id=11,
        answer=AsyncMock(),
    )


def _hub() -> SimpleNamespace:
    return SimpleNamespace(
        market_router=SimpleNamespace(get_price=AsyncMock(return_value={"price": 123.45, "change_24h": 4.2})),
    )


def _deps(**overrides) -> shortcut_command_flow.ShortcutCommandFlowDependencies:
    defaults = {
        "hub": _hub(),
        "dispatch_command_text": AsyncMock(return_value=True),
        "alpha_quick_menu": lambda: {"menu": "alpha"},
        "watch_quick_menu": lambda: {"menu": "watch"},
        "chart_quick_menu": lambda: {"menu": "chart"},
        "heatmap_quick_menu": lambda: {"menu": "heatmap"},
        "rsi_quick_menu": lambda: {"menu": "rsi"},
        "ema_quick_menu": lambda: {"menu": "ema"},
        "as_int": lambda value, default: int(value) if value is not None else default,
    }
    defaults.update(overrides)
    return shortcut_command_flow.ShortcutCommandFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_alpha_command_without_args_shows_menu() -> None:
    message = _message("/alpha")
    deps = _deps()

    await shortcut_command_flow.handle_alpha_command(message=message, deps=deps)

    message.answer.assert_awaited_once_with("Pick an analysis shortcut or tap Custom.", reply_markup={"menu": "alpha"})


@pytest.mark.asyncio
async def test_alpha_command_single_symbol_dispatches_watch() -> None:
    message = _message("/alpha btc")
    dispatch_command_text = AsyncMock(return_value=True)
    deps = _deps(dispatch_command_text=dispatch_command_text)

    await shortcut_command_flow.handle_alpha_command(message=message, deps=deps)

    dispatch_command_text.assert_awaited_once_with(message, "watch btc")


@pytest.mark.asyncio
async def test_price_command_returns_inline_quote_when_market_router_succeeds() -> None:
    message = _message("/price sol")
    deps = _deps()

    await shortcut_command_flow.handle_price_command(message=message, deps=deps)

    deps.hub.market_router.get_price.assert_awaited_once_with("SOL")
    message.answer.assert_awaited_once_with("<b>SOL</b> $123.45 (+4.20%)", reply_to_message_id=11)


@pytest.mark.asyncio
async def test_price_command_falls_back_to_watch_dispatch() -> None:
    message = _message("/price eth")
    dispatch_command_text = AsyncMock(return_value=True)
    hub = _hub()
    hub.market_router.get_price = AsyncMock(side_effect=RuntimeError("boom"))
    deps = _deps(hub=hub, dispatch_command_text=dispatch_command_text)

    await shortcut_command_flow.handle_price_command(message=message, deps=deps)

    dispatch_command_text.assert_awaited_once_with(message, "watch ETH")


@pytest.mark.asyncio
async def test_rsi_command_validates_mode_before_dispatch() -> None:
    message = _message("/rsi 1h sideways")
    deps = _deps()

    await shortcut_command_flow.handle_rsi_command(message=message, deps=deps)

    message.answer.assert_awaited_once_with("Mode must be `overbought` or `oversold`.")


@pytest.mark.asyncio
async def test_ema_command_dispatches_clamped_values() -> None:
    message = _message("/ema 700 4h 30")
    dispatch_command_text = AsyncMock(return_value=True)
    deps = _deps(dispatch_command_text=dispatch_command_text)

    await shortcut_command_flow.handle_ema_command(message=message, deps=deps)

    dispatch_command_text.assert_awaited_once_with(message, "ema 500 4h top 20")
