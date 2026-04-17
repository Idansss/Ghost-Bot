from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import trade_setup_command_flow


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=42),
        answer=AsyncMock(),
    )


def _deps(**overrides) -> trade_setup_command_flow.TradeSetupCommandFlowDependencies:
    defaults = {
        "set_wizard": AsyncMock(),
        "dispatch_command_text": AsyncMock(return_value=True),
        "findpair_quick_menu": lambda: {"menu": "findpair"},
        "setup_quick_menu": lambda: {"menu": "setup"},
    }
    defaults.update(overrides)
    return trade_setup_command_flow.TradeSetupCommandFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_tradecheck_sets_symbol_wizard() -> None:
    message = _message("/tradecheck")
    set_wizard = AsyncMock()
    deps = _deps(set_wizard=set_wizard)

    await trade_setup_command_flow.handle_tradecheck_command(message=message, deps=deps)

    set_wizard.assert_awaited_once_with(42, {"step": "symbol", "data": {}})
    message.answer.assert_awaited_once_with("Tradecheck wizard: send symbol (e.g., ETH).")


@pytest.mark.asyncio
async def test_findpair_without_query_shows_menu() -> None:
    message = _message("/findpair")
    deps = _deps()

    await trade_setup_command_flow.handle_findpair_command(message=message, deps=deps)

    message.answer.assert_awaited_once_with("Pick find mode.", reply_markup={"menu": "findpair"})


@pytest.mark.asyncio
async def test_findpair_numeric_query_dispatches_coin_around() -> None:
    message = _message("/findpair 0.155")
    dispatch = AsyncMock(return_value=True)
    deps = _deps(dispatch_command_text=dispatch)

    await trade_setup_command_flow.handle_findpair_command(message=message, deps=deps)

    dispatch.assert_awaited_once_with(message, "coin around 0.155")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_name", "command"),
    [
        ("handle_setup_command", "/setup entry 100 stop 90"),
        ("handle_margin_command", "/margin 5x risk 1"),
        ("handle_pnl_command", "/pnl btc long"),
    ],
)
async def test_setup_style_commands_dispatch_trimmed_text(handler_name: str, command: str) -> None:
    message = _message(command)
    dispatch = AsyncMock(return_value=True)
    deps = _deps(dispatch_command_text=dispatch)

    handler = getattr(trade_setup_command_flow, handler_name)
    await handler(message=message, deps=deps)

    dispatch.assert_awaited_once()
    assert dispatch.await_args.args[1] == command.split(maxsplit=1)[1]
