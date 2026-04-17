from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import position_command_flow


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=42),
        answer=AsyncMock(),
    )


def _deps(**overrides) -> position_command_flow.PositionCommandFlowDependencies:
    defaults = {
        "hub": SimpleNamespace(
            portfolio_service=SimpleNamespace(
                get_portfolio_summary=AsyncMock(),
                list_positions=AsyncMock(),
                delete_position=AsyncMock(),
                add_position=AsyncMock(),
            )
        ),
        "feature_flags_set": lambda: {"portfolio"},
    }
    defaults.update(overrides)
    return position_command_flow.PositionCommandFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_position_command_respects_feature_flag() -> None:
    message = _message("/position list")
    deps = _deps(feature_flags_set=lambda: set())

    await position_command_flow.handle_position_command(message=message, deps=deps)

    message.answer.assert_awaited_once_with("Portfolio feature is disabled.")


@pytest.mark.asyncio
async def test_position_summary_renders_best_and_worst() -> None:
    message = _message("/position summary")
    service = SimpleNamespace(
        get_portfolio_summary=AsyncMock(
            return_value={
                "positions": [{"symbol": "BTC", "side": "long", "allocation_pct": 60, "pnl_pct": 5.2}],
                "total_cost_usd": 1000.0,
                "total_value_usd": 1120.0,
                "total_pnl_usd": 120.0,
                "total_pnl_pct": 12.0,
                "best": {"symbol": "BTC", "pnl_pct": 5.2},
                "worst": {"symbol": "ETH", "pnl_pct": -2.1},
            }
        ),
        list_positions=AsyncMock(),
        delete_position=AsyncMock(),
        add_position=AsyncMock(),
    )
    deps = _deps(hub=SimpleNamespace(portfolio_service=service))

    await position_command_flow.handle_position_command(message=message, deps=deps)

    service.get_portfolio_summary.assert_awaited_once_with(42)
    reply = message.answer.await_args.args[0]
    assert "Portfolio Summary" in reply
    assert "BTC long" in reply
    assert "Best:" in reply
    assert "Worst:" in reply


@pytest.mark.asyncio
async def test_position_add_treats_non_numeric_leverage_as_notes() -> None:
    message = _message("/position add btc short 50000 1000 scalp entry")
    service = SimpleNamespace(
        get_portfolio_summary=AsyncMock(),
        list_positions=AsyncMock(),
        delete_position=AsyncMock(),
        add_position=AsyncMock(return_value=({"id": 7}, "low liquidity")),
    )
    deps = _deps(hub=SimpleNamespace(portfolio_service=service))

    await position_command_flow.handle_position_command(message=message, deps=deps)

    service.add_position.assert_awaited_once_with(
        42,
        symbol="BTC",
        side="short",
        entry_price=50000.0,
        size_quote=1000.0,
        leverage=1.0,
        notes="scalp entry",
    )
    reply = message.answer.await_args.args[0]
    assert "Position added" in reply
    assert "Warning: low liquidity" in reply
