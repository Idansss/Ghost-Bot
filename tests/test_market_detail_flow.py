from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import market_detail_flow


def _callback(data: str) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        message=SimpleNamespace(
            chat=SimpleNamespace(id=42),
            answer=AsyncMock(),
        ),
        answer=AsyncMock(),
        bot=SimpleNamespace(),
    )


async def _run_with_typing_lock(bot, chat_id: int, runner) -> None:
    await runner()


def _hub() -> SimpleNamespace:
    deriv_adapter = SimpleNamespace(get_funding_and_oi=AsyncMock(return_value={"funding_rate": 0.0012, "open_interest": 2500000000, "source": "Bybit", "source_line": "Bybit BTCUSDT"}))
    analysis_service = SimpleNamespace(deriv_adapter=deriv_adapter)
    news_service = SimpleNamespace(get_asset_headlines=AsyncMock(return_value=[]))
    wallet_service = SimpleNamespace(scan=AsyncMock())
    return SimpleNamespace(analysis_service=analysis_service, news_service=news_service, wallet_service=wallet_service)


def _deps(**overrides) -> market_detail_flow.MarketDetailFlowDependencies:
    defaults = {
        "hub": _hub(),
        "feature_flags_set": lambda: {"backtest"},
        "run_with_typing_lock": AsyncMock(side_effect=_run_with_typing_lock),
        "remember_source_context": AsyncMock(),
    }
    defaults.update(overrides)
    return market_detail_flow.MarketDetailFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_derivatives_callback_formats_payload_and_tracks_source() -> None:
    callback = _callback("derivatives:BTC")
    deps = _deps()

    await market_detail_flow.handle_derivatives_callback(callback=callback, deps=deps)

    deps.remember_source_context.assert_awaited_once_with(
        42,
        source_line="Bybit BTCUSDT",
        exchange="Bybit",
        market_kind="perp",
        symbol="BTC",
        context="derivatives",
    )
    text = callback.message.answer.await_args.args[0]
    assert "funding rate  <code>0.1200%</code>" in text
    assert "open interest <code>$2500.00B</code>" in text
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_catalysts_callback_handles_empty_headlines() -> None:
    callback = _callback("catalysts:SOL")
    deps = _deps()

    await market_detail_flow.handle_catalysts_callback(callback=callback, deps=deps)

    callback.message.answer.assert_awaited_once_with("no fresh catalysts for <b>SOL</b> right now. check back later.")
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_backtest_callback_requires_feature_flag() -> None:
    callback = _callback("backtest:ETH")
    deps = _deps(feature_flags_set=lambda: set())

    await market_detail_flow.handle_backtest_callback(callback=callback, deps=deps)

    callback.answer.assert_awaited_once_with("Backtest is disabled.", show_alert=True)
    callback.message.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_backtest_callback_prompts_for_trade_details() -> None:
    callback = _callback("backtest:ETH")
    deps = _deps()

    await market_detail_flow.handle_backtest_callback(callback=callback, deps=deps)

    text = callback.message.answer.await_args.args[0]
    assert "drop your <b>ETH</b> trade details" in text
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_save_wallet_callback_runs_scan_and_shows_alert() -> None:
    callback = _callback("save_wallet:solana:abc123")
    deps = _deps()

    await market_detail_flow.handle_save_wallet_callback(callback=callback, deps=deps)

    deps.hub.wallet_service.scan.assert_awaited_once_with("solana", "abc123", chat_id=42, save=True)
    callback.answer.assert_awaited_once_with("Wallet saved", show_alert=True)
