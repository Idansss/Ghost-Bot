from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import alert_command_flow


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=42),
        message_id=9,
        answer=AsyncMock(),
    )


def _alert(alert_id: int = 1, symbol: str = "BTC") -> SimpleNamespace:
    return SimpleNamespace(
        id=alert_id,
        symbol=symbol,
        condition="above",
        target_price=100.0,
        status="active",
        source_exchange="binance",
        market_kind="spot",
        instrument_id=f"{symbol}USDT",
    )


def _hub() -> SimpleNamespace:
    return SimpleNamespace(
        watchlist_service=SimpleNamespace(
            build_watchlist=AsyncMock(return_value={"items": ["BTC"], "summary": "watch", "source_line": "scanner"})
        ),
        news_service=SimpleNamespace(
            get_digest=AsyncMock(
                return_value={
                    "summary": "digest",
                    "headlines": [{"source": "CoinDesk", "url": "https://example.com/story"}],
                }
            )
        ),
        user_service=SimpleNamespace(get_settings=AsyncMock(return_value={"tone_mode": "wild"})),
        cycles_service=SimpleNamespace(cycle_check=AsyncMock(return_value={"summary": "cycle"})),
        rate_limiter=SimpleNamespace(check=AsyncMock(return_value=SimpleNamespace(allowed=True))),
        wallet_service=SimpleNamespace(scan=AsyncMock(return_value={"chain": "solana", "address": "abc"})),
        alerts_service=SimpleNamespace(
            list_alerts=AsyncMock(return_value=[_alert()]),
            clear_user_alerts=AsyncMock(return_value=3),
            pause_user_alerts=AsyncMock(return_value=2),
            resume_user_alerts=AsyncMock(return_value=4),
            delete_alert=AsyncMock(return_value=True),
            create_alert=AsyncMock(return_value=_alert()),
            delete_alerts_by_symbol=AsyncMock(return_value=5),
        ),
    )


def _deps(**overrides) -> alert_command_flow.AlertCommandFlowDependencies:
    defaults = {
        "hub": _hub(),
        "wallet_scan_limit_per_hour": 6,
        "remember_source_context": AsyncMock(),
        "watchlist_template": lambda payload: f"watchlist:{payload['summary']}",
        "news_template": lambda payload: f"news:{payload['summary']}",
        "cycle_template": lambda payload, settings: f"cycle:{payload['summary']}:{settings['tone_mode']}",
        "wallet_scan_template": lambda payload: f"wallet:{payload['chain']}",
        "wallet_actions": lambda chain, address: {"wallet": (chain, address)},
        "scan_quick_menu": lambda: {"menu": "scan"},
        "news_quick_menu": lambda: {"menu": "news"},
        "alert_quick_menu": lambda: {"menu": "alert"},
        "simple_followup": lambda options: {"options": options},
        "alert_created_menu": lambda symbol: {"menu": symbol},
    }
    defaults.update(overrides)
    return alert_command_flow.AlertCommandFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_watchlist_command_tracks_source_context() -> None:
    message = _message("/watchlist 12 long")
    remember_source_context = AsyncMock()
    deps = _deps(remember_source_context=remember_source_context)

    await alert_command_flow.handle_watchlist_command(message=message, deps=deps)

    deps.hub.watchlist_service.build_watchlist.assert_awaited_once_with(count=12, direction="long")
    remember_source_context.assert_awaited_once_with(42, source_line="scanner", context="watchlist")
    message.answer.assert_awaited_once_with("watchlist:watch")


@pytest.mark.asyncio
async def test_news_command_switches_to_openai_mode_and_tracks_first_headline() -> None:
    message = _message("/news chatgpt 8")
    remember_source_context = AsyncMock()
    deps = _deps(remember_source_context=remember_source_context)

    await alert_command_flow.handle_news_command(message=message, deps=deps)

    deps.hub.news_service.get_digest.assert_awaited_once_with(topic="openai", mode="openai", limit=8)
    remember_source_context.assert_awaited_once_with(
        42,
        source_line="CoinDesk | https://example.com/story",
        context="news",
    )
    message.answer.assert_awaited_once_with("news:digest", parse_mode="HTML")


@pytest.mark.asyncio
async def test_scan_command_without_address_shows_quick_menu() -> None:
    message = _message("/scan")
    deps = _deps()

    await alert_command_flow.handle_scan_command(message=message, deps=deps)

    message.answer.assert_awaited_once_with("Pick chain first, then paste address.", reply_markup={"menu": "scan"})


@pytest.mark.asyncio
async def test_alert_command_add_creates_alert_and_tracks_source() -> None:
    message = _message("/alert add sol above 155.5")
    remember_source_context = AsyncMock()
    deps = _deps(remember_source_context=remember_source_context)

    await alert_command_flow.handle_alert_command(message=message, deps=deps)

    deps.hub.alerts_service.create_alert.assert_awaited_once_with(42, "SOL", "above", 155.5, source="command")
    remember_source_context.assert_awaited_once_with(
        42,
        exchange="binance",
        market_kind="spot",
        instrument_id="BTCUSDT",
        symbol="SOL",
        context="alert",
    )
    message.answer.assert_awaited_once_with(
        "alert set <b>SOL</b> crosses above <b>$155.50</b>.\n"
        "i'll ping you the moment it hits. don't get liquidated.",
        reply_markup={"menu": "SOL"},
    )


@pytest.mark.asyncio
async def test_alertdel_command_without_id_builds_delete_options() -> None:
    message = _message("/alertdel")
    deps = _deps()

    await alert_command_flow.handle_alertdel_command(message=message, deps=deps)

    message.answer.assert_awaited_once_with(
        "Tap an alert to delete.",
        reply_markup={"options": [("Delete #1", "cmd:alertdel:1")]},
    )


@pytest.mark.asyncio
async def test_alertclear_command_by_symbol_clears_matching_alerts() -> None:
    message = _message("/alertclear eth")
    deps = _deps()

    await alert_command_flow.handle_alertclear_command(message=message, deps=deps)

    deps.hub.alerts_service.delete_alerts_by_symbol.assert_awaited_once_with(42, "ETH")
    message.answer.assert_awaited_once_with("Cleared 5 alerts for ETH.")
