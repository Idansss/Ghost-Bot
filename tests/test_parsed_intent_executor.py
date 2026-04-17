from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import parsed_intent_executor
from app.core.nlu import Intent


class _Lock:
    def __init__(self, acquired: bool = True) -> None:
        self.acquired = acquired

    async def __aenter__(self) -> bool:
        return self.acquired

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _message(text: str = "btc") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        answer=AsyncMock(),
        answer_photo=AsyncMock(),
        message_id=9,
        chat=SimpleNamespace(id=42),
        from_user=SimpleNamespace(id=7),
    )


def _parsed(intent: Intent, entities: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(intent=intent, entities=entities or {})


def _hub() -> SimpleNamespace:
    return SimpleNamespace(
        analysis_uc=SimpleNamespace(analyze=AsyncMock()),
        market_router=SimpleNamespace(get_price=AsyncMock(return_value={})),
        setup_review_service=SimpleNamespace(review=AsyncMock(return_value={})),
        rsi_scan_uc=SimpleNamespace(scan=AsyncMock()),
        ema_scan_uc=SimpleNamespace(scan=AsyncMock()),
        chart_service=SimpleNamespace(render_chart=AsyncMock()),
        orderbook_heatmap_service=SimpleNamespace(render_heatmap=AsyncMock()),
        discovery_service=SimpleNamespace(
            find_pair=AsyncMock(return_value={}),
            guess_by_price=AsyncMock(return_value={}),
        ),
        alerts_uc=SimpleNamespace(
            create=AsyncMock(),
            list=AsyncMock(return_value=[]),
            pause=AsyncMock(return_value=0),
            resume=AsyncMock(return_value=0),
            delete_by_symbol=AsyncMock(return_value=0),
            delete=AsyncMock(return_value=False),
        ),
        alerts_service=SimpleNamespace(list_alerts=AsyncMock(return_value=[])),
        giveaway_service=SimpleNamespace(
            join_active=AsyncMock(return_value={"giveaway_id": 1, "participants": 2}),
            status=AsyncMock(return_value={}),
            start_giveaway=AsyncMock(return_value={"id": 1, "prize": "Prize", "end_time": "later"}),
            end_giveaway=AsyncMock(return_value={"giveaway_id": 1, "winner_user_id": None, "note": "none"}),
            reroll=AsyncMock(return_value={"giveaway_id": 1, "winner_user_id": 2, "previous_winner_user_id": 1}),
        ),
        watchlist_service=SimpleNamespace(build_watchlist=AsyncMock(return_value={})),
        news_uc=SimpleNamespace(get_digest=AsyncMock()),
        wallet_service=SimpleNamespace(scan=AsyncMock(return_value={})),
        cycles_service=SimpleNamespace(cycle_check=AsyncMock(return_value={})),
        trade_verify_service=SimpleNamespace(verify=AsyncMock(return_value={"source_line": "Bybit BTCUSDT"})),
        correlation_service=SimpleNamespace(check_following=AsyncMock(return_value={})),
        user_service=SimpleNamespace(get_settings=AsyncMock(return_value={"tone": "ghost"})),
        rate_limiter=SimpleNamespace(check=AsyncMock(return_value=SimpleNamespace(allowed=True))),
        cache=SimpleNamespace(distributed_lock=lambda *args, **kwargs: _Lock(True)),
    )


def _deps(**overrides) -> parsed_intent_executor.ParsedIntentDependencies:
    defaults = {
        "hub": _hub(),
        "wallet_scan_limit_per_hour": 3,
        "maybe_send_market_warning": AsyncMock(),
        "analysis_timeframes_from_settings": lambda _settings: ["1h", "4h"],
        "parse_int_list": lambda value, fallback: list(value) if isinstance(value, list) else fallback,
        "append_last_symbol": AsyncMock(),
        "remember_analysis_context": AsyncMock(),
        "remember_source_context": AsyncMock(),
        "render_analysis_text": AsyncMock(return_value="analysis text"),
        "send_ghost_analysis": AsyncMock(),
        "as_float": lambda value, default=None: float(value) if value is not None else default,
        "trade_math_payload": lambda **kwargs: kwargs,
        "llm_fallback_reply": AsyncMock(return_value=None),
        "safe_exc": lambda exc: str(exc),
        "parse_duration_to_seconds": lambda raw: {"10m": 600}.get(raw),
        "save_trade_check": AsyncMock(),
    }
    defaults.update(overrides)
    return parsed_intent_executor.ParsedIntentDependencies(**defaults)


@pytest.mark.asyncio
async def test_analysis_cached_result_preserves_side_effects() -> None:
    message = _message("btc long")
    payload = {"data_source_line": "Bybit BTCUSDT", "summary": "trend up"}
    hub = _hub()
    hub.analysis_uc.analyze = AsyncMock(
        return_value=SimpleNamespace(kind="cached", payload=payload, error=None, fallback=None)
    )
    append_last_symbol = AsyncMock()
    remember_analysis_context = AsyncMock()
    remember_source_context = AsyncMock()
    render_analysis_text = AsyncMock(return_value="analysis text")
    send_ghost_analysis = AsyncMock()
    deps = _deps(
        hub=hub,
        append_last_symbol=append_last_symbol,
        remember_analysis_context=remember_analysis_context,
        remember_source_context=remember_source_context,
        render_analysis_text=render_analysis_text,
        send_ghost_analysis=send_ghost_analysis,
    )

    handled = await parsed_intent_executor.execute_parsed_intent(
        message=message,
        parsed=_parsed(Intent.ANALYSIS, {"symbol": "BTC", "direction": "long"}),
        settings={"preferred_ema_periods": [20, 50, 200], "preferred_rsi_periods": [14]},
        deps=deps,
    )

    assert handled is True
    append_last_symbol.assert_awaited_once_with(42, "BTC")
    remember_analysis_context.assert_awaited_once_with(42, "BTC", "long", payload)
    remember_source_context.assert_awaited_once_with(
        42,
        source_line="Bybit BTCUSDT",
        symbol="BTC",
        context="analysis",
    )
    message.answer.assert_awaited_once()
    send_ghost_analysis.assert_awaited_once_with(message, "BTC", "analysis text", direction="long")


@pytest.mark.asyncio
async def test_alert_create_includes_extra_conditions_and_source_context() -> None:
    message = _message("alert btc 65000")
    alert = SimpleNamespace(
        symbol="BTC",
        source_exchange="Bybit",
        market_kind="perp",
        instrument_id="BTCUSDT",
    )
    hub = _hub()
    hub.alerts_uc.create = AsyncMock(return_value=alert)
    remember_source_context = AsyncMock()
    deps = _deps(hub=hub, remember_source_context=remember_source_context)

    handled = await parsed_intent_executor.execute_parsed_intent(
        message=message,
        parsed=_parsed(
            Intent.ALERT_CREATE,
            {
                "symbol": "BTC",
                "target_price": 65000,
                "condition": "above",
                "extra_conditions": [{"type": "rsi", "operator": "lt", "value": 30, "timeframe": "1h"}],
            },
        ),
        settings={},
        deps=deps,
    )

    assert handled is True
    remember_source_context.assert_awaited_once_with(
        42,
        exchange="Bybit",
        market_kind="perp",
        instrument_id="BTCUSDT",
        symbol="BTC",
        context="alert",
    )
    message.answer.assert_awaited_once()
    text = message.answer.await_args.args[0]
    assert "extra conditions:" in text


@pytest.mark.asyncio
async def test_news_degraded_reply_tracks_source_context() -> None:
    message = _message("news")
    payload = {"headlines": [{"source": "CoinDesk", "url": "https://example.com"}]}
    hub = _hub()
    hub.news_uc.get_digest = AsyncMock(return_value=SimpleNamespace(payload=payload, degraded=True))
    remember_source_context = AsyncMock()
    deps = _deps(hub=hub, remember_source_context=remember_source_context)

    handled = await parsed_intent_executor.execute_parsed_intent(
        message=message,
        parsed=_parsed(Intent.NEWS, {"topic": "btc", "mode": "crypto", "limit": 6}),
        settings={},
        deps=deps,
    )

    assert handled is True
    assert message.answer.await_count == 2
    remember_source_context.assert_awaited_once_with(
        42,
        source_line="CoinDesk | https://example.com",
        context="news",
    )


@pytest.mark.asyncio
async def test_tradecheck_saves_and_reports_result() -> None:
    message = _message("check trade")
    hub = _hub()
    hub.trade_verify_service.verify = AsyncMock(return_value={"source_line": "Bybit BTCUSDT", "summary": "ok"})
    save_trade_check = AsyncMock()
    remember_source_context = AsyncMock()
    deps = _deps(hub=hub, save_trade_check=save_trade_check, remember_source_context=remember_source_context)

    handled = await parsed_intent_executor.execute_parsed_intent(
        message=message,
        parsed=_parsed(
            Intent.TRADECHECK,
            {
                "symbol": "BTC",
                "timeframe": "1h",
                "timestamp": datetime(2026, 4, 15, 12, 0, 0),
                "entry": 100,
                "stop": 95,
                "targets": [110, 120],
            },
        ),
        settings={"tone": "ghost"},
        deps=deps,
    )

    assert handled is True
    save_trade_check.assert_awaited_once()
    remember_source_context.assert_awaited_once_with(
        42,
        source_line="Bybit BTCUSDT",
        symbol="BTC",
        context="trade check",
    )
    message.answer.assert_awaited_once()
