from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.enums import ChatAction

from app.bot import routed_intent_executor


def _message(text: str = "btc setup?") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        answer=AsyncMock(),
        answer_photo=AsyncMock(),
        chat=SimpleNamespace(id=42),
        from_user=SimpleNamespace(id=7),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
    )


def _hub() -> SimpleNamespace:
    return SimpleNamespace(
        cache=SimpleNamespace(set_json=AsyncMock()),
        news_service=SimpleNamespace(
            get_asset_headlines=AsyncMock(return_value=[]),
            get_digest=AsyncMock(return_value={"headlines": []}),
        ),
        analysis_service=SimpleNamespace(analyze=AsyncMock(return_value={})),
        rsi_scanner_service=SimpleNamespace(scan=AsyncMock(return_value={})),
        ema_scanner_service=SimpleNamespace(scan=AsyncMock(return_value={"summary": "none", "items": [], "ema_length": 200})),
        chart_service=SimpleNamespace(render_chart=AsyncMock(return_value=(b"", {}))),
        orderbook_heatmap_service=SimpleNamespace(render_heatmap=AsyncMock(return_value=(b"", {}))),
        watchlist_service=SimpleNamespace(build_watchlist=AsyncMock(return_value={})),
        alerts_service=SimpleNamespace(
            create_alert=AsyncMock(),
            list_alerts=AsyncMock(return_value=[]),
            delete_alerts_by_symbol=AsyncMock(return_value=0),
            delete_alert=AsyncMock(return_value=False),
            clear_user_alerts=AsyncMock(return_value=0),
        ),
        discovery_service=SimpleNamespace(
            find_pair=AsyncMock(return_value={}),
            guess_by_price=AsyncMock(return_value={}),
        ),
        market_router=SimpleNamespace(get_price=AsyncMock(return_value={})),
        setup_review_service=SimpleNamespace(review=AsyncMock(return_value={})),
        giveaway_service=SimpleNamespace(
            join_active=AsyncMock(return_value={"giveaway_id": 1, "participants": 2}),
            status=AsyncMock(return_value={}),
            end_giveaway=AsyncMock(return_value={"giveaway_id": 1, "winner_user_id": None, "note": "no entries"}),
            reroll=AsyncMock(return_value={"giveaway_id": 1, "winner_user_id": 9}),
            start_giveaway=AsyncMock(return_value={"id": 1, "prize": "Prize", "end_time": "later"}),
        ),
        trade_journal_service=None,
    )


def _deps(**overrides) -> routed_intent_executor.RoutedIntentDependencies:
    defaults = {
        "hub": _hub(),
        "openai_router_min_confidence": 0.6,
        "bot_meta_re": re.compile(r"create alert", re.IGNORECASE),
        "try_answer_howto": lambda _text: None,
        "llm_fallback_reply": AsyncMock(return_value=None),
        "llm_market_chat_reply": AsyncMock(return_value=None),
        "send_llm_reply": AsyncMock(),
        "as_int": lambda value, default: int(value) if value is not None else default,
        "as_float": lambda value, default=None: float(value) if value is not None else default,
        "as_float_list": lambda value: [float(item) for item in value] if isinstance(value, list) else [],
        "extract_symbol": lambda params: params.get("symbol"),
        "normalize_symbol_value": lambda value: str(value).upper().lstrip("$") if value else None,
        "analysis_timeframes_from_settings": lambda _settings: ["1h", "4h"],
        "parse_int_list": lambda value, fallback: list(value) if isinstance(value, list) else fallback,
        "append_last_symbol": AsyncMock(),
        "remember_analysis_context": AsyncMock(),
        "remember_source_context": AsyncMock(),
        "render_analysis_text": AsyncMock(return_value="analysis text"),
        "send_ghost_analysis": AsyncMock(),
        "safe_exc": lambda exc: str(exc),
        "parse_duration_to_seconds": lambda raw: {"10m": 600}.get(raw),
        "trade_math_payload": lambda **kwargs: kwargs,
        "feature_flags_set": lambda: set(),
    }
    defaults.update(overrides)
    return routed_intent_executor.RoutedIntentDependencies(**defaults)


@pytest.mark.asyncio
async def test_low_confidence_route_returns_false() -> None:
    message = _message()
    deps = _deps()

    handled = await routed_intent_executor.execute_routed_intent(
        message=message,
        settings={},
        route={"intent": "market_chat", "confidence": 0.2},
        deps=deps,
    )

    assert handled is False
    deps.send_llm_reply.assert_not_awaited()
    message.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_bot_meta_smalltalk_prefers_howto_reply() -> None:
    message = _message("how do i create alert")
    deps = _deps(try_answer_howto=lambda _text: "Use the alert command.")

    handled = await routed_intent_executor.execute_routed_intent(
        message=message,
        settings={},
        route={"intent": "smalltalk", "confidence": 0.91},
        deps=deps,
    )

    assert handled is True
    message.bot.send_chat_action.assert_awaited_once_with(42, ChatAction.TYPING)
    message.answer.assert_awaited_once_with("Use the alert command.")
    deps.send_llm_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_market_chat_preserves_router_analytics() -> None:
    message = _message("btc still strong?")
    send_llm_reply = AsyncMock()
    deps = _deps(
        llm_market_chat_reply=AsyncMock(return_value="still bullish"),
        send_llm_reply=send_llm_reply,
        bot_meta_re=re.compile(r"$^"),
    )

    handled = await routed_intent_executor.execute_routed_intent(
        message=message,
        settings={"tone": "ghost"},
        route={"intent": "market_chat", "confidence": 0.87},
        deps=deps,
    )

    assert handled is True
    send_llm_reply.assert_awaited_once()
    kwargs = send_llm_reply.await_args.kwargs
    assert kwargs["analytics"]["route"] == "market_chat"
    assert kwargs["analytics"]["reply_kind"] == "market_chat"
    assert kwargs["analytics"]["router_intent"] == "market_chat"
    assert kwargs["analytics"]["router_confidence"] == 0.87


@pytest.mark.asyncio
async def test_market_analysis_persists_context_and_delegates_rendering() -> None:
    message = _message("analyze btc")
    payload = {"summary": "trend up", "data_source_line": "Bybit BTCUSDT"}
    hub = _hub()
    hub.analysis_service.analyze = AsyncMock(return_value=payload)
    append_last_symbol = AsyncMock()
    remember_analysis_context = AsyncMock()
    remember_source_context = AsyncMock()
    render_analysis_text = AsyncMock(return_value="analysis text")
    send_ghost_analysis = AsyncMock()
    deps = _deps(
        hub=hub,
        extract_symbol=lambda params: params.get("symbol"),
        append_last_symbol=append_last_symbol,
        remember_analysis_context=remember_analysis_context,
        remember_source_context=remember_source_context,
        render_analysis_text=render_analysis_text,
        send_ghost_analysis=send_ghost_analysis,
    )

    handled = await routed_intent_executor.execute_routed_intent(
        message=message,
        settings={
            "preferred_ema_periods": [20, 50, 200],
            "preferred_rsi_periods": [14],
        },
        route={
            "intent": "market_analysis",
            "confidence": 0.95,
            "params": {"symbol": "BTC", "timeframe": "4h", "direction": "long"},
        },
        deps=deps,
    )

    assert handled is True
    hub.analysis_service.analyze.assert_awaited_once_with(
        "BTC",
        direction="long",
        timeframe="4h",
        timeframes=["4h"],
        ema_periods=[20, 50, 200],
        rsi_periods=[14],
        include_derivatives=False,
        include_news=False,
    )
    hub.cache.set_json.assert_awaited_once_with("last_analysis:42:BTC", payload, ttl=1800)
    append_last_symbol.assert_awaited_once_with(42, "BTC")
    remember_analysis_context.assert_awaited_once_with(42, "BTC", "long", payload)
    remember_source_context.assert_awaited_once_with(
        42,
        source_line="Bybit BTCUSDT",
        symbol="BTC",
        context="analysis",
    )
    render_analysis_text.assert_awaited_once()
    send_ghost_analysis.assert_awaited_once_with(message, "BTC", "analysis text", direction="long")


@pytest.mark.asyncio
async def test_alert_create_falls_back_to_raw_text_and_tracks_source_context() -> None:
    message = _message("set alert for btc 65000 above")
    alert = SimpleNamespace(
        symbol="BTC",
        source_exchange="Bybit",
        market_kind="perp",
        instrument_id="BTCUSDT",
    )
    hub = _hub()
    hub.alerts_service.create_alert = AsyncMock(return_value=alert)
    remember_source_context = AsyncMock()
    deps = _deps(
        hub=hub,
        remember_source_context=remember_source_context,
        extract_symbol=lambda _params: None,
        as_float=lambda value, default=None: float(value) if value is not None else default,
    )

    handled = await routed_intent_executor.execute_routed_intent(
        message=message,
        settings={},
        route={
            "intent": "alert_create",
            "confidence": 0.9,
            "params": {"operator": "above"},
        },
        deps=deps,
    )

    assert handled is True
    hub.alerts_service.create_alert.assert_awaited_once_with(42, "BTC", "above", 65000.0)
    remember_source_context.assert_awaited_once_with(
        42,
        exchange="Bybit",
        market_kind="perp",
        instrument_id="BTCUSDT",
        symbol="BTC",
        context="alert",
    )
    message.answer.assert_awaited_once()
    kwargs = message.answer.await_args.kwargs
    assert kwargs["reply_markup"] is not None
