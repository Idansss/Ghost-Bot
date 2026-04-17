from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import quick_action_callback_flow


def _callback(data: str) -> SimpleNamespace:
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        answer=AsyncMock(),
        answer_photo=AsyncMock(),
    )
    return SimpleNamespace(
        data=data,
        message=message,
        answer=AsyncMock(),
        bot=SimpleNamespace(),
    )


def _hub() -> SimpleNamespace:
    return SimpleNamespace(
        user_service=SimpleNamespace(get_settings=AsyncMock(return_value={})),
        analysis_service=SimpleNamespace(
            analyze=AsyncMock(
                return_value={"side": "long", "data_source_line": "feed", "summary": "ok"}
            )
        ),
        cache=SimpleNamespace(set_json=AsyncMock()),
        chart_service=SimpleNamespace(render_chart=AsyncMock(return_value=(b"img", {}))),
        heatmap_service=SimpleNamespace(render=AsyncMock(return_value=(b"map", {}))),
        rsi_scanner_service=SimpleNamespace(scan=AsyncMock(return_value={"kind": "scan"})),
        news_service=SimpleNamespace(get_digest=AsyncMock(return_value={"summary": "news"})),
        orderbook_heatmap_service=SimpleNamespace(
            render_heatmap=AsyncMock(
                return_value=(
                    b"map",
                    {
                        "pair": "BTCUSDT",
                        "best_bid": 1.0,
                        "best_ask": 2.0,
                        "bid_wall": 3.0,
                        "ask_wall": 4.0,
                    },
                )
            )
        ),
    )


async def _run_with_typing_lock(bot, chat_id: int, runner) -> None:
    await runner()


def _deps(**overrides) -> quick_action_callback_flow.QuickActionCallbackDependencies:
    defaults = {
        "hub": _hub(),
        "run_with_typing_lock": AsyncMock(side_effect=_run_with_typing_lock),
        "analysis_timeframes_from_settings": lambda settings: ["1h", "4h"],
        "parse_int_list": lambda value, fallback: list(value) if isinstance(value, list) else list(fallback),
        "remember_analysis_context": AsyncMock(),
        "remember_source_context": AsyncMock(),
        "render_analysis_text": AsyncMock(return_value="analysis text"),
        "send_ghost_analysis": AsyncMock(),
        "set_pending_alert": AsyncMock(),
        "as_int": lambda value, default: int(value) if value is not None else default,
        "rsi_scan_template": lambda payload: f"scan:{payload['kind']}",
        "news_template": lambda payload: f"news:{payload['summary']}",
    }
    defaults.update(overrides)
    return quick_action_callback_flow.QuickActionCallbackDependencies(**defaults)


@pytest.mark.asyncio
async def test_quick_analysis_callback_tracks_cache_context_and_reply() -> None:
    callback = _callback("quick:analysis:SOL")
    deps = _deps()

    await quick_action_callback_flow.handle_quick_analysis_callback(callback=callback, deps=deps)

    deps.hub.analysis_service.analyze.assert_awaited_once_with(
        "SOL",
        timeframes=["1h", "4h"],
        ema_periods=[20, 50, 200],
        rsi_periods=[14],
        include_derivatives=False,
        include_news=False,
    )
    deps.hub.cache.set_json.assert_awaited_once_with(
        "last_analysis:42:SOL",
        {"side": "long", "data_source_line": "feed", "summary": "ok"},
        ttl=1800,
    )
    deps.remember_source_context.assert_awaited_once_with(
        42,
        source_line="feed",
        symbol="SOL",
        context="analysis",
    )
    deps.send_ghost_analysis.assert_awaited_once_with(
        callback.message,
        "SOL",
        "analysis text",
        direction="long",
    )


@pytest.mark.asyncio
async def test_quick_chart_callback_sends_chart_photo() -> None:
    callback = _callback("quick:chart:BTC:4h")
    deps = _deps()

    await quick_action_callback_flow.handle_quick_chart_callback(callback=callback, deps=deps)

    deps.hub.chart_service.render_chart.assert_awaited_once_with(symbol="BTC", timeframe="4h")
    callback.message.answer_photo.assert_awaited_once()
    assert callback.message.answer_photo.await_args.kwargs["caption"] == "BTC 4h chart."
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_quick_news_callback_uses_news_template() -> None:
    callback = _callback("quick:news:openai")
    deps = _deps()

    await quick_action_callback_flow.handle_quick_news_callback(callback=callback, deps=deps)

    deps.hub.news_service.get_digest.assert_awaited_once_with(topic="openai", mode="openai", limit=6)
    callback.message.answer.assert_awaited_once_with("news:news", parse_mode="HTML")
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_define_alert_sets_pending_alert_and_prompts() -> None:
    callback = _callback("define:alert")
    set_pending_alert = AsyncMock()
    deps = _deps(set_pending_alert=set_pending_alert)

    await quick_action_callback_flow.handle_define_easter_egg_callback(callback=callback, deps=deps)

    set_pending_alert.assert_awaited_once_with(42, "DEFINE")
    callback.message.answer.assert_awaited_once_with("Send alert level for DEFINE, e.g. DEFINE 0.50")
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_top_rsi_callback_scans_with_expected_defaults() -> None:
    callback = _callback("top:overbought:4h")
    deps = _deps()

    await quick_action_callback_flow.handle_top_rsi_callback(callback=callback, deps=deps)

    deps.hub.rsi_scanner_service.scan.assert_awaited_once_with(
        timeframe="4h",
        mode="overbought",
        limit=10,
        rsi_length=14,
        symbol=None,
    )
    callback.message.answer.assert_awaited_once_with("scan:scan")
    callback.answer.assert_awaited_once_with()
