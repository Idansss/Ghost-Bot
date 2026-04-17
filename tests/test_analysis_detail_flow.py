from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import analysis_detail_flow


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
    cache = SimpleNamespace(
        get_json=AsyncMock(),
        set_json=AsyncMock(),
    )
    user_service = SimpleNamespace(get_settings=AsyncMock(return_value={"preferred_ema_periods": [20, 50, 200], "preferred_rsi_periods": [14]}))
    analysis_service = SimpleNamespace(
        analyze=AsyncMock(return_value={"side": "long", "data_source_line": "Bybit BTCUSDT", "entry": "100", "tp1": "110", "tp2": "120", "sl": "95"}),
    )
    return SimpleNamespace(cache=cache, user_service=user_service, analysis_service=analysis_service)


def _deps(**overrides) -> analysis_detail_flow.AnalysisDetailFlowDependencies:
    defaults = {
        "hub": _hub(),
        "set_pending_alert": AsyncMock(),
        "run_with_typing_lock": AsyncMock(side_effect=_run_with_typing_lock),
        "analysis_timeframes_from_settings": lambda settings: ["1h", "4h"],
        "parse_int_list": lambda value, fallback: list(value or fallback),
        "remember_analysis_context": AsyncMock(),
        "remember_source_context": AsyncMock(),
        "render_analysis_text": AsyncMock(return_value="analysis text"),
        "send_ghost_analysis": AsyncMock(),
    }
    defaults.update(overrides)
    return analysis_detail_flow.AnalysisDetailFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_set_alert_callback_stores_symbol_and_prompts() -> None:
    callback = _callback("set_alert:SOL")
    deps = _deps()

    await analysis_detail_flow.handle_set_alert_callback(callback=callback, deps=deps)

    deps.set_pending_alert.assert_awaited_once_with(42, "SOL")
    callback.message.answer.assert_awaited_once()
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_show_levels_uses_cached_payload() -> None:
    callback = _callback("show_levels:BTC")
    hub = _hub()
    hub.cache.get_json = AsyncMock(return_value={"entry": "100", "tp1": "110", "tp2": "120", "sl": "95"})
    deps = _deps(hub=hub)

    await analysis_detail_flow.handle_show_levels_callback(callback=callback, deps=deps)

    callback.message.answer.assert_awaited_once()
    text = callback.message.answer.await_args.args[0]
    assert "<b>BTC</b> key levels" in text
    assert "entry    <code>100</code>" in text
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_why_callback_uses_bullets_when_present() -> None:
    callback = _callback("why:BTC")
    hub = _hub()
    hub.cache.get_json = AsyncMock(return_value={"why": ["trend strong", "breakout confirmed"]})
    deps = _deps(hub=hub)

    await analysis_detail_flow.handle_why_callback(callback=callback, deps=deps)

    text = callback.message.answer.await_args.args[0]
    assert "- trend strong" in text
    assert "- breakout confirmed" in text


@pytest.mark.asyncio
async def test_refresh_callback_reanalyzes_and_sends_non_detailed_reply() -> None:
    callback = _callback("refresh:BTC")
    deps = _deps()

    await analysis_detail_flow.handle_refresh_callback(callback=callback, deps=deps)

    deps.run_with_typing_lock.assert_awaited_once()
    deps.hub.analysis_service.analyze.assert_awaited_once()
    kwargs = deps.hub.analysis_service.analyze.await_args.kwargs
    assert kwargs["include_derivatives"] is False
    assert kwargs["include_news"] is False
    deps.render_analysis_text.assert_awaited_once()
    assert deps.render_analysis_text.await_args.kwargs["detailed"] is False
    deps.send_ghost_analysis.assert_awaited_once_with(callback.message, "BTC", "analysis text", direction="long")
    callback.answer.assert_awaited_once_with("Refreshed")


@pytest.mark.asyncio
async def test_details_callback_reanalyzes_with_detailed_mode() -> None:
    callback = _callback("details:ETH")
    deps = _deps()

    await analysis_detail_flow.handle_details_callback(callback=callback, deps=deps)

    kwargs = deps.hub.analysis_service.analyze.await_args.kwargs
    assert kwargs["include_derivatives"] is True
    assert kwargs["include_news"] is True
    assert deps.render_analysis_text.await_args.kwargs["detailed"] is True
    deps.send_ghost_analysis.assert_awaited_once_with(callback.message, "ETH", "analysis text", direction="long")
    callback.answer.assert_awaited_once_with("Detailed mode")
