from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import free_text_flow


def _message(text: str = "btc") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        answer=AsyncMock(),
        chat=SimpleNamespace(id=42),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
    )


def _hub() -> SimpleNamespace:
    return SimpleNamespace(
        llm_client=None,
        user_service=SimpleNamespace(
            get_settings=AsyncMock(return_value={"disclaimer_seen": True}),
            update_settings=AsyncMock(),
        ),
    )


def _deps(**overrides) -> free_text_flow.FreeTextFlowDependencies:
    defaults = {
        "hub": _hub(),
        "parse_message": lambda text: {"parsed": text},
        "openai_chat_mode": lambda: "hybrid",
        "route_free_text": AsyncMock(return_value=True),
        "send_llm_reply": AsyncMock(),
        "get_chat_history": AsyncMock(return_value=[]),
        "dispatch_command_text": AsyncMock(return_value=True),
        "recent_analysis_context": AsyncMock(return_value=None),
        "looks_like_analysis_followup": lambda _text, _ctx: False,
        "llm_followup_reply": AsyncMock(return_value=None),
        "llm_market_chat_reply": AsyncMock(return_value=None),
        "llm_route_message": AsyncMock(return_value=None),
        "handle_routed_intent": AsyncMock(return_value=False),
        "handle_parsed_intent": AsyncMock(return_value=False),
        "is_definition_question": lambda _text: False,
        "is_likely_english_phrase": lambda _text: False,
        "extract_action_symbol_hint": lambda _text: "BTC",
        "clarifying_question": lambda symbol: f"clarify:{symbol}",
        "smart_action_menu": lambda symbol: {"symbol": symbol},
        "analysis_symbol_followup_kb": lambda: {"kind": "analysis"},
        "define_keyboard": lambda: {"kind": "define"},
        "pause": AsyncMock(),
    }
    defaults.update(overrides)
    return free_text_flow.FreeTextFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_disclaimer_is_sent_once_then_router_runs() -> None:
    message = _message("btc")
    hub = _hub()
    hub.user_service.get_settings = AsyncMock(return_value={"disclaimer_seen": False})
    route_free_text = AsyncMock(return_value=True)
    deps = _deps(hub=hub, route_free_text=route_free_text)

    handled = await free_text_flow.handle_free_text_flow(
        message=message,
        text="btc",
        chat_id=42,
        start_ts=datetime.now(UTC),
        deps=deps,
    )

    assert handled is True
    message.answer.assert_awaited_once_with(
        "\u26a0\ufe0f <b>Disclaimer</b>: This bot is for info only, not financial advice. Data may be delayed. Trade at your own risk."
    )
    hub.user_service.update_settings.assert_awaited_once_with(42, {"disclaimer_seen": True})
    route_free_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_repair_flow_sends_llm_reply_with_analytics() -> None:
    message = _message("that's not what i meant")
    hub = _hub()
    hub.llm_client = SimpleNamespace(reply=AsyncMock(return_value="repaired"))
    send_llm_reply = AsyncMock()
    route_free_text = AsyncMock(return_value=True)
    deps = _deps(
        hub=hub,
        send_llm_reply=send_llm_reply,
        route_free_text=route_free_text,
        get_chat_history=AsyncMock(
            return_value=[
                {"role": "user", "content": "btc?"},
                {"role": "assistant", "content": "old reply"},
            ]
        ),
    )

    handled = await free_text_flow.handle_free_text_flow(
        message=message,
        text="that's not what i meant",
        chat_id=42,
        start_ts=datetime.now(UTC),
        deps=deps,
    )

    assert handled is True
    send_llm_reply.assert_awaited_once()
    kwargs = send_llm_reply.await_args.kwargs
    assert kwargs["analytics"]["route"] == "repair"
    assert kwargs["analytics"]["reply_kind"] == "repair"
    route_free_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_define_trading_short_circuits_with_keyboard() -> None:
    message = _message("define trading")
    pause = AsyncMock()
    route_free_text = AsyncMock(return_value=True)
    deps = _deps(pause=pause, route_free_text=route_free_text)

    handled = await free_text_flow.handle_free_text_flow(
        message=message,
        text="define trading",
        chat_id=42,
        start_ts=datetime.now(UTC),
        deps=deps,
    )

    assert handled is True
    message.bot.send_chat_action.assert_awaited_once()
    pause.assert_awaited_once_with(0.8)
    message.answer.assert_awaited_once()
    assert message.answer.await_args.kwargs["reply_markup"] == {"kind": "define"}
    route_free_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_scanner_shortcut_dispatches_synthetic_command() -> None:
    message = _message("show overbought list")
    dispatch_command_text = AsyncMock(return_value=True)
    route_free_text = AsyncMock(return_value=True)
    deps = _deps(dispatch_command_text=dispatch_command_text, route_free_text=route_free_text)

    handled = await free_text_flow.handle_free_text_flow(
        message=message,
        text="show overbought list",
        chat_id=42,
        start_ts=datetime.now(UTC),
        deps=deps,
    )

    assert handled is True
    dispatch_command_text.assert_awaited_once_with(message, "rsi top 10 1h overbought")
    route_free_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_analysis_followup_sends_followup_reply() -> None:
    message = _message("what about 99?")
    send_llm_reply = AsyncMock()
    route_free_text = AsyncMock(return_value=True)
    deps = _deps(
        send_llm_reply=send_llm_reply,
        route_free_text=route_free_text,
        recent_analysis_context=AsyncMock(return_value={"symbol": "BTC"}),
        looks_like_analysis_followup=lambda _text, _ctx: True,
        llm_followup_reply=AsyncMock(return_value="followup reply"),
    )

    handled = await free_text_flow.handle_free_text_flow(
        message=message,
        text="what about 99?",
        chat_id=42,
        start_ts=datetime.now(UTC),
        deps=deps,
    )

    assert handled is True
    send_llm_reply.assert_awaited_once()
    kwargs = send_llm_reply.await_args.kwargs
    assert kwargs["analytics"]["route"] == "analysis_followup"
    assert kwargs["analytics"]["reply_kind"] == "followup"
    route_free_text.assert_not_awaited()
