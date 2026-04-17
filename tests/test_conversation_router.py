from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import conversation_router
from app.core.nlu import Intent


def _message() -> SimpleNamespace:
    return SimpleNamespace(
        answer=AsyncMock(),
        message_id=9,
        chat=SimpleNamespace(id=42),
    )


def _parsed(
    *,
    intent: Intent = Intent.UNKNOWN,
    requires_followup: bool = False,
    entities: dict | None = None,
    followup_question: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        intent=intent,
        requires_followup=requires_followup,
        entities=entities or {},
        followup_question=followup_question,
    )


async def _route(**overrides) -> bool:
    defaults = {
        "message": _message(),
        "text": "btc setup?",
        "settings": {},
        "chat_id": 42,
        "start_ts": datetime.now(UTC),
        "hub_has_llm": True,
        "parsed": _parsed(),
        "chat_mode": "hybrid",
        "llm_market_chat_reply": AsyncMock(return_value=None),
        "llm_route_message": AsyncMock(return_value=None),
        "handle_routed_intent": AsyncMock(return_value=False),
        "handle_parsed_intent": AsyncMock(return_value=False),
        "send_llm_reply": AsyncMock(),
        "is_definition_question": lambda _text: False,
        "is_likely_english_phrase": lambda _text: False,
        "extract_action_symbol_hint": lambda _text: "BTC",
        "clarifying_question": lambda symbol: f"clarify:{symbol}",
        "smart_action_menu": lambda symbol: {"symbol": symbol},
        "analysis_symbol_followup_kb": lambda: "analysis-kb",
    }
    defaults.update(overrides)
    return await conversation_router.handle_free_text_routing(**defaults)


@pytest.mark.asyncio
async def test_chat_only_mode_sends_llm_reply() -> None:
    send_llm_reply = AsyncMock()

    handled = await _route(
        chat_mode="chat_only",
        llm_market_chat_reply=AsyncMock(return_value="btc still looks strong"),
        send_llm_reply=send_llm_reply,
    )

    assert handled is True
    send_llm_reply.assert_awaited_once()
    kwargs = send_llm_reply.await_args.kwargs
    assert kwargs["analytics"]["route"] == "chat_only"
    assert kwargs["analytics"]["reply_kind"] == "market_chat"


@pytest.mark.asyncio
async def test_llm_first_route_short_circuits_on_routed_intent() -> None:
    llm_market_chat_reply = AsyncMock(return_value="should not send")

    handled = await _route(
        chat_mode="llm_first",
        llm_route_message=AsyncMock(return_value={"intent": "analysis"}),
        handle_routed_intent=AsyncMock(return_value=True),
        llm_market_chat_reply=llm_market_chat_reply,
    )

    assert handled is True
    llm_market_chat_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_followup_without_llm_reply_prompts_for_action() -> None:
    message = _message()

    handled = await _route(
        message=message,
        parsed=_parsed(intent=Intent.UNKNOWN, requires_followup=True),
        llm_market_chat_reply=AsyncMock(return_value=None),
        is_likely_english_phrase=lambda _text: False,
        extract_action_symbol_hint=lambda _text: "BTC",
    )

    assert handled is True
    message.answer.assert_awaited_once_with(
        "pick an action for <b>BTC</b>:",
        reply_markup={"symbol": "BTC"},
        reply_to_message_id=9,
    )


@pytest.mark.asyncio
async def test_analysis_followup_without_symbol_prompts_quick_choices() -> None:
    message = _message()

    handled = await _route(
        message=message,
        parsed=_parsed(intent=Intent.ANALYSIS, requires_followup=True, entities={}),
    )

    assert handled is True
    message.answer.assert_awaited_once_with("Need one detail.", reply_markup="analysis-kb")


@pytest.mark.asyncio
async def test_parsed_fallback_sends_llm_reply_when_tool_path_does_not_handle() -> None:
    send_llm_reply = AsyncMock()

    handled = await _route(
        parsed=_parsed(intent=Intent.NEWS, requires_followup=False),
        llm_market_chat_reply=AsyncMock(return_value="macro still matters"),
        send_llm_reply=send_llm_reply,
    )

    assert handled is True
    send_llm_reply.assert_awaited_once()
    kwargs = send_llm_reply.await_args.kwargs
    assert kwargs["analytics"]["route"] == "parsed_fallback"
    assert kwargs["analytics"]["reply_kind"] == "market_chat"
