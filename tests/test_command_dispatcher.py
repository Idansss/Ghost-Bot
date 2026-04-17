from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import command_dispatcher
from app.core.nlu import Intent


def _message() -> SimpleNamespace:
    return SimpleNamespace(
        answer=AsyncMock(),
        text="alert BTC 65000",
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


async def _dispatch(**overrides) -> bool:
    hub = SimpleNamespace(
        user_service=SimpleNamespace(get_settings=AsyncMock(return_value={"tone": "ghost"})),
        alerts_uc=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(symbol="BTC")),
        ),
    )
    defaults = {
        "message": _message(),
        "synthetic_text": "alert BTC 65000",
        "hub": hub,
        "handle_parsed_intent": AsyncMock(return_value=False),
        "llm_fallback_reply": AsyncMock(return_value=None),
        "send_llm_reply": AsyncMock(),
        "clarifying_question": lambda symbol: f"clarify:{symbol}",
        "extract_action_symbol_hint": lambda _text: "BTC",
        "smart_action_menu": lambda *args, **kwargs: {"kind": "smart"},
        "analysis_symbol_followup_kb": lambda: {"kind": "analysis"},
        "safe_exc": lambda exc: str(exc),
        "alert_created_menu": lambda symbol: {"symbol": symbol},
    }
    defaults.update(overrides)
    return await command_dispatcher.dispatch_command_text(**defaults)


@pytest.mark.asyncio
async def test_alert_shortcut_creates_alert() -> None:
    message = _message()

    handled = await _dispatch(message=message)

    assert handled is True
    message.answer.assert_awaited_once()
    kwargs = message.answer.await_args.kwargs
    assert kwargs["reply_markup"] == {"symbol": "BTC"}


@pytest.mark.asyncio
async def test_analysis_followup_uses_quick_keyboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        command_dispatcher,
        "parse_message",
        lambda _text: _parsed(intent=Intent.ANALYSIS, requires_followup=True, entities={}),
    )
    message = _message()

    handled = await _dispatch(
        message=message,
        synthetic_text="analyze",
    )

    assert handled is True
    message.answer.assert_awaited_once_with("Need one detail.", reply_markup={"kind": "analysis"})


@pytest.mark.asyncio
async def test_parsed_handler_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(command_dispatcher, "parse_message", lambda _text: _parsed(intent=Intent.NEWS))
    handle_parsed_intent = AsyncMock(return_value=True)
    llm_fallback_reply = AsyncMock(return_value="unused")

    handled = await _dispatch(
        synthetic_text="news btc",
        handle_parsed_intent=handle_parsed_intent,
        llm_fallback_reply=llm_fallback_reply,
    )

    assert handled is True
    handle_parsed_intent.assert_awaited_once()
    llm_fallback_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_fallback_sends_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(command_dispatcher, "parse_message", lambda _text: _parsed(intent=Intent.UNKNOWN))
    send_llm_reply = AsyncMock()

    handled = await _dispatch(
        synthetic_text="random question",
        llm_fallback_reply=AsyncMock(return_value="fallback reply"),
        send_llm_reply=send_llm_reply,
    )

    assert handled is True
    send_llm_reply.assert_awaited_once()
    kwargs = send_llm_reply.await_args.kwargs
    assert kwargs["analytics"]["route"] == "synthetic_fallback"
    assert kwargs["analytics"]["reply_kind"] == "general_chat"
