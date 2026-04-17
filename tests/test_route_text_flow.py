from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import route_text_flow


class _Lock:
    def __init__(self, locked: bool = False) -> None:
        self._locked = locked

    def locked(self) -> bool:
        return self._locked

    async def __aenter__(self):
        self._locked = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self._locked = False
        return False


def _message(text: str, chat_type: str = "private") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        answer=AsyncMock(),
        message_id=9,
        chat=SimpleNamespace(id=42, type=chat_type),
        from_user=SimpleNamespace(id=7, first_name="fren"),
        reply_to_message=None,
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
    )


def _deps(**overrides) -> route_text_flow.RouteTextFlowDependencies:
    defaults = {
        "hub": SimpleNamespace(cache=SimpleNamespace(set_if_absent=AsyncMock(return_value=True)), bot_username="ghostbot"),
        "logger": Mock(),
        "record_abuse": Mock(),
        "is_blocked_subject": AsyncMock(return_value=False),
        "blocked_notice_ttl": AsyncMock(return_value=600),
        "is_group_admin": AsyncMock(return_value=True),
        "set_group_free_talk": AsyncMock(),
        "group_free_talk_enabled": AsyncMock(return_value=False),
        "mentions_bot": Mock(return_value=False),
        "strip_bot_mention": Mock(side_effect=lambda text, _username: text),
        "is_reply_to_bot": Mock(return_value=False),
        "looks_like_clear_intent": Mock(return_value=False),
        "is_source_query": Mock(return_value=False),
        "source_reply_for_chat": AsyncMock(return_value="Source: Bybit BTCUSDT"),
        "acquire_message_once": AsyncMock(return_value=True),
        "check_request_limit": AsyncMock(return_value=True),
        "record_strike_and_maybe_block": AsyncMock(return_value=False),
        "greeting_re": __import__("re").compile(r"^gm$", __import__("re").IGNORECASE),
        "choose_reply": lambda seq: seq[0],
        "gn_replies": ["gn"],
        "market_aware_gm_reply": AsyncMock(return_value="gm dynamic"),
        "chat_lock": lambda _chat_id: _Lock(False),
        "typing_loop": AsyncMock(side_effect=lambda _bot, _chat_id, stop: stop.wait()),
        "handle_pre_route_state": AsyncMock(return_value=False),
        "handle_free_text_flow": AsyncMock(return_value=False),
        "safe_exc": lambda exc: str(exc),
        "now_utc": lambda: datetime(2026, 4, 16, tzinfo=UTC),
        "blocked_message": lambda ttl: f"blocked {ttl}",
        "blocked_rate_limit_message": lambda ttl: f"rate blocked {ttl}",
        "rate_limit_notice": "slow down",
        "plain_text_prompt": "plain text prompt",
        "busy_notice": "busy",
    }
    defaults.update(overrides)
    return route_text_flow.RouteTextFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_route_text_short_circuits_source_queries() -> None:
    message = _message("source?")
    deps = _deps(is_source_query=Mock(return_value=True))

    await route_text_flow.handle_route_text(message, deps=deps)

    deps.source_reply_for_chat.assert_awaited_once_with(42, "source?")
    message.answer.assert_awaited_once_with("Source: Bybit BTCUSDT")
    deps.acquire_message_once.assert_not_called()


@pytest.mark.asyncio
async def test_route_text_handles_group_free_talk_toggle() -> None:
    message = _message("free talk mode on", chat_type="group")
    deps = _deps()

    await route_text_flow.handle_route_text(message, deps=deps)

    deps.is_group_admin.assert_awaited_once_with(message)
    deps.set_group_free_talk.assert_awaited_once_with(42, True)
    message.answer.assert_awaited_once_with("Group free talk mode ON.")


@pytest.mark.asyncio
async def test_route_text_handles_greeting_fast_path() -> None:
    message = _message("gm")
    deps = _deps()

    await route_text_flow.handle_route_text(message, deps=deps)

    deps.market_aware_gm_reply.assert_awaited_once_with("fren")
    message.answer.assert_awaited_once_with("gm dynamic", reply_to_message_id=9)


@pytest.mark.asyncio
async def test_route_text_delegates_through_typing_lock_path() -> None:
    message = _message("btc long")
    pre_route = AsyncMock(return_value=False)
    free_text = AsyncMock(return_value=True)
    deps = _deps(
        mentions_bot=Mock(return_value=True),
        handle_pre_route_state=pre_route,
        handle_free_text_flow=free_text,
        chat_lock=lambda _chat_id: _Lock(False),
    )
    message.chat.type = "group"
    deps.group_free_talk_enabled = AsyncMock(return_value=False)
    deps.strip_bot_mention = Mock(return_value="btc long")

    await route_text_flow.handle_route_text(message, deps=deps)

    pre_route.assert_awaited_once_with(message, "btc long", 42)
    free_text.assert_awaited_once()
