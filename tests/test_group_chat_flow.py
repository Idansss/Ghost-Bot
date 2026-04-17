from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import group_chat_flow


def _deps(**overrides) -> group_chat_flow.GroupChatFlowDependencies:
    defaults = {
        "cache": SimpleNamespace(get_json=AsyncMock(return_value=None), set_json=AsyncMock()),
        "bot": SimpleNamespace(get_chat_member=AsyncMock(return_value=SimpleNamespace(status="member"))),
        "admin_ids_list": lambda: [100],
        "parse_message": lambda text: SimpleNamespace(intent=text),
        "clear_intent_excluded": {"unknown", "smalltalk"},
    }
    defaults.update(overrides)
    return group_chat_flow.GroupChatFlowDependencies(**defaults)


def test_mentions_and_strip_bot_mention() -> None:
    assert group_chat_flow.mentions_bot("hey @ghostbot btc", "ghostbot") is True
    assert group_chat_flow.strip_bot_mention("hey @ghostbot btc", "ghostbot") == "hey  btc"


def test_is_reply_to_bot_checks_bot_id() -> None:
    message = SimpleNamespace(reply_to_message=SimpleNamespace(from_user=SimpleNamespace(id=55)))

    assert group_chat_flow.is_reply_to_bot(message, 55) is True
    assert group_chat_flow.is_reply_to_bot(message, 99) is False


@pytest.mark.asyncio
async def test_group_free_talk_persistence_and_admin_check() -> None:
    cache = SimpleNamespace(
        get_json=AsyncMock(return_value={"enabled": True}),
        set_json=AsyncMock(),
    )
    bot = SimpleNamespace(get_chat_member=AsyncMock(return_value=SimpleNamespace(status="administrator")))
    deps = _deps(cache=cache, bot=bot)
    message = SimpleNamespace(chat=SimpleNamespace(id=42), from_user=SimpleNamespace(id=7))

    enabled = await group_chat_flow.group_free_talk_enabled(42, deps=deps)
    await group_chat_flow.set_group_free_talk(42, False, deps=deps)
    is_admin = await group_chat_flow.is_group_admin(message, deps=deps)

    assert enabled is True
    cache.set_json.assert_awaited_once()
    assert is_admin is True


def test_looks_like_clear_intent_uses_parser_exclusions() -> None:
    deps = _deps(parse_message=lambda text: SimpleNamespace(intent=text))

    assert group_chat_flow.looks_like_clear_intent("analysis", deps=deps) is True
    assert group_chat_flow.looks_like_clear_intent("unknown", deps=deps) is False
