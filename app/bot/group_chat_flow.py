from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GroupChatFlowDependencies:
    cache: Any
    bot: Any
    admin_ids_list: Callable[[], list[int]]
    parse_message: Callable[[str], Any]
    clear_intent_excluded: set[Any]


def mentions_bot(text: str, bot_username: str | None) -> bool:
    if not bot_username:
        return False
    return f"@{bot_username.lower()}" in text.lower()


def strip_bot_mention(text: str, bot_username: str | None) -> str:
    if not bot_username:
        return text
    return re.sub(rf"@{re.escape(bot_username)}", "", text, flags=re.IGNORECASE).strip()


def is_reply_to_bot(message, bot_id: int) -> bool:
    reply = message.reply_to_message
    if not reply or not reply.from_user:
        return False
    return bool(reply.from_user.id == bot_id)


async def group_free_talk_enabled(chat_id: int, *, deps: GroupChatFlowDependencies) -> bool:
    payload = await deps.cache.get_json(f"group:free_talk:{chat_id}")
    return bool(payload and payload.get("enabled"))


async def set_group_free_talk(chat_id: int, enabled: bool, *, deps: GroupChatFlowDependencies) -> None:
    await deps.cache.set_json(
        f"group:free_talk:{chat_id}",
        {"enabled": bool(enabled)},
        ttl=60 * 60 * 24 * 365,
    )


def looks_like_clear_intent(text: str, *, deps: GroupChatFlowDependencies) -> bool:
    parsed = deps.parse_message(text)
    return parsed.intent not in deps.clear_intent_excluded


async def is_group_admin(message, *, deps: GroupChatFlowDependencies) -> bool:
    if not message.from_user:
        return False
    if int(message.from_user.id) in set(deps.admin_ids_list()):
        return True
    try:
        member = await deps.bot.get_chat_member(message.chat.id, message.from_user.id)
        return getattr(member, "status", "") in {"administrator", "creator"}
    except Exception:
        return False
