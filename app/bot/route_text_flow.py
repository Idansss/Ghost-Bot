from __future__ import annotations

import asyncio
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Sequence


@dataclass(frozen=True)
class RouteTextFlowDependencies:
    hub: Any
    logger: Any
    record_abuse: Callable[[str], None]
    is_blocked_subject: Callable[[int], Awaitable[bool]]
    blocked_notice_ttl: Callable[[int], Awaitable[int]]
    is_group_admin: Callable[[Any], Awaitable[bool]]
    set_group_free_talk: Callable[[int, bool], Awaitable[None]]
    group_free_talk_enabled: Callable[[int], Awaitable[bool]]
    mentions_bot: Callable[[str, str | None], bool]
    strip_bot_mention: Callable[[str, str | None], str]
    is_reply_to_bot: Callable[[Any], bool]
    looks_like_clear_intent: Callable[[str], bool]
    is_source_query: Callable[[str], bool]
    source_reply_for_chat: Callable[[int, str], Awaitable[str]]
    acquire_message_once: Callable[[Any], Awaitable[bool]]
    check_request_limit: Callable[[int], Awaitable[bool]]
    record_strike_and_maybe_block: Callable[[int], Awaitable[bool]]
    greeting_re: Any
    choose_reply: Callable[[Sequence[str]], str]
    gn_replies: Sequence[str]
    market_aware_gm_reply: Callable[[str], Awaitable[str]]
    chat_lock: Callable[[int], Any]
    typing_loop: Callable[[Any, int, asyncio.Event], Awaitable[None]]
    handle_pre_route_state: Callable[[Any, str, int], Awaitable[bool]]
    handle_free_text_flow: Callable[[Any, str, int, datetime], Awaitable[bool]]
    safe_exc: Callable[[Exception], str]
    now_utc: Callable[[], datetime]
    blocked_message: Callable[[int], str]
    blocked_rate_limit_message: Callable[[int], str]
    rate_limit_notice: str
    plain_text_prompt: str
    busy_notice: str


_GROUP_FREE_TALK_RE = re.compile(r"\bfree\s*talk\s*mode\s*(on|off)\b", flags=re.IGNORECASE)
_ID_QUERY_RE = re.compile(
    r"\b(my|show)\s+(user\s*)?id\b|\bwhat('?s| is)\s+my\s+id\b",
    flags=re.IGNORECASE,
)


async def handle_route_text(message, *, deps: RouteTextFlowDependencies) -> None:
    hub = deps.hub
    text = message.text or ""
    chat_id = message.chat.id
    raw_text = text.strip()
    subject_id = int(message.from_user.id) if message.from_user else int(chat_id)

    if raw_text.startswith("/"):
        return

    if await deps.is_blocked_subject(subject_id):
        ttl = await deps.blocked_notice_ttl(subject_id)
        deps.record_abuse("blocked_message")
        await message.answer(deps.blocked_message(ttl))
        return

    if message.chat.type in ("group", "supergroup"):
        ft_match = _GROUP_FREE_TALK_RE.search(raw_text)
        if ft_match:
            if not await deps.is_group_admin(message):
                await message.answer("Only group admins can toggle free talk mode.")
                return
            enabled = ft_match.group(1).lower() == "on"
            await deps.set_group_free_talk(message.chat.id, enabled)
            await message.answer(f"Group free talk mode {'ON' if enabled else 'OFF'}.")
            return

    if message.chat.type in ("group", "supergroup"):
        free_talk_enabled = await deps.group_free_talk_enabled(message.chat.id)
        mentioned = deps.mentions_bot(text, hub.bot_username)
        reply_to_bot = deps.is_reply_to_bot(message)
        clear_intent = deps.looks_like_clear_intent(text)
        if not (free_talk_enabled or mentioned or reply_to_bot or clear_intent):
            return
        text = deps.strip_bot_mention(text, hub.bot_username)
        if not text:
            await message.answer(deps.plain_text_prompt)
            return

    if _ID_QUERY_RE.search(text):
        if not message.from_user:
            await message.answer("Could not read your user id from this update.")
            return
        await message.answer(
            f"Your user id: {message.from_user.id}\n"
            f"Current chat id: {message.chat.id}"
        )
        return

    if deps.is_source_query(text):
        await message.answer(await deps.source_reply_for_chat(chat_id, text))
        return

    if not await deps.acquire_message_once(message):
        deps.logger.info(
            "duplicate_message_ignored",
            extra={
                "event": "duplicate_message_ignored",
                "chat_id": chat_id,
                "message_id": message.message_id,
            },
        )
        return

    if not await deps.check_request_limit(chat_id):
        if await deps.record_strike_and_maybe_block(subject_id):
            ttl = await deps.blocked_notice_ttl(subject_id)
            await message.answer(
                deps.blocked_rate_limit_message(ttl),
                reply_to_message_id=message.message_id,
            )
            return
        await message.answer(deps.rate_limit_notice, reply_to_message_id=message.message_id)
        return

    if deps.greeting_re.match(raw_text):
        low = raw_text.lower()
        rid = message.message_id
        if low.startswith("gn") or "night" in low:
            await message.answer(deps.choose_reply(deps.gn_replies), reply_to_message_id=rid)
        else:
            name = message.from_user.first_name if message.from_user else "fren"
            reply = await deps.market_aware_gm_reply(name)
            await message.answer(reply, reply_to_message_id=rid)
        return

    lock = deps.chat_lock(chat_id)
    if lock.locked():
        if await hub.cache.set_if_absent(f"busy_notice:{chat_id}", ttl=5):
            await message.answer(deps.busy_notice, reply_to_message_id=message.message_id)
        return

    start_ts = deps.now_utc()
    stop = asyncio.Event()
    typing_task = asyncio.create_task(deps.typing_loop(message.bot, chat_id, stop))
    try:
        async with lock:
            if await deps.handle_pre_route_state(message, text, chat_id):
                return
            if await deps.handle_free_text_flow(message, text, chat_id, start_ts):
                return
    except Exception as exc:
        deps.logger.exception(
            "route_text_unhandled_error",
            extra={"event": "route_text_unhandled_error", "chat_id": chat_id},
        )
        with suppress(Exception):
            await message.answer(
                f"something broke on my end. try again in a sec. (<i>{deps.safe_exc(exc)}</i>)",
                reply_to_message_id=message.message_id,
            )
    finally:
        stop.set()
        typing_task.cancel()
        with suppress(Exception):
            await typing_task
        latency_ms = int((deps.now_utc() - start_ts).total_seconds() * 1000)
        deps.logger.info(
            "message_processed",
            extra={"event": "message_processed", "chat_id": chat_id, "latency_ms": latency_ms},
        )
