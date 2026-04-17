from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from aiogram.enums import ChatAction


@dataclass(frozen=True)
class TransportRuntimeDependencies:
    cache: Any
    rate_limiter: Any
    logger: Any
    request_rate_limit_per_minute: int
    abuse_strike_window_sec: int
    abuse_strikes_to_block: int
    abuse_block_ttl_sec: int
    record_abuse: Callable[[str], None]
    chat_lock: Callable[[int], asyncio.Lock]


def abuse_block_key(subject_id: int) -> str:
    return f"abuse:block:{subject_id}"


def abuse_strike_key(subject_id: int) -> str:
    return f"abuse:strikes:{subject_id}"


async def acquire_message_once(message, *, deps: TransportRuntimeDependencies, ttl: int = 60 * 60 * 6) -> bool:
    key = f"seen:message:{message.chat.id}:{message.message_id}"
    try:
        return await deps.cache.set_if_absent(key, ttl=ttl)
    except Exception:
        deps.logger.exception(
            "dedupe_cache_error",
            extra={"event": "dedupe_cache_error", "chat_id": message.chat.id},
        )
        return True


async def acquire_callback_once(callback, *, deps: TransportRuntimeDependencies, ttl: int = 60 * 30) -> bool:
    cb_id = (callback.id or "").strip()
    if not cb_id:
        return True
    key = f"seen:callback:{cb_id}"
    return await deps.cache.set_if_absent(key, ttl=ttl)


async def typing_loop(bot, chat_id: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        with suppress(Exception):
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except TimeoutError:
            pass


async def is_blocked_subject(subject_id: int, *, deps: TransportRuntimeDependencies) -> bool:
    try:
        raw = await deps.cache.redis.get(abuse_block_key(subject_id))
        return bool(raw)
    except Exception:
        deps.logger.exception(
            "abuse_block_check_error",
            extra={"event": "abuse_block_check_error", "subject_id": subject_id},
        )
        return False


async def record_strike_and_maybe_block(subject_id: int, *, deps: TransportRuntimeDependencies) -> bool:
    if await is_blocked_subject(subject_id, deps=deps):
        return True
    try:
        strikes = await deps.cache.incr_with_expiry(
            abuse_strike_key(subject_id),
            ttl=int(deps.abuse_strike_window_sec),
        )
        if strikes >= int(deps.abuse_strikes_to_block):
            await deps.cache.set_if_absent(
                abuse_block_key(subject_id),
                ttl=int(deps.abuse_block_ttl_sec),
            )
            deps.record_abuse("auto_block")
            deps.logger.warning(
                "abuse_auto_blocked",
                extra={
                    "event": "abuse_auto_blocked",
                    "subject_id": subject_id,
                    "strikes": strikes,
                    "ttl_sec": int(deps.abuse_block_ttl_sec),
                },
            )
            return True
    except Exception:
        deps.logger.exception(
            "abuse_strike_error",
            extra={"event": "abuse_strike_error", "subject_id": subject_id},
        )
    return False


async def blocked_notice_ttl(subject_id: int, *, deps: TransportRuntimeDependencies) -> int:
    try:
        return await deps.cache.get_ttl(abuse_block_key(subject_id))
    except Exception:
        return -2


async def check_request_limit(chat_id: int, *, deps: TransportRuntimeDependencies) -> bool:
    try:
        result = await deps.rate_limiter.check(
            key=f"rl:req:{chat_id}:{datetime.now(UTC).strftime('%Y%m%d%H%M')}",
            limit=deps.request_rate_limit_per_minute,
            window_seconds=60,
        )
        return result.allowed
    except Exception:
        deps.logger.exception(
            "rate_limit_check_error",
            extra={"event": "rate_limit_check_error", "chat_id": chat_id},
        )
        return True


async def run_with_typing_lock(bot, chat_id: int, runner, *, deps: TransportRuntimeDependencies) -> None:
    stop = asyncio.Event()
    typing_task = asyncio.create_task(typing_loop(bot, chat_id, stop))
    lock = deps.chat_lock(chat_id)
    try:
        if int(chat_id) > 0 and await is_blocked_subject(int(chat_id), deps=deps):
            deps.record_abuse("blocked_callback")
            return
        async with lock:
            await runner()
    finally:
        stop.set()
        typing_task.cancel()
        with suppress(Exception):
            await typing_task
