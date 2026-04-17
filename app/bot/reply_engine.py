from __future__ import annotations

import re
from contextlib import suppress
from datetime import UTC, datetime
from typing import Callable

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardMarkup, Message


def reply_analytics_cache_key(chat_id: int, message_id: int) -> str:
    return f"reply:analytics:{chat_id}:{message_id}"


def feedback_reply_cache_key(chat_id: int, message_id: int) -> str:
    return f"feedback:reply:{chat_id}:{message_id}"


def normalize_feedback_reason(reason: str | None, allowed_reasons: set[str] | None = None) -> str:
    normalized = (reason or "").strip().lower()
    if allowed_reasons is None:
        return normalized or "other"
    return normalized if normalized in allowed_reasons else "other"


def reply_preview_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()[:500]


async def cache_feedback_reply_preview(
    cache,
    chat_id: int,
    message_id: int,
    reply_text: str,
    *,
    user_message: str | None = None,
    ttl: int = 60 * 60 * 24 * 7,
) -> None:
    with suppress(Exception):
        await cache.set_json(
            feedback_reply_cache_key(chat_id, message_id),
            {
                "reply_preview": reply_preview_text(reply_text),
                "user_message": (user_message or "").strip()[:500],
                "created_at": datetime.now(UTC).isoformat(),
            },
            ttl=ttl,
        )


async def cache_reply_analytics(cache, chat_id: int, message_id: int, payload: dict, ttl: int = 60 * 60 * 24 * 7) -> None:
    with suppress(Exception):
        await cache.set_json(reply_analytics_cache_key(chat_id, message_id), payload, ttl=ttl)


async def get_reply_analytics(cache, chat_id: int, message_id: int | None = None) -> dict | None:
    if message_id is None:
        return None
    payload = await cache.get_json(reply_analytics_cache_key(chat_id, message_id))
    return payload if isinstance(payload, dict) else None


async def resolve_feedback_reply_preview(cache, chat_id: int, message_id: int | None = None) -> str:
    if message_id is not None:
        payload = await cache.get_json(feedback_reply_cache_key(chat_id, message_id))
        if isinstance(payload, dict):
            preview = reply_preview_text(str(payload.get("reply_preview") or ""))
            if preview:
                return preview
    last = await cache.get_json(f"llm:last_reply:{chat_id}")
    if isinstance(last, str):
        return reply_preview_text(last)
    return reply_preview_text(str(last or ""))


def llm_model_metadata(llm_client) -> dict:
    if not llm_client:
        return {}
    primary_model = str(getattr(llm_client, "model", "") or "").strip()
    fallback_model = str(getattr(llm_client, "fallback_model", "") or "").strip()
    provider = primary_model.split("/", 1)[0] if "/" in primary_model else ("openai" if primary_model else "unknown")
    return {
        "provider": provider,
        "primary_model": primary_model or None,
        "fallback_model": fallback_model or None,
    }


async def log_bot_reply_event(
    *,
    cache,
    audit_service,
    llm_client,
    chat_mode: str,
    sent_message: Message,
    reply_text: str,
    user_message: str | None = None,
    analytics: dict | None = None,
) -> dict:
    payload = {
        "chat_id": sent_message.chat.id,
        "reply_message_id": sent_message.message_id,
        "chat_type": str(getattr(sent_message.chat, "type", "") or ""),
        "reply_preview": reply_preview_text(reply_text),
        "reply_chars": len(reply_text or ""),
        "reply_words": len((reply_text or "").split()),
        "user_message_preview": (user_message or "").strip()[:300] or None,
        "chat_mode": chat_mode,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload.update(llm_model_metadata(llm_client))
    for key, value in dict(analytics or {}).items():
        if key == "request_started_at":
            continue
        payload[key] = value
    started_at = (analytics or {}).get("request_started_at")
    if isinstance(started_at, datetime):
        payload["request_latency_ms"] = max(int((datetime.now(UTC) - started_at).total_seconds() * 1000), 0)

    await cache_reply_analytics(cache, sent_message.chat.id, sent_message.message_id, payload)
    with suppress(Exception):
        await audit_service.log("bot_reply", payload, success=True)
    return payload


async def log_feedback_event(
    *,
    cache,
    audit_service,
    chat_id: int,
    message_id: int | None,
    from_user_id: int | None,
    from_username: str | None,
    sentiment: str,
    source: str,
    reason: str | None,
    reply_preview: str,
    record_feedback_metric: Callable[..., None],
    allowed_reasons: set[str] | None = None,
    improvement_text: str | None = None,
) -> dict:
    reason_label = normalize_feedback_reason(reason, allowed_reasons)
    record_feedback_metric(sentiment=sentiment, source=source, reason=reason_label)
    reply_analytics = await get_reply_analytics(cache, chat_id, message_id)

    payload = {
        "chat_id": chat_id,
        "reply_message_id": message_id,
        "from_user_id": from_user_id,
        "from_username": from_username or "",
        "sentiment": sentiment,
        "source": source,
        "reason": reason_label,
        "reply_preview": reply_preview_text(reply_preview),
        "improvement_text": (improvement_text or "").strip()[:1000] or None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if reply_analytics:
        for key in ("route", "reply_kind", "provider", "primary_model", "fallback_model", "chat_mode"):
            value = reply_analytics.get(key)
            if value:
                payload[key] = value
    with suppress(Exception):
        await audit_service.log("user_feedback", payload, success=True)
    return payload


async def send_llm_reply(
    *,
    message: Message,
    reply: str,
    hub,
    sanitize_html: Callable[[str], str],
    llm_reply_keyboard_factory: Callable[[], InlineKeyboardMarkup],
    confirm_understanding_kb_factory: Callable[[], InlineKeyboardMarkup],
    chat_mode: str,
    settings: dict | None = None,
    user_message: str | None = None,
    add_quick_replies: bool = True,
    analytics: dict | None = None,
) -> None:
    cleaned = sanitize_html(reply)
    reply_to = message.message_id if message else None
    chat_id = message.chat.id if message else None
    if add_quick_replies and chat_id and user_message is not None:
        try:
            await hub.cache.set_json(f"llm:last_reply:{chat_id}", cleaned, ttl=3600)
            await hub.cache.set_json(f"llm:last_user:{chat_id}", user_message, ttl=3600)
        except Exception:
            pass
    use_confirm = add_quick_replies and cleaned.strip().lower().startswith("you want:")
    reply_markup = confirm_understanding_kb_factory() if use_confirm else (llm_reply_keyboard_factory() if add_quick_replies else None)

    in_group = getattr(message.chat, "type", None) in ("group", "supergroup")
    reply_in_dm = (settings or {}).get("reply_in_dm")
    if in_group and not reply_in_dm and getattr(message.from_user, "id", None):
        try:
            user_settings = await hub.user_service.get_settings(message.from_user.id)
            reply_in_dm = user_settings.get("reply_in_dm")
        except Exception:
            pass

    async def _record_sent(sent_message: Message, sent_text: str) -> None:
        await cache_feedback_reply_preview(
            hub.cache,
            sent_message.chat.id,
            sent_message.message_id,
            sent_text,
            user_message=user_message,
        )
        await log_bot_reply_event(
            cache=hub.cache,
            audit_service=hub.audit_service,
            llm_client=getattr(hub, "llm_client", None),
            chat_mode=chat_mode,
            sent_message=sent_message,
            reply_text=sent_text,
            user_message=user_message,
            analytics=analytics,
        )

    if in_group and reply_in_dm and getattr(message.from_user, "id", None):
        sent_message: Message | None = None
        try:
            sent_message = await message.bot.send_message(
                message.from_user.id, cleaned, reply_markup=reply_markup
            )
            if sent_message is not None:
                await _record_sent(sent_message, cleaned)
            await message.answer("Sent you a DM.", reply_to_message_id=reply_to)
        except TelegramBadRequest:
            plain = re.sub(r"<[^>]+>", "", reply)
            try:
                sent_message = await message.bot.send_message(
                    message.from_user.id, plain, reply_markup=reply_markup
                )
                if sent_message is not None:
                    await _record_sent(sent_message, plain)
                await message.answer("Sent you a DM.", reply_to_message_id=reply_to)
            except Exception:
                try:
                    sent_message = await message.answer(
                        cleaned, reply_to_message_id=reply_to, reply_markup=reply_markup
                    )
                    if sent_message is not None:
                        await _record_sent(sent_message, cleaned)
                except TelegramBadRequest:
                    sent_message = await message.answer(plain, reply_to_message_id=reply_to)
                    if sent_message is not None:
                        await _record_sent(sent_message, plain)
        return

    sent_message: Message | None = None
    try:
        sent_message = await message.answer(
            cleaned, reply_to_message_id=reply_to, reply_markup=reply_markup
        )
    except TelegramBadRequest:
        plain = re.sub(r"<[^>]+>", "", reply)
        with suppress(Exception):
            sent_message = await message.answer(
                plain, reply_to_message_id=reply_to, reply_markup=reply_markup
            )
            if sent_message is not None:
                await _record_sent(sent_message, plain)
            return

    if sent_message is not None:
        await _record_sent(sent_message, cleaned)
