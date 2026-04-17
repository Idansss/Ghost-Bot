from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


_NEGATIVE_REACTION_EMOJIS = {
    "\U0001F44E",
    "\U0001F44E\U0001F3FB",
    "\U0001F44E\U0001F3FC",
    "\U0001F44E\U0001F3FD",
    "\U0001F44E\U0001F3FE",
    "\U0001F44E\U0001F3FF",
    "\U0001F61E",
    "\U0001F92E",
}


@dataclass(frozen=True)
class FeedbackCallbackDependencies:
    acquire_callback_once: Callable[[Any], Awaitable[bool]]
    resolve_feedback_reply_preview: Callable[[int, int | None], Awaitable[str]]
    log_feedback_event: Callable[..., Awaitable[None]]
    set_pending_feedback_suggestion: Callable[[int, dict], Awaitable[None]]
    notify_admins_negative_feedback: Callable[..., Awaitable[None]]
    feedback_reason_kb: Callable[[], Any]
    get_user_settings: Callable[[int], Awaitable[dict]]
    update_user_settings: Callable[[int, dict], Awaitable[None]]


@dataclass(frozen=True)
class ReactionFeedbackDependencies:
    resolve_feedback_reply_preview: Callable[[int, int | None], Awaitable[str]]
    log_feedback_event: Callable[..., Awaitable[None]]
    notify_admins_negative_feedback: Callable[..., Awaitable[None]]


def _pending_feedback_suggestion_key(chat_id: int) -> str:
    return f"feedback:pending_suggestion:{chat_id}"


async def get_pending_feedback_suggestion(cache, chat_id: int) -> dict | None:
    payload = await cache.get_json(_pending_feedback_suggestion_key(chat_id))
    return payload if isinstance(payload, dict) else None


async def set_pending_feedback_suggestion(cache, chat_id: int, payload: dict, ttl: int = 300) -> None:
    await cache.set_json(_pending_feedback_suggestion_key(chat_id), payload, ttl=ttl)


async def clear_pending_feedback_suggestion(cache, chat_id: int) -> None:
    with suppress(Exception):
        await cache.delete(_pending_feedback_suggestion_key(chat_id))


async def notify_admins_negative_feedback(
    *,
    bot,
    admin_ids: list[int],
    logger,
    from_chat_id: int,
    from_username: str | None,
    reason: str,
    reply_preview: str,
    improvement_text: str | None = None,
) -> None:
    if not admin_ids:
        logger.warning(
            "feedback_no_admin_ids",
            extra={"event": "feedback_no_admin_ids", "from_chat_id": from_chat_id},
        )
        return
    username = from_username or "\u2014"
    preview = (reply_preview or "")[:400].replace("<", " ").replace(">", " ")
    lines = [
        "\U0001F44E <b>Negative feedback</b>",
        f"From: {username} (chat_id <code>{from_chat_id}</code>)",
        f"Reason: {reason}",
        "",
        f"Bot reply (preview): {preview}",
    ]
    if improvement_text and improvement_text.strip():
        lines.extend(["", "\U0001F4DD <b>Improvement suggestion:</b>", improvement_text.strip()[:1000]])
    text = "\n".join(lines)
    for admin_id in admin_ids:
        try:
            await bot.send_message(admin_id, text, parse_mode="HTML")
        except Exception as exc:
            logger.warning(
                "feedback_dm_failed",
                extra={
                    "event": "feedback_dm_failed",
                    "admin_id": admin_id,
                    "error": str(exc),
                    "hint": "Ensure ADMIN_CHAT_IDS is your Telegram user ID and you have started the bot in DMs.",
                },
            )


async def handle_feedback_callback(
    *,
    callback,
    deps: FeedbackCallbackDependencies,
) -> None:
    if not await deps.acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id
    from_username = getattr(callback.from_user, "username", None) or getattr(callback.from_user, "first_name", None) or ""
    data = (callback.data or "").strip()

    async def _reply_preview() -> str:
        try:
            return await deps.resolve_feedback_reply_preview(chat_id, getattr(callback.message, "message_id", None))
        except Exception:
            if callback.message.text:
                return (callback.message.text or "")[:500]
            return ""

    if data == "feedback:up":
        await deps.log_feedback_event(
            chat_id=chat_id,
            message_id=getattr(callback.message, "message_id", None),
            from_user_id=getattr(callback.from_user, "id", None),
            from_username=from_username,
            sentiment="positive",
            source="button",
            reason="thumbs_up",
            reply_preview=await _reply_preview(),
        )
        await callback.answer("Thanks!")
        return

    if data == "feedback:down":
        await callback.message.edit_reply_markup(reply_markup=deps.feedback_reason_kb())
        await callback.answer("What was wrong?")
        return

    if data == "feedback:suggest":
        with suppress(Exception):
            await callback.message.edit_reply_markup(reply_markup=None)
        await deps.set_pending_feedback_suggestion(
            chat_id,
            {"reason": "suggestion", "reply_preview": await _reply_preview(), "message_id": callback.message.message_id},
        )
        await callback.answer()
        await callback.message.answer("Type what we can do to improve in the chat \u2014 I'll pass it on personally.")
        return

    if data.startswith("feedback:reason:"):
        reason = data.replace("feedback:reason:", "", 1).lower()
        reply_preview = await _reply_preview()
        try:
            settings = await deps.get_user_settings(chat_id)
            prefs = dict(settings.get("feedback_prefs") or {})
            if reason == "long":
                prefs["prefers_shorter"] = True
                await deps.update_user_settings(chat_id, {"feedback_prefs": prefs})
        except Exception:
            pass
        with suppress(Exception):
            await callback.message.edit_reply_markup(reply_markup=None)
        await deps.log_feedback_event(
            chat_id=chat_id,
            message_id=getattr(callback.message, "message_id", None),
            from_user_id=getattr(callback.from_user, "id", None),
            from_username=from_username,
            sentiment="negative",
            source="button",
            reason=reason,
            reply_preview=reply_preview,
        )
        await deps.notify_admins_negative_feedback(
            from_chat_id=chat_id,
            from_username=from_username,
            reason=reason,
            reply_preview=reply_preview,
        )
        await deps.set_pending_feedback_suggestion(
            chat_id,
            {"reason": reason, "reply_preview": reply_preview, "message_id": callback.message.message_id},
        )
        msg = "Thanks \u2014 we'll keep it shorter next time." if reason == "long" else "Thanks for the feedback."
        await callback.answer(msg, show_alert=True)
        await callback.message.answer("Optional: type how we can improve and I'll pass it on personally.")
        return

    await callback.answer()


def is_negative_reaction(reaction_list: list) -> bool:
    if not reaction_list:
        return False
    for item in reaction_list:
        emoji = getattr(item, "emoji", None) or getattr(item, "type", None)
        if emoji and str(emoji).strip() in _NEGATIVE_REACTION_EMOJIS:
            return True
        if hasattr(item, "emoji") and item.emoji and "thumbs" in str(item.emoji).lower():
            return True
    return False


async def handle_message_reaction(
    *,
    reaction_update,
    deps: ReactionFeedbackDependencies,
) -> None:
    if not is_negative_reaction(reaction_update.new_reaction or []):
        return
    chat_id = reaction_update.chat.id
    user = reaction_update.user
    from_username = (getattr(user, "username", None) or getattr(user, "first_name", None) or "\u2014") if user else "\u2014"
    reply_preview = await deps.resolve_feedback_reply_preview(chat_id, getattr(reaction_update, "message_id", None))
    await deps.log_feedback_event(
        chat_id=chat_id,
        message_id=getattr(reaction_update, "message_id", None),
        from_user_id=getattr(user, "id", None) if user else None,
        from_username=from_username,
        sentiment="negative",
        source="reaction",
        reason="reaction",
        reply_preview=reply_preview or "(no preview)",
    )
    await deps.notify_admins_negative_feedback(
        from_chat_id=chat_id,
        from_username=from_username,
        reason="reaction (message reaction)",
        reply_preview=reply_preview or "(no preview)",
    )
