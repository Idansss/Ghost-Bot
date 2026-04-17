from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import feedback_admin


class _Cache:
    def __init__(self, initial: dict | None = None) -> None:
        self.store = dict(initial or {})
        self.get_json = AsyncMock(side_effect=self._get_json)
        self.set_json = AsyncMock(side_effect=self._set_json)
        self.delete = AsyncMock(side_effect=self._delete)

    async def _get_json(self, key: str):
        return self.store.get(key)

    async def _set_json(self, key: str, value, ttl: int | None = None) -> None:
        self.store[key] = value

    async def _delete(self, key: str) -> None:
        self.store.pop(key, None)


def _callback(data: str = "feedback:up", message_text: str = "reply text") -> SimpleNamespace:
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        message_id=9,
        text=message_text,
        edit_reply_markup=AsyncMock(),
        answer=AsyncMock(),
    )
    return SimpleNamespace(
        data=data,
        message=message,
        from_user=SimpleNamespace(id=7, username="ghost"),
        answer=AsyncMock(),
    )


def _feedback_deps(**overrides) -> feedback_admin.FeedbackCallbackDependencies:
    defaults = {
        "acquire_callback_once": AsyncMock(return_value=True),
        "resolve_feedback_reply_preview": AsyncMock(return_value="preview text"),
        "log_feedback_event": AsyncMock(),
        "set_pending_feedback_suggestion": AsyncMock(),
        "notify_admins_negative_feedback": AsyncMock(),
        "feedback_reason_kb": lambda: {"kind": "feedback"},
        "get_user_settings": AsyncMock(return_value={"feedback_prefs": {}}),
        "update_user_settings": AsyncMock(),
    }
    defaults.update(overrides)
    return feedback_admin.FeedbackCallbackDependencies(**defaults)


def _reaction_deps(**overrides) -> feedback_admin.ReactionFeedbackDependencies:
    defaults = {
        "resolve_feedback_reply_preview": AsyncMock(return_value="preview text"),
        "log_feedback_event": AsyncMock(),
        "notify_admins_negative_feedback": AsyncMock(),
    }
    defaults.update(overrides)
    return feedback_admin.ReactionFeedbackDependencies(**defaults)


@pytest.mark.asyncio
async def test_pending_feedback_suggestion_round_trip() -> None:
    cache = _Cache()

    await feedback_admin.set_pending_feedback_suggestion(cache, 42, {"reason": "long"})
    payload = await feedback_admin.get_pending_feedback_suggestion(cache, 42)
    await feedback_admin.clear_pending_feedback_suggestion(cache, 42)

    assert payload == {"reason": "long"}
    assert await feedback_admin.get_pending_feedback_suggestion(cache, 42) is None


@pytest.mark.asyncio
async def test_notify_admins_negative_feedback_sends_html_message() -> None:
    bot = SimpleNamespace(send_message=AsyncMock())
    logger = SimpleNamespace(warning=Mock())

    await feedback_admin.notify_admins_negative_feedback(
        bot=bot,
        admin_ids=[101, 202],
        logger=logger,
        from_chat_id=77,
        from_username="anon",
        reason="wrong",
        reply_preview="bad read",
        improvement_text="be more precise",
    )

    assert bot.send_message.await_count == 2
    assert bot.send_message.await_args.args[0] == 202
    text = bot.send_message.await_args.args[1]
    assert "Negative feedback" in text
    assert "be more precise" in text


@pytest.mark.asyncio
async def test_handle_feedback_callback_logs_positive_feedback() -> None:
    callback = _callback("feedback:up")
    deps = _feedback_deps()

    await feedback_admin.handle_feedback_callback(callback=callback, deps=deps)

    deps.log_feedback_event.assert_awaited_once()
    kwargs = deps.log_feedback_event.await_args.kwargs
    assert kwargs["sentiment"] == "positive"
    assert kwargs["reason"] == "thumbs_up"
    callback.answer.assert_awaited_once_with("Thanks!")


@pytest.mark.asyncio
async def test_handle_feedback_callback_reason_long_updates_prefs_and_stores_suggestion() -> None:
    callback = _callback("feedback:reason:long")
    update_user_settings = AsyncMock()
    set_pending_feedback_suggestion = AsyncMock()
    notify_admins_negative_feedback = AsyncMock()
    deps = _feedback_deps(
        update_user_settings=update_user_settings,
        set_pending_feedback_suggestion=set_pending_feedback_suggestion,
        notify_admins_negative_feedback=notify_admins_negative_feedback,
    )

    await feedback_admin.handle_feedback_callback(callback=callback, deps=deps)

    update_user_settings.assert_awaited_once_with(42, {"feedback_prefs": {"prefers_shorter": True}})
    deps.log_feedback_event.assert_awaited_once()
    notify_admins_negative_feedback.assert_awaited_once_with(
        from_chat_id=42,
        from_username="ghost",
        reason="long",
        reply_preview="preview text",
    )
    set_pending_feedback_suggestion.assert_awaited_once_with(
        42,
        {"reason": "long", "reply_preview": "preview text", "message_id": 9},
    )
    callback.answer.assert_awaited_once_with("Thanks \u2014 we'll keep it shorter next time.", show_alert=True)
    callback.message.answer.assert_awaited_once_with("Optional: type how we can improve and I'll pass it on personally.")


@pytest.mark.asyncio
async def test_handle_message_reaction_logs_and_notifies_on_negative_reaction() -> None:
    reaction_update = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        message_id=9,
        user=SimpleNamespace(id=7, username="ghost"),
        new_reaction=[SimpleNamespace(emoji="\U0001F44E")],
    )
    deps = _reaction_deps()

    await feedback_admin.handle_message_reaction(reaction_update=reaction_update, deps=deps)

    deps.log_feedback_event.assert_awaited_once()
    kwargs = deps.log_feedback_event.await_args.kwargs
    assert kwargs["source"] == "reaction"
    assert kwargs["reason"] == "reaction"
    deps.notify_admins_negative_feedback.assert_awaited_once_with(
        from_chat_id=42,
        from_username="ghost",
        reason="reaction (message reaction)",
        reply_preview="preview text",
    )
