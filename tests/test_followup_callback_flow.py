from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import followup_callback_flow


def _callback(data: str = "followup:simplify") -> SimpleNamespace:
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        edit_text=AsyncMock(),
        edit_reply_markup=AsyncMock(),
        answer=AsyncMock(),
    )
    return SimpleNamespace(
        data=data,
        message=message,
        answer=AsyncMock(),
    )


def _hub(last_reply="Original reply", last_user="Original question", llm_reply="Refined") -> SimpleNamespace:
    cache_values = {
        "llm:last_reply:42": last_reply,
        "llm:last_user:42": last_user,
    }
    cache = SimpleNamespace(get_json=AsyncMock(side_effect=lambda key: cache_values.get(key)))
    llm_client = SimpleNamespace(reply=AsyncMock(return_value=llm_reply))
    return SimpleNamespace(cache=cache, llm_client=llm_client)


@pytest.mark.asyncio
async def test_followup_callback_refines_last_reply() -> None:
    callback = _callback("followup:short")
    deps = followup_callback_flow.FollowupCallbackDependencies(
        hub=_hub(),
        sanitize_html=lambda text: f"clean:{text}",
        llm_reply_keyboard=lambda: {"kind": "llm"},
    )

    await followup_callback_flow.handle_followup_callback(callback=callback, deps=deps)

    deps.hub.llm_client.reply.assert_awaited_once()
    prompt = deps.hub.llm_client.reply.await_args.args[0]
    assert "Original question" in prompt
    assert "Original reply" in prompt
    callback.message.edit_text.assert_awaited_once_with("clean:Refined", reply_markup={"kind": "llm"})
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_followup_callback_without_last_reply_shows_alert() -> None:
    callback = _callback()
    deps = followup_callback_flow.FollowupCallbackDependencies(
        hub=_hub(last_reply=None),
        sanitize_html=lambda text: text,
        llm_reply_keyboard=lambda: None,
    )

    await followup_callback_flow.handle_followup_callback(callback=callback, deps=deps)

    callback.answer.assert_awaited_once_with("No previous reply to refine.", show_alert=True)
    callback.message.edit_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_understood_yes_clears_markup() -> None:
    callback = _callback("confirm:understood:yes")

    await followup_callback_flow.handle_confirm_understood_callback(callback=callback)

    callback.message.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
    callback.answer.assert_awaited_once_with("Got it.")


@pytest.mark.asyncio
async def test_confirm_understood_no_falls_back_to_new_message() -> None:
    callback = _callback("confirm:understood:no")
    callback.message.edit_text = AsyncMock(side_effect=RuntimeError("no edit"))

    await followup_callback_flow.handle_confirm_understood_callback(callback=callback)

    callback.message.answer.assert_awaited_once_with("Rephrase what you want - I'll match it.")
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_confirm_clear_alerts_clears_service_and_edits_message() -> None:
    callback = _callback("confirm:clear_alerts:3")
    deps = followup_callback_flow.ConfirmClearAlertsDependencies(clear_user_alerts=AsyncMock(return_value=5))

    await followup_callback_flow.handle_confirm_clear_alerts_callback(callback=callback, deps=deps)

    deps.clear_user_alerts.assert_awaited_once_with(42)
    callback.message.edit_text.assert_awaited_once_with("Cleared 5 alerts.", reply_markup=None)
    callback.answer.assert_awaited_once_with()
