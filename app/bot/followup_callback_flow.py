from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FollowupCallbackDependencies:
    hub: Any
    sanitize_html: Callable[[str], str]
    llm_reply_keyboard: Callable[[], Any]


@dataclass(frozen=True)
class ConfirmClearAlertsDependencies:
    clear_user_alerts: Callable[[int], Any]


async def handle_followup_callback(*, callback, deps: FollowupCallbackDependencies) -> None:
    chat_id = callback.message.chat.id
    data = (callback.data or "").strip()
    action = data.replace("followup:", "", 1).lower()
    last_reply = await deps.hub.cache.get_json(f"llm:last_reply:{chat_id}")
    last_user = await deps.hub.cache.get_json(f"llm:last_user:{chat_id}")
    if not last_reply or not isinstance(last_reply, str):
        await callback.answer("No previous reply to refine.", show_alert=True)
        return

    instructions = {
        "simplify": "Rewrite this in simpler, shorter form. Output only the simplified version.",
        "example": "Add one concrete example to illustrate this. Keep the original and add the example.",
        "short": "Give a one- or two-sentence version only.",
        "deeper": "Expand with one more paragraph of detail or context. Keep the original summary first.",
    }
    instruction = instructions.get(action, instructions["simplify"])
    prompt = (
        f'User originally asked: "{last_user or ""}". '
        f'Your previous reply: "{last_reply[:600]}". {instruction}'
    )
    try:
        reply = await deps.hub.llm_client.reply(prompt, history=[])
        if reply and reply.strip():
            await callback.message.edit_text(
                deps.sanitize_html(reply.strip())[:4000],
                reply_markup=deps.llm_reply_keyboard(),
            )
    except Exception:
        pass
    await callback.answer()


async def handle_confirm_understood_callback(*, callback) -> None:
    data = (callback.data or "").strip()
    if data.endswith(":yes"):
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        await callback.answer("Got it.")
        return

    if data.endswith(":no"):
        try:
            await callback.message.edit_text(
                "Rephrase what you want - I'll match it.",
                reply_markup=None,
            )
        except Exception:
            await callback.message.answer("Rephrase what you want - I'll match it.")
        await callback.answer()
        return

    await callback.answer()


async def handle_confirm_clear_alerts_callback(*, callback, deps: ConfirmClearAlertsDependencies) -> None:
    chat_id = callback.message.chat.id
    data = (callback.data or "").strip()
    suffix = data.replace("confirm:clear_alerts:", "", 1)
    if suffix == "no":
        try:
            await callback.message.edit_text("Cancelled. Alerts unchanged.", reply_markup=None)
        except Exception:
            await callback.message.answer("Cancelled. Alerts unchanged.")
        await callback.answer()
        return

    count = await deps.clear_user_alerts(chat_id)
    try:
        await callback.message.edit_text(f"Cleared {count} alerts.", reply_markup=None)
    except Exception:
        await callback.message.answer(f"Cleared {count} alerts.")
    await callback.answer()
