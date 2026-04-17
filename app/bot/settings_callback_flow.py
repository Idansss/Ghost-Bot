from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class SettingsCallbackDependencies:
    get_user_settings: Callable[[int], Awaitable[dict]]
    update_user_settings: Callable[[int, dict], Awaitable[dict]]
    settings_text: Callable[[dict], str]
    settings_menu: Callable[[dict], Any]


async def handle_settings_callback(*, callback, deps: SettingsCallbackDependencies) -> None:
    data = callback.data or ""
    chat_id = callback.message.chat.id

    if data == "settings:toggle:anon_mode":
        current = await deps.get_user_settings(chat_id)
        new_settings = await deps.update_user_settings(chat_id, {"anon_mode": not current.get("anon_mode", True)})
    elif data == "settings:toggle:formal_mode":
        current = await deps.get_user_settings(chat_id)
        new_settings = await deps.update_user_settings(chat_id, {"formal_mode": not current.get("formal_mode", False)})
    elif data == "settings:toggle:reply_in_dm":
        current = await deps.get_user_settings(chat_id)
        new_settings = await deps.update_user_settings(chat_id, {"reply_in_dm": not current.get("reply_in_dm", False)})
    elif data == "settings:toggle:ultra_brief":
        current = await deps.get_user_settings(chat_id)
        new_settings = await deps.update_user_settings(chat_id, {"ultra_brief": not current.get("ultra_brief", False)})
    elif data.startswith("settings:set:"):
        _, _, key, value = data.split(":", 3)
        new_settings = await deps.update_user_settings(chat_id, {key: value})
    else:
        await callback.answer("Unknown settings action", show_alert=True)
        return

    await callback.message.edit_text(
        deps.settings_text(new_settings),
        reply_markup=deps.settings_menu(new_settings),
    )
    await callback.answer("Updated")
