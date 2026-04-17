from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import settings_callback_flow


def _callback(data: str) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        message=SimpleNamespace(
            chat=SimpleNamespace(id=42),
            edit_text=AsyncMock(),
        ),
        answer=AsyncMock(),
    )


def _deps(**overrides) -> settings_callback_flow.SettingsCallbackDependencies:
    defaults = {
        "get_user_settings": AsyncMock(return_value={"anon_mode": True, "formal_mode": False, "reply_in_dm": False, "ultra_brief": False}),
        "update_user_settings": AsyncMock(return_value={"anon_mode": False}),
        "settings_text": lambda settings: f"text:{settings}",
        "settings_menu": lambda settings: {"menu": settings},
    }
    defaults.update(overrides)
    return settings_callback_flow.SettingsCallbackDependencies(**defaults)


@pytest.mark.asyncio
async def test_toggle_anon_mode_updates_and_rerenders() -> None:
    callback = _callback("settings:toggle:anon_mode")
    deps = _deps(
        update_user_settings=AsyncMock(return_value={"anon_mode": False}),
    )

    await settings_callback_flow.handle_settings_callback(callback=callback, deps=deps)

    deps.get_user_settings.assert_awaited_once_with(42)
    deps.update_user_settings.assert_awaited_once_with(42, {"anon_mode": False})
    callback.message.edit_text.assert_awaited_once_with("text:{'anon_mode': False}", reply_markup={"menu": {"anon_mode": False}})
    callback.answer.assert_awaited_once_with("Updated")


@pytest.mark.asyncio
async def test_set_value_updates_setting_directly() -> None:
    callback = _callback("settings:set:tone:ghost")
    deps = _deps(update_user_settings=AsyncMock(return_value={"tone": "ghost"}))

    await settings_callback_flow.handle_settings_callback(callback=callback, deps=deps)

    deps.update_user_settings.assert_awaited_once_with(42, {"tone": "ghost"})
    callback.answer.assert_awaited_once_with("Updated")


@pytest.mark.asyncio
async def test_unknown_action_shows_alert() -> None:
    callback = _callback("settings:toggle:unknown")
    deps = _deps()

    await settings_callback_flow.handle_settings_callback(callback=callback, deps=deps)

    callback.answer.assert_awaited_once_with("Unknown settings action", show_alert=True)
    callback.message.edit_text.assert_not_awaited()
