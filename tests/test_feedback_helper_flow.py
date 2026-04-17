from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import feedback_helper_flow


class _Cache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.get_json = AsyncMock(side_effect=self._get_json)
        self.set_json = AsyncMock(side_effect=self._set_json)
        self.delete = AsyncMock(side_effect=self._delete)

    async def _get_json(self, key: str):
        return self.store.get(key)

    async def _set_json(self, key: str, value, ttl: int | None = None) -> None:
        self.store[key] = value

    async def _delete(self, key: str) -> None:
        self.store.pop(key, None)


def _deps(**overrides) -> feedback_helper_flow.FeedbackHelperDependencies:
    defaults = {
        "hub": SimpleNamespace(
            cache=_Cache(),
            audit_service=SimpleNamespace(),
            bot=SimpleNamespace(send_message=AsyncMock()),
        ),
        "admin_ids_list": lambda: [101, 202],
        "logger": SimpleNamespace(warning=Mock()),
        "record_feedback_metric": Mock(),
        "allowed_reasons": {"long", "other", "reaction"},
    }
    defaults.update(overrides)
    return feedback_helper_flow.FeedbackHelperDependencies(**defaults)


@pytest.mark.asyncio
async def test_pending_feedback_suggestion_round_trip() -> None:
    deps = _deps()

    await feedback_helper_flow.set_pending_feedback_suggestion(chat_id=42, payload={"reason": "long"}, deps=deps)
    payload = await feedback_helper_flow.get_pending_feedback_suggestion(chat_id=42, deps=deps)
    await feedback_helper_flow.clear_pending_feedback_suggestion(chat_id=42, deps=deps)

    assert payload == {"reason": "long"}
    assert await feedback_helper_flow.get_pending_feedback_suggestion(chat_id=42, deps=deps) is None


@pytest.mark.asyncio
async def test_log_feedback_event_delegates_to_reply_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = _deps()
    delegate = AsyncMock()
    monkeypatch.setattr(feedback_helper_flow.reply_engine, "log_feedback_event", delegate)

    await feedback_helper_flow.log_feedback_event(
        chat_id=42,
        message_id=9,
        from_user_id=7,
        from_username="ghost",
        sentiment="negative",
        source="button",
        reason="long",
        reply_preview="preview",
        improvement_text="shorter",
        deps=deps,
    )

    delegate.assert_awaited_once()
    kwargs = delegate.await_args.kwargs
    assert kwargs["cache"] is deps.hub.cache
    assert kwargs["audit_service"] is deps.hub.audit_service
    assert kwargs["record_feedback_metric"] is deps.record_feedback_metric
    assert kwargs["allowed_reasons"] == deps.allowed_reasons


@pytest.mark.asyncio
async def test_notify_admins_negative_feedback_uses_hub_bot() -> None:
    bot = SimpleNamespace(send_message=AsyncMock())
    deps = _deps(hub=SimpleNamespace(cache=_Cache(), audit_service=SimpleNamespace(), bot=bot))

    await feedback_helper_flow.notify_admins_negative_feedback(
        from_chat_id=77,
        from_username="anon",
        reason="wrong",
        reply_preview="bad read",
        deps=deps,
    )

    assert bot.send_message.await_count == 2
    assert bot.send_message.await_args.args[0] == 202


@pytest.mark.asyncio
async def test_build_feedback_callback_dependencies_uses_helper_backed_storage() -> None:
    deps = _deps()
    callback_deps = feedback_helper_flow.build_feedback_callback_dependencies(
        helper_deps=deps,
        acquire_callback_once=AsyncMock(return_value=True),
        feedback_reason_kb=lambda: {"kind": "feedback"},
        get_user_settings=AsyncMock(return_value={}),
        update_user_settings=AsyncMock(),
    )

    await callback_deps.set_pending_feedback_suggestion(42, {"reason": "suggestion"})

    assert deps.hub.cache.store["feedback:pending_suggestion:42"] == {"reason": "suggestion"}
