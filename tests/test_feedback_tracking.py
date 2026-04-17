from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import handlers
from app.bot import reply_engine


def test_normalize_feedback_reason_falls_back_to_other() -> None:
    allowed = {"long", "other"}
    assert reply_engine.normalize_feedback_reason("long", allowed) == "long"
    assert reply_engine.normalize_feedback_reason("unknown-reason", allowed) == "other"


@pytest.mark.asyncio
async def test_cache_feedback_reply_preview_stores_plain_preview() -> None:
    cache = SimpleNamespace(set_json=AsyncMock())
    await reply_engine.cache_feedback_reply_preview(
        cache,
        42,
        7,
        "<b>BTC</b> is <i>moving</i>",
        user_message="btc?",
    )

    cache.set_json.assert_awaited_once()
    args = cache.set_json.await_args.args
    kwargs = cache.set_json.await_args.kwargs
    assert args[0] == "feedback:reply:42:7"
    assert args[1]["reply_preview"] == "BTC is moving"
    assert args[1]["user_message"] == "btc?"
    assert kwargs["ttl"] == 60 * 60 * 24 * 7


@pytest.mark.asyncio
async def test_resolve_feedback_reply_preview_prefers_reply_specific_cache() -> None:
    cache = SimpleNamespace(
        get_json=AsyncMock(return_value={"reply_preview": "<b>ETH</b> looks strong"}),
    )
    preview = await reply_engine.resolve_feedback_reply_preview(cache, 99, 11)

    assert preview == "ETH looks strong"
    cache.get_json.assert_awaited_once_with("feedback:reply:99:11")


@pytest.mark.asyncio
async def test_log_feedback_event_records_metric_and_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    audit_service = SimpleNamespace(log=AsyncMock())
    metric = Mock()
    monkeypatch.setattr(
        reply_engine,
        "get_reply_analytics",
        AsyncMock(return_value={"route": "market_chat", "reply_kind": "market_chat", "chat_mode": "hybrid"}),
    )
    cache = SimpleNamespace()

    await reply_engine.log_feedback_event(
        cache=cache,
        audit_service=audit_service,
        chat_id=123,
        message_id=456,
        from_user_id=789,
        from_username="ghost_user",
        sentiment="negative",
        source="button",
        reason="long",
        reply_preview="<b>Too much text</b>",
        record_feedback_metric=metric,
        allowed_reasons={"long", "other"},
        improvement_text="shorten it",
    )

    metric.assert_called_once_with(sentiment="negative", source="button", reason="long")
    audit_service.log.assert_awaited_once()
    payload = audit_service.log.await_args.args[1]
    assert payload["chat_id"] == 123
    assert payload["reply_message_id"] == 456
    assert payload["reason"] == "long"
    assert payload["reply_preview"] == "Too much text"
    assert payload["improvement_text"] == "shorten it"
    assert payload["route"] == "market_chat"
    assert payload["reply_kind"] == "market_chat"


@pytest.mark.asyncio
async def test_log_bot_reply_event_records_audit_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    audit_service = SimpleNamespace(log=AsyncMock())
    cache_reply = AsyncMock()
    monkeypatch.setattr(reply_engine, "cache_reply_analytics", cache_reply)
    monkeypatch.setattr(reply_engine, "llm_model_metadata", lambda _client: {"provider": "anthropic", "primary_model": "anthropic/foo"})

    sent_message = SimpleNamespace(message_id=77, chat=SimpleNamespace(id=42, type="private"))
    await reply_engine.log_bot_reply_event(
        cache=SimpleNamespace(),
        audit_service=audit_service,
        llm_client=object(),
        chat_mode="hybrid",
        sent_message=sent_message,
        reply_text="btc still looks strong",
        user_message="btc?",
        analytics={"route": "market_chat", "reply_kind": "market_chat"},
    )

    cache_reply.assert_awaited_once()
    audit_service.log.assert_awaited_once()
    payload = audit_service.log.await_args.args[1]
    assert payload["chat_id"] == 42
    assert payload["reply_message_id"] == 77
    assert payload["route"] == "market_chat"
    assert payload["reply_kind"] == "market_chat"
    assert payload["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_notify_admins_negative_feedback_uses_hub_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = SimpleNamespace(send_message=AsyncMock())
    monkeypatch.setattr(handlers, "_hub", SimpleNamespace(bot=bot))
    monkeypatch.setattr(handlers, "_settings", SimpleNamespace(admin_ids_list=lambda: [555]))

    await handlers._notify_admins_negative_feedback(
        from_chat_id=101,
        from_username="anon",
        reason="wrong",
        reply_preview="bad read",
    )

    bot.send_message.assert_awaited_once()
    assert bot.send_message.await_args.args[0] == 555
