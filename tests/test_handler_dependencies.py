from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.bot import (
    feedback_admin,
    greeting_flow,
    handler_dependencies,
    route_text_flow,
    source_query_flow,
    transport_runtime,
)


def _helpers() -> SimpleNamespace:
    return SimpleNamespace(
        acquire_callback_once=AsyncMock(return_value=True),
        feedback_reason_kb=lambda: {"kb": "feedback"},
        record_feedback_metric=Mock(),
        group_is_admin=AsyncMock(return_value=False),
        set_group_free_talk=AsyncMock(),
        group_free_talk_enabled=AsyncMock(return_value=False),
        mentions_bot=Mock(return_value=False),
        strip_bot_mention=Mock(side_effect=lambda text, _: text),
        is_reply_to_bot=Mock(return_value=False),
        looks_like_clear_intent=Mock(return_value=False),
        record_abuse=Mock(),
        handle_pre_route_state=AsyncMock(return_value=False),
        handle_free_text_flow=AsyncMock(return_value=False),
        safe_exc=lambda exc: str(exc),
        now_utc=lambda: datetime.now(UTC),
        blocked_message=lambda ttl: f"blocked:{ttl}",
        blocked_rate_limit_message=lambda ttl: f"rate:{ttl}",
        rate_limit_notice="slow down",
        plain_text_prompt="plain text only",
        busy_notice="busy",
        greeting_re=object(),
        choose_reply=lambda seq: seq[0],
        gn_replies=("gn",),
    )


def _factory() -> handler_dependencies.HandlerDependencyFactory:
    hub = SimpleNamespace(
        cache=SimpleNamespace(),
        rate_limiter=SimpleNamespace(),
        bot=SimpleNamespace(id=99),
        bot_username="ghost_bot",
        user_service=SimpleNamespace(
            get_settings=AsyncMock(return_value={"feedback_prefs": {}}),
            update_settings=AsyncMock(),
        ),
    )
    return handler_dependencies.HandlerDependencyFactory(
        hub=hub,
        settings_obj=SimpleNamespace(
            admin_ids_list=[1, 2],
            abuse_block_ttl_sec=300,
            feedback_reason_labels={"long", "wrong"},
        ),
        logger=Mock(),
        feedback_reason_labels={"long", "wrong"},
        transport_runtime_deps=transport_runtime.TransportRuntimeDependencies(
            cache=SimpleNamespace(),
            rate_limiter=SimpleNamespace(),
            logger=Mock(),
            request_rate_limit_per_minute=5,
            abuse_strike_window_sec=60,
            abuse_strikes_to_block=3,
            abuse_block_ttl_sec=300,
            record_abuse=Mock(),
            chat_lock=lambda chat_id: object(),
        ),
        source_query_deps=source_query_flow.SourceQueryFlowDependencies(cache=SimpleNamespace()),
        group_chat_deps=SimpleNamespace(),
        greeting_deps=greeting_flow.GreetingFlowDependencies(
            analysis_service=SimpleNamespace(),
            choose_reply=lambda seq: seq[0],
            gm_replies=("gm",),
            weekend_warning_text="warn",
            now_utc=lambda: datetime.now(UTC),
        ),
        helpers=_helpers(),
    )


def test_feedback_callback_dependencies_are_built_from_factory() -> None:
    deps = _factory().feedback_callback()

    assert isinstance(deps, feedback_admin.FeedbackCallbackDependencies)
    assert deps.feedback_reason_kb() == {"kb": "feedback"}


def test_route_text_flow_dependencies_use_factory_helpers() -> None:
    deps = _factory().route_text_flow()

    assert isinstance(deps, route_text_flow.RouteTextFlowDependencies)
    assert deps.rate_limit_notice == "slow down"
    assert deps.blocked_message(12) == "blocked:12"
