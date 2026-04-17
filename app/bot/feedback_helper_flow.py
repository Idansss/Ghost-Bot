from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.bot import feedback_admin, reply_engine


@dataclass(frozen=True)
class FeedbackHelperDependencies:
    hub: Any
    admin_ids_list: Callable[[], list[int]]
    logger: Any
    record_feedback_metric: Callable[..., None]
    allowed_reasons: set[str]


async def get_pending_feedback_suggestion(*, chat_id: int, deps: FeedbackHelperDependencies) -> dict | None:
    return await feedback_admin.get_pending_feedback_suggestion(deps.hub.cache, chat_id)


async def set_pending_feedback_suggestion(
    *,
    chat_id: int,
    payload: dict,
    deps: FeedbackHelperDependencies,
    ttl: int = 300,
) -> None:
    await feedback_admin.set_pending_feedback_suggestion(deps.hub.cache, chat_id, payload, ttl=ttl)


async def clear_pending_feedback_suggestion(*, chat_id: int, deps: FeedbackHelperDependencies) -> None:
    await feedback_admin.clear_pending_feedback_suggestion(deps.hub.cache, chat_id)


async def resolve_feedback_reply_preview(
    *,
    chat_id: int,
    deps: FeedbackHelperDependencies,
    message_id: int | None = None,
) -> str:
    return await reply_engine.resolve_feedback_reply_preview(deps.hub.cache, chat_id, message_id)


async def log_feedback_event(
    *,
    chat_id: int,
    sentiment: str,
    source: str,
    reply_preview: str,
    deps: FeedbackHelperDependencies,
    message_id: int | None = None,
    from_user_id: int | None = None,
    from_username: str | None = None,
    reason: str | None = None,
    improvement_text: str | None = None,
) -> None:
    await reply_engine.log_feedback_event(
        cache=deps.hub.cache,
        audit_service=deps.hub.audit_service,
        chat_id=chat_id,
        message_id=message_id,
        from_user_id=from_user_id,
        from_username=from_username,
        sentiment=sentiment,
        source=source,
        reason=reason,
        reply_preview=reply_preview,
        record_feedback_metric=deps.record_feedback_metric,
        allowed_reasons=deps.allowed_reasons,
        improvement_text=improvement_text,
    )


async def notify_admins_negative_feedback(
    *,
    from_chat_id: int,
    reason: str,
    reply_preview: str,
    deps: FeedbackHelperDependencies,
    from_username: str | None = None,
    improvement_text: str | None = None,
) -> None:
    await feedback_admin.notify_admins_negative_feedback(
        bot=deps.hub.bot,
        admin_ids=deps.admin_ids_list(),
        logger=deps.logger,
        from_chat_id=from_chat_id,
        from_username=from_username,
        reason=reason,
        reply_preview=reply_preview,
        improvement_text=improvement_text,
    )


def build_feedback_callback_dependencies(
    *,
    helper_deps: FeedbackHelperDependencies,
    acquire_callback_once,
    feedback_reason_kb,
    get_user_settings,
    update_user_settings,
) -> feedback_admin.FeedbackCallbackDependencies:
    async def _resolve(chat_id: int, message_id: int | None = None) -> str:
        return await resolve_feedback_reply_preview(chat_id=chat_id, message_id=message_id, deps=helper_deps)

    async def _log(**kwargs) -> None:
        await log_feedback_event(deps=helper_deps, **kwargs)

    async def _set(chat_id: int, payload: dict) -> None:
        await set_pending_feedback_suggestion(chat_id=chat_id, payload=payload, deps=helper_deps)

    async def _notify(**kwargs) -> None:
        await notify_admins_negative_feedback(deps=helper_deps, **kwargs)

    return feedback_admin.FeedbackCallbackDependencies(
        acquire_callback_once=acquire_callback_once,
        resolve_feedback_reply_preview=_resolve,
        log_feedback_event=_log,
        set_pending_feedback_suggestion=_set,
        notify_admins_negative_feedback=_notify,
        feedback_reason_kb=feedback_reason_kb,
        get_user_settings=get_user_settings,
        update_user_settings=update_user_settings,
    )


def build_reaction_feedback_dependencies(
    *,
    helper_deps: FeedbackHelperDependencies,
) -> feedback_admin.ReactionFeedbackDependencies:
    async def _resolve(chat_id: int, message_id: int | None = None) -> str:
        return await resolve_feedback_reply_preview(chat_id=chat_id, message_id=message_id, deps=helper_deps)

    async def _log(**kwargs) -> None:
        await log_feedback_event(deps=helper_deps, **kwargs)

    async def _notify(**kwargs) -> None:
        await notify_admins_negative_feedback(deps=helper_deps, **kwargs)

    return feedback_admin.ReactionFeedbackDependencies(
        resolve_feedback_reply_preview=_resolve,
        log_feedback_event=_log,
        notify_admins_negative_feedback=_notify,
    )
