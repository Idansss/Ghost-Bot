from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AdminCommandFlowDependencies:
    hub: Any
    is_bot_admin: Callable[[Any], bool]
    admin_ids_list: Callable[[], list[int]]
    format_feedback_summary: Callable[[dict], str]
    format_reply_stats_summary: Callable[[dict], str]
    format_quality_summary: Callable[[dict], str]
    abuse_block_ttl_sec: int
    abuse_block_key: Callable[[int], str]
    abuse_strike_key: Callable[[int], str]
    record_abuse: Callable[[str], None]


async def handle_admins_command(*, message, deps: AdminCommandFlowDependencies) -> None:
    admin_ids = sorted(set(deps.admin_ids_list()))
    if not admin_ids:
        await message.answer("no admin IDs configured.")
        return
    lines = ["<b>bot admins</b>\n"]
    lines.extend(f"- <code>{admin_id}</code>" for admin_id in admin_ids)
    await message.answer("\n".join(lines))


async def handle_feedback_command(*, message, deps: AdminCommandFlowDependencies) -> None:
    if not deps.is_bot_admin(message):
        await message.answer("Unauthorized.")
        return
    parts = (message.text or "").strip().split()
    hours = 24
    if len(parts) >= 2:
        with suppress(Exception):
            hours = int(parts[1])
    summary = await deps.hub.audit_service.feedback_summary(hours=hours)
    await message.answer(deps.format_feedback_summary(summary))


async def handle_replystats_command(*, message, deps: AdminCommandFlowDependencies) -> None:
    if not deps.is_bot_admin(message):
        await message.answer("Unauthorized.")
        return
    parts = (message.text or "").strip().split()
    hours = 24
    if len(parts) >= 2:
        with suppress(Exception):
            hours = int(parts[1])
    summary = await deps.hub.audit_service.reply_analytics_summary(hours=hours)
    await message.answer(deps.format_reply_stats_summary(summary))


async def handle_quality_command(*, message, deps: AdminCommandFlowDependencies) -> None:
    if not deps.is_bot_admin(message):
        await message.answer("Unauthorized.")
        return
    parts = (message.text or "").strip().split()
    hours = 24
    if len(parts) >= 2:
        with suppress(Exception):
            hours = int(parts[1])
    summary = await deps.hub.audit_service.quality_summary(hours=hours)
    await message.answer(deps.format_quality_summary(summary))


async def handle_block_command(*, message, deps: AdminCommandFlowDependencies) -> None:
    if not deps.is_bot_admin(message):
        await message.answer("Unauthorized.")
        return
    raw = (message.text or "").strip()
    parts = raw.split()
    if len(parts) < 2:
        await message.answer("Usage: /block <user_id> [minutes]")
        return
    try:
        subject_id = int(parts[1])
    except ValueError:
        await message.answer("Invalid user_id.")
        return
    ttl_min: int | None = None
    if len(parts) >= 3:
        with suppress(Exception):
            ttl_min = int(parts[2])
    ttl_sec = int(deps.abuse_block_ttl_sec if ttl_min is None else max(ttl_min, 1) * 60)
    await deps.hub.cache.set_if_absent(deps.abuse_block_key(subject_id), ttl=ttl_sec)
    deps.record_abuse("admin_block")
    await message.answer(f"Blocked <code>{subject_id}</code> for ~{max(ttl_sec // 60, 1)} min.")


async def handle_unblock_command(*, message, deps: AdminCommandFlowDependencies) -> None:
    if not deps.is_bot_admin(message):
        await message.answer("Unauthorized.")
        return
    raw = (message.text or "").strip()
    parts = raw.split()
    if len(parts) < 2:
        await message.answer("Usage: /unblock <user_id>")
        return
    try:
        subject_id = int(parts[1])
    except ValueError:
        await message.answer("Invalid user_id.")
        return
    await deps.hub.cache.delete(deps.abuse_block_key(subject_id), deps.abuse_strike_key(subject_id))
    deps.record_abuse("admin_unblock")
    await message.answer(f"Unblocked <code>{subject_id}</code>.")
