from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import admin_command_flow


def _message(text: str, *, user_id: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=42),
        from_user=SimpleNamespace(id=user_id),
        answer=AsyncMock(),
    )


def _hub() -> SimpleNamespace:
    return SimpleNamespace(
        audit_service=SimpleNamespace(
            feedback_summary=AsyncMock(return_value={"kind": "feedback"}),
            reply_analytics_summary=AsyncMock(return_value={"kind": "replystats"}),
            quality_summary=AsyncMock(return_value={"kind": "quality"}),
        ),
        cache=SimpleNamespace(
            set_if_absent=AsyncMock(),
            delete=AsyncMock(),
        ),
    )


def _deps(**overrides) -> admin_command_flow.AdminCommandFlowDependencies:
    defaults = {
        "hub": _hub(),
        "is_bot_admin": lambda message: True,
        "admin_ids_list": lambda: [99, 7],
        "format_feedback_summary": lambda summary: f"feedback:{summary['kind']}",
        "format_reply_stats_summary": lambda summary: f"reply:{summary['kind']}",
        "format_quality_summary": lambda summary: f"quality:{summary['kind']}",
        "abuse_block_ttl_sec": 1800,
        "abuse_block_key": lambda subject_id: f"block:{subject_id}",
        "abuse_strike_key": lambda subject_id: f"strike:{subject_id}",
        "record_abuse": Mock(),
    }
    defaults.update(overrides)
    return admin_command_flow.AdminCommandFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_admins_command_lists_sorted_unique_admins() -> None:
    message = _message("/admins")
    deps = _deps(admin_ids_list=lambda: [9, 3, 9])

    await admin_command_flow.handle_admins_command(message=message, deps=deps)

    message.answer.assert_awaited_once()
    text = message.answer.await_args.args[0]
    assert "<b>bot admins</b>" in text
    assert "<code>3</code>" in text
    assert "<code>9</code>" in text


@pytest.mark.asyncio
async def test_feedback_command_rejects_non_admins() -> None:
    message = _message("/feedback 48")
    deps = _deps(is_bot_admin=lambda message: False)

    await admin_command_flow.handle_feedback_command(message=message, deps=deps)

    message.answer.assert_awaited_once_with("Unauthorized.")
    deps.hub.audit_service.feedback_summary.assert_not_called()


@pytest.mark.asyncio
async def test_replystats_command_uses_hours_argument_and_formatter() -> None:
    message = _message("/replystats 72")
    deps = _deps()

    await admin_command_flow.handle_replystats_command(message=message, deps=deps)

    deps.hub.audit_service.reply_analytics_summary.assert_awaited_once_with(hours=72)
    message.answer.assert_awaited_once_with("reply:replystats")


@pytest.mark.asyncio
async def test_block_command_sets_ttl_and_records_abuse() -> None:
    message = _message("/block 555 12")
    record_abuse = Mock()
    deps = _deps(record_abuse=record_abuse)

    await admin_command_flow.handle_block_command(message=message, deps=deps)

    deps.hub.cache.set_if_absent.assert_awaited_once_with("block:555", ttl=720)
    record_abuse.assert_called_once_with("admin_block")
    message.answer.assert_awaited_once_with("Blocked <code>555</code> for ~12 min.")


@pytest.mark.asyncio
async def test_unblock_command_deletes_block_and_strike_keys() -> None:
    message = _message("/unblock 555")
    record_abuse = Mock()
    deps = _deps(record_abuse=record_abuse)

    await admin_command_flow.handle_unblock_command(message=message, deps=deps)

    deps.hub.cache.delete.assert_awaited_once_with("block:555", "strike:555")
    record_abuse.assert_called_once_with("admin_unblock")
    message.answer.assert_awaited_once_with("Unblocked <code>555</code>.")
