from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import giveaway_command_flow


def _message(text: str, *, user_id: int | None = 7) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=42),
        from_user=None if user_id is None else SimpleNamespace(id=user_id),
        answer=AsyncMock(),
    )


def _deps(**overrides) -> giveaway_command_flow.GiveawayCommandFlowDependencies:
    defaults = {
        "hub": SimpleNamespace(
            giveaway_service=SimpleNamespace(
                join_active=AsyncMock(return_value={"giveaway_id": 11, "participants": 23}),
                status=AsyncMock(return_value={"state": "live"}),
                end_giveaway=AsyncMock(return_value={"giveaway_id": 11, "winner_user_id": 77, "prize": "50 USDT"}),
                reroll=AsyncMock(return_value={"giveaway_id": 11, "winner_user_id": 99, "previous_winner_user_id": 77}),
                start_giveaway=AsyncMock(return_value={"id": 11, "prize": "50 USDT", "end_time": "2026-04-16T12:00:00Z"}),
                is_admin=Mock(return_value=True),
            )
        ),
        "safe_exc": lambda exc: f"safe:{exc}",
        "parse_duration_to_seconds": lambda value: 600 if value == "10m" else None,
        "giveaway_status_template": lambda payload: f"status:{payload['state']}",
        "giveaway_menu": lambda **kwargs: {"is_admin": kwargs["is_admin"]},
    }
    defaults.update(overrides)
    return giveaway_command_flow.GiveawayCommandFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_join_requires_identified_user() -> None:
    message = _message("/join", user_id=None)
    deps = _deps()

    await giveaway_command_flow.handle_join_command(message=message, deps=deps)

    message.answer.assert_awaited_once_with("Could not identify user for join.")


@pytest.mark.asyncio
async def test_join_error_uses_safe_exception_text() -> None:
    message = _message("/join")
    service = SimpleNamespace(
        join_active=AsyncMock(side_effect=RuntimeError("boom")),
        status=AsyncMock(),
        end_giveaway=AsyncMock(),
        reroll=AsyncMock(),
        start_giveaway=AsyncMock(),
        is_admin=Mock(return_value=True),
    )
    deps = _deps(hub=SimpleNamespace(giveaway_service=service))

    await giveaway_command_flow.handle_join_command(message=message, deps=deps)

    message.answer.assert_awaited_once_with("couldn't join: safe:boom")


@pytest.mark.asyncio
async def test_giveaway_status_uses_template() -> None:
    message = _message("/giveaway status")
    deps = _deps()

    await giveaway_command_flow.handle_giveaway_command(message=message, deps=deps)

    deps.hub.giveaway_service.status.assert_awaited_once_with(42)
    message.answer.assert_awaited_once_with("status:live")


@pytest.mark.asyncio
async def test_giveaway_start_includes_multiwinner_note() -> None:
    message = _message('/giveaway start 10m prize "50 USDT" winners=3')
    deps = _deps()

    await giveaway_command_flow.handle_giveaway_command(message=message, deps=deps)

    deps.hub.giveaway_service.start_giveaway.assert_awaited_once_with(
        group_chat_id=42,
        admin_chat_id=7,
        duration_seconds=600,
        prize="50 USDT",
    )
    reply = message.answer.await_args.args[0]
    assert "Giveaway #11 started." in reply
    assert "multi-winner draw" in reply


@pytest.mark.asyncio
async def test_giveaway_end_reports_winner_and_prize() -> None:
    message = _message("/giveaway end")
    deps = _deps()

    await giveaway_command_flow.handle_giveaway_command(message=message, deps=deps)

    deps.hub.giveaway_service.end_giveaway.assert_awaited_once_with(42, 7)
    reply = message.answer.await_args.args[0]
    assert "giveaway <b>#11</b> closed." in reply
    assert "winner: <code>77</code>" in reply
    assert "prize: <b>50 USDT</b>" in reply
