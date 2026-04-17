from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import giveaway_menu_flow


def _callback(data: str, *, user_id: int | None = 7) -> SimpleNamespace:
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        answer=AsyncMock(),
    )
    from_user = None if user_id is None else SimpleNamespace(id=user_id)
    return SimpleNamespace(
        data=data,
        message=message,
        from_user=from_user,
        answer=AsyncMock(),
        bot=SimpleNamespace(),
    )


def _hub(*, is_admin: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        giveaway_service=SimpleNamespace(
            is_admin=Mock(return_value=is_admin),
            join_active=AsyncMock(return_value={"giveaway_id": 11, "participants": 23}),
            status=AsyncMock(return_value={"status": "live"}),
            end_giveaway=AsyncMock(return_value={"giveaway_id": 11, "winner_user_id": 77, "prize": "50 USDT"}),
            reroll=AsyncMock(return_value={"giveaway_id": 11, "winner_user_id": 99, "previous_winner_user_id": 77}),
        ),
    )


async def _run_with_typing_lock(bot, chat_id: int, runner) -> None:
    await runner()


def _deps(**overrides) -> giveaway_menu_flow.GiveawayMenuFlowDependencies:
    defaults = {
        "hub": _hub(),
        "run_with_typing_lock": AsyncMock(side_effect=_run_with_typing_lock),
        "set_cmd_wizard": AsyncMock(),
        "as_int": lambda value, default: int(value) if value is not None else default,
        "giveaway_duration_menu": lambda: {"menu": "duration"},
        "giveaway_winners_menu": lambda duration: {"menu": duration},
        "giveaway_status_template": lambda payload: f"status:{payload['status']}",
    }
    defaults.update(overrides)
    return giveaway_menu_flow.GiveawayMenuFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_start_requires_admin() -> None:
    callback = _callback("gw:start")
    deps = _deps(hub=_hub(is_admin=False))

    await giveaway_menu_flow.handle_giveaway_menu_callback(callback=callback, deps=deps)

    callback.answer.assert_awaited_once_with("Admin only", show_alert=True)
    callback.message.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_duration_menu_uses_clamped_minimum_seconds() -> None:
    callback = _callback("gw:dur:5")
    winners_menu = Mock(return_value={"menu": "winners"})
    deps = _deps(giveaway_winners_menu=winners_menu)

    await giveaway_menu_flow.handle_giveaway_menu_callback(callback=callback, deps=deps)

    winners_menu.assert_called_once_with(30)
    callback.message.answer.assert_awaited_once_with("Pick number of winners.", reply_markup={"menu": "winners"})
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_win_sets_giveaway_prize_wizard() -> None:
    callback = _callback("gw:win:600:9")
    set_cmd_wizard = AsyncMock()
    deps = _deps(set_cmd_wizard=set_cmd_wizard)

    await giveaway_menu_flow.handle_giveaway_menu_callback(callback=callback, deps=deps)

    set_cmd_wizard.assert_awaited_once_with(
        42,
        {"step": "giveaway_prize", "duration_seconds": 600, "winners": 5},
    )
    callback.message.answer.assert_awaited_once_with("Send giveaway prize text, e.g. `50 USDT`.")
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_join_requires_user_identity() -> None:
    callback = _callback("gw:join", user_id=None)
    deps = _deps()

    await giveaway_menu_flow.handle_giveaway_menu_callback(callback=callback, deps=deps)

    callback.answer.assert_awaited_once_with("No user", show_alert=True)


@pytest.mark.asyncio
async def test_join_runs_through_typing_lock_and_sends_participant_count() -> None:
    callback = _callback("gw:join")
    run_with_typing_lock = AsyncMock(side_effect=_run_with_typing_lock)
    deps = _deps(run_with_typing_lock=run_with_typing_lock)

    await giveaway_menu_flow.handle_giveaway_menu_callback(callback=callback, deps=deps)

    run_with_typing_lock.assert_awaited_once()
    deps.hub.giveaway_service.join_active.assert_awaited_once_with(42, 7)
    text = callback.message.answer.await_args.args[0]
    assert "giveaway <b>#11</b>" in text
    assert "participants so far: <b>23</b>" in text
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_status_uses_template_inside_typing_lock() -> None:
    callback = _callback("gw:status")
    run_with_typing_lock = AsyncMock(side_effect=_run_with_typing_lock)
    deps = _deps(run_with_typing_lock=run_with_typing_lock)

    await giveaway_menu_flow.handle_giveaway_menu_callback(callback=callback, deps=deps)

    run_with_typing_lock.assert_awaited_once()
    deps.hub.giveaway_service.status.assert_awaited_once_with(42)
    callback.message.answer.assert_awaited_once_with("status:live")
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_end_reports_winner_payload() -> None:
    callback = _callback("gw:end")
    run_with_typing_lock = AsyncMock(side_effect=_run_with_typing_lock)
    deps = _deps(run_with_typing_lock=run_with_typing_lock)

    await giveaway_menu_flow.handle_giveaway_menu_callback(callback=callback, deps=deps)

    deps.hub.giveaway_service.end_giveaway.assert_awaited_once_with(42, 7)
    text = callback.message.answer.await_args.args[0]
    assert "giveaway <b>#11</b> closed." in text
    assert "winner: <code>77</code>" in text
    assert "prize: <b>50 USDT</b>" in text
    callback.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_reroll_reports_new_and_previous_winner() -> None:
    callback = _callback("gw:reroll")
    deps = _deps(run_with_typing_lock=AsyncMock(side_effect=_run_with_typing_lock))

    await giveaway_menu_flow.handle_giveaway_menu_callback(callback=callback, deps=deps)

    deps.hub.giveaway_service.reroll.assert_awaited_once_with(42, 7)
    text = callback.message.answer.await_args.args[0]
    assert "new winner: <code>99</code>" in text
    assert "prev winner: <code>77</code>" in text
    callback.answer.assert_awaited_once_with()
