from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class GiveawayMenuFlowDependencies:
    hub: Any
    run_with_typing_lock: Callable[[Any, int, Callable[[], Awaitable[None]]], Awaitable[None]]
    set_cmd_wizard: Callable[[int, dict], Awaitable[None]]
    as_int: Callable[[Any, int], int]
    giveaway_duration_menu: Callable[[], Any]
    giveaway_winners_menu: Callable[[int], Any]
    giveaway_status_template: Callable[[dict], str]


def _is_admin(*, deps: GiveawayMenuFlowDependencies, user_id: int | None) -> bool:
    return deps.hub.giveaway_service.is_admin(user_id or 0)


async def _run_and_answer(*, callback, chat_id: int, runner, deps: GiveawayMenuFlowDependencies) -> None:
    async def _run() -> None:
        await runner()
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_giveaway_menu_callback(*, callback, deps: GiveawayMenuFlowDependencies) -> None:
    chat_id = callback.message.chat.id
    data = callback.data or ""
    parts = data.split(":")
    action = parts[1] if len(parts) > 1 else ""
    user_id = callback.from_user.id if callback.from_user else None

    if action == "start":
        if not _is_admin(deps=deps, user_id=user_id):
            await callback.answer("Admin only", show_alert=True)
            return
        await callback.message.answer("Pick duration.", reply_markup=deps.giveaway_duration_menu())
        await callback.answer()
        return

    if action == "dur" and len(parts) >= 3:
        if not _is_admin(deps=deps, user_id=user_id):
            await callback.answer("Admin only", show_alert=True)
            return
        duration_seconds = max(30, deps.as_int(parts[2], 600))
        await callback.message.answer(
            "Pick number of winners.",
            reply_markup=deps.giveaway_winners_menu(duration_seconds),
        )
        await callback.answer()
        return

    if action == "win" and len(parts) >= 4:
        if not _is_admin(deps=deps, user_id=user_id):
            await callback.answer("Admin only", show_alert=True)
            return
        duration_seconds = max(30, deps.as_int(parts[2], 600))
        winners = max(1, min(deps.as_int(parts[3], 1), 5))
        await deps.set_cmd_wizard(
            chat_id,
            {"step": "giveaway_prize", "duration_seconds": duration_seconds, "winners": winners},
        )
        await callback.message.answer("Send giveaway prize text, e.g. `50 USDT`.")
        await callback.answer()
        return

    if action == "join":
        if not user_id:
            await callback.answer("No user", show_alert=True)
            return

        async def _runner() -> None:
            try:
                payload = await deps.hub.giveaway_service.join_active(chat_id, user_id)
            except Exception as exc:
                await callback.message.answer(f"couldn't join: {exc}")
                return
            gw_id = payload.get("giveaway_id", "?")
            participants = payload.get("participants", "?")
            await callback.message.answer(
                "you're in\n\n"
                f"giveaway <b>#{gw_id}</b>\n"
                f"participants so far: <b>{participants}</b>\n\n"
                "<i>good luck, fren.</i>"
            )

        await _run_and_answer(callback=callback, chat_id=chat_id, runner=_runner, deps=deps)
        return

    if action == "status":
        async def _runner() -> None:
            payload = await deps.hub.giveaway_service.status(chat_id)
            await callback.message.answer(deps.giveaway_status_template(payload))

        await _run_and_answer(callback=callback, chat_id=chat_id, runner=_runner, deps=deps)
        return

    if action == "end":
        if not _is_admin(deps=deps, user_id=user_id):
            await callback.answer("Admin only", show_alert=True)
            return

        async def _runner() -> None:
            try:
                payload = await deps.hub.giveaway_service.end_giveaway(chat_id, user_id)
            except Exception as exc:
                await callback.message.answer(f"couldn't end giveaway: {exc}")
                return
            if payload.get("winner_user_id"):
                await callback.message.answer(
                    f"giveaway <b>#{payload.get('giveaway_id')}</b> closed.\n\n"
                    f"winner: <code>{payload.get('winner_user_id')}</code>\n"
                    f"prize: <b>{payload.get('prize', '-')}</b>"
                )
                return
            note = payload.get("note") or "no participants"
            await callback.message.answer(f"giveaway ended with no winner. {note}")

        await _run_and_answer(callback=callback, chat_id=chat_id, runner=_runner, deps=deps)
        return

    if action == "reroll":
        if not _is_admin(deps=deps, user_id=user_id):
            await callback.answer("Admin only", show_alert=True)
            return

        async def _runner() -> None:
            try:
                payload = await deps.hub.giveaway_service.reroll(chat_id, user_id)
            except Exception as exc:
                await callback.message.answer(f"reroll failed: {exc}")
                return
            await callback.message.answer(
                f"reroll done for giveaway <b>#{payload.get('giveaway_id')}</b>\n\n"
                f"new winner: <code>{payload.get('winner_user_id')}</code>\n"
                f"prev winner: <code>{payload.get('previous_winner_user_id', '-')}</code>"
            )

        await _run_and_answer(callback=callback, chat_id=chat_id, runner=_runner, deps=deps)
        return

    await callback.answer()
