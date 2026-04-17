from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GiveawayCommandFlowDependencies:
    hub: Any
    safe_exc: Callable[[Exception], str]
    parse_duration_to_seconds: Callable[[str], int | None]
    giveaway_status_template: Callable[[dict], str]
    giveaway_menu: Callable[..., Any]


async def handle_join_command(*, message, deps: GiveawayCommandFlowDependencies) -> None:
    if not message.from_user:
        await message.answer("Could not identify user for join.")
        return
    try:
        payload = await deps.hub.giveaway_service.join_active(message.chat.id, message.from_user.id)
    except Exception as exc:
        await message.answer(f"couldn't join: {deps.safe_exc(exc)}")
        return
    await message.answer(
        f"you're in giveaway <b>#{payload['giveaway_id']}</b> | participants: <b>{payload['participants']}</b>"
    )


async def handle_giveaway_command(*, message, deps: GiveawayCommandFlowDependencies) -> None:
    text = (message.text or "").strip()
    if not message.from_user:
        await message.answer("Could not identify sender.")
        return

    if re.search(r"^/giveaway\s+status\b", text, flags=re.IGNORECASE):
        payload = await deps.hub.giveaway_service.status(message.chat.id)
        await message.answer(deps.giveaway_status_template(payload))
        return

    if re.search(r"^/giveaway\s+join\b", text, flags=re.IGNORECASE):
        await handle_join_command(message=message, deps=deps)
        return

    if re.search(r"^/giveaway\s+end\b", text, flags=re.IGNORECASE):
        try:
            payload = await deps.hub.giveaway_service.end_giveaway(message.chat.id, message.from_user.id)
        except Exception as exc:
            await message.answer(f"couldn't end giveaway: {deps.safe_exc(exc)}")
            return
        if payload.get("winner_user_id"):
            await message.answer(
                f"giveaway <b>#{payload.get('giveaway_id')}</b> closed.\n"
                f"winner: <code>{payload.get('winner_user_id')}</code>\n"
                f"prize: <b>{payload.get('prize', 'Prize')}</b>"
            )
        else:
            await message.answer(f"giveaway ended with no winner. {payload.get('note')}")
        return

    if re.search(r"^/giveaway\s+reroll\b", text, flags=re.IGNORECASE):
        try:
            payload = await deps.hub.giveaway_service.reroll(message.chat.id, message.from_user.id)
        except Exception as exc:
            await message.answer(f"reroll failed: {deps.safe_exc(exc)}")
            return
        await message.answer(
            f"reroll done for giveaway <b>#{payload.get('giveaway_id')}</b>\n"
            f"new winner: <code>{payload.get('winner_user_id')}</code>\n"
            f"prev: <code>{payload.get('previous_winner_user_id', '-')}</code>"
        )
        return

    start_match = re.search(r"^/giveaway\s+start\s+(\S+)(?:\s+(.+))?$", text, flags=re.IGNORECASE)
    if start_match:
        duration_seconds = deps.parse_duration_to_seconds(start_match.group(1))
        if duration_seconds is None:
            await message.answer("Invalid duration. Example: /giveaway start 10m prize \"50 USDT\"")
            return
        tail = (start_match.group(2) or "").strip()
        winners_match = re.search(r"\bwinners?\s*=?\s*(\d+)\b", tail, flags=re.IGNORECASE)
        winners_requested = int(winners_match.group(1)) if winners_match else 1
        tail = re.sub(r"\bwinners?\s*=?\s*\d+\b", "", tail, flags=re.IGNORECASE).strip()
        tail = re.sub(r"^\s*prize\s+", "", tail, flags=re.IGNORECASE).strip()
        prize = (tail or "Prize").strip("'\"")
        try:
            payload = await deps.hub.giveaway_service.start_giveaway(
                group_chat_id=message.chat.id,
                admin_chat_id=message.from_user.id,
                duration_seconds=duration_seconds,
                prize=prize,
            )
        except Exception as exc:
            await message.answer(f"couldn't start giveaway: {deps.safe_exc(exc)}")
            return
        note = ""
        if winners_requested > 1:
            note = "\nNote: multi-winner draw will run as sequential rerolls after the first winner."
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\n"
            f"Ends at: {payload['end_time']}\nUsers enter with /join or /giveaway join{note}"
        )
        return

    await message.answer(
        "Pick giveaway action.",
        reply_markup=deps.giveaway_menu(is_admin=deps.hub.giveaway_service.is_admin(message.from_user.id)),
    )
