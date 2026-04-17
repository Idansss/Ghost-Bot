from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class PreRouteStateDependencies:
    hub: Any
    as_int: Callable[[Any, int], int]
    dispatch_command_text: Callable[[Any, str], Awaitable[bool]]
    get_pending_feedback_suggestion: Callable[[int], Awaitable[dict | None]]
    clear_pending_feedback_suggestion: Callable[[int], Awaitable[None]]
    log_feedback_event: Callable[..., Awaitable[None]]
    notify_admins_negative_feedback: Callable[..., Awaitable[None]]
    get_cmd_wizard: Callable[[int], Awaitable[dict | None]]
    clear_cmd_wizard: Callable[[int], Awaitable[None]]
    get_wizard: Callable[[int], Awaitable[dict | None]]
    set_wizard: Callable[[int, dict], Awaitable[None]]
    clear_wizard: Callable[[int], Awaitable[None]]
    parse_timestamp: Callable[[str], Any]
    save_trade_check: Callable[[int, dict, dict], Awaitable[None]]
    remember_source_context: Callable[..., Awaitable[None]]
    get_pending_alert: Callable[[int], Awaitable[str | None]]
    clear_pending_alert: Callable[[int], Awaitable[None]]
    trade_verification_template: Callable[[dict, dict], str]
    giveaway_menu: Callable[..., Any]
    alert_created_menu: Callable[[str], Any]


async def handle_pre_route_state(
    *,
    message,
    text: str,
    chat_id: int,
    deps: PreRouteStateDependencies,
) -> bool:
    pending_feedback = await deps.get_pending_feedback_suggestion(chat_id)
    if pending_feedback:
        await deps.clear_pending_feedback_suggestion(chat_id)
        from_username = getattr(message.from_user, "username", None) or getattr(message.from_user, "first_name", None) or ""
        await deps.log_feedback_event(
            chat_id=chat_id,
            message_id=deps.as_int(pending_feedback.get("message_id"), 0) or None,
            from_user_id=getattr(message.from_user, "id", None),
            from_username=from_username,
            sentiment="suggestion",
            source="message",
            reason=str(pending_feedback.get("reason") or "suggestion"),
            reply_preview=str(pending_feedback.get("reply_preview") or ""),
            improvement_text=text.strip() or None,
        )
        await deps.notify_admins_negative_feedback(
            from_chat_id=chat_id,
            from_username=from_username,
            reason=str(pending_feedback.get("reason") or "other"),
            reply_preview=str(pending_feedback.get("reply_preview") or ""),
            improvement_text=text.strip() or None,
        )
        await message.answer("Thanks \u2014 I've passed that on personally. We'll use it to improve.")
        return True

    cmd_wizard = await deps.get_cmd_wizard(chat_id)
    if cmd_wizard:
        step = str(cmd_wizard.get("step") or "").strip().lower()
        if step == "dispatch_text":
            prefix = str(cmd_wizard.get("prefix") or "")
            await deps.clear_cmd_wizard(chat_id)
            typed = text.strip()
            if not typed:
                await message.answer("send the details and i'll run it.")
                return True
            await deps.dispatch_command_text(message, f"{prefix}{typed}".strip())
            return True
        if step == "giveaway_prize":
            await deps.clear_cmd_wizard(chat_id)
            if not message.from_user:
                await message.answer("Could not identify sender.")
                return True
            prize = text.strip().strip("'\"") or "Prize"
            duration_seconds = max(30, deps.as_int(cmd_wizard.get("duration_seconds"), 600))
            winners_requested = max(1, min(deps.as_int(cmd_wizard.get("winners"), 1), 5))
            try:
                payload = await deps.hub.giveaway_service.start_giveaway(
                    group_chat_id=chat_id,
                    admin_chat_id=message.from_user.id,
                    duration_seconds=duration_seconds,
                    prize=prize,
                )
            except Exception as exc:
                await message.answer(str(exc))
                return True
            note = ""
            if winners_requested > 1:
                note = "\nNote: multi-winner draw runs as sequential rerolls after first winner."
            await message.answer(
                f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\n"
                f"Ends at: {payload['end_time']}\nUsers enter with /join or /giveaway join{note}",
                reply_markup=deps.giveaway_menu(is_admin=True),
            )
            return True
        if step == "alert_clear_symbol":
            await deps.clear_cmd_wizard(chat_id)
            symbol = text.strip().upper().lstrip("$")
            if not re.fullmatch(r"[A-Z0-9]{2,20}", symbol):
                await message.answer("Invalid symbol. Send a ticker like SOL.")
                return True
            count = await deps.hub.alerts_service.delete_alerts_by_symbol(chat_id, symbol)
            await message.answer(f"Cleared {count} alerts for {symbol}.")
            return True

    wizard = await deps.get_wizard(chat_id)
    if wizard:
        step = wizard.get("step")
        data = wizard.get("data", {})
        if step == "symbol":
            data["symbol"] = text.strip().upper()
            await deps.set_wizard(chat_id, {"step": "timeframe", "data": data})
            await message.answer("Timeframe? (15m / 1h / 4h)")
            return True
        if step == "timeframe":
            data["timeframe"] = text.strip().lower()
            await deps.set_wizard(chat_id, {"step": "timestamp", "data": data})
            await message.answer("Timestamp? (ISO, yesterday, or 2 hours ago)")
            return True
        if step == "timestamp":
            timestamp = deps.parse_timestamp(text)
            if not timestamp:
                await message.answer("Could not parse timestamp. Try 'yesterday' or ISO datetime.")
                return True
            data["timestamp"] = timestamp.isoformat()
            await deps.set_wizard(chat_id, {"step": "levels", "data": data})
            await message.answer("Send levels: entry <x> stop <y> targets <a> <b> ...")
            return True
        if step == "levels":
            entry_match = re.search(r"entry\s*([0-9.]+)", text, re.IGNORECASE)
            stop_match = re.search(r"stop\s*([0-9.]+)", text, re.IGNORECASE)
            targets_match = re.search(r"targets?\s*([0-9.\s]+)", text, re.IGNORECASE)
            if not entry_match or not stop_match or not targets_match:
                await message.answer("Format: entry <x> stop <y> targets <a> <b>")
                return True
            targets = [float(item) for item in re.findall(r"[0-9.]+", targets_match.group(1))]
            data.update(
                {
                    "entry": float(entry_match.group(1)),
                    "stop": float(stop_match.group(1)),
                    "targets": targets,
                    "timestamp": datetime.fromisoformat(data["timestamp"]),
                    "mode": "ambiguous",
                }
            )
            result = await deps.hub.trade_verify_service.verify(**data)
            await deps.save_trade_check(chat_id, data, result)
            await deps.remember_source_context(
                chat_id,
                source_line=str(result.get("source_line") or ""),
                symbol=data["symbol"],
                context="trade check",
            )
            wizard_settings = await deps.hub.user_service.get_settings(chat_id)
            await message.answer(deps.trade_verification_template(result, wizard_settings))
            await deps.clear_wizard(chat_id)
            return True

    pending_alert_symbol = await deps.get_pending_alert(chat_id)
    if pending_alert_symbol:
        price_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if price_match:
            target = float(price_match.group(1))
            alert = await deps.hub.alerts_service.create_alert(
                chat_id,
                pending_alert_symbol,
                "cross",
                target,
                source="button",
            )
            await deps.clear_pending_alert(chat_id)
            await deps.remember_source_context(
                chat_id,
                exchange=alert.source_exchange,
                market_kind=alert.market_kind,
                instrument_id=alert.instrument_id,
                symbol=pending_alert_symbol,
                context="alert",
            )
            await message.answer(
                f"\U0001F514 alert set \u2014 <b>{pending_alert_symbol}</b> crosses <b>${target:,.2f}</b>.\n"
                "i'll ping you the moment it hits. don't get liquidated.",
                reply_markup=deps.alert_created_menu(pending_alert_symbol),
            )
            return True

    return False
