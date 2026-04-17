from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from app.core.nlu import Intent, _extract_prices, parse_message

_ALERT_SHORTCUT_RE = re.compile(
    r"^alert\s+([A-Za-z0-9]{2,20})\s+([\d.,]+[kKmM]?)\s*(above|below|cross|crosses|over|under)?\s*$",
    re.IGNORECASE,
)


async def dispatch_command_text(
    *,
    message,
    synthetic_text: str,
    hub: Any,
    handle_parsed_intent: Callable[[Any, Any, dict], Awaitable[bool]],
    llm_fallback_reply: Callable[[str, dict | None, int | None], Awaitable[str | None]],
    send_llm_reply: Callable[..., Awaitable[None]],
    clarifying_question: Callable[[str | None], str],
    extract_action_symbol_hint: Callable[[str], str | None],
    smart_action_menu: Callable[..., Any],
    analysis_symbol_followup_kb: Callable[[], Any],
    safe_exc: Callable[[BaseException], str],
    alert_created_menu: Callable[[str], Any],
) -> bool:
    chat_id = message.chat.id
    settings = await hub.user_service.get_settings(chat_id)

    alert_match = _ALERT_SHORTCUT_RE.match(synthetic_text.strip())
    if alert_match:
        symbol = alert_match.group(1).upper()
        raw_price_str = alert_match.group(2)
        raw_cond = (alert_match.group(3) or "cross").lower()
        prices = _extract_prices(raw_price_str)
        price = prices[0] if prices else None
        if symbol and price is not None:
            condition = (
                "above"
                if raw_cond in ("above", "over")
                else ("below" if raw_cond in ("below", "under") else "cross")
            )
            try:
                if not hub.alerts_uc:
                    raise RuntimeError("alerts use-case not configured")
                alert = await hub.alerts_uc.create(
                    chat_id=chat_id,
                    symbol=symbol,
                    condition=condition,
                    target_price=float(price),
                )
                cond_word = {
                    "above": "crosses above",
                    "below": "crosses below",
                }.get(condition, "crosses")
                await message.answer(
                    f"alert set - <b>{alert.symbol}</b> {cond_word} <b>${float(price):,.2f}</b>.\n"
                    "i'll ping you the moment it hits. don't get liquidated.",
                    reply_markup=alert_created_menu(alert.symbol),
                )
            except RuntimeError as exc:
                await message.answer(f"couldn't set that alert - {safe_exc(exc)}")
            except Exception:
                await message.answer("alert creation failed. try again.")
            return True

    parsed = parse_message(synthetic_text)

    if parsed.requires_followup:
        if parsed.intent == Intent.ANALYSIS and not parsed.entities.get("symbol"):
            await message.answer(
                parsed.followup_question or "Need one detail.",
                reply_markup=analysis_symbol_followup_kb(),
            )
            return True
        await message.answer(
            parsed.followup_question or clarifying_question(extract_action_symbol_hint(message.text or "")),
            reply_markup=smart_action_menu(),
        )
        return True

    if await handle_parsed_intent(message, parsed, settings):
        return True

    llm_reply = await llm_fallback_reply(synthetic_text, settings, chat_id=chat_id)
    if llm_reply:
        await send_llm_reply(
            message,
            llm_reply,
            settings,
            user_message=synthetic_text,
            analytics={"route": "synthetic_fallback", "reply_kind": "general_chat"},
        )
        return True
    return False
