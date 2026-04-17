from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable

from app.core.nlu import Intent

logger = logging.getLogger(__name__)


async def handle_free_text_routing(
    *,
    message,
    text: str,
    settings: dict,
    chat_id: int,
    start_ts: datetime,
    hub_has_llm: bool,
    parsed,
    chat_mode: str,
    llm_market_chat_reply: Callable[[str, dict | None, int | None], Awaitable[str | None]],
    llm_route_message: Callable[[str], Awaitable[dict | None]],
    handle_routed_intent: Callable[[Any, dict, dict], Awaitable[bool]],
    handle_parsed_intent: Callable[[Any, Any, dict], Awaitable[bool]],
    send_llm_reply: Callable[..., Awaitable[None]],
    is_definition_question: Callable[[str], bool],
    is_likely_english_phrase: Callable[[str], bool],
    extract_action_symbol_hint: Callable[[str], str | None],
    clarifying_question: Callable[[str | None], str],
    smart_action_menu: Callable[[str | None], Any],
    analysis_symbol_followup_kb: Callable[[], Any],
) -> bool:
    if hub_has_llm and chat_mode == "chat_only":
        llm_reply = await llm_market_chat_reply(text, settings, chat_id=chat_id)
        if llm_reply:
            await send_llm_reply(
                message,
                llm_reply,
                settings,
                user_message=text,
                analytics={
                    "route": "chat_only",
                    "reply_kind": "market_chat",
                    "request_started_at": start_ts,
                },
            )
            return True
        fallback = (
            "couldn't reach the brain - try again in a sec, fren."
            if is_definition_question(text)
            else "signal unclear fren - try rephrasing or drop a ticker."
        )
        await message.answer(fallback, reply_to_message_id=message.message_id)
        return True

    if hub_has_llm and chat_mode == "llm_first":
        routed = await llm_route_message(text)
        if routed:
            try:
                if await handle_routed_intent(message, settings, routed):
                    return True
            except Exception:
                logger.exception("router_llm_first_handle_error", extra={"event": "router_llm_first_handle_error", "chat_id": chat_id})
        llm_reply = await llm_market_chat_reply(text, settings, chat_id=chat_id)
        if llm_reply:
            await send_llm_reply(
                message,
                llm_reply,
                settings,
                user_message=text,
                analytics={
                    "route": "llm_first_fallback",
                    "reply_kind": "market_chat",
                    "request_started_at": start_ts,
                },
            )
            return True

    if chat_mode in {"hybrid", "tool_first"} and (parsed.intent == Intent.UNKNOWN or (parsed.requires_followup and parsed.intent == Intent.UNKNOWN)):
        routed = await llm_route_message(text)
        if routed:
            try:
                if await handle_routed_intent(message, settings, routed):
                    return True
            except Exception:
                logger.exception("router_hybrid_handle_error", extra={"event": "router_hybrid_handle_error", "chat_id": chat_id})

    if parsed.requires_followup:
        if parsed.intent == Intent.UNKNOWN:
            llm_reply = await llm_market_chat_reply(text, settings, chat_id=chat_id)
            if llm_reply:
                await send_llm_reply(
                    message,
                    llm_reply,
                    settings,
                    user_message=text,
                    analytics={
                        "route": "unknown_followup",
                        "reply_kind": "market_chat",
                        "request_started_at": start_ts,
                    },
                )
                return True
            english_phrase = is_likely_english_phrase(text)
            symbol_hint = None if english_phrase else extract_action_symbol_hint(text)
            if symbol_hint:
                await message.answer(
                    f"pick an action for <b>{symbol_hint}</b>:",
                    reply_markup=smart_action_menu(symbol_hint),
                    reply_to_message_id=message.message_id,
                )
            else:
                await message.answer(
                    clarifying_question(None),
                    reply_markup=smart_action_menu(None),
                    reply_to_message_id=message.message_id,
                )
            return True
        if parsed.intent == Intent.ANALYSIS and not parsed.entities.get("symbol"):
            await message.answer(parsed.followup_question or "Need one detail.", reply_markup=analysis_symbol_followup_kb())
            return True
        await message.answer(
            parsed.followup_question or clarifying_question(None),
            reply_to_message_id=message.message_id,
        )
        return True

    try:
        if await handle_parsed_intent(message, parsed, settings):
            return True
        llm_reply = await llm_market_chat_reply(text, settings, chat_id=chat_id)
        if llm_reply:
            await send_llm_reply(
                message,
                llm_reply,
                settings,
                user_message=text,
                analytics={
                    "route": "parsed_fallback",
                    "reply_kind": "market_chat",
                    "request_started_at": start_ts,
                },
            )
            return True
        symbol_hint = extract_action_symbol_hint(text)
        await message.answer(
            parsed.followup_question or clarifying_question(symbol_hint),
            reply_markup=smart_action_menu(symbol_hint) if symbol_hint else None,
            reply_to_message_id=message.message_id,
        )
        return True
    except Exception as exc:
        logger.exception("router_handle_parsed_intent_error", extra={"event": "router_handle_parsed_intent_error", "chat_id": chat_id})
        safe_exc = (
            str(exc)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        await message.answer(
            f"couldn't complete that - <i>{safe_exc}</i>\n"
            "try again with a bit more detail."
        )
        return True
