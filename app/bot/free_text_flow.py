from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from aiogram.enums import ChatAction


_REPAIR_RE = re.compile(
    r"\b(that'?s not what i meant|no(pe)?\s*(that'?s wrong|that'?s not it)|wrong|i meant|i wanted|not that|something else)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FreeTextFlowDependencies:
    hub: Any
    parse_message: Callable[[str], Any]
    openai_chat_mode: Callable[[], str]
    route_free_text: Callable[..., Awaitable[bool]]
    send_llm_reply: Callable[..., Awaitable[None]]
    get_chat_history: Callable[[int], Awaitable[list[dict[str, str]]]]
    dispatch_command_text: Callable[[Any, str], Awaitable[bool]]
    recent_analysis_context: Callable[[int], Awaitable[dict | None]]
    looks_like_analysis_followup: Callable[[str, dict | None], bool]
    llm_followup_reply: Callable[[str, dict, int], Awaitable[str | None]]
    llm_market_chat_reply: Callable[[str, dict | None, int | None], Awaitable[str | None]]
    llm_route_message: Callable[[str], Awaitable[dict | None]]
    handle_routed_intent: Callable[[Any, dict, dict], Awaitable[bool]]
    handle_parsed_intent: Callable[[Any, Any, dict], Awaitable[bool]]
    is_definition_question: Callable[[str], bool]
    is_likely_english_phrase: Callable[[str], bool]
    extract_action_symbol_hint: Callable[[str], str | None]
    clarifying_question: Callable[[str | None], str]
    smart_action_menu: Callable[[str | None], Any]
    analysis_symbol_followup_kb: Callable[[], Any]
    define_keyboard: Callable[[], Any]
    pause: Callable[[float], Awaitable[None]]


async def handle_free_text_flow(
    *,
    message,
    text: str,
    chat_id: int,
    start_ts,
    deps: FreeTextFlowDependencies,
) -> bool:
    settings = await deps.hub.user_service.get_settings(chat_id)
    if not settings.get("disclaimer_seen"):
        await message.answer(
            "\u26a0\ufe0f <b>Disclaimer</b>: This bot is for info only, not financial advice. "
            "Data may be delayed. Trade at your own risk."
        )
        await deps.hub.user_service.update_settings(chat_id, {"disclaimer_seen": True})

    text_lower = text.lower().strip()

    if _REPAIR_RE.search(text_lower) and deps.hub.llm_client:
        history = await deps.get_chat_history(chat_id)
        if len(history) >= 2:
            last_user = next((item["content"] for item in reversed(history) if item.get("role") == "user"), "")
            last_assistant = next((item["content"] for item in reversed(history) if item.get("role") == "assistant"), "")
            repair_prompt = (
                f'The user just said: "{text[:200]}". '
                f'Your previous reply was: "{last_assistant[:400]}". '
                f'Their previous message was: "{last_user[:200]}". '
                "Restate in one short line what they likely want, then answer that. Be brief."
            )
            try:
                repair_reply = await deps.hub.llm_client.reply(repair_prompt, history=history[-6:])
            except Exception:
                repair_reply = None
            if repair_reply and repair_reply.strip():
                await deps.send_llm_reply(
                    message,
                    repair_reply.strip(),
                    settings,
                    user_message=text,
                    add_quick_replies=True,
                    analytics={
                        "route": "repair",
                        "reply_kind": "repair",
                        "request_started_at": start_ts,
                    },
                )
                return True

    if "define trading" in text_lower or ("define" in text_lower and len(text_lower.split()) <= 3):
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        await deps.pause(0.8)
        await message.answer(
            "the art of losing money faster than a casino while staring at candles "
            "until your eyes bleed. buy low, sell high, don't get rekt",
            reply_markup=deps.define_keyboard(),
        )
        return True

    if "rsi top" in text_lower or "overbought list" in text_lower or "strong coins" in text_lower:
        if "oversold" in text_lower:
            await deps.dispatch_command_text(message, "rsi top 10 1h oversold")
        elif "overbought" in text_lower:
            await deps.dispatch_command_text(message, "rsi top 10 1h overbought")
        elif "rsi top" in text_lower:
            await deps.dispatch_command_text(message, text_lower)
        else:
            await deps.dispatch_command_text(message, "coins to watch 5")
        return True

    followup_context = await deps.recent_analysis_context(chat_id)
    if deps.looks_like_analysis_followup(text, followup_context):
        followup_reply = await deps.llm_followup_reply(text, followup_context or {}, chat_id=chat_id)
        if followup_reply:
            await deps.send_llm_reply(
                message,
                followup_reply,
                settings,
                user_message=text,
                analytics={
                    "route": "analysis_followup",
                    "reply_kind": "followup",
                    "request_started_at": start_ts,
                },
            )
            return True

    parsed = deps.parse_message(text)
    chat_mode = deps.openai_chat_mode()
    return await deps.route_free_text(
        message=message,
        text=text,
        settings=settings,
        chat_id=chat_id,
        start_ts=start_ts,
        hub_has_llm=bool(deps.hub.llm_client),
        parsed=parsed,
        chat_mode=chat_mode,
        llm_market_chat_reply=deps.llm_market_chat_reply,
        llm_route_message=deps.llm_route_message,
        handle_routed_intent=deps.handle_routed_intent,
        handle_parsed_intent=deps.handle_parsed_intent,
        send_llm_reply=deps.send_llm_reply,
        is_definition_question=deps.is_definition_question,
        is_likely_english_phrase=deps.is_likely_english_phrase,
        extract_action_symbol_hint=deps.extract_action_symbol_hint,
        clarifying_question=deps.clarifying_question,
        smart_action_menu=deps.smart_action_menu,
        analysis_symbol_followup_kb=deps.analysis_symbol_followup_kb,
    )
