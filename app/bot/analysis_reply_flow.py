from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from aiogram.enums import ChatAction


_FOLLOWUP_RE = re.compile(
    r"\b("
    r"how about|what if|too risky|risky|is\s+[0-9.]+\s+(a\s+)?good\s+entry|good entry|"
    r"sl|stop(?:\s+loss)?|entry|target|tp\d*|take profit|"
    r"leverage|[0-9]+x|risk|rr|send it|thoughts?|better|worse"
    r")\b",
    re.IGNORECASE,
)
_FOLLOWUP_VALUE_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\b")


@dataclass(frozen=True)
class AnalysisReplyFlowDependencies:
    format_as_ghost: Callable[[dict], Awaitable[str]]
    llm_analysis_reply: Callable[..., Awaitable[str | None]]
    trade_plan_template: Callable[..., str]
    analysis_progressive_menu: Callable[[str, str | None], Any]
    pause: Callable[[float], Awaitable[None]]


def analysis_context_payload(symbol: str, direction: str | None, payload: dict) -> dict:
    return {
        "symbol": symbol.upper(),
        "direction": (direction or payload.get("side") or "").strip().lower() or None,
        "analysis_summary": str(payload.get("summary") or "").strip(),
        "key_levels": {
            "entry": str(payload.get("entry") or "").strip(),
            "tp1": str(payload.get("tp1") or "").strip(),
            "tp2": str(payload.get("tp2") or "").strip(),
            "sl": str(payload.get("sl") or "").strip(),
            "price": payload.get("price"),
        },
        "market_context": payload.get("market_context", {}),
        "market_context_text": str(payload.get("market_context_text") or "").strip(),
    }


async def remember_analysis_context(
    *,
    cache,
    chat_id: int,
    symbol: str,
    direction: str | None,
    payload: dict,
) -> None:
    context = analysis_context_payload(symbol, direction, payload)
    await cache.set_json(f"last_analysis_context:{chat_id}", context, ttl=300)
    await cache.set_json(f"last_analysis_context:{chat_id}:{symbol.upper()}", context, ttl=300)


async def recent_analysis_context(*, cache, chat_id: int) -> dict | None:
    payload = await cache.get_json(f"last_analysis_context:{chat_id}")
    return payload if isinstance(payload, dict) else None


def looks_like_analysis_followup(text: str, context: dict | None) -> bool:
    cleaned = (text or "").strip()
    if not cleaned or not context:
        return False
    lower = cleaned.lower()
    symbol = str(context.get("symbol") or "").lower()
    if symbol and re.search(rf"\b{re.escape(symbol)}\b", lower) and _FOLLOWUP_VALUE_RE.search(lower):
        return True
    if _FOLLOWUP_RE.search(lower):
        return True
    if lower.endswith("?") and _FOLLOWUP_VALUE_RE.search(lower):
        return True
    return False


async def render_analysis_text(
    *,
    payload: dict,
    symbol: str,
    direction: str | None,
    settings: dict,
    chat_id: int,
    detailed: bool = False,
    deps: AnalysisReplyFlowDependencies,
) -> str:
    try:
        return await deps.format_as_ghost(payload)
    except Exception:
        llm_text = await deps.llm_analysis_reply(
            payload=payload,
            symbol=symbol,
            direction=direction,
            chat_id=chat_id,
        )
        if llm_text:
            return llm_text
        return deps.trade_plan_template(payload, settings, detailed=detailed)


async def send_ghost_analysis(
    message,
    symbol: str,
    text: str,
    *,
    direction: str | None = None,
    deps: AnalysisReplyFlowDependencies,
) -> None:
    with suppress(Exception):
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    await deps.pause(1.3)
    await message.answer(text, reply_markup=deps.analysis_progressive_menu(symbol, direction))
