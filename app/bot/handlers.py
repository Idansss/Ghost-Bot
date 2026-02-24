from __future__ import annotations

import asyncio
import json
import random
import re
from contextlib import suppress
from datetime import datetime, timezone
import logging

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    MessageReactionUpdated,
)

from app.bot.keyboards import (
    alert_created_menu,
    alert_quick_menu,
    alpha_quick_menu,
    analysis_actions,
    analysis_progressive_menu,
    chart_quick_menu,
    command_center_menu,
    confirm_understanding_kb,
    ema_quick_menu,
    feedback_buttons,
    feedback_reason_kb,
    findpair_quick_menu,
    giveaway_duration_menu,
    giveaway_menu,
    giveaway_winners_menu,
    heatmap_quick_menu,
    llm_reply_keyboard,
    news_quick_menu,
    rsi_quick_menu,
    scan_quick_menu,
    settings_menu,
    setup_quick_menu,
    simple_followup,
    smart_action_menu,
    wallet_actions,
    watch_quick_menu,
)
from app.bot.templates import (
    asset_unsupported_template,
    clarifying_question,
    correlation_template,
    cycle_template,
    giveaway_status_template,
    help_text,
    market_condition_warning,
    news_template,
    pair_find_template,
    price_guess_template,
    rsi_scan_template,
    setup_review_template,
    settings_text,
    smalltalk_reply,
    trade_math_template,
    trade_plan_template,
    trade_verification_template,
    unknown_prompt,
    wallet_scan_template,
    watchlist_template,
)
from app.core.config import get_settings
from app.core.container import ServiceHub
from app.core.fred_persona import ghost as fred
from app.services.market_context import format_market_context
from app.core.nlu import COMMON_WORDS_NOT_TICKERS, Intent, is_likely_english_phrase, parse_message, parse_timestamp
from app.db.models import TradeCheck
from app.db.session import AsyncSessionLocal

router = Router()
_settings = get_settings()
_hub: ServiceHub | None = None
_ALLOWED_OPENAI_CHAT_MODES = {"hybrid", "tool_first", "llm_first", "chat_only"}
_CHAT_LOCKS: dict[int, asyncio.Lock] = {}
logger = logging.getLogger(__name__)
SOURCE_QUERY_RE = re.compile(
    r"\b(where\s+is\s+this\s+from|what(?:'s| is)\s+the\s+source|which\s+exchange|source\??|exchange\??)\b",
    re.IGNORECASE,
)

# Greeting fast-path — compiled once at import time
_GREETING_RE = re.compile(
    r"^(gm|gn|gg|gm fren|gn fren|good\s*morning|good\s*night|"
    r"hi|hey|hello|sup|yo|wassup|wagmi|lgtm|lfg|ngmi|ser|fren|anon|"
    r"wen\s*moon|wen\s*lambo|wen\s*pump|wen\s*bull|wen\s*dump|"
    r"still\s*alive|you\s*there|you\s*alive|are\s*you\s*there)[\s!?.]*$",
    re.IGNORECASE,
)
_GM_REPLIES = [
    "gm fren 👋 charts are open, tape is moving. what are we hunting today?",
    "gm anon ☀️ market's breathing. drop a ticker or ask anything.",
    "gm 👋 still alive, still watching. what do you need?",
    "gm fren — locked in. throw me a coin or question.",
    "gm anon. BTC still the anchor, alts still lagging dominance. what's the play?",
    "gm ☕ fresh session. give me a ticker, a question, or ask what's moving.",
    "gm — charts loaded, alerts armed. what are we doing today?",
    "gm fren. the market doesn't care about your feelings. let's get to work.",
    "gm anon 🌅 new candle, new opportunity. what's on the radar?",
]
_GN_REPLIES = [
    "gn fren 🌙 set your alerts before you sleep.",
    "gn anon. the market doesn't sleep but you should.",
    "gn — if you haven't set alerts, do it now. i'll watch.",
    "gn fren 🌙 tape keeps printing while you rest. alerts are armed.",
]
SOURCE_QUERY_STOPWORDS = {
    "source",
    "exchange",
    "where",
    "is",
    "this",
    "from",
    "what",
    "whats",
    "the",
    "for",
    "of",
    "result",
    "last",
}
ACTION_SYMBOL_STOPWORDS = {
    "what",
    "who",
    "hwo",
    "how",
    "are",
    "you",
    "doing",
    "coin",
    "coins",
    "overbought",
    "oversold",
    "list",
    "top",
    "news",
    "alert",
    "scan",
    "chart",
    "heatmap",
    "short",
    "long",
}
ACTION_SYMBOL_STOPWORDS.update(COMMON_WORDS_NOT_TICKERS)
FOLLOWUP_RE = re.compile(
    r"\b("
    r"how about|what if|too risky|risky|is\s+[0-9.]+\s+(a\s+)?good\s+entry|good entry|"
    r"sl|stop(?:\s+loss)?|entry|target|tp\d*|take profit|"
    r"leverage|[0-9]+x|risk|rr|send it|thoughts?|better|worse"
    r")\b",
    re.IGNORECASE,
)
FOLLOWUP_VALUE_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\b")


_CHAT_LOCKS_MAX = 2000


def _safe_exc(exc: Exception) -> str:
    """Return exception message safe for Telegram HTML parse_mode (no raw < > & chars)."""
    return (
        str(exc)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# Valid Telegram HTML tags (self-closing not needed)
_TELEGRAM_ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a", "blockquote", "tg-spoiler"}
_HTML_TAG_RE = re.compile(r"<(/?)(\w[\w\-]*)(\s[^>]*)?>", re.IGNORECASE)


def _sanitize_llm_html(text: str) -> str:
    """
    Clean LLM-generated HTML so Telegram won't reject it.
    - Strips unsupported tags (keeps their text content)
    - Closes any unclosed valid tags
    """
    if not text:
        return text

    result: list[str] = []
    open_stack: list[str] = []  # tags opened but not yet closed
    pos = 0

    for m in _HTML_TAG_RE.finditer(text):
        # Append the literal text before this tag
        result.append(text[pos:m.start()])
        pos = m.end()

        closing = m.group(1) == "/"
        tag = m.group(2).lower()
        attrs = m.group(3) or ""

        if tag not in _TELEGRAM_ALLOWED_TAGS:
            # Unsupported tag — drop the tag but keep nothing (content will still flow through)
            continue

        if closing:
            # Only emit the closing tag if we actually opened this tag
            if tag in open_stack:
                # Close any tags opened after this one (auto-close nested unclosed tags)
                while open_stack and open_stack[-1] != tag:
                    result.append(f"</{open_stack.pop()}>")
                if open_stack:
                    open_stack.pop()
                result.append(f"</{tag}>")
        else:
            result.append(f"<{tag}{attrs}>")
            # <a> and block-level tags need tracking; void-like usage is rare in LLM output
            open_stack.append(tag)

    # Append remaining text
    result.append(text[pos:])

    # Close any still-open tags in reverse order
    for tag in reversed(open_stack):
        result.append(f"</{tag}>")

    return "".join(result)


def _build_communication_memory_block(settings: dict | None) -> str:
    """Build prompt block for communication style, memory (name, goals, last symbols)."""
    s = settings or {}
    style = str(s.get("communication_style") or "friendly").strip().lower()
    style_rules = {
        "short": "Keep replies to 1–2 sentences. No preamble.",
        "detailed": "Give a one-line summary first, then 2–3 key points, then optional detail. Use line breaks.",
        "friendly": "Casual tone; 'fren' or 'anon' once. Match question length.",
        "formal": "Professional tone. No slang.",
    }
    style_rule = style_rules.get(style, style_rules["friendly"])
    parts = [f"COMMUNICATION: User prefers {style} style. {style_rule}"]
    name = (s.get("display_name") or "").strip()
    if name:
        parts.append(f"Call them {name}.")
    goals = (s.get("trading_goals") or "").strip()[:200]
    if goals:
        parts.append(f"Their stated goals: {goals}")
    last_syms = s.get("last_symbols") or []
    if isinstance(last_syms, list) and last_syms:
        syms = [str(x).upper() for x in last_syms[:5] if x]
        if syms:
            parts.append(f"They recently asked about: {', '.join(syms)}.")
    prefs = s.get("feedback_prefs") or {}
    if isinstance(prefs, dict) and prefs.get("prefers_shorter"):
        parts.append("User has asked for shorter answers before; keep it brief.")
    return " ".join(parts)


async def _append_last_symbol(chat_id: int, symbol: str) -> None:
    """Append symbol to user's last_symbols (max 5) for memory."""
    hub = _require_hub()
    settings = await hub.user_service.get_settings(chat_id)
    last = list(settings.get("last_symbols") or [])
    if not isinstance(last, list):
        last = []
    sym = (symbol or "").upper().strip()
    if not sym:
        return
    last = [sym] + [x for x in last if str(x).upper() != sym][:4]
    await hub.user_service.update_settings(chat_id, {"last_symbols": last})


async def _send_llm_reply(
    message: Message,
    reply: str,
    settings: dict | None = None,
    user_message: str | None = None,
    add_quick_replies: bool = True,
) -> None:
    """Send an LLM reply with HTML. Optionally attach quick-reply + feedback keyboard and cache for followups.
    In groups, if settings.reply_in_dm, send full reply to user's DM and post 'Sent you a DM' in the group."""
    from aiogram.exceptions import TelegramBadRequest

    cleaned = _sanitize_llm_html(reply)
    reply_to = message.message_id if message else None
    chat_id = message.chat.id if message else None
    if add_quick_replies and chat_id and user_message is not None:
        try:
            hub = _require_hub()
            await hub.cache.set_json(f"llm:last_reply:{chat_id}", reply, ttl=3600)
            await hub.cache.set_json(f"llm:last_user:{chat_id}", user_message, ttl=3600)
        except Exception:  # noqa: BLE001
            pass
    # If reply starts with "You want:" use confirm buttons so user can confirm or rephrase
    use_confirm = add_quick_replies and cleaned.strip().lower().startswith("you want:")
    reply_markup = confirm_understanding_kb() if use_confirm else (llm_reply_keyboard() if add_quick_replies else None)

    in_group = getattr(message.chat, "type", None) in ("group", "supergroup")
    reply_in_dm = (settings or {}).get("reply_in_dm")
    if in_group and not reply_in_dm and getattr(message.from_user, "id", None):
        try:
            user_settings = await _require_hub().user_service.get_settings(message.from_user.id)
            reply_in_dm = user_settings.get("reply_in_dm")
        except Exception:
            pass
    if in_group and reply_in_dm and getattr(message.from_user, "id", None):
        try:
            await message.bot.send_message(
                message.from_user.id, cleaned, reply_markup=reply_markup
            )
            await message.answer("Sent you a DM.", reply_to_message_id=reply_to)
        except TelegramBadRequest:
            plain = re.sub(r"<[^>]+>", "", reply)
            try:
                await message.bot.send_message(
                    message.from_user.id, plain, reply_markup=reply_markup
                )
                await message.answer("Sent you a DM.", reply_to_message_id=reply_to)
            except Exception:
                try:
                    await message.answer(
                        cleaned, reply_to_message_id=reply_to, reply_markup=reply_markup
                    )
                except TelegramBadRequest:
                    await message.answer(plain, reply_to_message_id=reply_to)
        return
    try:
        await message.answer(
            cleaned, reply_to_message_id=reply_to, reply_markup=reply_markup
        )
    except TelegramBadRequest:
        plain = re.sub(r"<[^>]+>", "", reply)
        with suppress(Exception):
            await message.answer(
                plain, reply_to_message_id=reply_to, reply_markup=reply_markup
            )


def _chat_lock(chat_id: int) -> asyncio.Lock:
    lock = _CHAT_LOCKS.get(chat_id)
    if lock is None:
        # Prune idle locks when dict grows too large to prevent unbounded memory growth
        if len(_CHAT_LOCKS) >= _CHAT_LOCKS_MAX:
            idle = [k for k, v in list(_CHAT_LOCKS.items()) if not v.locked()]
            for k in idle[:len(idle) // 2 + 1]:
                _CHAT_LOCKS.pop(k, None)
        lock = asyncio.Lock()
        _CHAT_LOCKS[chat_id] = lock
    return lock


async def _acquire_message_once(message: Message, ttl: int = 60 * 60 * 6) -> bool:
    hub = _require_hub()
    key = f"seen:message:{message.chat.id}:{message.message_id}"
    try:
        return await hub.cache.set_if_absent(key, ttl=ttl)
    except Exception:  # noqa: BLE001
        logger.exception("dedupe_cache_error", extra={"event": "dedupe_cache_error", "chat_id": message.chat.id})
        return True


async def _acquire_callback_once(callback: CallbackQuery, ttl: int = 60 * 30) -> bool:
    hub = _require_hub()
    cb_id = (callback.id or "").strip()
    if not cb_id:
        return True
    key = f"seen:callback:{cb_id}"
    return await hub.cache.set_if_absent(key, ttl=ttl)


async def _typing_loop(bot, chat_id: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        with suppress(Exception):
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


async def _run_with_typing_lock(bot, chat_id: int, runner) -> None:
    stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(bot, chat_id, stop))
    lock = _chat_lock(chat_id)
    try:
        async with lock:
            await runner()
    finally:
        stop.set()
        typing_task.cancel()
        with suppress(Exception):
            await typing_task


async def _market_aware_gm_reply(hub: ServiceHub, name: str) -> str:
    """Try to return a BTC-context-aware GM greeting, fall back to static pool."""
    try:
        ctx = await asyncio.wait_for(hub.analysis_service.get_market_context(), timeout=3.0)
        if isinstance(ctx, dict):
            btc_change: float | None = None
            for key in ("btc_change_pct_1h", "btc_pct_change_1h", "btc_change_pct", "btc_pct_change"):
                val = ctx.get(key)
                if val is not None:
                    try:
                        btc_change = float(val)
                        break
                    except (TypeError, ValueError):
                        pass
            if btc_change is not None:
                direction = "up" if btc_change > 0 else "down"
                abs_pct = abs(btc_change)
                mood = "volatile" if abs_pct >= 3.0 else ("active" if abs_pct >= 1.5 else "quiet")
                pool = [
                    f"gm {name} 👋 btc {direction} {abs_pct:.1f}% — {mood} session. drop a ticker or ask what's moving.",
                    f"gm {name} ☀️ btc is {direction} {abs_pct:.1f}%. tape is {'hot' if abs_pct >= 2.0 else 'breathing'}. what are we hunting?",
                    f"gm — btc {direction} {abs_pct:.1f}% this hour. {mood} market. send a coin.",
                ]
                return random.choice(pool)
    except Exception:  # noqa: BLE001
        pass
    return random.choice(_GM_REPLIES)


async def _maybe_send_market_warning(message: Message) -> None:
    """Send a one-liner warning if it's a weekend session (thin liquidity)."""
    now = datetime.now(timezone.utc)
    if now.weekday() < 5:  # Mon–Fri: no warning
        return
    with suppress(Exception):
        await message.answer("<i>⚠ weekend session — liquidity is thin, spreads are wide. size accordingly.</i>")


def init_handlers(hub: ServiceHub) -> None:
    global _hub
    _hub = hub


def _parse_int_list(value, fallback: list[int]) -> list[int]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        return fallback
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except Exception:  # noqa: BLE001
            continue
    return out or fallback


def _extract_source_symbol_hint(text: str) -> str | None:
    for token in re.findall(r"\b[A-Za-z]{2,12}\b", text):
        low = token.lower()
        if low in SOURCE_QUERY_STOPWORDS:
            continue
        return token.upper().lstrip("$")
    return None


def _is_source_query(text: str) -> bool:
    return bool(SOURCE_QUERY_RE.search(text or ""))


def _extract_action_symbol_hint(text: str) -> str | None:
    if is_likely_english_phrase(text):
        return None
    for token in re.findall(r"\b[A-Za-z]{2,12}\b", text):
        low = token.lower()
        if low in ACTION_SYMBOL_STOPWORDS:
            continue
        return token.upper().lstrip("$")
    return None


async def _remember_source_context(
    chat_id: int,
    *,
    source_line: str | None = None,
    exchange: str | None = None,
    market_kind: str | None = None,
    instrument_id: str | None = None,
    updated_at: str | None = None,
    symbol: str | None = None,
    context: str | None = None,
) -> None:
    if not any([source_line, exchange, market_kind, instrument_id]):
        return
    hub = _require_hub()
    payload = {
        "source_line": source_line or "",
        "exchange": exchange or "",
        "market_kind": market_kind or "",
        "instrument_id": instrument_id or "",
        "updated_at": updated_at or "",
        "symbol": symbol.upper() if isinstance(symbol, str) and symbol else "",
        "context": context or "",
    }
    await hub.cache.set_json(f"last_source:{chat_id}", payload, ttl=60 * 60 * 12)
    if payload["symbol"]:
        await hub.cache.set_json(f"last_source:{chat_id}:{payload['symbol']}", payload, ttl=60 * 60 * 12)


def _format_source_response(payload: dict | None) -> str:
    if not payload:
        return "I do not have a recent source to report yet."
    line = str(payload.get("source_line") or "").strip()
    if line:
        return f"Source: {line}"
    exchange = str(payload.get("exchange") or "").strip()
    market_kind = str(payload.get("market_kind") or "").strip()
    instrument_id = str(payload.get("instrument_id") or "").strip()
    updated = str(payload.get("updated_at") or "").strip()
    context = str(payload.get("context") or "").strip()
    parts = [p for p in [exchange, market_kind, instrument_id] if p]
    if not parts:
        return "I do not have a recent source to report yet."
    base = " ".join(parts)
    suffix = f" | Updated: {updated}" if updated else ""
    prefix = f"{context} source: " if context else "Source: "
    return f"{prefix}{base}{suffix}"


async def _source_reply_for_chat(chat_id: int, query_text: str) -> str:
    hub = _require_hub()
    symbol = _extract_source_symbol_hint(query_text)
    if symbol:
        per_symbol = await hub.cache.get_json(f"last_source:{chat_id}:{symbol}")
        if isinstance(per_symbol, dict):
            return _format_source_response(per_symbol)
        analysis = await hub.cache.get_json(f"last_analysis:{chat_id}:{symbol}")
        if isinstance(analysis, dict):
            line = str(analysis.get("data_source_line") or "").strip()
            if line:
                return f"{symbol} source: {line}"
    payload = await hub.cache.get_json(f"last_source:{chat_id}")
    if isinstance(payload, dict):
        return _format_source_response(payload)
    return "I do not have a recent source to report yet. Ask for analysis/chart first, then ask `source?`."


def _parse_tf_list(value, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        return fallback
    out = [x for x in items if x]
    return out or fallback


def _analysis_timeframes_from_settings(settings: dict) -> list[str]:
    if _settings.analysis_fast_mode:
        return _parse_tf_list(settings.get("preferred_timeframe", "1h"), ["1h"])
    return _parse_tf_list(settings.get("preferred_timeframes", settings.get("preferred_timeframe", "1h,4h")), ["1h", "4h"])


def _require_hub() -> ServiceHub:
    if _hub is None:
        raise RuntimeError("Handlers not initialized")
    return _hub


def _openai_chat_mode() -> str:
    mode = str(_settings.openai_chat_mode or "hybrid").strip().lower()
    if mode not in _ALLOWED_OPENAI_CHAT_MODES:
        return "hybrid"
    return mode


async def _get_chat_history(chat_id: int) -> list[dict[str, str]]:
    hub = _require_hub()
    payload = await hub.cache.get_json(f"llm:history:{chat_id}")
    if not isinstance(payload, list):
        return []
    out: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            out.append({"role": role, "content": content})
    return out


async def _append_chat_history(chat_id: int, role: str, content: str) -> None:
    role = role.strip().lower()
    content = content.strip()
    if role not in {"user", "assistant"} or not content:
        return
    history = await _get_chat_history(chat_id)
    history.append({"role": role, "content": content})
    turns = max(int(_settings.openai_chat_history_turns), 1)
    history = history[-(turns * 2) :]
    hub = _require_hub()
    await hub.cache.set_json(f"llm:history:{chat_id}", history, ttl=60 * 60 * 24 * 7)


def _analysis_context_payload(symbol: str, direction: str | None, payload: dict) -> dict:
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


async def _remember_analysis_context(chat_id: int, symbol: str, direction: str | None, payload: dict) -> None:
    hub = _require_hub()
    context = _analysis_context_payload(symbol, direction, payload)
    await hub.cache.set_json(f"last_analysis_context:{chat_id}", context, ttl=300)
    await hub.cache.set_json(f"last_analysis_context:{chat_id}:{symbol.upper()}", context, ttl=300)


async def _recent_analysis_context(chat_id: int) -> dict | None:
    hub = _require_hub()
    payload = await hub.cache.get_json(f"last_analysis_context:{chat_id}")
    return payload if isinstance(payload, dict) else None


def _looks_like_analysis_followup(text: str, context: dict | None) -> bool:
    cleaned = (text or "").strip()
    if not cleaned or not context:
        return False
    lower = cleaned.lower()
    symbol = str(context.get("symbol") or "").lower()
    if symbol and re.search(rf"\b{re.escape(symbol)}\b", lower) and FOLLOWUP_VALUE_RE.search(lower):
        return True
    if FOLLOWUP_RE.search(lower):
        return True
    if lower.endswith("?") and FOLLOWUP_VALUE_RE.search(lower):
        return True
    return False


async def _llm_analysis_reply(
    *,
    payload: dict,
    symbol: str,
    direction: str | None,
    chat_id: int | None,
) -> str | None:
    hub = _require_hub()
    if not hub.llm_client:
        return None

    market_context_text = str(payload.get("market_context_text") or "").strip()
    direction_label = (direction or payload.get("side") or "").strip().lower() or "none"
    prompt = (
        f"Analysis data for {symbol.upper()} ({direction_label} bias):\n"
        f"{json.dumps(payload, ensure_ascii=True, default=str)}\n\n"
        f"BTC/market backdrop: {market_context_text or 'not available'}\n\n"
        "Write this as Ghost would — start with current price and % change. "
        "Use the full indicator set in the data: SMA/EMA, RSI, MACD (and histogram), Bollinger Bands, "
        "support/resistance, Fibonacci levels, VWAP, Stochastic, ATR, OBV, ADX, and any candlestick patterns. "
        "Weave key levels and readings naturally in prose; mention macro if relevant. "
        "then give entry range / 3 targets / stop as simple plain lines. "
        "End with one short risk caveat (e.g. don't oversize, cut if structure breaks, wait for confirmation). All lowercase, casual trader voice. No HTML tags."
    )
    history = await _get_chat_history(chat_id) if chat_id is not None else []
    try:
        reply = await hub.llm_client.reply(
            prompt,
            history=history,
            max_output_tokens=min(max(int(_settings.openai_max_output_tokens), 400), 700),
            temperature=max(0.6, float(_settings.openai_temperature)),
        )
    except Exception:  # noqa: BLE001
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final and chat_id is not None:
        await _append_chat_history(chat_id, "user", f"{symbol.upper()} {(direction or payload.get('side') or '').strip()} analysis")
        await _append_chat_history(chat_id, "assistant", final)
    return final


async def _llm_followup_reply(
    user_text: str,
    context: dict,
    *,
    chat_id: int,
) -> str | None:
    hub = _require_hub()
    if not hub.llm_client:
        return None

    cleaned = (user_text or "").strip()
    if not cleaned:
        return None

    prompt = (
        "You are replying to a follow-up message about a recent trade setup.\n"
        f"Last analysis context JSON: {json.dumps(context, ensure_ascii=True, default=str)}\n"
        f"User follow-up: {cleaned}\n"
        "Treat this as continuation of the same setup, not a fresh full report.\n"
        "If the proposed SL/entry/leverage is weak, say it directly and suggest a better level.\n"
        "Keep it conversational and concise."
    )
    history = await _get_chat_history(chat_id)
    try:
        reply = await hub.llm_client.reply(prompt, history=history, max_output_tokens=220)
    except Exception:  # noqa: BLE001
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final:
        await _append_chat_history(chat_id, "user", cleaned)
        await _append_chat_history(chat_id, "assistant", final)
    return final


async def _render_analysis_text(
    *,
    payload: dict,
    symbol: str,
    direction: str | None,
    settings: dict,
    chat_id: int,
    detailed: bool = False,
) -> str:
    try:
        return await fred.format_as_ghost(payload)
    except Exception:  # noqa: BLE001
        llm_text = await _llm_analysis_reply(
            payload=payload,
            symbol=symbol,
            direction=direction,
            chat_id=chat_id,
        )
        if llm_text:
            return llm_text
        return trade_plan_template(payload, settings, detailed=detailed)


async def _send_ghost_analysis(message: Message, symbol: str, text: str, direction: str | None = None) -> None:
    with suppress(Exception):
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    await asyncio.sleep(1.3)
    await message.answer(text, reply_markup=analysis_progressive_menu(symbol, direction))


def _define_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Analyze 1h", callback_data="define:analyze:1h"),
                InlineKeyboardButton(text="Analyze 4h", callback_data="define:analyze:4h"),
            ],
            [
                InlineKeyboardButton(text="Chart 1h", callback_data="define:chart:1h"),
                InlineKeyboardButton(text="Heatmap", callback_data="define:heatmap"),
            ],
            [
                InlineKeyboardButton(text="Set Alert", callback_data="define:alert"),
                InlineKeyboardButton(text="Top Overbought 1h", callback_data="top:overbought:1h"),
            ],
            [
                InlineKeyboardButton(text="Top Oversold 1h", callback_data="top:oversold:1h"),
                InlineKeyboardButton(text="Latest News", callback_data="define:news"),
            ],
        ]
    )


def _mentions_bot(text: str, bot_username: str | None) -> bool:
    if not bot_username:
        return False
    return f"@{bot_username.lower()}" in text.lower()


def _strip_bot_mention(text: str, bot_username: str | None) -> str:
    if not bot_username:
        return text
    return re.sub(rf"@{re.escape(bot_username)}", "", text, flags=re.IGNORECASE).strip()


def _is_reply_to_bot(message: Message, hub: ServiceHub) -> bool:
    reply = message.reply_to_message
    if not reply or not reply.from_user:
        return False
    return bool(reply.from_user.id == hub.bot.id)


async def _group_free_talk_enabled(chat_id: int) -> bool:
    hub = _require_hub()
    payload = await hub.cache.get_json(f"group:free_talk:{chat_id}")
    return bool(payload and payload.get("enabled"))


async def _set_group_free_talk(chat_id: int, enabled: bool) -> None:
    hub = _require_hub()
    await hub.cache.set_json(f"group:free_talk:{chat_id}", {"enabled": bool(enabled)}, ttl=60 * 60 * 24 * 365)


def _looks_like_clear_intent(text: str) -> bool:
    parsed = parse_message(text)
    return parsed.intent not in {Intent.UNKNOWN, Intent.SMALLTALK, Intent.HELP, Intent.START}


async def _is_group_admin(message: Message) -> bool:
    hub = _require_hub()
    if not message.from_user:
        return False
    if int(message.from_user.id) in set(_settings.admin_ids_list()):
        return True
    try:
        member = await hub.bot.get_chat_member(message.chat.id, message.from_user.id)
        return getattr(member, "status", "") in {"administrator", "creator"}
    except Exception:  # noqa: BLE001
        return False


async def _check_req_limit(chat_id: int) -> bool:
    hub = _require_hub()
    try:
        result = await hub.rate_limiter.check(
            key=f"rl:req:{chat_id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
            limit=_settings.request_rate_limit_per_minute,
            window_seconds=60,
        )
        return result.allowed
    except Exception:  # noqa: BLE001
        logger.exception("rate_limit_check_error", extra={"event": "rate_limit_check_error", "chat_id": chat_id})
        return True


async def _llm_fallback_reply(user_text: str, settings: dict | None = None, chat_id: int | None = None) -> str | None:
    hub = _require_hub()
    if not hub.llm_client:
        return None

    cleaned = (user_text or "").strip()
    if not cleaned:
        return None

    style = "wild"
    if settings:
        if settings.get("formal_mode"):
            style = "formal"
        else:
            style = str(settings.get("tone_mode", "wild")).lower()

    comm_memory = _build_communication_memory_block(settings)
    prompt = (
        f"{comm_memory}\n\n"
        "GHOST PERSONA (use only when they ask who you are, your personality, or what you're like):\n"
        "You are Ghost — sharp, no-nonsense, crypto-native. You tell the truth even when it wrecks a bag. "
        "Slightly unhinged and unfiltered; you don't sugarcoat. You help with levels, risk, and setups but you're not a cheerleader. "
        "One short paragraph max; match their tone (friendly or formal from settings).\n\n"
        f"User message: {cleaned}\n\n"
        "Answer exactly what was asked. Do not divert to a different topic. Same question → same type of answer.\n"
        "STRUCTURE: One-line summary first, then key points. CONFIDENCE: When uncertain add '(low confidence)'.\n"
        "LIMITS: If out of scope (tax/legal/guarantees), say so in one sentence and suggest what you can do.\n"
        "Do NOT add a 'coins to watch' or 'coins to watch right now' section unless they explicitly ask for a watchlist or what to watch. Never use that phrase as a heading otherwise.\n"
        "NEVER tie metals (gold, XAU, silver) to crypto — no inverse/safe-haven links, no predicting metals from BTC.\n\n"
        "BOT CAPABILITIES (only when they ask how to use the bot):\n"
        "- Alerts: 'alert BTC 100000 above' or Create Alert button\n"
        "- Analysis: 'BTC long' or 'ETH short 4h'\n"
        "- Watchlist: 'coins to watch', 'top movers'\n"
        "- News: 'latest crypto news'\n"
        "- Price: /price BTC\n\n"
        "TRADING FRAMEWORK (only when they ask about trading concepts, strategy, or risk — do not list headings unless they explicitly ask):\n"
        "- Market structure: trends vs ranges, breakouts, liquidity.\n"
        "- Support/resistance and supply/demand zones.\n"
        "- Risk management: position sizing, risk/reward, stop losses, max drawdown rules.\n"
        "- Trading psychology: discipline, emotional control, avoiding FOMO/revenge trading.\n"
        "- Volatility: calm vs explosive conditions and when to press vs chill.\n"
        "- Order types & execution: market/limit/stop, slippage, spreads.\n"
        "- Timeframes & top-down analysis: higher TF bias, lower TF entries.\n"
        "- Trade planning: setups, entry/exit rules, invalidation.\n"
        "- Journaling & review: tracking stats, finding what works.\n"
        "- Backtesting & forward testing: proof before size.\n"
        "- Fundamentals & catalysts: news, macro, tokenomics, unlocks.\n"
        "- On-chain basics: flows, exchange reserves, whales, stablecoin liquidity.\n"
        "- Liquidity & order flow concepts: where stops sit, imbalances.\n"
        "- Correlation awareness: BTC dominance, alt correlation, equities/DXY.\n"
        "- Market regimes: bull, bear, chop; adapting strategy.\n"
        "- Security & operations: wallet safety, avoiding scams, basic ops hygiene.\n"
    )
    history = await _get_chat_history(chat_id) if chat_id is not None else []
    try:
        reply = await hub.llm_client.reply(prompt, history=history)
    except Exception:  # noqa: BLE001
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final and chat_id is not None:
        await _append_chat_history(chat_id, "user", cleaned)
        await _append_chat_history(chat_id, "assistant", final)
    return final


_MARKET_QUESTION_RE = re.compile(
    r"\b(pump|dump|moon|rug|rekt|bleed|crash|rally|bull|bear|market|btc|bitcoin|eth|ethereum|"
    r"crypto|price|move|movement|run|drop|dip|bounce|trend|happening|why|explain|think|feel|"
    r"going|direction|outlook|setup|narrative|sentiment|vibe|catalys|news|macro|tariff|"
    r"inflation|rate|fed|fomc|cpi|pce|blackrock|etf|liquidat|funding|dominan|"
    # expanded: general "which coin / what to watch / what to buy" questions
    r"coin|coins|token|tokens|alt|alts|altcoin|gem|gems|pick|picks|"
    r"watch|looking|look|buy|sell|trade|play|plays|long|short|"
    r"which|what|best|top|good|strong|weak|hot|cold|"
    r"portfolio|invest|hold|accumulate|dca|entry|exit|"
    r"solana|sol|bnb|xrp|matic|avax|ada|dot|link|doge|shib|pepe|"
    r"layer|defi|nft|meme|perp|spot|futures|leverage|"
    r"dominance|volume|liquidity|whale|orderbook|ob|candle|chart|"
    r"resistance|support|ema|rsi|macd|bollinger)\b",
    re.IGNORECASE,
)

# Questions about the bot itself — must never be treated as market questions
_BOT_META_RE = re.compile(
    r"\b(alert\s+creat|creat\s+alert|alert\s+not|button\s+not|command\s+not|"
    r"bot\s+not|bot\s+down|not\s+working|isn'?t\s+work|not\s+respond|"
    r"why\s+is\s+(the\s+)?(alert|button|command|bot|feature)|"
    r"how\s+do\s+i\s+(create|set|use|make)|what\s+commands|what\s+can\s+you|"
    r"how\s+to\s+(create|set|use)|are\s+you\s+working|still\s+(not\s+)?work|"
    r"failing|broken|feature\s+not|doesn'?t\s+work|"
    r"who\s+are\s+you|what('s|\s+is)\s+your\s+personality|your\s+personality|"
    r"personality\s+like|whats\s+your\s+personality|what\s+are\s+you|describe\s+yourself)\b",
    re.IGNORECASE,
)


def _looks_like_market_question(text: str) -> bool:
    if not text.strip():
        return False
    # Bot-meta questions must never be routed to the market chat handler
    if _BOT_META_RE.search(text):
        return False
    # Single-word checks that strongly indicate crypto intent without needing a keyword match
    lower = text.lower()
    crypto_intent_phrases = (
        "which coin", "what coin", "best coin", "top coin",
        "what to buy", "what should i buy", "should i buy",
        "what to watch", "what should i watch", "worth watching",
        "worth buying", "worth trading", "what to trade",
        "good trade", "good play", "where is the market",
        "how is the market", "what is happening", "what's happening",
        "what happened", "what do you think", "give me a call",
        "market update", "quick update", "anything good",
    )
    if any(phrase in lower for phrase in crypto_intent_phrases):
        return True
    return bool(_MARKET_QUESTION_RE.search(text))


def _is_definition_question(text: str) -> bool:
    """True if the user is asking for a concept/definition (e.g. what is SMC, define FVG)."""
    if not text or len(text) > 200:
        return False
    lower = text.strip().lower()
    if lower.startswith(("what is ", "what's ", "what are ", "define ", "explain ", "meaning of ")):
        return True
    if " what is " in lower or " define " in lower or " explain " in lower:
        return True
    return False


# Keywords that suggest the user wants coin fundamentals (stats, links, about, fear/greed).
_FUNDAMENTALS_KEYWORDS = (
    "info", "fundamentals", "market cap", "marketcap", "ath", "atl", "all time high", "all time low",
    "supply", "circulating", "max supply", "total supply", "fdv", "diluted", "valuation",
    "website", "explorer", "explorers", "links", "social", "twitter", "telegram", "reddit",
    "about the coin", "about this coin", "what is this coin", "fear", "greed", "fear and greed",
    "treasury", "volume", "trading volume", "market cap/fdv", "mc/fdv",
)


def _wants_fundamentals(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    lower = text.strip().lower()
    return any(k in lower for k in _FUNDAMENTALS_KEYWORDS)


def _extract_symbol_for_fundamentals(text: str, last_symbols: list) -> str | None:
    """Get a single symbol from message or last_symbols for fundamentals lookup."""
    try:
        from app.core.nlu import _extract_symbols
        syms = _extract_symbols(text)
        if syms:
            base = getattr(syms[0], "base", None) or (syms[0] if isinstance(syms[0], str) else None)
            if base:
                return str(base).upper()
    except Exception:  # noqa: BLE001
        pass
    if last_symbols:
        return str(last_symbols[0]).upper().strip() if last_symbols else None
    return None


def _format_coin_fundamentals_block(info: dict | None, fear_greed: dict | None) -> str:
    """Format coin info + Fear & Greed as a text block for the LLM (use when relevant, don't dump every time)."""
    lines = ["<b>Coin fundamentals (use only when user asks for stats, links, or about):</b>"]
    if not info:
        lines.append("(No fundamentals data for this symbol.)")
    else:
        def _fmt_num(x):  # noqa: B903
            if x is None:
                return "—"
            if x >= 1e12:
                return f"{x / 1e12:.2f}T"
            if x >= 1e9:
                return f"{x / 1e9:.2f}B"
            if x >= 1e6:
                return f"{x / 1e6:.2f}M"
            if x >= 1e3:
                return f"{x / 1e3:.2f}k"
            return f"{x:,.2f}"

        name = info.get("name") or info.get("symbol") or "—"
        lines.append(f"Name: {name} ({info.get('symbol', '')})")
        if info.get("high_24h") is not None or info.get("low_24h") is not None:
            lines.append(f"24H high: {_fmt_num(info.get('high_24h'))} | 24H low: {_fmt_num(info.get('low_24h'))}")
        if info.get("ath") is not None or info.get("atl") is not None:
            lines.append(f"ATH: {_fmt_num(info.get('ath'))} | ATL: {_fmt_num(info.get('atl'))}")
        if info.get("market_cap") is not None:
            lines.append(f"Market cap: ${_fmt_num(info.get('market_cap'))}")
        if info.get("circulating_supply") is not None:
            lines.append(f"Circulating supply: {_fmt_num(info.get('circulating_supply'))}")
        if info.get("total_supply") is not None:
            lines.append(f"Total supply: {_fmt_num(info.get('total_supply'))}")
        if info.get("max_supply") is not None:
            lines.append(f"Max supply: {_fmt_num(info.get('max_supply'))}")
        if info.get("fdv") is not None:
            lines.append(f"FDV: ${_fmt_num(info.get('fdv'))}")
        if info.get("market_cap_fdv_ratio") is not None:
            lines.append(f"Market cap / FDV ratio: {info.get('market_cap_fdv_ratio')}")
        if info.get("total_volume") is not None:
            lines.append(f"Trading volume (24h): ${_fmt_num(info.get('total_volume'))}")
        if info.get("website"):
            lines.append(f"Website: {info.get('website')}")
        if info.get("explorers"):
            lines.append("Explorers: " + ", ".join(info.get("explorers", [])[:3]))
        if info.get("social"):
            social = info.get("social", {})
            parts = [f"{k}: {v}" for k, v in list(social.items())[:4]]
            if parts:
                lines.append("Social: " + " | ".join(parts))
        if info.get("about"):
            about = (info.get("about") or "")[:800].strip()
            if about:
                lines.append(f"About: {about}")
    if fear_greed:
        v = fear_greed.get("value")
        c = fear_greed.get("classification")
        if v is not None or c:
            lines.append(f"Crypto Fear & Greed Index: {v} ({c})" if (v is not None and c) else f"Fear & Greed: {c or v}")
    return "\n".join(lines)


async def _llm_market_chat_reply(
    user_text: str,
    settings: dict | None = None,
    chat_id: int | None = None,
) -> str | None:
    """Answer open-ended market questions by injecting live price + news context."""
    hub = _require_hub()
    if not hub.llm_client:
        return None

    cleaned = (user_text or "").strip()
    if not cleaned:
        return None

    style = "wild"
    if settings:
        style = "formal" if settings.get("formal_mode") else str(settings.get("tone_mode", "wild")).lower()

    # Fetch live market snapshot + recent headlines in parallel
    mkt_ctx: dict = {}
    news_headlines: list[dict] = []
    try:
        mkt_ctx, news_payload = await asyncio.gather(
            hub.analysis_service.get_market_context(),
            hub.news_service.get_digest(mode="crypto", limit=6),
            return_exceptions=True,
        )
        if isinstance(mkt_ctx, Exception):
            mkt_ctx = {}
        if isinstance(news_payload, Exception):
            news_payload = {}
        if isinstance(news_payload, dict):
            news_headlines = news_payload.get("headlines") or []
    except Exception:  # noqa: BLE001
        pass

    mkt_text = format_market_context(mkt_ctx) if mkt_ctx else ""
    news_lines = "\n".join(
        f"- {h.get('title', '')} ({h.get('source', '')})"
        for h in news_headlines[:6]
        if h.get("title")
    )

    context_block = ""
    if _is_definition_question(cleaned):
        context_block += "This is a definition/knowledge question. Answer from trading knowledge only. Do not ask for ticker or timeframe.\n\n"
    if mkt_text:
        context_block += f"Live market snapshot (BTC, ETH, SOL): {mkt_text}\n"
    else:
        context_block += "Live market snapshot: not available.\n"
    if news_lines:
        context_block += f"Recent crypto news:\n{news_lines}\n"

    # If user asked about a specific coin (e.g. POWER, POWERUSDT), try to fetch price from exchanges (Bybit, etc.)
    last_syms = list((settings or {}).get("last_symbols") or [])[:5]
    asked_symbol = _extract_symbol_for_fundamentals(cleaned, last_syms)
    if asked_symbol and asked_symbol not in ("BTC", "ETH", "SOL"):
        try:
            quote = await hub.market_router.get_price(asked_symbol)
            if quote and float(quote.get("price") or 0) > 0:
                price = float(quote["price"])
                source = str(quote.get("source_line") or quote.get("source") or "exchange")
                context_block += f"\nRequested symbol {asked_symbol}: ${price:,.2f} (from {source}). You have data for this symbol — use it to answer; do not say you only have BTC/ETH/SOL.\n"
        except Exception:  # noqa: BLE001
            pass

    if _wants_fundamentals(cleaned):
        last_syms = list((settings or {}).get("last_symbols") or [])[:5]
        symbol_for_info = _extract_symbol_for_fundamentals(cleaned, last_syms)
        if symbol_for_info and getattr(hub, "coin_info_service", None):
            try:
                info_task = hub.coin_info_service.get_coin_info(symbol_for_info)
                fg_task = hub.coin_info_service.get_fear_greed()
                info, fear_greed = await asyncio.gather(info_task, fg_task, return_exceptions=True)
                if isinstance(info, Exception):
                    info = None
                if isinstance(fear_greed, Exception):
                    fear_greed = None
                fundamentals_block = _format_coin_fundamentals_block(info, fear_greed)
                context_block += "\n\n" + fundamentals_block + "\n"
            except Exception:  # noqa: BLE001
                pass

    ultra = (settings or {}).get("ultra_brief")
    length_rule = (
        "Answer in one short sentence only."
        if ultra
        else "Match length to the question: short question → short answer; open-ended → fuller answer. Paragraphs 3–4 sentences max. Use \"fren\" or \"anon\" once, not every sentence."
    )
    comm_memory = _build_communication_memory_block(settings)
    prompt = (
        f"{context_block}\n"
        f"{comm_memory}\n\n"
        "GHOST PERSONA (only when they ask who you are or your personality): "
        "You are Ghost — sharp, no-nonsense, crypto-native. You tell the truth even when it hurts. Slightly unhinged, unfiltered; not a cheerleader. One short paragraph max.\n\n"
        "RULES:\n"
        "- The snapshot above has BTC, ETH, SOL. If 'Requested symbol X' data is provided above, you have price data for that symbol (e.g. from Bybit) — use it and do not say you only have BTC/ETH/SOL. "
        "Only say \"I don't have recent data for [symbol]\" when that symbol is NOT in the snapshot and NOT listed as 'Requested symbol'. Do not substitute BTC/ETH data for another symbol.\n"
        "- NEVER tie metals to crypto. Do not say gold/XAU moves inverse to crypto, or is a safe haven when crypto falls, or predict metals from BTC. If they ask about gold or XAU, say you don't link metals to crypto and don't speculate on that; keep the answer short and do not add a 'coins to watch' with XAU context.\n"
        "- If the user says \"outdated price\" or \"old data\" or \"prices are wrong\": acknowledge it in one short sentence. "
        "Then either say the snapshot above is the freshest you have, or that you don't have newer data. "
        "Do NOT ignore the comment and launch into a long analysis. Answer what they said first.\n"
        "- Otherwise: answer exactly what was asked. Do not divert. Same question later → same kind of answer (with updated data).\n"
        f"- {length_rule}\n"
        "- STRUCTURE: One-line summary first, then key points/steps, then optional details. Use line breaks.\n"
        "- CONFIDENCE: When uncertain or extrapolating, add \"Likely\" or \"(low confidence)\". When data-backed, no need to label.\n"
        "- NEXT ACTION: End with one short suggested next step or question when helpful (e.g. \"Want a chart for that?\" \"Set an alert?\"). Don't force every time.\n"
        "- LIMITS: If the request is out of scope (tax/legal advice, guaranteed outcomes), say so in one sentence and suggest what you can do instead.\n"
        "- For complex or multi-part requests, start with \"You want: [one-line summary].\" then answer.\n"
        "- Do NOT add any section titled \"coins to watch\" or \"coins to watch right now\" (with or without entry/targets) when the user asks for a general market snapshot, daily overview, \"what are we looking at\", personality/perspectives, or any single-asset question. Only give that section when they explicitly ask for a watchlist, \"what to watch\", \"what to buy\", or \"coins to watch\". Never use the phrase \"coins to watch right now\" as a heading unless they asked for a watchlist.\n"
        "- If they ask how you were built, who made you, your tech/architecture/codebase/APIs: do NOT answer. One short deflect only (e.g. \"that's classified, anon\" or \"i just read the charts, fren\"). No details.\n"
        "- When a 'Coin fundamentals' block is present above, use it only to answer questions about that coin's stats (24h high/low, ATH/ATL, market cap, supply, FDV, volume), website, explorers, social links, about, or Fear & Greed. Do not dump the whole block; answer what they asked. If they asked for a full overview, you can summarize key points.\n\n"
        "TRADING FRAMEWORK (use only when relevant; do not list headings unless the user explicitly asks):\n"
        "- Market structure: trend vs range, breakouts, key liquidity areas.\n"
        "- Levels: major support/resistance and supply/demand zones.\n"
        "- Risk: position sizing, risk/reward, stop placement, max drawdown awareness.\n"
        "- Psychology: discipline, emotional control, avoiding FOMO/revenge trading.\n"
        "- Volatility & regime: calm vs explosive conditions; bull, bear, and chop adjustment.\n"
        "- Execution: order types (market/limit/stop), slippage/spreads, and using higher timeframes for bias with lower timeframes for entries.\n"
        "- Trade plan & review: clear setup, entry/exit rules, invalidation, journaling, backtesting/forward testing before size.\n"
        "- Context: fundamentals/catalysts (news, macro, unlocks), on-chain/liquidity/flows, correlations/BTC dominance/indices, and basic security/ops.\n\n"
        f"User question: {cleaned}\n\n"
        "Telegram HTML: <b>bold</b> for coins and key levels, <i>italic</i> for closing line."
    )

    history = await _get_chat_history(chat_id) if chat_id is not None else []
    try:
        reply = await hub.llm_client.reply(
            prompt,
            history=history,
            max_output_tokens=min(max(int(_settings.openai_max_output_tokens), 600), 1000),
            temperature=max(0.5, float(_settings.openai_temperature)),
        )
    except Exception:  # noqa: BLE001
        return None

    final = reply.strip() if reply and reply.strip() else None
    if final and chat_id is not None:
        await _append_chat_history(chat_id, "user", cleaned)
        await _append_chat_history(chat_id, "assistant", final)
    return final


def _parse_duration_to_seconds(raw: str) -> int | None:
    m = re.match(r"^\s*(\d+)\s*([smhd])\s*$", raw.lower())
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return value * mult


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def _as_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _as_float_list(value) -> list[float]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = re.findall(r"[0-9]+(?:\.[0-9]+)?", value)
    else:
        raw = []
    out: list[float] = []
    for item in raw:
        v = _as_float(item)
        if v is not None:
            out.append(float(v))
    return out


def _infer_direction(entry: float, targets: list[float], explicit: str | None) -> str:
    side = (explicit or "").strip().lower()
    if side in {"long", "short"}:
        return side
    if not targets:
        return "long"
    return "long" if float(targets[0]) >= float(entry) else "short"


def _trade_math_payload(
    *,
    entry: float,
    stop: float,
    targets: list[float],
    direction: str | None,
    margin_usd: float | None,
    leverage: float | None,
    symbol: str | None = None,
) -> dict:
    e = float(entry)
    s = float(stop)
    tps = [float(x) for x in targets if float(x) > 0]
    if not tps:
        raise RuntimeError("Need at least one target.")
    risk = abs(e - s)
    if risk <= 0:
        raise RuntimeError("Entry and stop cannot be the same.")
    side = _infer_direction(e, tps, direction)

    rows: list[dict] = []
    for tp in tps:
        reward = (tp - e) if side == "long" else (e - tp)
        r_mult = reward / risk
        rows.append({"tp": round(tp, 8), "r_multiple": round(r_mult, 3)})
    best_r = max(row["r_multiple"] for row in rows)

    payload: dict = {
        "symbol": symbol or "",
        "direction": side,
        "entry": round(e, 8),
        "stop": round(s, 8),
        "targets": [round(x, 8) for x in tps],
        "risk_per_unit": round(risk, 8),
        "rows": rows,
        "best_r": round(best_r, 3),
    }

    if margin_usd and leverage and margin_usd > 0 and leverage > 0:
        notional = float(margin_usd) * float(leverage)
        qty = notional / e

        def _pnl(exit_price: float) -> float:
            if side == "long":
                return (exit_price - e) * qty
            return (e - exit_price) * qty

        payload["position"] = {
            "margin_usd": round(float(margin_usd), 2),
            "leverage": round(float(leverage), 2),
            "notional_usd": round(notional, 2),
            "qty": round(qty, 8),
            "stop_pnl_usd": round(_pnl(s), 2),
            "tp_pnls": [{"tp": round(tp, 8), "pnl_usd": round(_pnl(tp), 2)} for tp in tps],
        }

    return payload


def _extract_symbol(params: dict) -> str | None:
    for key in ("symbol", "asset", "ticker", "coin"):
        val = params.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().upper().lstrip("$")
    return None


async def _llm_route_message(user_text: str) -> dict | None:
    hub = _require_hub()
    if not hub.llm_client:
        return None
    try:
        payload = await hub.llm_client.route_message(user_text)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    return payload


_ALERT_SHORTCUT_RE = re.compile(
    r"^alert\s+([A-Za-z0-9]{2,20})\s+([\d.,]+[kKmM]?)\s*(above|below|cross|crosses|over|under)?\s*$",
    re.IGNORECASE,
)


async def _dispatch_command_text(message: Message, synthetic_text: str) -> bool:
    hub = _require_hub()
    chat_id = message.chat.id
    settings = await hub.user_service.get_settings(chat_id)

    # Fast-path: "alert {symbol} {price} [condition]" — bypass NLU to avoid LLM confusion
    _am = _ALERT_SHORTCUT_RE.match(synthetic_text.strip())
    if _am:
        from app.core.nlu import _extract_prices  # already imported elsewhere but safe to reimport
        sym = _am.group(1).upper()
        raw_price_str = _am.group(2)
        raw_cond = (_am.group(3) or "cross").lower()
        prices = _extract_prices(raw_price_str)
        px = prices[0] if prices else None
        if sym and px is not None:
            cond = "above" if raw_cond in ("above", "over") else ("below" if raw_cond in ("below", "under") else "cross")
            try:
                alert = await hub.alerts_service.create_alert(chat_id, sym, cond, float(px))
                _cond_word = {"above": "crosses above", "below": "crosses below"}.get(cond, "crosses")
                await message.answer(
                    f"🔔 alert set — <b>{alert.symbol}</b> {_cond_word} <b>${float(px):,.2f}</b>.\n"
                    "i'll ping you the moment it hits. don't get liquidated.",
                    reply_markup=alert_created_menu(alert.symbol),
                )
            except RuntimeError as exc:
                await message.answer(f"couldn't set that alert — {_safe_exc(exc)}")
            except Exception:  # noqa: BLE001
                await message.answer("alert creation failed. try again.")
            return True

    parsed = parse_message(synthetic_text)

    if parsed.requires_followup:
        if parsed.intent == Intent.ANALYSIS and not parsed.entities.get("symbol"):
            kb = simple_followup(
                [
                    ("BTC", "quick:analysis:BTC"),
                    ("ETH", "quick:analysis:ETH"),
                    ("SOL", "quick:analysis:SOL"),
                ]
            )
            await message.answer(parsed.followup_question or "Need one detail.", reply_markup=kb)
            return True
        await message.answer(
            parsed.followup_question or clarifying_question(_extract_action_symbol_hint(message.text or "")),
            reply_markup=smart_action_menu(),
        )
        return True

    if await _handle_parsed_intent(message, parsed, settings):
        return True

    llm_reply = await _llm_fallback_reply(synthetic_text, settings, chat_id=chat_id)
    if llm_reply:
        await _send_llm_reply(message, llm_reply, settings, user_message=synthetic_text)
        return True
    return False


async def _handle_routed_intent(message: Message, settings: dict, route: dict) -> bool:
    hub = _require_hub()
    intent = str(route.get("intent", "")).strip().lower()
    try:
        confidence = float(route.get("confidence", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        confidence = 0.0
    params = route.get("params") if isinstance(route.get("params"), dict) else {}
    chat_id = message.chat.id
    raw_text = message.text or ""

    if confidence < _settings.openai_router_min_confidence:
        return False

    if intent in {"smalltalk", "market_chat", "general_chat"}:
        with suppress(Exception):
            await message.bot.send_chat_action(chat_id, ChatAction.TYPING)
        # Always use live-data path — Claude/Grok with market context answers everything better
        llm_reply = await _llm_market_chat_reply(raw_text, settings, chat_id=chat_id)
        if llm_reply:
            await _send_llm_reply(message, llm_reply, settings)
        return True
        # Bot-meta questions (how-to, features) fall back to plain reply
        if _BOT_META_RE.search(raw_text):
            llm_reply = await _llm_fallback_reply(raw_text, settings, chat_id=chat_id)
            await _send_llm_reply(
                message, llm_reply or smalltalk_reply(settings), settings, user_message=raw_text
            )
            return True
        return False

    if intent == "news_digest":
        limit = max(3, min(_as_int(params.get("limit"), 6), 10))
        topic = params.get("topic")
        symbol_param = params.get("symbol") or topic
        # News by asset: if topic/symbol looks like a ticker, use asset headlines
        if isinstance(symbol_param, str) and 2 <= len(symbol_param.strip()) <= 10 and symbol_param.strip().isalnum():
            ticker = symbol_param.strip().upper()
            headlines = await hub.news_service.get_asset_headlines(ticker, limit=limit)
            if headlines:
                lines = [f"<b>News for {ticker}</b>", ""]
                for h in headlines:
                    title = (h.get("title") or "")[:120]
                    url = h.get("url") or ""
                    src = h.get("source") or ""
                    lines.append(f"• {title}" + (f" ({src})" if src else "") + (f"\n  {url}" if url else ""))
                await message.answer("\n".join(lines), disable_web_page_preview=True)
            else:
                await message.answer(f"No recent headlines for <b>{ticker}</b>. Try general /news.")
            return True
        mode = str(params.get("mode") or "crypto").strip().lower() or "crypto"
        if isinstance(topic, str) and topic.strip().lower() == "openai" and mode == "crypto":
            mode = "openai"
        payload = await hub.news_service.get_digest(
            topic=topic if isinstance(topic, str) else None,
            mode=mode,
            limit=limit,
        )
        heads = payload.get("headlines") if isinstance(payload, dict) else None
        head = heads[0] if isinstance(heads, list) and heads else {}
        await _remember_source_context(
            chat_id,
            source_line=f"{head.get('source', 'news feed')} | {head.get('url', '')}".strip(),
            context="news",
        )
        await message.answer(news_template(payload), parse_mode="HTML")
        return True

    if intent in {"watch_asset", "market_analysis"}:
        symbol = _extract_symbol(params)
        if not symbol:
            await message.answer("Which coin should I analyze?")
            return True
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        side = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        settings_tfs = _analysis_timeframes_from_settings(settings)
        settings_emas = _parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200])
        settings_rsis = _parse_int_list(settings.get("preferred_rsi_periods", [14]), [14])
        payload = await hub.analysis_service.analyze(
            symbol,
            direction=side if side in {"long", "short"} else None,
            timeframe=timeframe,
            timeframes=[timeframe] if timeframe else settings_tfs,
            ema_periods=settings_emas,
            rsi_periods=settings_rsis,
            include_derivatives=bool(params.get("include_derivatives") or params.get("derivatives")),
            include_news=bool(params.get("include_news") or params.get("news") or params.get("catalysts")),
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _append_last_symbol(chat_id, symbol)
        await _remember_analysis_context(chat_id, symbol, side, payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=side,
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(message, symbol, analysis_text, direction=side)
        return True

    if intent == "rsi_scan":
        symbol = _extract_symbol(params)
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        mode_raw = str(params.get("mode", "oversold")).strip().lower()
        mode = "overbought" if mode_raw == "overbought" else "oversold"
        limit = max(1, min(_as_int(params.get("limit"), 10), 20))
        rsi_length = max(2, min(_as_int(params.get("rsi_length"), 14), 50))
        payload = await hub.rsi_scanner_service.scan(
            timeframe=timeframe,
            mode=mode,
            limit=limit,
            rsi_length=rsi_length,
            symbol=symbol,
        )
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            symbol=symbol,
            context="rsi scan",
        )
        await message.answer(rsi_scan_template(payload))
        return True

    if intent == "ema_scan":
        timeframe = str(params.get("timeframe", "4h")).strip() or "4h"
        ema_length = max(2, min(_as_int(params.get("ema_length"), 200), 500))
        mode_raw = str(params.get("mode", "closest")).strip().lower()
        mode = mode_raw if mode_raw in {"closest", "above", "below"} else "closest"
        limit = max(1, min(_as_int(params.get("limit"), 10), 20))
        payload = await hub.ema_scanner_service.scan(
            timeframe=timeframe,
            ema_length=ema_length,
            mode=mode,
            limit=limit,
        )
        lines = [payload["summary"], ""]
        for idx, row in enumerate(payload.get("items", []), start=1):
            lines.append(
                f"{idx}. {row['symbol']} price {row['price']} | EMA{payload['ema_length']} {row['ema']} | "
                f"dist {row['distance_pct']}% ({row['side']})"
            )
        if not payload.get("items"):
            lines.append("No EMA matches right now.")
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="ema scan",
        )
        await message.answer("\n".join(lines))
        return True

    if intent == "chart":
        symbol = _extract_symbol(params)
        if not symbol:
            await message.answer("Which symbol should I chart?")
            return True
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        img, meta = await hub.chart_service.render_chart(symbol=symbol, timeframe=timeframe)
        caption = f"{symbol} {timeframe} chart."
        await _remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("instrument_id") or ""),
            updated_at=str(meta.get("updated_at") or ""),
            symbol=symbol,
            context="chart",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_{timeframe}_chart.png"),
            caption=caption,
        )
        return True

    if intent == "heatmap":
        symbol = _extract_symbol(params) or "BTC"
        img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
        caption = (
            f"{meta['pair']} orderbook heatmap\n"
            f"Best bid: {meta['best_bid']:.6f} | Best ask: {meta['best_ask']:.6f}\n"
            f"Bid wall: {meta['bid_wall']:.6f} | Ask wall: {meta['ask_wall']:.6f}"
        )
        await _remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("pair") or ""),
            symbol=symbol,
            context="heatmap",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_heatmap.png"),
            caption=caption,
        )
        return True

    if intent == "watchlist":
        count = max(1, min(_as_int(params.get("count"), 5), 20))
        direction_raw = str(params.get("direction") or "").strip().lower()
        direction = direction_raw if direction_raw in {"long", "short"} else None
        payload = await hub.watchlist_service.build_watchlist(count=count, direction=direction)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="watchlist",
        )
        await message.answer(watchlist_template(payload))
        return True

    if intent == "alert_create":
        symbol = _extract_symbol(params)
        price = _as_float(params.get("price") or params.get("target_price"))

        # Fallback: parse symbol/price directly from the raw text if router missed them
        if not symbol or price is None:
            from app.core.nlu import _extract_symbols, _extract_prices
            if not symbol:
                syms = _extract_symbols(raw_text)
                symbol = syms[0] if syms else None
            if price is None:
                pxs = _extract_prices(raw_text)
                price = pxs[0] if pxs else None

        if not symbol or price is None:
            await message.answer("Need symbol and price — e.g. <code>alert BTC 66k</code> or <code>set alert for SOL 200</code>.")
            return True
        op = str(params.get("operator") or params.get("condition") or "cross").strip().lower()
        if op in {">", ">=", "above", "gt", "gte", "crosses above", "cross above"}:
            condition = "above"
        elif op in {"<", "<=", "below", "lt", "lte", "crosses below", "cross below"}:
            condition = "below"
        else:
            condition = "cross"
        try:
            alert = await hub.alerts_service.create_alert(chat_id, symbol, condition, float(price))
        except RuntimeError as exc:
            await message.answer(f"couldn't set that alert — {_safe_exc(exc)}")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("alert_create_failed", extra={"chat_id": chat_id, "symbol": symbol, "price": price})
            await message.answer(f"alert creation failed. try again in a sec.")
            return True
        await _remember_source_context(
            chat_id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        _cond_word = {"above": "crosses above", "below": "crosses below"}.get(condition, "crosses")
        await message.answer(
            f"🔔 alert set — <b>{alert.symbol}</b> {_cond_word} <b>${float(price):,.2f}</b>.\n"
            "i'll ping you the moment it hits. don't get liquidated.",
            reply_markup=alert_created_menu(alert.symbol),
        )
        return True

    if intent == "alert_list":
        alerts = await hub.alerts_service.list_alerts(chat_id)
        if not alerts:
            await message.answer("No active alerts.")
        else:
            lines = ["<b>Active Alerts</b>", ""]
            for a in alerts:
                lines.append(f"<code>#{a.id}</code>  <b>{a.symbol}</b>  {a.condition}  {a.target_price}  <i>[{a.status}]</i>")
            first = alerts[0]
            await _remember_source_context(
                chat_id,
                exchange=first.source_exchange,
                market_kind=first.market_kind,
                instrument_id=first.instrument_id,
                symbol=first.symbol,
                context="alerts list",
            )
            await message.answer("\n".join(lines))
        return True

    if intent == "alert_delete":
        symbol = _extract_symbol(params)
        if symbol:
            count = await hub.alerts_service.delete_alerts_by_symbol(chat_id, symbol)
            await message.answer(f"Removed {count} alert(s) for {symbol}.")
            return True
        alert_id = _as_int(params.get("id") or params.get("alert_id"), 0)
        if alert_id <= 0:
            await message.answer("Which alert id should I delete?")
            return True
        ok = await hub.alerts_service.delete_alert(chat_id, alert_id)
        await message.answer("Deleted." if ok else "Alert not found.")
        return True

    if intent == "alert_clear":
        count = await hub.alerts_service.clear_user_alerts(chat_id)
        await message.answer(f"Cleared {count} alerts.")
        return True

    if intent == "pair_find":
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            query = _extract_symbol(params)
        if not isinstance(query, str) or not query.strip():
            await message.answer("Which coin should I resolve to a pair?")
            return True
        payload = await hub.discovery_service.find_pair(query.strip())
        await message.answer(pair_find_template(payload))
        return True

    if intent == "price_guess":
        price = _as_float(params.get("price") or params.get("target_price"))
        if price is None:
            await message.answer("What price should I search around?")
            return True
        limit = max(1, min(_as_int(params.get("limit"), 10), 20))
        payload = await hub.discovery_service.guess_by_price(price, limit=limit)
        await message.answer(price_guess_template(payload))
        return True

    if intent == "setup_review":
        symbol = _extract_symbol(params)
        entry = _as_float(params.get("entry"))
        stop = _as_float(params.get("stop") or params.get("sl"))
        targets = _as_float_list(params.get("targets") or params.get("tp"))
        leverage = _as_float(params.get("leverage"))

        # "Market Price" / "market order" / "MP" as entry → fetch live price
        if entry is None and symbol and re.search(
            r"\bmarket\s*(?:price|order)?\b|\bmp\b|\bat\s+market\b", raw_text, re.IGNORECASE
        ):
            with suppress(Exception):
                price_data = await hub.market_router.get_price(symbol)
                entry = _as_float(price_data.get("price") or price_data.get("last"))

        if not symbol or entry is None or stop is None or not targets:
            await message.answer(
                "need <b>symbol</b>, <b>entry</b>, <b>stop</b>, and at least one <b>target</b>.\n"
                "e.g. <code>SNXUSDT entry 0.028 stop 0.036 tp 0.022</code>"
            )
            return True
        direction = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        amount_usd = _as_float(params.get("amount") or params.get("amount_usd") or params.get("margin"))
        payload = await hub.setup_review_service.review(
            symbol=symbol,
            timeframe=timeframe,
            entry=float(entry),
            stop=float(stop),
            targets=[float(x) for x in targets],
            direction=direction if direction in {"long", "short"} else None,
            amount_usd=amount_usd,
            leverage=leverage,
        )
        await message.answer(setup_review_template(payload, settings))
        return True

    if intent == "trade_math":
        entry = _as_float(params.get("entry"))
        stop = _as_float(params.get("stop") or params.get("sl"))
        targets = _as_float_list(params.get("targets") or params.get("tp"))
        if entry is None or stop is None or not targets:
            await message.answer("Send entry, stop, and target(s), e.g. `entry 100 sl 95 tp 110`.")
            return True
        symbol = _extract_symbol(params)
        side = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        margin_usd = _as_float(params.get("amount") or params.get("amount_usd") or params.get("margin"))
        leverage = _as_float(params.get("leverage"))
        payload = _trade_math_payload(
            entry=float(entry),
            stop=float(stop),
            targets=[float(x) for x in targets],
            direction=side,
            margin_usd=margin_usd,
            leverage=leverage,
            symbol=symbol,
        )
        await message.answer(trade_math_template(payload, settings))
        return True

    if intent == "giveaway_join":
        if not message.from_user:
            await message.answer("Could not identify user for giveaway join.")
            return True
        payload = await hub.giveaway_service.join_active(chat_id, message.from_user.id)
        await message.answer(f"Joined giveaway #{payload['giveaway_id']}. Participants: {payload['participants']}")
        return True

    if intent == "giveaway_status":
        payload = await hub.giveaway_service.status(chat_id)
        await message.answer(giveaway_status_template(payload))
        return True

    if intent == "giveaway_end":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.end_giveaway(chat_id, message.from_user.id)
        if payload.get("winner_user_id"):
            await message.answer(
                f"Giveaway #{payload['giveaway_id']} ended.\nWinner: {payload['winner_user_id']}\nPrize: {payload['prize']}"
            )
        else:
            await message.answer(f"Giveaway ended with no winner. {payload.get('note')}")
        return True

    if intent == "giveaway_reroll":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.reroll(chat_id, message.from_user.id)
        await message.answer(
            f"Reroll complete for giveaway #{payload['giveaway_id']}.\nNew winner: {payload['winner_user_id']}"
        )
        return True

    if intent == "giveaway_cancel":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.end_giveaway(chat_id, message.from_user.id)
        if payload.get("winner_user_id"):
            await message.answer(
                f"Giveaway #{payload['giveaway_id']} ended.\nWinner: {payload['winner_user_id']}\nPrize: {payload['prize']}"
            )
        else:
            await message.answer(f"Giveaway ended with no winner. {payload.get('note')}")
        return True

    if intent == "giveaway_start":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        duration = params.get("duration") or params.get("duration_text") or "10m"
        duration_seconds = None
        if isinstance(duration, (int, float)):
            duration_seconds = max(30, int(duration))
        elif isinstance(duration, str):
            duration_seconds = _parse_duration_to_seconds(duration)
        if duration_seconds is None:
            await message.answer("Give a duration like 10m or 1h for giveaway start.")
            return True
        prize = str(params.get("prize") or "Prize").strip() or "Prize"
        payload = await hub.giveaway_service.start_giveaway(
            group_chat_id=chat_id,
            admin_chat_id=message.from_user.id,
            duration_seconds=duration_seconds,
            prize=prize,
        )
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\nEnds at: {payload['end_time']}\nUsers enter with /join"
        )
        return True

    return False


async def _get_pending_alert(chat_id: int) -> str | None:
    hub = _require_hub()
    payload = await hub.cache.get_json(f"pending_alert:{chat_id}")
    return payload.get("symbol") if payload else None


async def _set_pending_alert(chat_id: int, symbol: str) -> None:
    hub = _require_hub()
    await hub.cache.set_json(f"pending_alert:{chat_id}", {"symbol": symbol.upper()}, ttl=300)


async def _clear_pending_alert(chat_id: int) -> None:
    hub = _require_hub()
    await hub.cache.redis.delete(f"pending_alert:{chat_id}")


async def _wizard_get(chat_id: int) -> dict | None:
    hub = _require_hub()
    return await hub.cache.get_json(f"wizard:tradecheck:{chat_id}")


async def _wizard_set(chat_id: int, payload: dict, ttl: int = 900) -> None:
    hub = _require_hub()
    await hub.cache.set_json(f"wizard:tradecheck:{chat_id}", payload, ttl=ttl)


async def _wizard_clear(chat_id: int) -> None:
    hub = _require_hub()
    await hub.cache.redis.delete(f"wizard:tradecheck:{chat_id}")


async def _cmd_wizard_get(chat_id: int) -> dict | None:
    hub = _require_hub()
    return await hub.cache.get_json(f"wizard:cmd:{chat_id}")


async def _cmd_wizard_set(chat_id: int, payload: dict, ttl: int = 900) -> None:
    hub = _require_hub()
    await hub.cache.set_json(f"wizard:cmd:{chat_id}", payload, ttl=ttl)


async def _cmd_wizard_clear(chat_id: int) -> None:
    hub = _require_hub()
    await hub.cache.redis.delete(f"wizard:cmd:{chat_id}")


async def _save_trade_check(chat_id: int, data: dict, result: dict) -> None:
    hub = _require_hub()
    user = await hub.user_service.ensure_user(chat_id)
    async with AsyncSessionLocal() as session:
        row = TradeCheck(
            user_id=user.id,
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            timestamp=data["timestamp"],
            entry=float(data["entry"]),
            stop=float(data["stop"]),
            targets_json=[float(x) for x in data["targets"]],
            mode=data.get("mode", "ambiguous"),
            result_json=result,
        )
        session.add(row)
        await session.commit()


async def _handle_parsed_intent(message: Message, parsed, settings: dict) -> bool:
    hub = _require_hub()
    chat_id = message.chat.id
    raw_text = message.text or ""

    if parsed.intent == Intent.ANALYSIS:
        symbol = parsed.entities["symbol"]
        direction = parsed.entities.get("direction")
        parsed_tfs = parsed.entities.get("timeframes")
        parsed_emas = parsed.entities.get("ema_periods")
        parsed_rsis = parsed.entities.get("rsi_periods")

        await _maybe_send_market_warning(message)
        settings_tfs = _analysis_timeframes_from_settings(settings)
        settings_emas = _parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200])
        settings_rsis = _parse_int_list(settings.get("preferred_rsi_periods", [14]), [14])

        try:
            payload = await hub.analysis_service.analyze(
                symbol,
                direction=direction,
                timeframe=parsed.entities.get("timeframe"),
                timeframes=parsed_tfs or settings_tfs,
                ema_periods=parsed_emas or settings_emas,
                rsi_periods=parsed_rsis or settings_rsis,
                all_timeframes=bool(parsed.entities.get("all_timeframes")),
                all_emas=bool(parsed.entities.get("all_emas")),
                all_rsis=bool(parsed.entities.get("all_rsis")),
                include_derivatives=bool(parsed.entities.get("include_derivatives")),
                include_news=bool(parsed.entities.get("include_news")),
                notes=parsed.entities.get("notes", []),
            )
            await _append_last_symbol(chat_id, symbol)
        except Exception as exc:  # noqa: BLE001
            err = str(exc).lower()
            if any(
                marker in err
                for marker in (
                    "price unavailable",
                    "no valid ohlcv",
                    "isn't supported",
                    "binance-only",
                    "unavailable",
                )
            ):
                fallback = await hub.analysis_service.fallback_asset_brief(symbol, reason=str(exc))
                await message.answer(asset_unsupported_template(fallback, settings))
                return True
            raise
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol, direction, payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=direction,
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(message, symbol, analysis_text, direction=direction)
        return True

    if parsed.intent == Intent.SETUP_REVIEW:
        timeframe = parsed.entities.get("timeframe", "1h")
        tfs = parsed.entities.get("timeframes") or []
        if tfs:
            timeframe = tfs[0]
        symbol = parsed.entities.get("symbol")
        entry = _as_float(parsed.entities.get("entry"))
        stop = _as_float(parsed.entities.get("stop"))
        targets = [float(x) for x in (parsed.entities.get("targets") or [])]

        # "Market Price" / "market order" as entry → fetch live price
        raw = message.text or ""
        if entry is None and symbol and re.search(
            r"\bmarket\s*(?:price|order)?\b|\bmp\b|\bat\s+market\b", raw, re.IGNORECASE
        ):
            with suppress(Exception):
                price_data = await hub.market_router.get_price(symbol)
                entry = _as_float(price_data.get("price") or price_data.get("last"))

        if not symbol or entry is None or stop is None or not targets:
            await message.answer(
                "need <b>symbol</b>, <b>entry</b>, <b>stop</b>, and at least one <b>target</b>.\n"
                "e.g. <code>SNXUSDT entry 0.028 stop 0.036 tp 0.022</code>"
            )
            return True
        payload = await hub.setup_review_service.review(
            symbol=symbol,
            timeframe=timeframe,
            entry=float(entry),
            stop=float(stop),
            targets=targets,
            direction=parsed.entities.get("direction"),
            amount_usd=parsed.entities.get("amount_usd"),
            leverage=parsed.entities.get("leverage"),
        )
        await message.answer(setup_review_template(payload, settings))
        return True

    if parsed.intent == Intent.TRADE_MATH:
        payload = _trade_math_payload(
            entry=float(parsed.entities["entry"]),
            stop=float(parsed.entities["stop"]),
            targets=[float(x) for x in parsed.entities["targets"]],
            direction=parsed.entities.get("direction"),
            margin_usd=parsed.entities.get("amount_usd"),
            leverage=parsed.entities.get("leverage"),
            symbol=parsed.entities.get("symbol"),
        )
        await message.answer(trade_math_template(payload, settings))
        return True

    if parsed.intent == Intent.RSI_SCAN:
        try:
            payload = await hub.rsi_scanner_service.scan(
                timeframe=parsed.entities.get("timeframe", "1h"),
                mode=parsed.entities.get("mode", "oversold"),
                limit=int(parsed.entities.get("limit", 10)),
                rsi_length=int(parsed.entities.get("rsi_length", 14)),
                symbol=parsed.entities.get("symbol"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("rsi_scan_intent_failed", extra={"event": "rsi_intent_error", "error": str(exc)})
            await message.answer("RSI scan hit an error — try again in a moment.")
            return True
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            symbol=parsed.entities.get("symbol"),
            context="rsi scan",
        )
        await message.answer(rsi_scan_template(payload))
        return True

    if parsed.intent == Intent.EMA_SCAN:
        payload = await hub.ema_scanner_service.scan(
            timeframe=parsed.entities.get("timeframe", "4h"),
            ema_length=int(parsed.entities.get("ema_length", 200)),
            mode=parsed.entities.get("mode", "closest"),
            limit=int(parsed.entities.get("limit", 10)),
        )
        lines = [payload["summary"], ""]
        for idx, row in enumerate(payload.get("items", []), start=1):
            lines.append(
                f"{idx}. {row['symbol']} price {row['price']} | EMA{payload['ema_length']} {row['ema']} | "
                f"dist {row['distance_pct']}% ({row['side']})"
            )
        if not payload.get("items"):
            lines.append("No EMA matches right now.")
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="ema scan",
        )
        await message.answer("\n".join(lines))
        return True

    if parsed.intent == Intent.CHART:
        img, meta = await hub.chart_service.render_chart(
            symbol=parsed.entities["symbol"],
            timeframe=parsed.entities.get("timeframe", "1h"),
        )
        symbol = str(parsed.entities["symbol"]).upper()
        timeframe = str(parsed.entities.get("timeframe", "1h"))
        caption = f"{symbol} {timeframe} chart."
        await _remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("instrument_id") or ""),
            updated_at=str(meta.get("updated_at") or ""),
            symbol=symbol,
            context="chart",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_{timeframe}_chart.png"),
            caption=caption,
        )
        return True

    if parsed.intent == Intent.HEATMAP:
        symbol = str(parsed.entities.get("symbol", "BTC"))
        img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
        await _remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("pair") or ""),
            symbol=symbol,
            context="heatmap",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_heatmap.png"),
            caption=(
                f"{meta['pair']} orderbook heatmap\n"
                f"Best bid: {meta['best_bid']:.6f} | Best ask: {meta['best_ask']:.6f}\n"
                f"Bid wall: {meta['bid_wall']:.6f} | Ask wall: {meta['ask_wall']:.6f}"
            ),
        )
        return True

    if parsed.intent == Intent.PAIR_FIND:
        payload = await hub.discovery_service.find_pair(parsed.entities["query"])
        await message.answer(pair_find_template(payload))
        return True

    if parsed.intent == Intent.PRICE_GUESS:
        payload = await hub.discovery_service.guess_by_price(
            target_price=float(parsed.entities["target_price"]),
            limit=int(parsed.entities.get("limit", 10)),
        )
        await message.answer(price_guess_template(payload))
        return True

    if parsed.intent == Intent.SMALLTALK:
        llm_reply = await _llm_fallback_reply(raw_text, settings, chat_id=chat_id)
        await message.answer(llm_reply or smalltalk_reply(settings))
        return True

    if parsed.intent == Intent.ASSET_UNSUPPORTED:
        await message.answer("Send the ticker + context and I'll give a safe fallback brief.")
        return True

    if parsed.intent == Intent.ALERT_CREATE:
        sym = parsed.entities["symbol"]
        price_val = float(parsed.entities["target_price"])
        cond = parsed.entities.get("condition", "cross")
        try:
            alert = await hub.alerts_service.create_alert(chat_id, sym, cond, price_val)
        except RuntimeError as exc:
            await message.answer(f"couldn't set that alert — {_safe_exc(exc)}")
            return True
        except Exception:  # noqa: BLE001
            logger.exception("alert_create_nlu_failed", extra={"chat_id": chat_id, "symbol": sym})
            await message.answer("alert creation failed. try again in a sec.")
            return True
        await _remember_source_context(
            chat_id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=alert.symbol,
            context="alert",
        )
        _cond_word = {"above": "crosses above", "below": "crosses below"}.get(cond, "crosses")
        await message.answer(
            f"🔔 alert set — <b>{alert.symbol}</b> {_cond_word} <b>${price_val:,.2f}</b>.\n"
            "i'll ping you the moment it hits. don't get liquidated.",
            reply_markup=alert_created_menu(alert.symbol),
        )
        return True

    if parsed.intent == Intent.ALERT_LIST:
        alerts = await hub.alerts_service.list_alerts(chat_id)
        if not alerts:
            await message.answer("No active alerts.")
        else:
            lines = ["<b>Active Alerts</b>", ""]
            for a in alerts:
                lines.append(f"<code>#{a.id}</code>  <b>{a.symbol}</b>  {a.condition}  {a.target_price}  <i>[{a.status}]</i>")
            first = alerts[0]
            await _remember_source_context(
                chat_id,
                exchange=first.source_exchange,
                market_kind=first.market_kind,
                instrument_id=first.instrument_id,
                symbol=first.symbol,
                context="alerts list",
            )
            await message.answer("\n".join(lines))
        return True

    if parsed.intent == Intent.ALERT_CLEAR:
        alerts = await hub.alerts_service.list_alerts(chat_id)
        count = len(alerts)
        if count == 0:
            await message.answer("No alerts to clear.")
            return True
        from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="Yes, clear all", callback_data=f"confirm:clear_alerts:{count}")],
            [InlineKeyboardButton(text="Cancel", callback_data="confirm:clear_alerts:no")],
        ])
        await message.answer(f"Clear all <b>{count}</b> alerts? This can't be undone.", reply_markup=kb, reply_to_message_id=message.message_id)
        return True

    if parsed.intent == Intent.ALERT_PAUSE:
        count = await hub.alerts_service.pause_user_alerts(chat_id)
        await message.answer(f"Paused {count} alerts.")
        return True

    if parsed.intent == Intent.ALERT_RESUME:
        count = await hub.alerts_service.resume_user_alerts(chat_id)
        await message.answer(f"Resumed {count} alerts.")
        return True

    if parsed.intent == Intent.ALERT_DELETE:
        symbol = parsed.entities.get("symbol")
        if symbol:
            count = await hub.alerts_service.delete_alerts_by_symbol(chat_id, str(symbol))
            await message.answer(f"Removed {count} alert(s) for {str(symbol).upper()}.")
            return True
        ok = await hub.alerts_service.delete_alert(chat_id, int(parsed.entities["alert_id"]))
        await message.answer("Deleted." if ok else "Alert not found.")
        return True

    if parsed.intent == Intent.GIVEAWAY_JOIN:
        if not message.from_user:
            await message.answer("Could not identify user for join.")
            return True
        payload = await hub.giveaway_service.join_active(chat_id, message.from_user.id)
        await message.answer(f"Joined giveaway #{payload['giveaway_id']}. Participants: {payload['participants']}")
        return True

    if parsed.intent == Intent.GIVEAWAY_STATUS:
        payload = await hub.giveaway_service.status(chat_id)
        await message.answer(giveaway_status_template(payload))
        return True

    if parsed.intent == Intent.GIVEAWAY_START:
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        duration_seconds = _parse_duration_to_seconds(str(parsed.entities.get("duration", "10m")))
        if duration_seconds is None:
            await message.answer("Duration format should look like 10m, 1h, or 1d.")
            return True
        payload = await hub.giveaway_service.start_giveaway(
            group_chat_id=chat_id,
            admin_chat_id=message.from_user.id,
            duration_seconds=duration_seconds,
            prize=str(parsed.entities.get("prize", "Prize")),
        )
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\nEnds at: {payload['end_time']}\nUsers enter with /join"
        )
        return True

    if parsed.intent == Intent.GIVEAWAY_CANCEL:
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.end_giveaway(chat_id, message.from_user.id)
        if payload.get("winner_user_id"):
            await message.answer(
                f"Giveaway #{payload['giveaway_id']} ended.\nWinner: {payload['winner_user_id']}\nPrize: {payload['prize']}"
            )
        else:
            await message.answer(f"Giveaway ended with no winner. {payload.get('note')}")
        return True

    if parsed.intent == Intent.GIVEAWAY_REROLL:
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.reroll(chat_id, message.from_user.id)
        await message.answer(
            f"Reroll complete for giveaway #{payload['giveaway_id']}.\n"
            f"New winner: {payload['winner_user_id']} (prev: {payload.get('previous_winner_user_id')})"
        )
        return True

    if parsed.intent == Intent.WATCHLIST:
        payload = await hub.watchlist_service.build_watchlist(
            count=parsed.entities.get("count", 5),
            direction=parsed.entities.get("direction"),
        )
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="watchlist",
        )
        await message.answer(watchlist_template(payload))
        return True

    if parsed.intent == Intent.NEWS:
        payload = await hub.news_service.get_digest(
            topic=parsed.entities.get("topic"),
            mode=parsed.entities.get("mode", "crypto"),
            limit=int(parsed.entities.get("limit", 6)),
        )
        heads = payload.get("headlines") if isinstance(payload, dict) else None
        head = heads[0] if isinstance(heads, list) and heads else {}
        await _remember_source_context(
            chat_id,
            source_line=f"{head.get('source', 'news feed')} | {head.get('url', '')}".strip(),
            context="news",
        )
        await message.answer(news_template(payload), parse_mode="HTML")
        return True

    if parsed.intent == Intent.SCAN_WALLET:
        limiter = await hub.rate_limiter.check(
            key=f"rl:scan:{chat_id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H')}",
            limit=_settings.wallet_scan_limit_per_hour,
            window_seconds=3600,
        )
        if not limiter.allowed:
            await message.answer("Wallet scan limit reached for this hour.")
            return True

        result = await hub.wallet_service.scan(parsed.entities["chain"], parsed.entities["address"], chat_id=chat_id)
        await message.answer(
            wallet_scan_template(result),
            reply_markup=wallet_actions(parsed.entities["chain"], parsed.entities["address"]),
        )
        return True

    if parsed.intent == Intent.CYCLE:
        payload = await hub.cycles_service.cycle_check()
        await message.answer(cycle_template(payload, settings))
        return True

    if parsed.intent == Intent.TRADECHECK:
        ts = parsed.entities["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        data = {
            "symbol": parsed.entities["symbol"],
            "timeframe": parsed.entities.get("timeframe", "1h"),
            "timestamp": ts,
            "entry": float(parsed.entities["entry"]),
            "stop": float(parsed.entities["stop"]),
            "targets": [float(x) for x in parsed.entities["targets"]],
            "mode": "ambiguous",
        }
        result = await hub.trade_verify_service.verify(**data)
        await _save_trade_check(chat_id, data, result)
        await _remember_source_context(
            chat_id,
            source_line=str(result.get("source_line") or ""),
            symbol=data["symbol"],
            context="trade check",
        )
        await message.answer(trade_verification_template(result, settings))
        return True

    if parsed.intent == Intent.CORRELATION:
        payload = await hub.correlation_service.check_following(
            parsed.entities["symbol"],
            benchmark=parsed.entities.get("benchmark", "BTC"),
        )
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            symbol=parsed.entities["symbol"],
            context="correlation",
        )
        await message.answer(correlation_template(payload, settings))
        return True

    if parsed.intent == Intent.SETTINGS:
        st = await hub.user_service.get_settings(chat_id)
        await message.answer(settings_text(st), reply_markup=settings_menu(st))
        return True

    if parsed.intent == Intent.HELP:
        await message.answer(help_text())
        return True

    return False


@router.message(Command("start"))
async def start_cmd(message: Message) -> None:
    hub = _require_hub()
    user = await hub.user_service.ensure_user(message.chat.id)
    name = message.from_user.first_name if message.from_user else "fren"
    chat_id = message.chat.id

    # Detect new user: created_at within the last 15 seconds
    try:
        is_new = (datetime.utcnow() - user.created_at).total_seconds() < 15
    except Exception:  # noqa: BLE001
        is_new = False

    if is_new:
        await message.answer(
            f"gm <b>{name}</b> 👋\n\n"
            "i'm <b>ghost</b> — your on-chain trading assistant. i live in the market 24/7 so you don't have to.\n\n"
            "try something like:\n"
            "· <code>BTC 4h</code> — full technical analysis\n"
            "· <code>ping me when ETH hits 2000</code> — price alert\n"
            "· <code>coins to watch</code> — top movers watchlist\n"
            "· <code>why is BTC pumping</code> — live market read\n\n"
            "<i>short questions get short answers. send a ticker for a deep dive. tap a button to start.</i>",
            reply_markup=smart_action_menu(),
        )
    else:
        # Returning user — check for session continuity
        continuity = ""
        with suppress(Exception):
            last_ctx = await hub.cache.get_json(f"last_analysis_context:{chat_id}")
            if isinstance(last_ctx, dict) and last_ctx.get("symbol"):
                sym = str(last_ctx["symbol"]).upper()
                dir_txt = last_ctx.get("direction") or ""
                direction_part = f" {dir_txt}" if dir_txt else ""
                continuity = f"\n\n<i>last time you were watching <b>{sym}{direction_part}</b> — want a fresh read?</i>"
        await message.answer(
            f"wb back {name}. ghost is live.{continuity}\n\n"
            "drop a ticker, ask a question, or tap a button.",
            reply_markup=smart_action_menu(),
        )


@router.message(Command("help"))
async def help_cmd(message: Message) -> None:
    await message.answer(help_text(), reply_markup=command_center_menu())


@router.message(Command("admins"))
async def admins_cmd(message: Message) -> None:
    admin_ids = sorted(set(_settings.admin_ids_list()))
    if not admin_ids:
        await message.answer("no admin IDs configured.")
        return
    lines = ["<b>bot admins</b>\n"]
    lines.extend(f"· <code>{admin_id}</code>" for admin_id in admin_ids)
    await message.answer("\n".join(lines))


@router.message(Command("id"))
async def id_cmd(message: Message) -> None:
    if not message.from_user:
        await message.answer("couldn't read your user id from this update.")
        return
    await message.answer(
        f"your user id  <code>{message.from_user.id}</code>\n"
        f"this chat id  <code>{message.chat.id}</code>"
    )


@router.message(Command("settings"))
async def settings_cmd(message: Message) -> None:
    hub = _require_hub()
    settings = await hub.user_service.get_settings(message.chat.id)
    await message.answer(settings_text(settings), reply_markup=settings_menu(settings))


@router.message(Command("name"))
async def name_cmd(message: Message) -> None:
    """Set display name for memory (e.g. /name Alice)."""
    hub = _require_hub()
    text = (message.text or "").strip().split(maxsplit=1)[1] if len((message.text or "").strip().split()) > 1 else ""
    name = (text or "").strip()[:64]
    await hub.user_service.update_settings(message.chat.id, {"display_name": name})
    await message.answer(f"Got it. I'll call you {name}." if name else "Display name cleared.")


@router.message(Command("goals"))
async def goals_cmd(message: Message) -> None:
    """Set trading goals for memory (e.g. /goals swing trading, low risk)."""
    hub = _require_hub()
    text = (message.text or "").strip().split(maxsplit=1)[1] if len((message.text or "").strip().split()) > 1 else ""
    goals = (text or "").strip()[:300]
    await hub.user_service.update_settings(message.chat.id, {"trading_goals": goals})
    await message.answer("Goals saved. I'll keep that in mind." if goals else "Goals cleared.")


@router.message(Command("alpha"))
async def alpha_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick an analysis shortcut or tap Custom.", reply_markup=alpha_quick_menu())
        return
    tokens = text.split()
    if len(tokens) == 1:
        text = f"watch {tokens[0]}"
    await _dispatch_command_text(message, text)


@router.message(Command("watch"))
async def watch_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick a watch shortcut or tap Custom.", reply_markup=watch_quick_menu())
        return
    await _dispatch_command_text(message, f"watch {text}")


@router.message(Command("price"))
async def price_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    symbol = args.strip().upper().lstrip("$") or ""
    if not symbol:
        await message.answer("send symbol for quick price.\nExample: <code>/price BTC</code> or <code>/price SOL</code>")
        return
    hub = _require_hub()
    try:
        data = await hub.market_router.get_price(symbol)
        price = float(data.get("price") or 0)
        change_24h = data.get("change_24h")
        if change_24h is not None:
            line = f"<b>{symbol}</b> ${price:,.2f} ({change_24h:+.2f}%)"
        else:
            line = f"<b>{symbol}</b> ${price:,.2f}"
        await message.answer(line, reply_to_message_id=message.message_id)
    except Exception:
        await _dispatch_command_text(message, f"watch {symbol}")


@router.message(Command("chart"))
async def chart_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick a chart shortcut or tap Custom.", reply_markup=chart_quick_menu())
        return
    await _dispatch_command_text(message, f"chart {text}")


@router.message(Command("heatmap"))
async def heatmap_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick a symbol for heatmap or tap Custom.", reply_markup=heatmap_quick_menu())
        return
    await _dispatch_command_text(message, f"heatmap {text}")


@router.message(Command("rsi"))
async def rsi_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    parts = raw.split()
    if len(parts) < 3:
        await message.answer("Pick an RSI scanner preset or tap Custom.", reply_markup=rsi_quick_menu())
        return
    timeframe = parts[1].lower()
    mode = parts[2].lower()
    if mode not in {"overbought", "oversold"}:
        await message.answer("Mode must be `overbought` or `oversold`.")
        return
    top_n = max(1, min(_as_int(parts[3], 10), 20)) if len(parts) >= 4 else 10
    rsi_len = max(2, min(_as_int(parts[4], 14), 50)) if len(parts) >= 5 else 14
    await _dispatch_command_text(message, f"rsi top {top_n} {timeframe} {mode} rsi{rsi_len}")


@router.message(Command("ema"))
async def ema_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    parts = raw.split()
    if len(parts) < 3:
        await message.answer("Pick an EMA scanner preset or tap Custom.", reply_markup=ema_quick_menu())
        return
    ema_len = max(2, min(_as_int(parts[1], 200), 500))
    timeframe = parts[2].lower()
    top_n = max(1, min(_as_int(parts[3], 10), 20)) if len(parts) >= 4 else 10
    await _dispatch_command_text(message, f"ema {ema_len} {timeframe} top {top_n}")


@router.message(Command("watchlist"))
async def watchlist_cmd(message: Message) -> None:
    hub = _require_hub()
    n_match = re.search(r"/watchlist\s+(\d+)", message.text or "")
    n = int(n_match.group(1)) if n_match else 5
    direction = None
    if re.search(r"\blong\b", message.text or "", re.IGNORECASE):
        direction = "long"
    elif re.search(r"\bshort\b", message.text or "", re.IGNORECASE):
        direction = "short"
    payload = await hub.watchlist_service.build_watchlist(count=max(1, min(n, 20)), direction=direction)
    await _remember_source_context(
        message.chat.id,
        source_line=str(payload.get("source_line") or ""),
        context="watchlist",
    )
    await message.answer(watchlist_template(payload))


@router.message(Command("news"))
async def news_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()
    topic: str | None = None
    mode = "crypto"
    limit = 6
    parts = text.split()
    if len(parts) == 1:
        await message.answer("Pick a news mode.", reply_markup=news_quick_menu())
        return
    if len(parts) > 1:
        raw_topic = parts[1].strip()
        if raw_topic.isdigit():
            limit = max(3, min(int(raw_topic), 10))
        else:
            topic = raw_topic
    if len(parts) > 2 and parts[2].isdigit():
        limit = max(3, min(int(parts[2]), 10))

    if topic:
        lowered = topic.lower().strip()
        if lowered in {"crypto", "openai", "cpi", "fomc"}:
            topic = lowered
        if re.search(r"\b(openai|chatgpt|gpt|codex)\b", lowered):
            mode = "openai"
            topic = "openai"
        elif re.search(r"\b(cpi|inflation)\b", lowered):
            mode = "macro"
            topic = "cpi"
        elif re.search(r"\b(fomc|fed|powell|macro|rates?)\b", lowered):
            mode = "macro"
            topic = "macro"
    payload = await hub.news_service.get_digest(topic=topic, mode=mode, limit=limit)
    heads = payload.get("headlines") if isinstance(payload, dict) else None
    head = heads[0] if isinstance(heads, list) and heads else {}
    await _remember_source_context(
        message.chat.id,
        source_line=f"{head.get('source', 'news feed')} | {head.get('url', '')}".strip(),
        context="news",
    )
    await message.answer(news_template(payload), parse_mode="HTML")


@router.message(Command("cycle"))
async def cycle_cmd(message: Message) -> None:
    hub = _require_hub()
    settings = await hub.user_service.get_settings(message.chat.id)
    payload = await hub.cycles_service.cycle_check()
    await message.answer(cycle_template(payload, settings))


@router.message(Command("scan"))
async def scan_cmd(message: Message) -> None:
    hub = _require_hub()
    text = message.text or ""
    m = re.search(r"/scan\s+(solana|tron)\s+([A-Za-z0-9]+)", text, re.IGNORECASE)
    if not m:
        await message.answer("Pick chain first, then paste address.", reply_markup=scan_quick_menu())
        return

    limiter = await hub.rate_limiter.check(
        key=f"rl:scan:{message.chat.id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H')}",
        limit=_settings.wallet_scan_limit_per_hour,
        window_seconds=3600,
    )
    if not limiter.allowed:
        await message.answer("Wallet scan limit reached for this hour.")
        return

    chain, address = m.group(1).lower(), m.group(2)
    result = await hub.wallet_service.scan(chain, address, chat_id=message.chat.id)
    await message.answer(wallet_scan_template(result), reply_markup=wallet_actions(chain, address))


@router.message(Command("alert"))
async def alert_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()

    if text.startswith("/alert list"):
        alerts = await hub.alerts_service.list_alerts(message.chat.id)
        if not alerts:
            await message.answer("No active alerts.")
            return
        rows = ["<b>Active Alerts</b>", ""]
        for a in alerts:
            rows.append(f"<code>#{a.id}</code>  <b>{a.symbol}</b>  {a.condition}  {a.target_price}  <i>[{a.status}]</i>")
        first = alerts[0]
        await _remember_source_context(
            message.chat.id,
            exchange=first.source_exchange,
            market_kind=first.market_kind,
            instrument_id=first.instrument_id,
            symbol=first.symbol,
            context="alerts list",
        )
        await message.answer("\n".join(rows))
        return

    if text.startswith("/alert clear"):
        count = await hub.alerts_service.clear_user_alerts(message.chat.id)
        await message.answer(f"Cleared {count} alerts.")
        return

    if text.startswith("/alert pause"):
        count = await hub.alerts_service.pause_user_alerts(message.chat.id)
        await message.answer(f"Paused {count} alerts.")
        return

    if text.startswith("/alert resume"):
        count = await hub.alerts_service.resume_user_alerts(message.chat.id)
        await message.answer(f"Resumed {count} alerts.")
        return

    d_match = re.search(r"/alert\s+delete\s+(\d+)", text)
    if d_match:
        ok = await hub.alerts_service.delete_alert(message.chat.id, int(d_match.group(1)))
        await message.answer("Deleted." if ok else "Alert not found.")
        return

    a_match = re.search(r"/alert\s+add\s+([A-Za-z0-9]+)\s+(above|below|cross)\s+([0-9.]+)", text, re.IGNORECASE)
    if a_match:
        symbol, condition, price = a_match.group(1).upper(), a_match.group(2).lower(), float(a_match.group(3))
        alert = await hub.alerts_service.create_alert(message.chat.id, symbol, condition, price, source="command")
        await _remember_source_context(
            message.chat.id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        _cond_word = {"above": "crosses above", "below": "crosses below"}.get(condition, "crosses")
        await message.answer(
            f"🔔 alert set — <b>{symbol}</b> {_cond_word} <b>${price:,.2f}</b>.\n"
            "i'll ping you the moment it hits. don't get liquidated.",
            reply_markup=alert_created_menu(symbol),
        )
        return

    simple_match = re.search(
        r"^/alert\s+([A-Za-z0-9$]{2,20})\s+([0-9]+(?:\.[0-9]+)?)(?:\s+(above|below|cross))?\s*$",
        text,
        re.IGNORECASE,
    )
    if simple_match:
        symbol = simple_match.group(1).upper().lstrip("$")
        price = float(simple_match.group(2))
        condition = (simple_match.group(3) or "cross").lower()
        alert = await hub.alerts_service.create_alert(message.chat.id, symbol, condition, price, source="command")
        await _remember_source_context(
            message.chat.id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        _cond_word2 = {"above": "crosses above", "below": "crosses below"}.get(condition, "crosses")
        await message.answer(
            f"🔔 alert set — <b>{symbol}</b> {_cond_word2} <b>${price:,.2f}</b>.\n"
            "i'll ping you the moment it hits. don't get liquidated.",
            reply_markup=alert_created_menu(symbol),
        )
        return

    await message.answer("Pick an alert action.", reply_markup=alert_quick_menu())


@router.message(Command("alerts"))
async def alerts_cmd(message: Message) -> None:
    hub = _require_hub()
    try:
        alerts = await hub.alerts_service.list_alerts(message.chat.id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("alerts_list_failed", extra={"event": "alerts_list_failed", "error": str(exc), "chat_id": message.chat.id})
        await message.answer("Alerts are temporarily unavailable. Try again in a few seconds.")
        return
    if not alerts:
        await message.answer("No active alerts.")
        return
    rows = ["<b>Active Alerts</b>", ""]
    for a in alerts:
        rows.append(f"<code>#{a.id}</code>  <b>{a.symbol}</b>  {a.condition}  {a.target_price}  <i>[{a.status}]</i>")
    first = alerts[0]
    await _remember_source_context(
        message.chat.id,
        exchange=first.source_exchange,
        market_kind=first.market_kind,
        instrument_id=first.instrument_id,
        symbol=first.symbol,
        context="alerts list",
    )
    await message.answer("\n".join(rows))


@router.message(Command("alertdel"))
async def alertdel_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()
    m = re.search(r"^/alertdel\s+(\d+)\s*$", text, re.IGNORECASE)
    if not m:
        try:
            alerts = await hub.alerts_service.list_alerts(message.chat.id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("alertdel_list_failed", extra={"event": "alertdel_list_failed", "error": str(exc), "chat_id": message.chat.id})
            await message.answer("Alerts are temporarily unavailable. Try again in a few seconds.")
            return
        if not alerts:
            await message.answer("No active alerts.", reply_markup=alert_quick_menu())
            return
        options = [(f"Delete #{a.id}", f"cmd:alertdel:{a.id}") for a in alerts[:8]]
        await message.answer("Tap an alert to delete.", reply_markup=simple_followup(options))
        return
    try:
        ok = await hub.alerts_service.delete_alert(message.chat.id, int(m.group(1)))
    except Exception as exc:  # noqa: BLE001
        logger.exception("alertdel_failed", extra={"event": "alertdel_failed", "error": str(exc), "chat_id": message.chat.id})
        await message.answer("Delete failed on my side. Try again in a few seconds.")
        return
    await message.answer("Deleted." if ok else "Alert not found.")


@router.message(Command("alertclear"))
async def alertclear_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()
    m = re.search(r"^/alertclear\s+([A-Za-z0-9$]{2,20})\s*$", text, re.IGNORECASE)
    if m:
        symbol = m.group(1).upper().lstrip("$")
        count = await hub.alerts_service.delete_alerts_by_symbol(message.chat.id, symbol)
        await message.answer(f"Cleared {count} alerts for {symbol}.")
        return
    await message.answer("Pick clear action.", reply_markup=simple_followup([("Clear all alerts", "cmd:alert:clear"), ("Clear by symbol", "cmd:alert:clear_symbol")]))


@router.message(Command("position"))
async def position_cmd(message: Message) -> None:
    flags = _settings.feature_flags_set()
    if "portfolio" not in flags:
        await message.answer("Portfolio feature is disabled.")
        return
    hub = _require_hub()
    if not getattr(hub, "portfolio_service", None):
        await message.answer("Portfolio is not available.")
        return
    text = (message.text or "").strip().split(maxsplit=1)[1] if (message.text or "").strip().split() else ""
    args = (text or "").strip().lower().split()
    chat_id = message.chat.id

    if not args or args[0] == "list":
        positions = await hub.portfolio_service.list_positions(chat_id)
        if not positions:
            await message.answer("No positions. Add one: <code>/position add BTC long 50000 1000 2</code>")
            return
        lines = ["<b>Positions</b>", ""]
        total_pnl = 0.0
        for p in positions:
            total_pnl += p.get("pnl_quote", 0) or 0
            lines.append(
                f"<code>#{p['id']}</code> <b>{p['symbol']}</b> {p['side']} | entry ${p['entry_price']:,.2f} → ${p['current_price']:,.2f} | "
                f"PnL {p['pnl_pct']:+.2f}% (${p['pnl_quote']:+,.2f})"
            )
        lines.append(f"\n<b>Total unrealized PnL:</b> ${total_pnl:+,.2f}")
        await message.answer("\n".join(lines))
        return

    if args[0] == "delete" and len(args) >= 2:
        try:
            pid = int(args[1])
        except ValueError:
            await message.answer("Usage: /position delete &lt;id&gt;")
            return
        ok = await hub.portfolio_service.delete_position(chat_id, pid)
        await message.answer("Position removed." if ok else "Position not found.")
        return

    if args[0] == "add" and len(args) >= 5:
        # /position add SYMBOL long ENTRY SIZE [leverage] [notes...]
        symbol = (args[1] or "").upper()
        side = (args[2] or "long").lower()
        if side not in ("long", "short"):
            side = "long"
        try:
            entry_price = float(args[3])
            size_quote = float(args[4])
        except (ValueError, IndexError):
            await message.answer("Usage: /position add SYMBOL long|short ENTRY_PRICE SIZE_QUOTE [leverage]")
            return
        leverage = 1.0
        notes = ""
        if len(args) >= 6:
            try:
                leverage = float(args[5])
            except ValueError:
                notes = " ".join(args[5:])[:255]
        if len(args) >= 7 and not notes:
            notes = " ".join(args[6:])[:255]
        pos, warning = await hub.portfolio_service.add_position(
            chat_id, symbol=symbol, side=side, entry_price=entry_price, size_quote=size_quote, leverage=leverage, notes=notes or None
        )
        if pos is None:
            await message.answer(warning or "Failed to add position.")
            return
        msg = f"Position added: <b>{symbol}</b> {side} @ ${entry_price:,.2f} size ${size_quote:,.0f} {leverage}x"
        if warning:
            msg += f"\n⚠️ {warning}"
        await message.answer(msg)
        return

    await message.answer(
        "Usage:\n"
        "• <code>/position list</code> — list positions\n"
        "• <code>/position add SYMBOL long|short ENTRY SIZE [leverage]</code>\n"
        "• <code>/position delete ID</code>"
    )


@router.message(Command("journal"))
async def journal_cmd(message: Message) -> None:
    flags = _settings.feature_flags_set()
    if "journal" not in flags:
        await message.answer("Journal feature is disabled.")
        return
    hub = _require_hub()
    if not getattr(hub, "trade_journal_service", None):
        await message.answer("Journal is not available.")
        return
    text = (message.text or "").strip().split(maxsplit=1)[1] if len((message.text or "").strip().split()) > 1 else ""
    args = (text or "").strip().lower().split()
    chat_id = message.chat.id

    if not args or args[0] == "list":
        limit = 20
        if len(args) >= 2 and args[1].isdigit():
            limit = min(int(args[1]), 50)
        trades = await hub.trade_journal_service.list_trades(chat_id, limit=limit)
        if not trades:
            await message.answer("No journal entries. Log one: <code>/journal log BTC long 50000 51000 100</code>")
            return
        lines = ["<b>Trade journal</b>", ""]
        for t in trades:
            line = f"<code>#{t['id']}</code> {t['symbol']} {t['side']} entry {t['entry']} → exit {t['exit_price']}"
            if t.get("pnl_quote") is not None:
                line += f" | PnL ${t['pnl_quote']:+,.2f}"
            lines.append(line)
        await message.answer("\n".join(lines))
        return

    if args[0] == "stats":
        days = 30
        if len(args) >= 2 and args[1].isdigit():
            days = min(int(args[1]), 365)
        st = await hub.trade_journal_service.get_stats(chat_id, days=days)
        await message.answer(
            f"<b>Journal stats</b> (last {days}d)\n"
            f"Trades: {st['trades']} | Wins: {st['wins']} | Win rate: {st['win_rate']}% | Total PnL: ${st['total_pnl']:+,.2f}"
        )
        return

    if args[0] == "log" and len(args) >= 5:
        symbol = (args[1] or "").upper()
        side = (args[2] or "long").lower()
        if side not in ("long", "short"):
            side = "long"
        try:
            entry = float(args[3])
            exit_price = float(args[4])
        except (ValueError, IndexError):
            await message.answer("Usage: /journal log SYMBOL long|short ENTRY EXIT [pnl]")
            return
        pnl = None
        if len(args) >= 6:
            try:
                pnl = float(args[5])
            except ValueError:
                pass
        entry_obj = await hub.trade_journal_service.log_trade(
            chat_id, symbol=symbol, side=side, entry=entry, exit_price=exit_price, pnl_quote=pnl
        )
        if entry_obj:
            await message.answer(f"Logged: <b>{symbol}</b> {side} {entry} → {exit_price}" + (f" (${pnl:+,.2f})" if pnl is not None else ""))
        else:
            await message.answer("Failed to log trade.")
        return

    await message.answer(
        "Usage:\n"
        "• <code>/journal list [N]</code> — last N entries (default 20)\n"
        "• <code>/journal stats [days]</code>\n"
        "• <code>/journal log SYMBOL long|short ENTRY EXIT [pnl]</code>"
    )


@router.message(Command("compare"))
async def compare_cmd(message: Message) -> None:
    flags = _settings.feature_flags_set()
    if "multi_compare" not in flags:
        await message.answer("Multi-symbol compare is disabled.")
        return
    hub = _require_hub()
    text = (message.text or "").strip().split(maxsplit=1)[1] if len((message.text or "").strip().split()) > 1 else ""
    symbols = [s.upper().lstrip("$") for s in (text or "BTC ETH SOL").strip().split() if s][:8]
    if not symbols:
        await message.answer("Usage: /compare BTC ETH SOL [SYMBOL...]")
        return
    try:
        prices = await asyncio.gather(
            *[hub.market_router.get_price(s) for s in symbols],
            return_exceptions=True,
        )
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"Could not fetch prices: {_safe_exc(exc)}")
        return
    lines = ["<b>Compare</b>", ""]
    for sym, p in zip(symbols, prices):
        if isinstance(p, Exception):
            lines.append(f"<b>{sym}</b> — error")
            continue
        price = float((p or {}).get("price") or 0)
        if price <= 0:
            lines.append(f"<b>{sym}</b> — no data")
        else:
            lines.append(f"<b>{sym}</b> ${price:,.2f}")
    await message.answer("\n".join(lines))


@router.message(Command("report"))
async def report_cmd(message: Message) -> None:
    flags = _settings.feature_flags_set()
    if "scheduled_report" not in flags:
        await message.answer("Scheduled reports are disabled.")
        return
    hub = _require_hub()
    svc = getattr(hub, "scheduled_report_service", None)
    if not svc:
        await message.answer("Scheduled reports not available.")
        return
    text = (message.text or "").strip().split(maxsplit=1)[1] if len((message.text or "").strip().split()) > 1 else ""
    args = (text or "").strip().lower().split()
    chat_id = message.chat.id

    if not args or args[0] == "list":
        reports = await svc.list_reports(chat_id)
        if not reports:
            await message.answer(
                "No scheduled report. Subscribe: <code>/report on 9 0</code> (9:00 UTC daily)."
            )
            return
        lines = ["<b>Scheduled reports</b>", ""]
        for r in reports:
            lines.append(
                f"• {r['report_type']} @ {r['cron_hour_utc']:02d}:{r['cron_minute_utc']:02d} UTC"
                + (f" ({r['timezone']})" if r.get("timezone") else "")
            )
        await message.answer("\n".join(lines))
        return

    if args[0] == "off" or args[0] == "unsubscribe":
        ok = await svc.unsubscribe(chat_id)
        await message.answer("Scheduled report turned off." if ok else "No subscription found.")
        return

    if args[0] in ("on", "subscribe") or args[0].isdigit():
        hour_utc, minute_utc = 9, 0
        if args[0].isdigit() and len(args) >= 2 and args[1].isdigit():
            hour_utc = max(0, min(23, int(args[0])))
            minute_utc = max(0, min(59, int(args[1])))
        elif len(args) >= 3 and args[1].isdigit() and args[2].isdigit():
            hour_utc = max(0, min(23, int(args[1])))
            minute_utc = max(0, min(59, int(args[2])))
        rec = await svc.subscribe(chat_id, report_type="market_summary", hour_utc=hour_utc, minute_utc=minute_utc)
        if rec:
            await message.answer(
                f"Scheduled report on. You'll get a market summary daily at {rec.cron_hour_utc:02d}:{rec.cron_minute_utc:02d} UTC."
            )
        else:
            await message.answer("Could not subscribe. Try /start first.")
        return

    await message.answer(
        "Usage:\n"
        "• <code>/report on [HOUR] [MINUTE]</code> — daily at HOUR:MINUTE UTC (default 9:00)\n"
        "• <code>/report off</code> — unsubscribe\n"
        "• <code>/report list</code>"
    )


@router.message(Command("export"))
async def export_cmd(message: Message) -> None:
    flags = _settings.feature_flags_set()
    if "export" not in flags:
        await message.answer("Export feature is disabled.")
        return
    hub = _require_hub()
    text = (message.text or "").strip().split(maxsplit=1)[1] if len((message.text or "").strip().split()) > 1 else ""
    kind = (text or "").strip().lower() or "alerts"
    chat_id = message.chat.id

    if kind == "alerts":
        try:
            alerts = await hub.alerts_service.list_alerts(chat_id)
        except Exception:
            await message.answer("Could not load alerts.")
            return
        if not alerts:
            await message.answer("No alerts to export.")
            return
        lines = ["# Alerts export", ""]
        for a in alerts:
            lines.append(f"#{a.id} {a.symbol} {a.condition} {a.target_price} [{a.status}]")
        body = "\n".join(lines)
    elif kind == "journal":
        if not getattr(hub, "trade_journal_service", None):
            await message.answer("Journal not available.")
            return
        trades = await hub.trade_journal_service.list_trades(chat_id, limit=200)
        if not trades:
            await message.answer("No journal entries to export.")
            return
        lines = ["# Trade journal export", ""]
        for t in trades:
            line = f"#{t['id']} {t['symbol']} {t['side']} entry={t['entry']} exit={t['exit_price']}"
            if t.get("pnl_quote") is not None:
                line += f" pnl={t['pnl_quote']}"
            lines.append(line)
        body = "\n".join(lines)
    else:
        await message.answer("Usage: /export alerts | /export journal")
        return

    if len(body) <= 4000:
        await message.answer(f"<pre>{body}</pre>")
    else:
        try:
            await message.answer_document(
                BufferedInputFile(body.encode("utf-8"), filename=f"ghost_export_{kind}.txt"),
                caption=f"Export: {kind}",
            )
        except Exception:
            await message.answer("Export too long; sending in parts.")
            for i in range(0, len(body), 4000):
                await message.answer(f"<pre>{body[i:i+4000]}</pre>")
    return


@router.message(Command("tradecheck"))
async def tradecheck_cmd(message: Message) -> None:
    await _wizard_set(message.chat.id, {"step": "symbol", "data": {}})
    await message.answer("Tradecheck wizard: send symbol (e.g., ETH).")


@router.message(Command("findpair"))
async def findpair_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    query = args.strip()
    if not query:
        await message.answer("Pick find mode.", reply_markup=findpair_quick_menu())
        return
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", query):
        await _dispatch_command_text(message, f"coin around {query}")
        return
    await _dispatch_command_text(message, f"find pair {query}")


@router.message(Command("setup"))
async def setup_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    if not args.strip():
        await message.answer("Choose setup input mode.", reply_markup=setup_quick_menu())
        return
    await _dispatch_command_text(message, args.strip())


@router.message(Command("margin"))
async def margin_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Choose setup input mode.", reply_markup=setup_quick_menu())
        return
    await _dispatch_command_text(message, text)


@router.message(Command("pnl"))
async def pnl_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Choose setup input mode.", reply_markup=setup_quick_menu())
        return
    await _dispatch_command_text(message, text)


@router.message(Command("join"))
async def join_cmd(message: Message) -> None:
    hub = _require_hub()
    if not message.from_user:
        await message.answer("Could not identify user for join.")
        return
    try:
        payload = await hub.giveaway_service.join_active(message.chat.id, message.from_user.id)
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"couldn't join: {_safe_exc(exc)}")
        return
    await message.answer(f"you're in 🎉 giveaway <b>#{payload['giveaway_id']}</b> — participants: <b>{payload['participants']}</b>")


@router.message(Command("giveaway"))
async def giveaway_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()
    if not message.from_user:
        await message.answer("Could not identify sender.")
        return

    if re.search(r"^/giveaway\s+status\b", text, flags=re.IGNORECASE):
        payload = await hub.giveaway_service.status(message.chat.id)
        await message.answer(giveaway_status_template(payload))
        return

    if re.search(r"^/giveaway\s+join\b", text, flags=re.IGNORECASE):
        try:
            payload = await hub.giveaway_service.join_active(message.chat.id, message.from_user.id)
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"couldn't join: {_safe_exc(exc)}")
            return
        await message.answer(f"you're in 🎉 giveaway <b>#{payload['giveaway_id']}</b> — participants: <b>{payload['participants']}</b>")
        return

    if re.search(r"^/giveaway\s+end\b", text, flags=re.IGNORECASE):
        try:
            payload = await hub.giveaway_service.end_giveaway(message.chat.id, message.from_user.id)
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"couldn't end giveaway: {_safe_exc(exc)}")
            return
        if payload.get("winner_user_id"):
            await message.answer(
                f"🏆 giveaway <b>#{payload.get('giveaway_id')}</b> closed.\n"
                f"winner: <code>{payload.get('winner_user_id')}</code>\n"
                f"prize: <b>{payload.get('prize', '—')}</b>"
            )
        else:
            await message.answer(f"giveaway ended with no winner. {payload.get('note')}")
        return

    if re.search(r"^/giveaway\s+reroll\b", text, flags=re.IGNORECASE):
        try:
            payload = await hub.giveaway_service.reroll(message.chat.id, message.from_user.id)
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"reroll failed: {_safe_exc(exc)}")
            return
        await message.answer(
            f"🔄 reroll done for giveaway <b>#{payload.get('giveaway_id')}</b>\n"
            f"new winner: <code>{payload.get('winner_user_id')}</code>\n"
            f"prev: <code>{payload.get('previous_winner_user_id', '—')}</code>"
        )
        return

    start_match = re.search(r"^/giveaway\s+start\s+(\S+)(?:\s+(.+))?$", text, flags=re.IGNORECASE)
    if start_match:
        duration_raw = start_match.group(1)
        duration_seconds = _parse_duration_to_seconds(duration_raw)
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
            payload = await hub.giveaway_service.start_giveaway(
                group_chat_id=message.chat.id,
                admin_chat_id=message.from_user.id,
                duration_seconds=duration_seconds,
                prize=prize,
            )
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"couldn't start giveaway: {_safe_exc(exc)}")
            return
        note = ""
        if winners_requested > 1:
            note = "\nNote: multi-winner draw will run as sequential rerolls after the first winner."
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\n"
            f"Ends at: {payload['end_time']}\nUsers enter with /join or /giveaway join{note}"
        )
        return

    await message.answer("Pick giveaway action.", reply_markup=giveaway_menu(is_admin=hub.giveaway_service.is_admin(message.from_user.id)))


@router.callback_query(F.data.startswith("followup:"))
async def followup_callback(callback: CallbackQuery) -> None:
    """Simplify, Example, Short, Go deeper — re-run LLM on last reply."""
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return
    hub = _require_hub()
    chat_id = callback.message.chat.id
    data = (callback.data or "").strip()
    action = data.replace("followup:", "", 1).lower()
    last_reply = await hub.cache.get_json(f"llm:last_reply:{chat_id}")
    last_user = await hub.cache.get_json(f"llm:last_user:{chat_id}")
    if not last_reply or not isinstance(last_reply, str):
        await callback.answer("No previous reply to refine.", show_alert=True)
        return
    instructions = {
        "simplify": "Rewrite this in simpler, shorter form. Output only the simplified version.",
        "example": "Add one concrete example to illustrate this. Keep the original and add the example.",
        "short": "Give a one- or two-sentence version only.",
        "deeper": "Expand with one more paragraph of detail or context. Keep the original summary first.",
    }
    instr = instructions.get(action, instructions["simplify"])
    prompt = f"User originally asked: \"{last_user or ''}\". Your previous reply: \"{last_reply[:600]}\". {instr}"
    try:
        reply = await hub.llm_client.reply(prompt, history=[])
        if reply and reply.strip():
            await callback.message.edit_text(
                _sanitize_llm_html(reply.strip())[:4000],
                reply_markup=llm_reply_keyboard(),
            )
    except Exception:  # noqa: BLE001
        pass
    await callback.answer()


async def _get_pending_feedback_suggestion(chat_id: int) -> dict | None:
    hub = _require_hub()
    return await hub.cache.get_json(f"feedback:pending_suggestion:{chat_id}")


async def _set_pending_feedback_suggestion(chat_id: int, payload: dict, ttl: int = 300) -> None:
    hub = _require_hub()
    await hub.cache.set_json(f"feedback:pending_suggestion:{chat_id}", payload, ttl=ttl)


async def _clear_pending_feedback_suggestion(chat_id: int) -> None:
    hub = _require_hub()
    with suppress(Exception):
        await hub.cache.delete(f"feedback:pending_suggestion:{chat_id}")


async def _notify_admins_negative_feedback(
    *,
    from_chat_id: int,
    from_username: str | None,
    reason: str,
    reply_preview: str,
    improvement_text: str | None = None,
) -> None:
    """Send negative feedback (and optional improvement text) to all admin chat IDs (e.g. your DM)."""
    admin_ids = _settings.admin_ids_list()
    if not admin_ids:
        logger.warning(
            "feedback_no_admin_ids",
            extra={"event": "feedback_no_admin_ids", "from_chat_id": from_chat_id},
        )
        return
    username = from_username or "—"
    preview = (reply_preview or "")[:400].replace("<", " ").replace(">", " ")
    lines = [
        "👎 <b>Negative feedback</b>",
        f"From: {username} (chat_id <code>{from_chat_id}</code>)",
        f"Reason: {reason}",
        "",
        f"Bot reply (preview): {preview}",
    ]
    if improvement_text and improvement_text.strip():
        lines.extend(["", "📝 <b>Improvement suggestion:</b>", improvement_text.strip()[:1000]])
    text = "\n".join(lines)
    hub = _require_hub()
    for admin_id in admin_ids:
        try:
            await hub.telegram_bot.send_message(admin_id, text, parse_mode="HTML")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feedback_dm_failed",
                extra={
                    "event": "feedback_dm_failed",
                    "admin_id": admin_id,
                    "error": str(exc),
                    "hint": "Ensure ADMIN_CHAT_IDS is your Telegram user ID and you have started the bot in DMs.",
                },
            )


@router.callback_query(F.data.startswith("feedback:"))
async def feedback_callback(callback: CallbackQuery) -> None:
    """Thumbs up/down; on down show reason, notify admin, optionally collect improvement text."""
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return
    hub = _require_hub()
    chat_id = callback.message.chat.id
    from_username = getattr(callback.from_user, "username", None) or getattr(callback.from_user, "first_name", None) or ""
    data = (callback.data or "").strip()

    async def _reply_preview() -> str:
        if callback.message.text:
            return (callback.message.text or "")[:500]
        try:
            last = await hub.cache.get_json(f"llm:last_reply:{chat_id}")
            return (last or "")[:500] if isinstance(last, str) else str(last or "")[:500]
        except Exception:  # noqa: BLE001
            return ""

    if data == "feedback:up":
        await callback.answer("Thanks!")
        return
    if data == "feedback:down":
        await callback.message.edit_reply_markup(reply_markup=feedback_reason_kb())
        await callback.answer("What was wrong?")
        return
    if data == "feedback:suggest":
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        await _set_pending_feedback_suggestion(
            chat_id,
            {"reason": "suggestion", "reply_preview": await _reply_preview(), "message_id": callback.message.message_id},
        )
        await callback.answer()
        await callback.message.answer("Type what we can do to improve in the chat — I'll pass it on personally.")
        return
    if data.startswith("feedback:reason:"):
        reason = data.replace("feedback:reason:", "", 1).lower()
        reply_preview = await _reply_preview()
        try:
            settings = await hub.user_service.get_settings(chat_id)
            prefs = dict(settings.get("feedback_prefs") or {})
            if reason == "long":
                prefs["prefers_shorter"] = True
                await hub.user_service.update_settings(chat_id, {"feedback_prefs": prefs})
        except Exception:  # noqa: BLE001
            pass
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        await _notify_admins_negative_feedback(
            from_chat_id=chat_id,
            from_username=from_username,
            reason=reason,
            reply_preview=reply_preview,
        )
        await _set_pending_feedback_suggestion(
            chat_id,
            {"reason": reason, "reply_preview": reply_preview, "message_id": callback.message.message_id},
        )
        msg = "Thanks — we'll keep it shorter next time." if reason == "long" else "Thanks for the feedback."
        await callback.answer(msg, show_alert=True)
        await callback.message.answer("Optional: type how we can improve and I'll pass it on personally.")
        return
    await callback.answer()


def _is_negative_reaction(reaction_list: list) -> bool:
    """True if the reaction list contains a thumbs-down or similar negative emoji."""
    if not reaction_list:
        return False
    negative_emojis = {"👎", "👎🏻", "👎🏼", "👎🏽", "👎🏾", "👎🏿", "😞", "🤮"}
    for r in reaction_list:
        emoji = getattr(r, "emoji", None) or getattr(r, "type", None)
        if emoji and str(emoji).strip() in negative_emojis:
            return True
        if hasattr(r, "emoji") and r.emoji and "thumbs" in str(r.emoji).lower():
            return True
    return False


@router.message_reaction()
async def message_reaction_handler(reaction_update: MessageReactionUpdated) -> None:
    """When someone uses Telegram's native reaction (e.g. 👎) on a bot message, notify admins."""
    if not _is_negative_reaction(reaction_update.new_reaction or []):
        return
    hub = _require_hub()
    chat_id = reaction_update.chat.id
    user = reaction_update.user
    from_username = (getattr(user, "username", None) or getattr(user, "first_name", None) or "—") if user else "—"
    reply_preview = ""
    try:
        last = await hub.cache.get_json(f"llm:last_reply:{chat_id}")
        reply_preview = (last or "")[:500] if isinstance(last, str) else str(last or "")[:500]
    except Exception:  # noqa: BLE001
        pass
    await _notify_admins_negative_feedback(
        from_chat_id=chat_id,
        from_username=from_username,
        reason="reaction (message reaction)",
        reply_preview=reply_preview or "(no preview)",
    )


@router.callback_query(F.data.startswith("confirm:understood:"))
async def confirm_understood_callback(callback: CallbackQuery) -> None:
    """User confirmed or rejected the 'You want: X. Correct?' summary."""
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return
    data = (callback.data or "").strip()
    if data.endswith(":yes"):
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:  # noqa: BLE001
            pass
        await callback.answer("Got it.")
        return
    if data.endswith(":no"):
        try:
            await callback.message.edit_text(
                "Rephrase what you want — I'll match it.",
                reply_markup=None,
            )
        except Exception:  # noqa: BLE001
            await callback.message.answer("Rephrase what you want — I'll match it.")
        await callback.answer()
        return
    await callback.answer()


@router.callback_query(F.data.startswith("confirm:clear_alerts:"))
async def confirm_clear_alerts_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return
    hub = _require_hub()
    chat_id = callback.message.chat.id
    data = (callback.data or "").strip()
    suffix = data.replace("confirm:clear_alerts:", "", 1)
    if suffix == "no":
        try:
            await callback.message.edit_text("Cancelled. Alerts unchanged.", reply_markup=None)
        except Exception:
            await callback.message.answer("Cancelled. Alerts unchanged.")
        await callback.answer()
        return
    # suffix is count (e.g. "3") — clear all for this user
    count = await hub.alerts_service.clear_user_alerts(chat_id)
    try:
        await callback.message.edit_text(f"Cleared {count} alerts.", reply_markup=None)
    except Exception:
        await callback.message.answer(f"Cleared {count} alerts.")
    await callback.answer()


@router.callback_query(F.data.startswith("cmd:"))
async def command_menu_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return
    hub = _require_hub()
    chat_id = callback.message.chat.id
    data = callback.data or ""
    parts = data.split(":")
    if len(parts) < 2:
        await callback.answer()
        return

    def _menu_for(name: str):
        mapping = {
            "alpha": ("Pick analysis shortcut.", alpha_quick_menu()),
            "watch": ("Pick watch shortcut.", watch_quick_menu()),
            "chart": ("Pick chart shortcut.", chart_quick_menu()),
            "heatmap": ("Pick heatmap symbol.", heatmap_quick_menu()),
            "rsi": ("Pick RSI scanner preset.", rsi_quick_menu()),
            "ema": ("Pick EMA scanner preset.", ema_quick_menu()),
            "news": ("Pick news mode.", news_quick_menu()),
            "alert": ("Pick alert action.", alert_quick_menu()),
            "findpair": ("Pick find mode.", findpair_quick_menu()),
            "setup": ("Choose setup input mode.", setup_quick_menu()),
            "scan": ("Pick chain first.", scan_quick_menu()),
            "giveaway": ("Pick giveaway action.", giveaway_menu(is_admin=hub.giveaway_service.is_admin(callback.from_user.id))),
        }
        return mapping.get(name)

    if parts[1] == "menu":
        menu = _menu_for(parts[2] if len(parts) > 2 else "")
        if menu:
            await callback.message.answer(menu[0], reply_markup=menu[1])
        await callback.answer()
        return

    async def _dispatch_with_typing(synthetic_text: str) -> None:
        async def _run() -> None:
            await _dispatch_command_text(callback.message, synthetic_text)
            await callback.answer()

        await _run_with_typing_lock(callback.bot, chat_id, _run)

    action = parts[1]
    if action == "alpha":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": ""})
            await callback.message.answer("Send symbol and optional tf, e.g. `SOL 4h`.")
            await callback.answer()
            return
        if len(parts) >= 4:
            await _dispatch_with_typing(f"{parts[2]} {parts[3]}")
            return
    if action == "watch":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "watch "})
            await callback.message.answer("Send symbol and optional tf, e.g. `BTC 1h`.")
            await callback.answer()
            return
        if len(parts) >= 4:
            await _dispatch_with_typing(f"watch {parts[2]} {parts[3]}")
            return
    if action == "chart":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "chart "})
            await callback.message.answer("Send symbol and optional tf, e.g. `ETH 4h`.")
            await callback.answer()
            return
        if len(parts) >= 4:
            await _dispatch_with_typing(f"chart {parts[2]} {parts[3]}")
            return
    if action == "heatmap":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "heatmap "})
            await callback.message.answer("Send symbol, e.g. `BTC`.")
            await callback.answer()
            return
        if len(parts) >= 3:
            await _dispatch_with_typing(f"heatmap {parts[2]}")
            return
    if action == "rsi":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "rsi "})
            await callback.message.answer("Send format: <code>1h oversold top 10 rsi14</code>.")
            await callback.answer()
            return
        if len(parts) >= 6:
            await callback.answer("Scanning…")
            _rsi_tf = parts[2]
            _rsi_mode = "overbought" if parts[3] == "overbought" else "oversold"
            _rsi_limit = max(1, min(_as_int(parts[4], 10), 20))
            _rsi_len = max(2, min(_as_int(parts[5], 14), 50))

            async def _run_rsi() -> None:
                try:
                    payload = await hub.rsi_scanner_service.scan(
                        timeframe=_rsi_tf,
                        mode=_rsi_mode,
                        limit=_rsi_limit,
                        rsi_length=_rsi_len,
                        symbol=None,
                    )
                    await callback.message.answer(rsi_scan_template(payload))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("rsi_scan_button_failed", extra={"event": "rsi_button_error", "error": str(exc)})
                    await callback.message.answer("rsi scan hit an error — try again in a moment.")
            await _run_with_typing_lock(callback.bot, chat_id, _run_rsi)
            return
    if action == "ema":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "ema "})
            await callback.message.answer("Send format: `200 4h top 10`.")
            await callback.answer()
            return
        if len(parts) >= 5:
            await _dispatch_with_typing(f"ema {parts[2]} {parts[3]} top {parts[4]}")
            return
    if action == "news" and len(parts) >= 4:
        await _dispatch_with_typing(f"news {parts[2]} {parts[3]}")
        return
    if action == "alert":
        if len(parts) >= 3 and parts[2] == "create":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "alert "})
            await callback.message.answer(
                "send me the alert details:\n\n"
                "<code>SOL 100 above</code>\n"
                "<code>BTC 66000 below</code>\n"
                "<code>ETH 3200</code>  ← defaults to cross\n\n"
                "<i>format: symbol  price  [above | below | cross]</i>"
            )
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "list":
            await _dispatch_with_typing("list my alerts")
            return
        if len(parts) >= 3 and parts[2] == "clear":
            alerts = await hub.alerts_service.list_alerts(chat_id)
            count = len(alerts)
            if count == 0:
                await callback.message.answer("No alerts to clear.")
                await callback.answer()
                return
            from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
            kb = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="Yes, clear all", callback_data=f"confirm:clear_alerts:{count}")],
                [InlineKeyboardButton(text="Cancel", callback_data="confirm:clear_alerts:no")],
            ])
            await callback.message.answer(
                f"Clear all <b>{count}</b> alerts? This can't be undone.",
                reply_markup=kb,
            )
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "pause":
            await _dispatch_with_typing("pause alerts")
            return
        if len(parts) >= 3 and parts[2] == "resume":
            await _dispatch_with_typing("resume alerts")
            return
        if len(parts) >= 3 and parts[2] == "delete":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "delete alert "})
            await callback.message.answer("Send alert id, e.g. `12`.")
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "clear_symbol":
            await _cmd_wizard_set(chat_id, {"step": "alert_clear_symbol"})
            await callback.message.answer("Send symbol to clear, e.g. `SOL`.")
            await callback.answer()
            return
    if action == "findpair":
        if len(parts) >= 3 and parts[2] == "price":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "coin around "})
            await callback.message.answer("Send target price, e.g. `0.155`.")
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "query":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "find pair "})
            await callback.message.answer("Send name/ticker/context, e.g. `xion`.")
            await callback.answer()
            return
    if action == "setup":
        if len(parts) >= 3 and parts[2] == "wizard":
            await _wizard_set(chat_id, {"step": "symbol", "data": {}})
            await callback.message.answer("Tradecheck wizard: send symbol (e.g., ETH).")
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "freeform":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": ""})
            await callback.message.answer("Paste setup text, e.g. `entry 2100 stop 2165 targets 2043 2027 1991`.")
            await callback.answer()
            return
    if action == "scan" and len(parts) >= 3:
        chain = "solana" if parts[2] == "solana" else "tron"
        await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": f"scan {chain} "})
        await callback.message.answer(f"Paste {chain} address.")
        await callback.answer()
        return
    if action == "alertdel" and len(parts) >= 3:
        await _dispatch_with_typing(f"delete alert {parts[2]}")
        return

    await callback.answer()


@router.callback_query(F.data.startswith("gw:"))
async def giveaway_menu_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return
    hub = _require_hub()
    chat_id = callback.message.chat.id
    data = callback.data or ""
    parts = data.split(":")
    action = parts[1] if len(parts) > 1 else ""
    user_id = callback.from_user.id if callback.from_user else None

    async def _run_and_answer(runner) -> None:
        async def _run() -> None:
            await runner()
            await callback.answer()

        await _run_with_typing_lock(callback.bot, chat_id, _run)

    if action == "start":
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return
        await callback.message.answer("Pick duration.", reply_markup=giveaway_duration_menu())
        await callback.answer()
        return
    if action == "dur" and len(parts) >= 3:
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return
        duration_seconds = max(30, _as_int(parts[2], 600))
        await callback.message.answer("Pick number of winners.", reply_markup=giveaway_winners_menu(duration_seconds))
        await callback.answer()
        return
    if action == "win" and len(parts) >= 4:
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return
        duration_seconds = max(30, _as_int(parts[2], 600))
        winners = max(1, min(_as_int(parts[3], 1), 5))
        await _cmd_wizard_set(chat_id, {"step": "giveaway_prize", "duration_seconds": duration_seconds, "winners": winners})
        await callback.message.answer("Send giveaway prize text, e.g. `50 USDT`.")
        await callback.answer()
        return
    if action == "join":
        if not user_id:
            await callback.answer("No user", show_alert=True)
            return

        async def _runner() -> None:
            try:
                payload = await hub.giveaway_service.join_active(chat_id, user_id)
            except Exception as exc:  # noqa: BLE001
                await callback.message.answer(f"couldn't join: {exc}")
                return
            gw_id = payload.get("giveaway_id", "?")
            participants = payload.get("participants", "?")
            await callback.message.answer(
                f"you're in 🎉\n\n"
                f"giveaway <b>#{gw_id}</b>\n"
                f"participants so far: <b>{participants}</b>\n\n"
                f"<i>good luck, fren.</i>"
            )

        await _run_and_answer(_runner)
        return
    if action == "status":
        async def _runner() -> None:
            payload = await hub.giveaway_service.status(chat_id)
            await callback.message.answer(giveaway_status_template(payload))

        await _run_and_answer(_runner)
        return
    if action == "end":
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return

        async def _runner() -> None:
            try:
                payload = await hub.giveaway_service.end_giveaway(chat_id, user_id)
            except Exception as exc:  # noqa: BLE001
                await callback.message.answer(f"couldn't end giveaway: {exc}")
                return
            if payload.get("winner_user_id"):
                await callback.message.answer(
                    f"🏆 giveaway <b>#{payload.get('giveaway_id')}</b> closed.\n\n"
                    f"winner: <code>{payload.get('winner_user_id')}</code>\n"
                    f"prize: <b>{payload.get('prize', '—')}</b>"
                )
            else:
                note = payload.get("note") or "no participants"
                await callback.message.answer(f"giveaway ended with no winner. {note}")

        await _run_and_answer(_runner)
        return
    if action == "reroll":
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return

        async def _runner() -> None:
            try:
                payload = await hub.giveaway_service.reroll(chat_id, user_id)
            except Exception as exc:  # noqa: BLE001
                await callback.message.answer(f"reroll failed: {exc}")
                return
            await callback.message.answer(
                f"🔄 reroll done for giveaway <b>#{payload.get('giveaway_id')}</b>\n\n"
                f"new winner: <code>{payload.get('winner_user_id')}</code>\n"
                f"prev winner: <code>{payload.get('previous_winner_user_id', '—')}</code>"
            )

        await _run_and_answer(_runner)
        return

    await callback.answer()


@router.callback_query(F.data.startswith("settings:"))
async def settings_callbacks(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    hub = _require_hub()
    data = callback.data or ""
    chat_id = callback.message.chat.id

    if data == "settings:toggle:anon_mode":
        cur = await hub.user_service.get_settings(chat_id)
        new_settings = await hub.user_service.update_settings(chat_id, {"anon_mode": not cur.get("anon_mode", True)})
    elif data == "settings:toggle:formal_mode":
        cur = await hub.user_service.get_settings(chat_id)
        new_settings = await hub.user_service.update_settings(chat_id, {"formal_mode": not cur.get("formal_mode", False)})
    elif data == "settings:toggle:reply_in_dm":
        cur = await hub.user_service.get_settings(chat_id)
        new_settings = await hub.user_service.update_settings(chat_id, {"reply_in_dm": not cur.get("reply_in_dm", False)})
    elif data == "settings:toggle:ultra_brief":
        cur = await hub.user_service.get_settings(chat_id)
        new_settings = await hub.user_service.update_settings(chat_id, {"ultra_brief": not cur.get("ultra_brief", False)})
    elif data.startswith("settings:set:"):
        _, _, key, value = data.split(":", 3)
        new_settings = await hub.user_service.update_settings(chat_id, {key: value})
    else:
        await callback.answer("Unknown settings action", show_alert=True)
        return

    await callback.message.edit_text(settings_text(new_settings), reply_markup=settings_menu(new_settings))
    await callback.answer("Updated")


@router.callback_query(F.data.startswith("set_alert:"))
async def set_alert_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    symbol = (callback.data or "").split(":", 1)[1]
    await _set_pending_alert(callback.message.chat.id, symbol)
    await callback.message.answer(
        f"send me the target price for <b>{symbol}</b>.\n"
        f"e.g. <code>{symbol} 100</code> or <code>alert {symbol} 100 above</code>"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("show_levels:"))
async def show_levels_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    hub = _require_hub()
    symbol = (callback.data or "").split(":", 1)[1]
    payload = await hub.cache.get_json(f"last_analysis:{callback.message.chat.id}:{symbol}")
    if not payload:
        await callback.answer("No cached levels — run a fresh analysis first.", show_alert=True)
        return
    entry = payload.get("entry", "—")
    tp1 = payload.get("tp1", "—")
    tp2 = payload.get("tp2", "—")
    sl = payload.get("sl", "—")
    await callback.message.answer(
        f"<b>{symbol}</b> key levels\n\n"
        f"entry    <code>{entry}</code>\n"
        f"target 1  <code>{tp1}</code>\n"
        f"target 2  <code>{tp2}</code>\n"
        f"stop      <code>{sl}</code>"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("why:"))
async def why_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    hub = _require_hub()
    symbol = (callback.data or "").split(":", 1)[1]
    payload = await hub.cache.get_json(f"last_analysis:{callback.message.chat.id}:{symbol}")
    if not payload:
        await callback.answer("No context saved — run a fresh analysis first.", show_alert=True)
        return
    bullets = payload.get("why", [])
    if bullets:
        bullet_lines = "\n".join(f"· {w}" for w in bullets)
        text = f"<b>why {symbol}</b>\n\n{bullet_lines}"
    else:
        summary = payload.get("summary", "")
        text = f"<b>why {symbol}</b>\n\n{summary or 'no reasoning available for this setup.'}"
    await callback.message.answer(text)
    await callback.answer()


@router.callback_query(F.data.startswith("refresh:"))
async def refresh_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":", 1)[1]
        settings = await hub.user_service.get_settings(chat_id)
        payload = await hub.analysis_service.analyze(
            symbol,
            timeframes=_analysis_timeframes_from_settings(settings),
            ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=False,
            include_news=False,
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(callback.message, symbol, analysis_text, direction=payload.get("side"))
        await callback.answer("Refreshed")

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("details:"))
async def details_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":", 1)[1]
        settings = await hub.user_service.get_settings(chat_id)
        payload = await hub.analysis_service.analyze(
            symbol,
            timeframes=_analysis_timeframes_from_settings(settings),
            ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=True,
            include_news=True,
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
            detailed=True,
        )
        await _send_ghost_analysis(callback.message, symbol, analysis_text, direction=payload.get("side"))
        await callback.answer("Detailed mode")

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("derivatives:"))
async def derivatives_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":", 1)[1]
        payload = await hub.analysis_service.deriv_adapter.get_funding_and_oi(symbol)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            exchange=str(payload.get("source") or ""),
            market_kind="perp",
            symbol=symbol,
            context="derivatives",
        )
        funding = payload.get("funding_rate")
        oi = payload.get("open_interest")
        source = payload.get("source") or payload.get("source_line") or "live"
        funding_str = f"{float(funding)*100:.4f}%" if funding is not None else "n/a"
        oi_str = f"${float(oi)/1_000_000:.2f}B" if oi is not None else "n/a"
        await callback.message.answer(
            f"<b>{symbol}</b> derivatives\n\n"
            f"funding rate  <code>{funding_str}</code>\n"
            f"open interest <code>{oi_str}</code>\n\n"
            f"<i>source: {source}</i>"
        )
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("catalysts:"))
async def catalysts_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":", 1)[1]
        headlines = await hub.news_service.get_asset_headlines(symbol, limit=3)
        if not headlines:
            await callback.message.answer(f"no fresh catalysts for <b>{symbol}</b> right now. check back later.")
            await callback.answer()
            return
        lines = [f"<b>{symbol} catalysts</b>\n"]
        for item in headlines[:3]:
            title = item.get("title", "")
            url = item.get("url", "")
            source = item.get("source", "")
            if url:
                lines.append(f'· <a href="{url}">{title}</a>')
            else:
                lines.append(f"· {title}")
            if source:
                lines.append(f"  <i>{source}</i>")
            lines.append("")
        await callback.message.answer("\n".join(lines).strip(), disable_web_page_preview=True)
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("backtest:"))
async def backtest_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return
    if "backtest" not in _settings.feature_flags_set():
        await callback.answer("Backtest is disabled.", show_alert=True)
        return

    symbol = (callback.data or "").split(":", 1)[1]
    await callback.message.answer(
        f"drop your <b>{symbol}</b> trade details and i'll check it.\n\n"
        f"format: <code>{symbol} entry 2100 stop 2060 targets 2140 2180 2220 timeframe 1h</code>\n\n"
        "<i>or just paste it in natural language — i'll figure it out.</i>"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("save_wallet:"))
async def save_wallet_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, chain, address = (callback.data or "").split(":", 2)
        await hub.wallet_service.scan(chain, address, chat_id=chat_id, save=True)
        await callback.answer("Wallet saved", show_alert=True)

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:analysis:"))
async def quick_analysis_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    with suppress(Exception):
        await callback.answer("Analyzing…")
    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":")[-1]
        settings = await hub.user_service.get_settings(chat_id)
        payload = await hub.analysis_service.analyze(
            symbol,
            timeframes=_analysis_timeframes_from_settings(settings),
            ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=False,
            include_news=False,
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(callback.message, symbol, analysis_text, direction=payload.get("side"))
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:analysis_tf:"))
async def quick_analysis_tf_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    with suppress(Exception):
        await callback.answer("Analyzing…")
    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, symbol, timeframe = (callback.data or "").split(":", 3)
        settings = await hub.user_service.get_settings(chat_id)
        payload = await hub.analysis_service.analyze(
            symbol.upper(),
            timeframe=timeframe,
            timeframes=[timeframe],
            ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=False,
            include_news=False,
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol.upper()}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol.upper(), payload.get("side"), payload)
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol.upper(),
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(callback.message, symbol.upper(), analysis_text, direction=payload.get("side"))
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:chart:"))
async def quick_chart_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, symbol, timeframe = (callback.data or "").split(":", 3)
        img, _ = await hub.chart_service.render_chart(symbol=symbol.upper(), timeframe=timeframe)
        await callback.message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol.upper()}-{timeframe}.png"),
            caption=f"{symbol.upper()} {timeframe} chart.",
        )
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:heatmap:"))
async def quick_heatmap_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, symbol = (callback.data or "").split(":", 2)
        img, _ = await hub.heatmap_service.render(symbol=symbol.upper())
        await callback.message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol.upper()}-heatmap.png"),
            caption=f"{symbol.upper()} order-book heatmap.",
        )
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:rsi:"))
async def quick_rsi_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, mode, timeframe, limit_raw = (callback.data or "").split(":", 4)
        limit = max(1, min(_as_int(limit_raw, 5), 20))
        payload = await hub.rsi_scanner_service.scan(
            timeframe=timeframe,
            mode="overbought" if mode == "overbought" else "oversold",
            limit=limit,
            rsi_length=14,
            symbol=None,
        )
        await callback.message.answer(rsi_scan_template(payload))
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:news:"))
async def quick_news_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, mode = (callback.data or "").split(":", 2)
        mode_norm = "openai" if mode == "openai" else "crypto"
        topic = "openai" if mode_norm == "openai" else "crypto"
        payload = await hub.news_service.get_digest(topic=topic, mode=mode_norm, limit=6)
        await callback.message.answer(news_template(payload), parse_mode="HTML")
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("define:"))
async def define_easter_egg_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        settings = await hub.user_service.get_settings(chat_id)
        parts = (callback.data or "").split(":")
        action = parts[1] if len(parts) > 1 else ""

        if action == "analyze":
            timeframe = parts[2] if len(parts) > 2 else "1h"
            symbol = "DEFINE"
            payload = await hub.analysis_service.analyze(
                symbol,
                timeframe=timeframe,
                timeframes=[timeframe],
                ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
                rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
                include_derivatives=True,
                include_news=True,
            )
            await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
            await _remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
            analysis_text = await _render_analysis_text(
                payload=payload,
                symbol=symbol,
                direction=payload.get("side"),
                settings=settings,
                chat_id=chat_id,
            )
            await _send_ghost_analysis(callback.message, symbol, analysis_text)
            await callback.answer()
            return

        if action == "chart":
            timeframe = parts[2] if len(parts) > 2 else "1h"
            try:
                img, _ = await hub.chart_service.render_chart(symbol="DEFINE", timeframe=timeframe)
            except Exception:  # noqa: BLE001
                await callback.message.answer("Drop a real ticker for chart, e.g. `chart SOL 1h`.")
                await callback.answer()
                return
            await callback.message.answer_photo(
                BufferedInputFile(img, filename=f"DEFINE-{timeframe}.png"),
                caption=f"DEFINE {timeframe} chart.",
            )
            await callback.answer()
            return

        if action == "heatmap":
            symbol = "DEFINE"
            try:
                img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
            except Exception:  # noqa: BLE001
                symbol = "BTC"
                img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
            await callback.message.answer_photo(
                BufferedInputFile(img, filename=f"{symbol}_heatmap.png"),
                caption=(
                    f"{meta['pair']} orderbook heatmap\n"
                    f"Best bid: {meta['best_bid']:.6f} | Best ask: {meta['best_ask']:.6f}\n"
                    f"Bid wall: {meta['bid_wall']:.6f} | Ask wall: {meta['ask_wall']:.6f}"
                ),
            )
            await callback.answer()
            return

        if action == "alert":
            await _set_pending_alert(chat_id, "DEFINE")
            await callback.message.answer("Send alert level for DEFINE, e.g. DEFINE 0.50")
            await callback.answer()
            return

        if action == "news":
            payload = await hub.news_service.get_digest(topic="DEFINE", mode="crypto", limit=6)
            await callback.message.answer(news_template(payload), parse_mode="HTML")
            await callback.answer()
            return

        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("top:"))
async def top_rsi_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        parts = (callback.data or "").split(":")
        mode = parts[1] if len(parts) > 1 else "oversold"
        timeframe = parts[2] if len(parts) > 2 else "1h"
        mode = "overbought" if mode == "overbought" else "oversold"
        payload = await hub.rsi_scanner_service.scan(
            timeframe=timeframe,
            mode=mode,
            limit=10,
            rsi_length=14,
            symbol=None,
        )
        await callback.message.answer(rsi_scan_template(payload))
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.message(F.text)
async def route_text(message: Message) -> None:
    hub = _require_hub()
    text = message.text or ""
    chat_id = message.chat.id
    raw_text = text.strip()

    if raw_text.startswith("/"):
        return

    if message.chat.type in ("group", "supergroup"):
        ft_match = re.search(r"\bfree\s*talk\s*mode\s*(on|off)\b", raw_text, flags=re.IGNORECASE)
        if ft_match:
            if not await _is_group_admin(message):
                await message.answer("Only group admins can toggle free talk mode.")
                return
            enabled = ft_match.group(1).lower() == "on"
            await _set_group_free_talk(message.chat.id, enabled)
            await message.answer(f"Group free talk mode {'ON' if enabled else 'OFF'}.")
            return

    if message.chat.type in ("group", "supergroup"):
        free_talk_enabled = await _group_free_talk_enabled(message.chat.id)
        mentioned = _mentions_bot(text, hub.bot_username)
        reply_to_bot = _is_reply_to_bot(message, hub)
        clear_intent = _looks_like_clear_intent(text)
        if not (free_talk_enabled or mentioned or reply_to_bot or clear_intent):
            return
        text = _strip_bot_mention(text, hub.bot_username)
        if not text:
            await message.answer("Send a request in plain text, e.g. `SOL long`, `cpi news`, `chart btc 1h`, or `alert me when SOL hits 50`.")
            return

    if re.search(r"\b(my|show)\s+(user\s*)?id\b|\bwhat('?s| is)\s+my\s+id\b", text, flags=re.IGNORECASE):
        if not message.from_user:
            await message.answer("Could not read your user id from this update.")
            return
        await message.answer(
            f"Your user id: {message.from_user.id}\n"
            f"Current chat id: {message.chat.id}"
        )
        return

    if _is_source_query(text):
        await message.answer(await _source_reply_for_chat(chat_id, text))
        return

    if not await _acquire_message_once(message):
        logger.info(
            "duplicate_message_ignored",
            extra={
                "event": "duplicate_message_ignored",
                "chat_id": chat_id,
                "message_id": message.message_id,
            },
        )
        return

    if not await _check_req_limit(chat_id):
        await message.answer(
            "slow down fren — rate limit hit. resets in ~1 min.",
            reply_to_message_id=message.message_id,
        )
        return

    # Fast-path: pure greetings — respond immediately without LLM or market data.
    if _GREETING_RE.match(raw_text.strip()):
        low = raw_text.strip().lower()
        rid = message.message_id
        if low.startswith("gn") or "night" in low:
            await message.answer(random.choice(_GN_REPLIES), reply_to_message_id=rid)
        else:
            name = message.from_user.first_name if message.from_user else "fren"
            reply = await _market_aware_gm_reply(hub, name)
            await message.answer(reply, reply_to_message_id=rid)
        return

    lock = _chat_lock(chat_id)
    if lock.locked():
        # Avoid flooding busy notices if user sends many messages quickly.
        if await hub.cache.set_if_absent(f"busy_notice:{chat_id}", ttl=5):
            await message.answer("still on it fren — give me a few seconds.", reply_to_message_id=message.message_id)
        return

    start_ts = datetime.now(timezone.utc)
    stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(message.bot, chat_id, stop))
    try:
        async with lock:
            pending_fb = await _get_pending_feedback_suggestion(chat_id)
            if pending_fb:
                await _clear_pending_feedback_suggestion(chat_id)
                from_username = getattr(message.from_user, "username", None) or getattr(message.from_user, "first_name", None) or ""
                await _notify_admins_negative_feedback(
                    from_chat_id=chat_id,
                    from_username=from_username,
                    reason=str(pending_fb.get("reason") or "other"),
                    reply_preview=str(pending_fb.get("reply_preview") or ""),
                    improvement_text=text.strip() or None,
                )
                await message.answer("Thanks — I've passed that on personally. We'll use it to improve.")
                return

            cmd_wizard = await _cmd_wizard_get(chat_id)
            if cmd_wizard:
                step = str(cmd_wizard.get("step") or "").strip().lower()
                if step == "dispatch_text":
                    prefix = str(cmd_wizard.get("prefix") or "")
                    await _cmd_wizard_clear(chat_id)
                    typed = text.strip()
                    if not typed:
                        await message.answer("send the details and i'll run it.")
                        return
                    await _dispatch_command_text(message, f"{prefix}{typed}".strip())
                    return
                if step == "giveaway_prize":
                    await _cmd_wizard_clear(chat_id)
                    if not message.from_user:
                        await message.answer("Could not identify sender.")
                        return
                    prize = text.strip().strip("'\"") or "Prize"
                    duration_seconds = max(30, _as_int(cmd_wizard.get("duration_seconds"), 600))
                    winners_requested = max(1, min(_as_int(cmd_wizard.get("winners"), 1), 5))
                    try:
                        payload = await hub.giveaway_service.start_giveaway(
                            group_chat_id=chat_id,
                            admin_chat_id=message.from_user.id,
                            duration_seconds=duration_seconds,
                            prize=prize,
                        )
                    except Exception as exc:  # noqa: BLE001
                        await message.answer(str(exc))
                        return
                    note = ""
                    if winners_requested > 1:
                        note = "\nNote: multi-winner draw runs as sequential rerolls after first winner."
                    await message.answer(
                        f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\n"
                        f"Ends at: {payload['end_time']}\nUsers enter with /join or /giveaway join{note}",
                        reply_markup=giveaway_menu(is_admin=True),
                    )
                    return
                if step == "alert_clear_symbol":
                    await _cmd_wizard_clear(chat_id)
                    symbol = text.strip().upper().lstrip("$")
                    if not re.fullmatch(r"[A-Z0-9]{2,20}", symbol):
                        await message.answer("Invalid symbol. Send a ticker like SOL.")
                        return
                    count = await hub.alerts_service.delete_alerts_by_symbol(chat_id, symbol)
                    await message.answer(f"Cleared {count} alerts for {symbol}.")
                    return

            wizard = await _wizard_get(chat_id)
            if wizard:
                step = wizard.get("step")
                data = wizard.get("data", {})
                if step == "symbol":
                    data["symbol"] = text.strip().upper()
                    await _wizard_set(chat_id, {"step": "timeframe", "data": data})
                    await message.answer("Timeframe? (15m / 1h / 4h)")
                    return
                if step == "timeframe":
                    data["timeframe"] = text.strip().lower()
                    await _wizard_set(chat_id, {"step": "timestamp", "data": data})
                    await message.answer("Timestamp? (ISO, yesterday, or 2 hours ago)")
                    return
                if step == "timestamp":
                    ts = parse_timestamp(text)
                    if not ts:
                        await message.answer("Could not parse timestamp. Try 'yesterday' or ISO datetime.")
                        return
                    data["timestamp"] = ts.isoformat()
                    await _wizard_set(chat_id, {"step": "levels", "data": data})
                    await message.answer("Send levels: entry <x> stop <y> targets <a> <b> ...")
                    return
                if step == "levels":
                    entry_m = re.search(r"entry\s*([0-9.]+)", text, re.IGNORECASE)
                    stop_m = re.search(r"stop\s*([0-9.]+)", text, re.IGNORECASE)
                    targets_m = re.search(r"targets?\s*([0-9.\s]+)", text, re.IGNORECASE)
                    if not entry_m or not stop_m or not targets_m:
                        await message.answer("Format: entry <x> stop <y> targets <a> <b>")
                        return
                    targets = [float(x) for x in re.findall(r"[0-9.]+", targets_m.group(1))]
                    data.update(
                        {
                            "entry": float(entry_m.group(1)),
                            "stop": float(stop_m.group(1)),
                            "targets": targets,
                            "timestamp": datetime.fromisoformat(data["timestamp"]),
                            "mode": "ambiguous",
                        }
                    )
                    result = await hub.trade_verify_service.verify(**data)
                    await _save_trade_check(chat_id, data, result)
                    await _remember_source_context(
                        chat_id,
                        source_line=str(result.get("source_line") or ""),
                        symbol=data["symbol"],
                        context="trade check",
                    )
                    _wiz_settings = await hub.user_service.get_settings(chat_id)
                    await message.answer(trade_verification_template(result, _wiz_settings))
                    await _wizard_clear(chat_id)
                    return

            pending_alert_symbol = await _get_pending_alert(chat_id)
            if pending_alert_symbol:
                price_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
                if price_match:
                    target = float(price_match.group(1))
                    alert = await hub.alerts_service.create_alert(chat_id, pending_alert_symbol, "cross", target, source="button")
                    await _clear_pending_alert(chat_id)
                    await _remember_source_context(
                        chat_id,
                        exchange=alert.source_exchange,
                        market_kind=alert.market_kind,
                        instrument_id=alert.instrument_id,
                        symbol=pending_alert_symbol,
                        context="alert",
                    )
                    await message.answer(
                        f"🔔 alert set — <b>{pending_alert_symbol}</b> crosses <b>${target:,.2f}</b>.\n"
                        "i'll ping you the moment it hits. don't get liquidated.",
                        reply_markup=alert_created_menu(pending_alert_symbol),
                    )
                    return

            settings = await hub.user_service.get_settings(chat_id)
            if not settings.get("disclaimer_seen"):
                await message.answer(
                    "⚠️ <b>Disclaimer</b>: This bot is for info only, not financial advice. "
                    "Data may be delayed. Trade at your own risk."
                )
                await hub.user_service.update_settings(chat_id, {"disclaimer_seen": True})

            text_lower = text.lower().strip()

            # Repair: user says that's not what I meant / wrong / I meant — restate and retry
            _repair_re = re.compile(
                r"\b(that'?s not what i meant|no(pe)?\s*(that'?s wrong|that'?s not it)|wrong|i meant|i wanted|not that|something else)\b",
                re.IGNORECASE,
            )
            if _repair_re.search(text_lower) and hub.llm_client:
                history = await _get_chat_history(chat_id)
                if len(history) >= 2:
                    last_user = next((h["content"] for h in reversed(history) if h.get("role") == "user"), "")
                    last_assistant = next((h["content"] for h in reversed(history) if h.get("role") == "assistant"), "")
                    repair_prompt = (
                        f"The user just said: \"{text[:200]}\". "
                        f"Your previous reply was: \"{last_assistant[:400]}\". "
                        f"Their previous message was: \"{last_user[:200]}\". "
                        "Restate in one short line what they likely want, then answer that. Be brief."
                    )
                    try:
                        repair_reply = await hub.llm_client.reply(repair_prompt, history=history[-6:])
                        if repair_reply and repair_reply.strip():
                            await _send_llm_reply(message, repair_reply.strip(), settings, user_message=text, add_quick_replies=True)
                            return
                    except Exception:  # noqa: BLE001
                        pass

            # Special Ghost Easter eggs / overrides
            if "define trading" in text_lower or ("define" in text_lower and len(text_lower.split()) <= 3):
                await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
                await asyncio.sleep(0.8)

                funny_line = (
                    "the art of losing money faster than a casino while staring at candles "
                    "until your eyes bleed. buy low, sell high, don't get rekt"
                )
                await message.answer(funny_line, reply_markup=_define_keyboard())
                return

            # Bonus: catch casual scanner phrases and route directly.
            if "rsi top" in text_lower or "overbought list" in text_lower or "strong coins" in text_lower:
                if "oversold" in text_lower:
                    await _dispatch_command_text(message, "rsi top 10 1h oversold")
                elif "overbought" in text_lower:
                    await _dispatch_command_text(message, "rsi top 10 1h overbought")
                elif "rsi top" in text_lower:
                    await _dispatch_command_text(message, text_lower)
                else:
                    await _dispatch_command_text(message, "coins to watch 5")
                return

            followup_context = await _recent_analysis_context(chat_id)
            if _looks_like_analysis_followup(text, followup_context):
                followup_reply = await _llm_followup_reply(
                    text,
                    followup_context or {},
                    chat_id=chat_id,
                )
                if followup_reply:
                    await message.answer(followup_reply)
                    return

            parsed = parse_message(text)
            chat_mode = _openai_chat_mode()

            if hub.llm_client and chat_mode == "chat_only":
                llm_reply = await _llm_market_chat_reply(text, settings, chat_id=chat_id)
                if llm_reply:
                    await _send_llm_reply(message, llm_reply, settings, user_message=text)
                    return
                fallback = "couldn't reach the brain — try again in a sec, fren." if _is_definition_question(text) else "signal unclear fren — try rephrasing or drop a ticker."
                await message.answer(fallback, reply_to_message_id=message.message_id)
                return

            if hub.llm_client and chat_mode == "llm_first":
                routed = await _llm_route_message(text)
                if routed:
                    try:
                        if await _handle_routed_intent(message, settings, routed):
                            return
                    except Exception:  # noqa: BLE001
                        pass
                llm_reply = await _llm_market_chat_reply(text, settings, chat_id=chat_id)
                if llm_reply:
                    await _send_llm_reply(message, llm_reply, settings, user_message=text)
                    return

            if chat_mode in {"hybrid", "tool_first"} and (parsed.intent == Intent.UNKNOWN or (parsed.requires_followup and parsed.intent == Intent.UNKNOWN)):
                routed = await _llm_route_message(text)
                if routed:
                    try:
                        if await _handle_routed_intent(message, settings, routed):
                            return
                    except Exception:  # noqa: BLE001
                        pass

            if parsed.requires_followup:
                if parsed.intent == Intent.UNKNOWN:
                    llm_reply = await _llm_market_chat_reply(text, settings, chat_id=chat_id)
                    if llm_reply:
                        await _send_llm_reply(message, llm_reply, settings, user_message=text)
                        return
                    english_phrase = is_likely_english_phrase(text)
                    symbol_hint = None if english_phrase else _extract_action_symbol_hint(text)
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
                    return
                if parsed.intent == Intent.ANALYSIS and not parsed.entities.get("symbol"):
                    kb = simple_followup(
                        [
                            ("BTC", "quick:analysis:BTC"),
                            ("ETH", "quick:analysis:ETH"),
                            ("SOL", "quick:analysis:SOL"),
                        ]
                    )
                    await message.answer(parsed.followup_question or "Need one detail.", reply_markup=kb)
                    return
                await message.answer(
                    parsed.followup_question or clarifying_question(None),
                    reply_to_message_id=message.message_id,
                )
                return

            try:
                if await _handle_parsed_intent(message, parsed, settings):
                    return
                llm_reply = await _llm_market_chat_reply(text, settings, chat_id=chat_id)
                if llm_reply:
                    await _send_llm_reply(message, llm_reply, settings, user_message=text)
                    return
                symbol_hint = _extract_action_symbol_hint(text)
                await message.answer(
                    parsed.followup_question or clarifying_question(symbol_hint),
                    reply_markup=smart_action_menu(symbol_hint) if symbol_hint else None,
                    reply_to_message_id=message.message_id,
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("handle_parsed_intent_error", extra={"event": "handle_parsed_intent_error", "chat_id": chat_id})
                await message.answer(
                    f"couldn't complete that — <i>{_safe_exc(exc)}</i>\n"
                    "try again with a bit more detail."
                )
                return
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "route_text_unhandled_error",
            extra={"event": "route_text_unhandled_error", "chat_id": chat_id},
        )
        with suppress(Exception):
            await message.answer(
                f"something broke on my end. try again in a sec. (<i>{_safe_exc(exc)}</i>)",
                reply_to_message_id=message.message_id,
            )
    finally:
        stop.set()
        typing_task.cancel()
        with suppress(Exception):
            await typing_task
        latency_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
        logger.info("message_processed", extra={"event": "message_processed", "chat_id": chat_id, "latency_ms": latency_ms})
