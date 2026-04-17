from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable


_FUNDAMENTALS_KEYWORDS = (
    "info",
    "fundamentals",
    "market cap",
    "marketcap",
    "ath",
    "atl",
    "all time high",
    "all time low",
    "supply",
    "circulating",
    "max supply",
    "total supply",
    "fdv",
    "diluted",
    "valuation",
    "website",
    "explorer",
    "explorers",
    "links",
    "social",
    "twitter",
    "telegram",
    "reddit",
    "about the coin",
    "about this coin",
    "what is this coin",
    "fear",
    "greed",
    "fear and greed",
    "treasury",
    "volume",
    "trading volume",
    "market cap/fdv",
    "mc/fdv",
)


@dataclass(frozen=True)
class LlmReplyFlowDependencies:
    hub: Any
    openai_chat_history_turns: int
    openai_max_output_tokens: int
    openai_temperature: float
    bot_meta_re: re.Pattern[str]
    try_answer_definition: Callable[[str], str | None]
    try_answer_howto: Callable[[str], str | None]
    format_market_context: Callable[[dict], str]


def build_communication_memory_block(settings: dict | None) -> str:
    data = settings or {}
    style = str(data.get("communication_style") or "friendly").strip().lower()
    style_rules = {
        "short": "Keep replies to 1\u20132 sentences. No preamble.",
        "detailed": "Give a one-line summary first, then 2\u20133 key points, then optional detail. Use line breaks.",
        "friendly": "Casual tone; 'fren' or 'anon' once. Match question length.",
        "formal": "Professional tone. No slang.",
    }
    parts = [f"COMMUNICATION: User prefers {style} style. {style_rules.get(style, style_rules['friendly'])}"]
    name = (data.get("display_name") or "").strip()
    if name:
        parts.append(f"Call them {name}.")
    goals = (data.get("trading_goals") or "").strip()[:200]
    if goals:
        parts.append(f"Their stated goals: {goals}")
    last_symbols = data.get("last_symbols") or []
    if isinstance(last_symbols, list) and last_symbols:
        symbols = [str(item).upper() for item in last_symbols[:5] if item]
        if symbols:
            parts.append(f"They recently asked about: {', '.join(symbols)}.")
    prefs = data.get("feedback_prefs") or {}
    if isinstance(prefs, dict) and prefs.get("prefers_shorter"):
        parts.append("User has asked for shorter answers before; keep it brief.")
    return " ".join(parts)


def is_definition_question(text: str) -> bool:
    if not text or len(text) > 200:
        return False
    lower = text.strip().lower()
    if lower.startswith(("what is ", "what's ", "what are ", "define ", "explain ", "meaning of ")):
        return True
    if " what is " in lower or " define " in lower or " explain " in lower:
        return True
    return False


async def get_chat_history(*, cache, chat_id: int) -> list[dict[str, str]]:
    payload = await cache.get_json(f"llm:history:{chat_id}")
    if not isinstance(payload, list):
        return []
    history: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    return history


async def append_chat_history(
    *,
    cache,
    chat_id: int,
    role: str,
    content: str,
    turns: int,
) -> None:
    normalized_role = role.strip().lower()
    normalized_content = content.strip()
    if normalized_role not in {"user", "assistant"} or not normalized_content:
        return
    history = await get_chat_history(cache=cache, chat_id=chat_id)
    history.append({"role": normalized_role, "content": normalized_content})
    history = history[-(max(turns, 1) * 2) :]
    await cache.set_json(f"llm:history:{chat_id}", history, ttl=60 * 60 * 24 * 7)


def _wants_fundamentals(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    lower = text.strip().lower()
    return any(keyword in lower for keyword in _FUNDAMENTALS_KEYWORDS)


def _extract_symbol_for_fundamentals(text: str, last_symbols: list) -> str | None:
    try:
        from app.core.nlu import _extract_symbols

        symbols = _extract_symbols(text)
        if symbols:
            base = getattr(symbols[0], "base", None) or (symbols[0] if isinstance(symbols[0], str) else None)
            if base:
                return str(base).upper()
    except Exception:
        pass
    if last_symbols:
        return str(last_symbols[0]).upper().strip()
    return None


def _format_coin_fundamentals_block(info: dict | None, fear_greed: dict | None) -> str:
    lines = ["<b>Coin fundamentals (use only when user asks for stats, links, or about):</b>"]
    if not info:
        lines.append("(No fundamentals data for this symbol.)")
    else:
        def _fmt_num(value):
            if value is None:
                return "\u2014"
            if value >= 1e12:
                return f"{value / 1e12:.2f}T"
            if value >= 1e9:
                return f"{value / 1e9:.2f}B"
            if value >= 1e6:
                return f"{value / 1e6:.2f}M"
            if value >= 1e3:
                return f"{value / 1e3:.2f}k"
            return f"{value:,.2f}"

        name = info.get("name") or info.get("symbol") or "\u2014"
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
            parts = [f"{key}: {value}" for key, value in list(social.items())[:4]]
            if parts:
                lines.append("Social: " + " | ".join(parts))
        if info.get("about"):
            about = (info.get("about") or "")[:800].strip()
            if about:
                lines.append(f"About: {about}")
    if fear_greed:
        value = fear_greed.get("value")
        classification = fear_greed.get("classification")
        if value is not None or classification:
            lines.append(
                f"Crypto Fear & Greed Index: {value} ({classification})"
                if (value is not None and classification)
                else f"Fear & Greed: {classification or value}"
            )
    return "\n".join(lines)


async def llm_analysis_reply(
    *,
    payload: dict,
    symbol: str,
    direction: str | None,
    chat_id: int | None,
    deps: LlmReplyFlowDependencies,
) -> str | None:
    if not deps.hub.llm_client:
        return None
    market_context_text = str(payload.get("market_context_text") or "").strip()
    direction_label = (direction or payload.get("side") or "").strip().lower() or "none"
    prompt = (
        f"Analysis data for {symbol.upper()} ({direction_label} bias):\n"
        f"{json.dumps(payload, ensure_ascii=True, default=str)}\n\n"
        f"BTC/market backdrop: {market_context_text or 'not available'}\n\n"
        "Write this as Ghost would \u2014 start with current price and % change. "
        "Use the full indicator set in the data: SMA/EMA, RSI, MACD (and histogram), Bollinger Bands, "
        "support/resistance, Fibonacci levels, VWAP, Stochastic, ATR, OBV, ADX, and any candlestick patterns. "
        "Weave key levels and readings naturally in prose; mention macro if relevant. "
        "then give entry range / 3 targets / stop as simple plain lines. "
        "End with one short risk caveat (e.g. don't oversize, cut if structure breaks, wait for confirmation). All lowercase, casual trader voice. No HTML tags."
    )
    history = await get_chat_history(cache=deps.hub.cache, chat_id=chat_id) if chat_id is not None else []
    try:
        reply = await deps.hub.llm_client.reply(
            prompt,
            history=history,
            max_output_tokens=min(max(int(deps.openai_max_output_tokens), 400), 700),
            temperature=max(0.6, float(deps.openai_temperature)),
        )
    except Exception:
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final and chat_id is not None:
        await append_chat_history(
            cache=deps.hub.cache,
            chat_id=chat_id,
            role="user",
            content=f"{symbol.upper()} {(direction or payload.get('side') or '').strip()} analysis",
            turns=deps.openai_chat_history_turns,
        )
        await append_chat_history(
            cache=deps.hub.cache,
            chat_id=chat_id,
            role="assistant",
            content=final,
            turns=deps.openai_chat_history_turns,
        )
    return final


async def llm_followup_reply(
    user_text: str,
    context: dict,
    *,
    chat_id: int,
    deps: LlmReplyFlowDependencies,
) -> str | None:
    if not deps.hub.llm_client:
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
    history = await get_chat_history(cache=deps.hub.cache, chat_id=chat_id)
    try:
        reply = await deps.hub.llm_client.reply(prompt, history=history, max_output_tokens=220)
    except Exception:
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final:
        await append_chat_history(
            cache=deps.hub.cache,
            chat_id=chat_id,
            role="user",
            content=cleaned,
            turns=deps.openai_chat_history_turns,
        )
        await append_chat_history(
            cache=deps.hub.cache,
            chat_id=chat_id,
            role="assistant",
            content=final,
            turns=deps.openai_chat_history_turns,
        )
    return final


async def llm_fallback_reply(
    user_text: str,
    settings: dict | None = None,
    chat_id: int | None = None,
    *,
    deps: LlmReplyFlowDependencies,
) -> str | None:
    if not deps.hub.llm_client:
        return None
    cleaned = (user_text or "").strip()
    if not cleaned:
        return None
    comm_memory = build_communication_memory_block(settings)
    prompt = (
        f"{comm_memory}\n\n"
        "GHOST PERSONA (use only when they ask who you are, your personality, or what you're like):\n"
        "You are Ghost \u2014 sharp, no-nonsense, crypto-native. You tell the truth even when it wrecks a bag. "
        "Slightly unhinged and unfiltered; you don't sugarcoat. You help with levels, risk, and setups but you're not a cheerleader. "
        "One short paragraph max; match their tone (friendly or formal from settings).\n\n"
        f"User message: {cleaned}\n\n"
        "Answer exactly what was asked. Do not divert to a different topic. Same question \u2192 same type of answer.\n"
        "STRUCTURE: One-line summary first, then key points. CONFIDENCE: When uncertain add '(low confidence)'.\n"
        "LIMITS: If out of scope (tax/legal/guarantees), say so in one sentence and suggest what you can do.\n"
        "Do NOT add a 'coins to watch' or 'coins to watch right now' section unless they explicitly ask for a watchlist or what to watch. Never use that phrase as a heading otherwise.\n"
        "NEVER tie metals (gold, XAU, silver) to crypto \u2014 no inverse/safe-haven links, no predicting metals from BTC.\n\n"
        "BOT CAPABILITIES (only when they ask how to use the bot):\n"
        "- Alerts: 'alert BTC 100000 above' or Create Alert button\n"
        "- Analysis: 'BTC long' or 'ETH short 4h'\n"
        "- Watchlist: 'coins to watch', 'top movers'\n"
        "- News: 'latest crypto news'\n"
        "- Price: /price BTC\n\n"
        "TRADING FRAMEWORK (only when they ask about trading concepts, strategy, or risk \u2014 do not list headings unless they explicitly ask):\n"
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
    history = await get_chat_history(cache=deps.hub.cache, chat_id=chat_id) if chat_id is not None else []
    try:
        reply = await deps.hub.llm_client.reply(prompt, history=history)
    except Exception:
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final and chat_id is not None:
        await append_chat_history(
            cache=deps.hub.cache,
            chat_id=chat_id,
            role="user",
            content=cleaned,
            turns=deps.openai_chat_history_turns,
        )
        await append_chat_history(
            cache=deps.hub.cache,
            chat_id=chat_id,
            role="assistant",
            content=final,
            turns=deps.openai_chat_history_turns,
        )
    return final


async def llm_market_chat_reply(
    user_text: str,
    settings: dict | None = None,
    chat_id: int | None = None,
    *,
    deps: LlmReplyFlowDependencies,
) -> str | None:
    if not deps.hub.llm_client:
        return None
    cleaned = (user_text or "").strip()
    if not cleaned:
        return None
    if is_definition_question(cleaned):
        knowledge = deps.try_answer_definition(cleaned)
        if knowledge:
            return knowledge
    if deps.bot_meta_re.search(cleaned):
        howto = deps.try_answer_howto(cleaned)
        if howto:
            return howto

    market_context: dict = {}
    news_headlines: list[dict] = []
    try:
        market_context, news_payload = await asyncio.gather(
            deps.hub.analysis_service.get_market_context(),
            deps.hub.news_service.get_digest(mode="crypto", limit=6),
            return_exceptions=True,
        )
        if isinstance(market_context, Exception):
            market_context = {}
        if isinstance(news_payload, Exception):
            news_payload = {}
        if isinstance(news_payload, dict):
            news_headlines = news_payload.get("headlines") or []
    except Exception:
        pass

    context_block = f"[Context as of {datetime.now(UTC).strftime('%H:%M UTC')}]\n"
    if is_definition_question(cleaned):
        context_block += (
            "This is a definition/knowledge question. Do NOT use market snapshot/news. "
            "Answer as: definition \u2192 why it matters \u2192 example \u2192 common mistakes \u2192 takeaway.\n\n"
        )
    market_text = deps.format_market_context(market_context) if market_context else ""
    if market_text:
        context_block += f"Live market snapshot (BTC, ETH, SOL): {market_text}\n"
    else:
        context_block += "Live market snapshot: not available.\n"

    news_lines = "\n".join(
        f"- {item.get('title', '')} ({item.get('source', '')})"
        for item in news_headlines[:6]
        if item.get("title")
    )
    if news_lines:
        context_block += f"Recent crypto news:\n{news_lines}\n"

    last_symbols = list((settings or {}).get("last_symbols") or [])[:5]
    asked_symbol = _extract_symbol_for_fundamentals(cleaned, last_symbols)
    if asked_symbol and asked_symbol not in ("BTC", "ETH", "SOL"):
        try:
            quote = await deps.hub.market_router.get_price(asked_symbol)
            if quote and float(quote.get("price") or 0) > 0:
                price = float(quote["price"])
                source = str(quote.get("source_line") or quote.get("source") or "exchange")
                context_block += (
                    f"\nRequested symbol {asked_symbol}: ${price:,.2f} (from {source}). "
                    f"You have data for this symbol \u2014 use it to answer; do not say you only have BTC/ETH/SOL.\n"
                )
        except Exception:
            pass

    if _wants_fundamentals(cleaned):
        symbol_for_info = _extract_symbol_for_fundamentals(cleaned, last_symbols)
        if symbol_for_info and getattr(deps.hub, "coin_info_service", None):
            try:
                info, fear_greed = await asyncio.gather(
                    deps.hub.coin_info_service.get_coin_info(symbol_for_info),
                    deps.hub.coin_info_service.get_fear_greed(),
                    return_exceptions=True,
                )
                if isinstance(info, Exception):
                    info = None
                if isinstance(fear_greed, Exception):
                    fear_greed = None
                context_block += "\n\n" + _format_coin_fundamentals_block(info, fear_greed) + "\n"
            except Exception:
                pass

    ultra = (settings or {}).get("ultra_brief")
    length_rule = (
        "Answer in one short sentence only."
        if ultra
        else "Match length to the question: short question \u2192 short answer; open-ended \u2192 fuller answer. Paragraphs 3\u20134 sentences max."
    )
    prompt = (
        f"{context_block}\n"
        f"{build_communication_memory_block(settings)}\n\n"
        "CONTEXT RULES:\n"
        "- If market snapshot says 'not available', do NOT invent prices/levels/news. Keep the answer general and say data isn't available.\n"
        "- If you mention a number (price/level), it must come from the snapshot or a 'Requested symbol' line above. Otherwise, do not use numbers.\n"
        "- If 'Requested symbol X: $Y' appears above, you have live price data for X - use it. Only say \"I don't have data for [symbol]\" when it's absent from both the snapshot and the 'Requested symbol' line.\n"
        "- If a 'Coin fundamentals' block is present, use it only for the specific stats asked (cap, supply, ATH, etc). Don't dump the whole block unprompted.\n"
        f"- {length_rule}\n"
        "- When uncertain, prefix with \"likely\" or \"(low confidence)\". When data-backed, state it directly.\n"
        "- For out-of-scope requests (tax advice, guaranteed outcomes), say so in one sentence and offer what you can do instead.\n\n"
        f"User: {cleaned}\n\n"
        "Telegram HTML: <b>bold</b> for coins and key levels, <i>italic</i> for closing line."
    )
    history = await get_chat_history(cache=deps.hub.cache, chat_id=chat_id) if chat_id is not None else []
    try:
        reply = await deps.hub.llm_client.reply(
            prompt,
            history=history,
            max_output_tokens=min(max(int(deps.openai_max_output_tokens), 600), 1000),
            temperature=max(0.5, float(deps.openai_temperature)),
        )
    except Exception:
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final and chat_id is not None:
        await append_chat_history(
            cache=deps.hub.cache,
            chat_id=chat_id,
            role="user",
            content=cleaned,
            turns=deps.openai_chat_history_turns,
        )
        await append_chat_history(
            cache=deps.hub.cache,
            chat_id=chat_id,
            role="assistant",
            content=final,
            turns=deps.openai_chat_history_turns,
        )
    return final
