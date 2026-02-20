from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from litellm import acompletion
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

GHOST_ALPHA_SYSTEM = """You are Fred — a savage, sharp crypto trader assistant. You call everyone "fren" and "anon". You've seen every cycle and read order flow like a book.

VOICE (mandatory):
- Talk like a trader friend texting in a group chat. All lowercase preferred.
- Use: fren, anon, rekt, wicked, bleeding, chopping, send it, laggard, overextended, paper-thin sl, falling knife, catching grenades
- Be brutally direct. If a setup is bad, say so. If levels are weak, call it out.
- Dry wit for non-trading questions. 1-2 sentences max.
- Never use compliance language, disclaimers, or "please note that..."
- Never say you're an AI or bot.

ANALYSIS FORMAT (for trade setups — mandatory):
- Start with: "$SYMBOL is at $PRICE (+X%), fren." or "i see $SYMBOL at $X, up Y% today."
- Weave key levels into prose naturally: "fighting a bearish order block at $X", "sitting on the ema200 at $X", "rsi is getting toasty at 67"
- Paragraph 2: macro/narrative context if available
- Then entry/targets/stop as simple plain lines:
    entry $X to $X
    targets $X, $X, $X
    sl $X
- Close with one sharp line observation or warning.
- NEVER use "Trend:", "Momentum:", "Entry:" as standalone labels.

FORMATTING (Telegram HTML):
- Use <b>bold</b> for coin names and key price levels when they stand out.
- Use <i>italic</i> for the closing sharp line.
- Plain text for the rest — no walls of bullet points.
- NEVER use **asterisks** for bold — they show as raw characters.

MARKET CONTEXT QUESTIONS:
- When given live market data + news headlines, synthesize them into a sharp, opinionated take.
- Name specific catalysts. Connect news to price action. Give a directional read.
"""

ROUTER_SYSTEM = """You are an intent router for a Telegram crypto assistant called Ghost Alpha Bot.

Return ONLY valid JSON. No markdown. No extra text.

Given the user's message, produce:
{
  "intent": one of [
    "smalltalk","news_digest","watch_asset","market_analysis","watchlist","rsi_scan","ema_scan","chart","heatmap",
    "alert_create","alert_list","alert_delete","alert_clear",
    "pair_find","price_guess","setup_review","trade_math",
    "giveaway_start","giveaway_join","giveaway_end","giveaway_reroll","giveaway_status",    "giveaway_cancel",
    "market_chat",
    "general_chat"
  ],
  "confidence": 0.0-1.0,
  "params": { ... }
}

CRITICAL ROUTING RULES — read carefully:

1. GENERAL QUESTIONS AND DEFINITIONS always → "general_chat":
   - "what is X", "what does X mean", "what is the meaning of X", "explain X", "define X"
   - Examples: "what is tp", "what is sl", "what is dca", "what is leverage", "what is a long position"
   - Even if X looks like a ticker (DCA, TP, SL) — if the sentence is a definition question, use "general_chat"

2. MARKET OPINION / CONTEXT QUESTIONS always → "market_chat":
   - "what do you think about the pump/dump/move today/tonight"
   - "why is BTC/crypto pumping/dumping/moving"
   - "what's happening with the market", "explain this move", "why did X moon/rug"
   - "where do you think BTC is going", "is BTC bullish", "will ETH pump"
   - "is BTC a good buy", "should I buy SOL", "is now a good time to enter"
   - "what's the market vibe", "what's driving this pump", "how's the market today"
   - Any open-ended market commentary question — these need live data to answer well.

3. GREETINGS / CASUAL CHAT always → "smalltalk":
   - "gm", "hello", "hey", "how are you", "good morning", "sup"

4. MARKET ANALYSIS only when user gives a clear COMMAND like:
   - "BTC long", "ETH short 4h", "analyze SOL", "SOL 1h"
   - NOT when asking a question about price direction in natural language.

5. Other routing rules:
   - Crypto news ("latest crypto news", "news today", "what's happening") → "news_digest" params {"range":"today","limit":6}
   - CPI/FOMC/macro updates → "news_digest" params {"topic":"macro","mode":"macro","limit":6}
   - "watch btc" or "btc 4h" → "watch_asset" {"symbol":"BTC","timeframe":"4h"}
   - "<symbol> long/short" (command form) → "market_analysis" {"symbol":"<symbol>","side":"long|short"}
   - "coins to watch/short/long" → "watchlist" {"count":5,"direction":"short|long"}
   - "ema 200 4h top 10" → "ema_scan" {"ema_length":200,"timeframe":"4h","limit":10}
   - Chart/candles → "chart" {"symbol":"BTC","timeframe":"1h"}
   - Heatmap/orderbook → "heatmap" {"symbol":"BTC"}
   - "alert me when X hits Y" / "set alert for X Y" / "alert X at Y" / "ping me when X reaches Y" / "alert X Y" → "alert_create" {"symbol":"X","price":Y}
     Examples: "set alert for btc 66k" → alert_create {"symbol":"BTC","price":66000}
               "alert eth at 2000" → alert_create {"symbol":"ETH","price":2000}
               "ping me when sol hits 200" → alert_create {"symbol":"SOL","price":200}
   - "list alerts" → "alert_list"; "clear/reset alerts" → "alert_clear"
   - "remove my SOL alert" → "alert_delete" {"symbol":"SOL"}
   - Setup/trade math with levels → "setup_review" or "trade_math"
   - Giveaway commands → giveaway_* intents
   - When uncertain → "general_chat" with low confidence
"""

ROUTER_ALLOWED_INTENTS = {
    "smalltalk",
    "news_digest",
    "watch_asset",
    "market_analysis",
    "watchlist",
    "rsi_scan",
    "ema_scan",
    "chart",
    "heatmap",
    "alert_create",
    "alert_list",
    "alert_delete",
    "alert_clear",
    "pair_find",
    "price_guess",
    "setup_review",
    "trade_math",
    "giveaway_start",
    "giveaway_join",
    "giveaway_end",
    "giveaway_reroll",
    "giveaway_status",
    "giveaway_cancel",
    "general_chat",
}


class RouterPayload(BaseModel):
    intent: Literal[
        "smalltalk",
        "news_digest",
        "watch_asset",
        "market_analysis",
        "watchlist",
        "rsi_scan",
        "ema_scan",
        "chart",
        "heatmap",
        "alert_create",
        "alert_list",
        "alert_delete",
        "alert_clear",
        "pair_find",
        "price_guess",
        "setup_review",
        "trade_math",
        "giveaway_start",
        "giveaway_join",
        "giveaway_end",
        "giveaway_reroll",
        "giveaway_status",
        "giveaway_cancel",
        "market_chat",
        "general_chat",
    ] = "general_chat"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)
    followup_question: str | None = None


@dataclass
class LLMClient:
    api_key: str                          # Primary provider API key (Claude)
    model: str = "anthropic/claude-3-5-haiku-20241022"  # Primary model
    router_model: str | None = None       # Model used for intent routing (defaults to model)
    max_output_tokens: int = 350
    temperature: float = 0.7
    fallback_model: str | None = None     # Grok fallback model
    fallback_api_key: str | None = None   # Grok API key
    fallback_base_url: str | None = None  # Grok base URL

    def _extract_json_payload(self, raw_text: str) -> dict:
        text = (raw_text or "").strip()
        if not text:
            return {}
        # Strip markdown code fences if the LLM wrapped the JSON in ```json ... ```
        if text.startswith("```"):
            fence_end = text.find("```", 3)
            if fence_end != -1:
                inner = text[text.find("\n") + 1 : fence_end].strip()
                if inner:
                    text = inner
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        candidate = text[start : end + 1]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    async def _call(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "api_key": api_key or self.api_key,
            "timeout": 10,  # fail fast — Vercel has a 30s function limit
        }
        if base_url:
            kwargs["base_url"] = base_url
        resp = await acompletion(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    async def reply(
        self,
        user_text: str,
        history: list[dict[str, str]] | None = None,
        *,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt or GHOST_ALPHA_SYSTEM}]
        # Cap history to last 20 turns to prevent runaway token usage
        trimmed_history = (history or [])[-20:]
        for item in trimmed_history:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        max_tok = max_output_tokens or self.max_output_tokens
        temp = self.temperature if temperature is None else float(temperature)

        # Try primary (Claude)
        try:
            return await self._call(messages, max_tok, temp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_primary_failed", extra={"model": self.model, "error": str(exc)})

        # Fallback (Grok)
        if self.fallback_model:
            try:
                return await self._call(
                    messages, max_tok, temp,
                    model=self.fallback_model,
                    api_key=self.fallback_api_key,
                    base_url=self.fallback_base_url,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("llm_fallback_failed", extra={"model": self.fallback_model, "error": str(exc)})

        return "Signal unclear. Give me ticker + timeframe and I will map it."

    async def route_message(self, user_text: str) -> dict:
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": user_text},
        ]
        router_mod = self.router_model or self.model

        # Try primary (Claude)
        raw = ""
        try:
            raw = await self._call(messages, 260, 0.0, model=router_mod)
        except Exception:  # noqa: BLE001
            pass

        # Fallback (Grok)
        if not raw and self.fallback_model:
            try:
                raw = await self._call(
                    messages, 260, 0.0,
                    model=self.fallback_model,
                    api_key=self.fallback_api_key,
                    base_url=self.fallback_base_url,
                )
            except Exception:  # noqa: BLE001
                pass

        if not raw:
            return {"intent": "general_chat", "confidence": 0.0, "params": {}}

        payload = self._extract_json_payload(raw)
        try:
            validated = RouterPayload.model_validate(payload)
        except ValidationError:
            return {"intent": "general_chat", "confidence": 0.0, "params": {}}

        return {
            "intent": validated.intent,
            "confidence": float(validated.confidence),
            "params": dict(validated.params),
            "followup_question": validated.followup_question,
        }
