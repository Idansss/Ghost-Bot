from __future__ import annotations

import re


_SOURCE_QUERY_RE = re.compile(
    r"\b(where\s+is\s+this\s+from|what(?:'s| is)\s+the\s+source|which\s+exchange|source\??|exchange\??)\b",
    re.IGNORECASE,
)
_SOURCE_QUERY_STOPWORDS = {
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


def extract_source_symbol_hint(text: str) -> str | None:
    for token in re.findall(r"\b[A-Za-z]{2,12}\b", text):
        low = token.lower()
        if low in _SOURCE_QUERY_STOPWORDS:
            continue
        return token.upper().lstrip("$")
    return None


def is_source_query(text: str) -> bool:
    return bool(_SOURCE_QUERY_RE.search(text or ""))


async def remember_source_context(
    *,
    cache,
    chat_id: int,
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
    payload = {
        "source_line": source_line or "",
        "exchange": exchange or "",
        "market_kind": market_kind or "",
        "instrument_id": instrument_id or "",
        "updated_at": updated_at or "",
        "symbol": symbol.upper() if isinstance(symbol, str) and symbol else "",
        "context": context or "",
    }
    await cache.set_json(f"last_source:{chat_id}", payload, ttl=60 * 60 * 12)
    if payload["symbol"]:
        await cache.set_json(f"last_source:{chat_id}:{payload['symbol']}", payload, ttl=60 * 60 * 12)


def format_source_response(payload: dict | None) -> str:
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
    parts = [part for part in [exchange, market_kind, instrument_id] if part]
    if not parts:
        return "I do not have a recent source to report yet."
    base = " ".join(parts)
    suffix = f" | Updated: {updated}" if updated else ""
    prefix = f"{context} source: " if context else "Source: "
    return f"{prefix}{base}{suffix}"


async def source_reply_for_chat(*, cache, chat_id: int, query_text: str) -> str:
    symbol = extract_source_symbol_hint(query_text)
    if symbol:
        per_symbol = await cache.get_json(f"last_source:{chat_id}:{symbol}")
        if isinstance(per_symbol, dict):
            return format_source_response(per_symbol)
        analysis = await cache.get_json(f"last_analysis:{chat_id}:{symbol}")
        if isinstance(analysis, dict):
            line = str(analysis.get("data_source_line") or "").strip()
            if line:
                return f"{symbol} source: {line}"
    payload = await cache.get_json(f"last_source:{chat_id}")
    if isinstance(payload, dict):
        return format_source_response(payload)
    return "I do not have a recent source to report yet. Ask for analysis/chart first, then ask `source?`."
