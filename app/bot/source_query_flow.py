from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.bot import source_context


@dataclass(frozen=True)
class SourceQueryFlowDependencies:
    cache: Any


def extract_source_symbol_hint(text: str) -> str | None:
    return source_context.extract_source_symbol_hint(text)


def is_source_query(text: str) -> bool:
    return source_context.is_source_query(text)


async def remember_source_context(
    chat_id: int,
    *,
    deps: SourceQueryFlowDependencies,
    source_line: str | None = None,
    exchange: str | None = None,
    market_kind: str | None = None,
    instrument_id: str | None = None,
    updated_at: str | None = None,
    symbol: str | None = None,
    context: str | None = None,
) -> None:
    await source_context.remember_source_context(
        cache=deps.cache,
        chat_id=chat_id,
        source_line=source_line,
        exchange=exchange,
        market_kind=market_kind,
        instrument_id=instrument_id,
        updated_at=updated_at,
        symbol=symbol,
        context=context,
    )


def format_source_response(payload: dict | None) -> str:
    return source_context.format_source_response(payload)


async def source_reply_for_chat(
    chat_id: int,
    query_text: str,
    *,
    deps: SourceQueryFlowDependencies,
) -> str:
    return await source_context.source_reply_for_chat(
        cache=deps.cache,
        chat_id=chat_id,
        query_text=query_text,
    )
