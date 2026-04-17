from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.bot import source_context


class _Cache:
    def __init__(self, initial: dict | None = None) -> None:
        self.store = dict(initial or {})
        self.get_json = AsyncMock(side_effect=self._get_json)
        self.set_json = AsyncMock(side_effect=self._set_json)

    async def _get_json(self, key: str):
        return self.store.get(key)

    async def _set_json(self, key: str, value, ttl: int | None = None) -> None:
        self.store[key] = value


def test_is_source_query_detects_source_prompts() -> None:
    assert source_context.is_source_query("what's the source?") is True
    assert source_context.is_source_query("which exchange is this from") is True
    assert source_context.is_source_query("gm fren") is False


@pytest.mark.asyncio
async def test_remember_source_context_persists_global_and_symbol_entries() -> None:
    cache = _Cache()

    await source_context.remember_source_context(
        cache=cache,
        chat_id=42,
        source_line="Bybit BTCUSDT",
        exchange="Bybit",
        market_kind="perp",
        instrument_id="BTCUSDT",
        updated_at="2026-04-16T12:00:00Z",
        symbol="btc",
        context="analysis",
    )

    assert cache.store["last_source:42"] == {
        "source_line": "Bybit BTCUSDT",
        "exchange": "Bybit",
        "market_kind": "perp",
        "instrument_id": "BTCUSDT",
        "updated_at": "2026-04-16T12:00:00Z",
        "symbol": "BTC",
        "context": "analysis",
    }
    assert cache.store["last_source:42:BTC"] == cache.store["last_source:42"]


@pytest.mark.asyncio
async def test_source_reply_prefers_symbol_specific_context() -> None:
    cache = _Cache(
        {
            "last_source:42:BTC": {
                "source_line": "Bybit BTCUSDT",
                "exchange": "",
                "market_kind": "",
                "instrument_id": "",
                "updated_at": "",
                "symbol": "BTC",
                "context": "analysis",
            }
        }
    )

    reply = await source_context.source_reply_for_chat(
        cache=cache,
        chat_id=42,
        query_text="btc source?",
    )

    assert reply == "Source: Bybit BTCUSDT"


@pytest.mark.asyncio
async def test_source_reply_falls_back_to_analysis_payload_then_global_context() -> None:
    analysis_cache = _Cache(
        {
            "last_analysis:42:BTC": {"data_source_line": "Binance BTCUSDT"},
        }
    )

    analysis_reply = await source_context.source_reply_for_chat(
        cache=analysis_cache,
        chat_id=42,
        query_text="btc source?",
    )

    assert analysis_reply == "BTC source: Binance BTCUSDT"

    global_cache = _Cache(
        {
            "last_source:42": {
                "source_line": "",
                "exchange": "Bybit",
                "market_kind": "spot",
                "instrument_id": "SOLUSDT",
                "updated_at": "2026-04-16T12:00:00Z",
                "symbol": "SOL",
                "context": "chart",
            }
        }
    )

    global_reply = await source_context.source_reply_for_chat(
        cache=global_cache,
        chat_id=42,
        query_text="where is this from?",
    )

    assert global_reply == "chart source: Bybit spot SOLUSDT | Updated: 2026-04-16T12:00:00Z"


@pytest.mark.asyncio
async def test_source_reply_returns_default_when_nothing_is_cached() -> None:
    cache = _Cache()

    reply = await source_context.source_reply_for_chat(
        cache=cache,
        chat_id=42,
        query_text="source?",
    )

    assert reply == "I do not have a recent source to report yet. Ask for analysis/chart first, then ask `source?`."
