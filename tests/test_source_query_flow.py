from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.bot import source_query_flow


class _Cache:
    def __init__(self, initial: dict | None = None) -> None:
        self.store = dict(initial or {})
        self.get_json = AsyncMock(side_effect=self._get_json)
        self.set_json = AsyncMock(side_effect=self._set_json)

    async def _get_json(self, key: str):
        return self.store.get(key)

    async def _set_json(self, key: str, value, ttl: int | None = None) -> None:
        self.store[key] = value


def test_extract_source_symbol_hint_delegates_to_source_context() -> None:
    assert source_query_flow.extract_source_symbol_hint("btc source?") == "BTC"
    assert source_query_flow.extract_source_symbol_hint("source?") is None


def test_is_source_query_delegates_to_source_context() -> None:
    assert source_query_flow.is_source_query("what's the source?") is True
    assert source_query_flow.is_source_query("gm fren") is False


def test_format_source_response_delegates_to_source_context() -> None:
    payload = {
        "source_line": "",
        "exchange": "Bybit",
        "market_kind": "perp",
        "instrument_id": "BTCUSDT",
        "updated_at": "2026-04-16T12:00:00Z",
        "context": "analysis",
    }

    assert (
        source_query_flow.format_source_response(payload)
        == "analysis source: Bybit perp BTCUSDT | Updated: 2026-04-16T12:00:00Z"
    )


@pytest.mark.asyncio
async def test_remember_source_context_uses_dependency_cache() -> None:
    cache = _Cache()
    deps = source_query_flow.SourceQueryFlowDependencies(cache=cache)

    await source_query_flow.remember_source_context(
        chat_id=42,
        deps=deps,
        source_line="Bybit BTCUSDT",
        exchange="Bybit",
        market_kind="perp",
        instrument_id="BTCUSDT",
        updated_at="2026-04-16T12:00:00Z",
        symbol="btc",
        context="analysis",
    )

    assert cache.store["last_source:42"]["symbol"] == "BTC"
    assert cache.store["last_source:42:BTC"]["context"] == "analysis"


@pytest.mark.asyncio
async def test_source_reply_for_chat_uses_dependency_cache() -> None:
    cache = _Cache(
        {
            "last_source:42": {
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
    deps = source_query_flow.SourceQueryFlowDependencies(cache=cache)

    reply = await source_query_flow.source_reply_for_chat(
        chat_id=42,
        query_text="where is this from?",
        deps=deps,
    )

    assert reply == "Source: Bybit BTCUSDT"
