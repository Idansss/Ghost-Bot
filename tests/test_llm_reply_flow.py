from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import llm_reply_flow


class _Cache:
    def __init__(self, initial: dict | None = None) -> None:
        self.store = dict(initial or {})
        self.get_json = AsyncMock(side_effect=self._get_json)
        self.set_json = AsyncMock(side_effect=self._set_json)

    async def _get_json(self, key: str):
        return self.store.get(key)

    async def _set_json(self, key: str, value, ttl: int | None = None) -> None:
        self.store[key] = value


def _hub(cache: _Cache | None = None, llm_reply: str = "llm reply") -> SimpleNamespace:
    return SimpleNamespace(
        cache=cache or _Cache(),
        llm_client=SimpleNamespace(reply=AsyncMock(return_value=llm_reply)),
        analysis_service=SimpleNamespace(get_market_context=AsyncMock(return_value={"btc": "up"})),
        news_service=SimpleNamespace(get_digest=AsyncMock(return_value={"headlines": []})),
        market_router=SimpleNamespace(get_price=AsyncMock(return_value={})),
        coin_info_service=SimpleNamespace(
            get_coin_info=AsyncMock(return_value=None),
            get_fear_greed=AsyncMock(return_value=None),
        ),
    )


def _deps(**overrides) -> llm_reply_flow.LlmReplyFlowDependencies:
    defaults = {
        "hub": _hub(),
        "openai_chat_history_turns": 2,
        "openai_max_output_tokens": 700,
        "openai_temperature": 0.7,
        "bot_meta_re": re.compile(r"create alert", re.IGNORECASE),
        "try_answer_definition": lambda _text: None,
        "try_answer_howto": lambda _text: None,
        "format_market_context": lambda payload: f"btc: {payload.get('btc')}",
    }
    defaults.update(overrides)
    return llm_reply_flow.LlmReplyFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_get_chat_history_filters_invalid_entries() -> None:
    cache = _Cache(
        {
            "llm:history:42": [
                {"role": " user ", "content": " hi "},
                {"role": "system", "content": "skip"},
                {"role": "assistant", "content": " ok "},
                "bad",
                {"role": "assistant", "content": "   "},
            ]
        }
    )

    history = await llm_reply_flow.get_chat_history(cache=cache, chat_id=42)

    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


@pytest.mark.asyncio
async def test_append_chat_history_trims_to_turn_budget() -> None:
    cache = _Cache(
        {
            "llm:history:42": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ]
        }
    )

    await llm_reply_flow.append_chat_history(
        cache=cache,
        chat_id=42,
        role="assistant",
        content=" five ",
        turns=2,
    )

    assert cache.store["llm:history:42"] == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
        {"role": "assistant", "content": "five"},
    ]


@pytest.mark.asyncio
async def test_llm_analysis_reply_uses_history_and_persists_synthetic_turns() -> None:
    cache = _Cache(
        {
            "llm:history:42": [
                {"role": "user", "content": "btc?"},
                {"role": "assistant", "content": "old"},
            ]
        }
    )
    hub = _hub(cache=cache, llm_reply="analysis reply")
    deps = _deps(hub=hub, openai_chat_history_turns=3, openai_max_output_tokens=650, openai_temperature=0.5)

    reply = await llm_reply_flow.llm_analysis_reply(
        payload={"market_context_text": "btc strong", "side": "long"},
        symbol="BTC",
        direction="long",
        chat_id=42,
        deps=deps,
    )

    assert reply == "analysis reply"
    hub.llm_client.reply.assert_awaited_once()
    kwargs = hub.llm_client.reply.await_args.kwargs
    assert kwargs["history"] == [
        {"role": "user", "content": "btc?"},
        {"role": "assistant", "content": "old"},
    ]
    assert kwargs["max_output_tokens"] == 650
    assert kwargs["temperature"] == 0.6
    assert cache.store["llm:history:42"][-2:] == [
        {"role": "user", "content": "BTC long analysis"},
        {"role": "assistant", "content": "analysis reply"},
    ]


@pytest.mark.asyncio
async def test_llm_followup_reply_appends_user_and_assistant_history() -> None:
    cache = _Cache({"llm:history:42": []})
    hub = _hub(cache=cache, llm_reply="followup reply")
    deps = _deps(hub=hub)

    reply = await llm_reply_flow.llm_followup_reply(
        "what about 91?",
        {"symbol": "BTC", "entry": 90},
        chat_id=42,
        deps=deps,
    )

    assert reply == "followup reply"
    prompt = hub.llm_client.reply.await_args.args[0]
    assert '"symbol": "BTC"' in prompt
    assert cache.store["llm:history:42"] == [
        {"role": "user", "content": "what about 91?"},
        {"role": "assistant", "content": "followup reply"},
    ]


@pytest.mark.asyncio
async def test_llm_fallback_reply_includes_memory_block_and_persists_history() -> None:
    cache = _Cache({"llm:history:42": []})
    hub = _hub(cache=cache, llm_reply="fallback reply")
    deps = _deps(hub=hub)

    reply = await llm_reply_flow.llm_fallback_reply(
        "how should i think about risk?",
        settings={
            "communication_style": "short",
            "display_name": "Ada",
            "trading_goals": "Protect capital first.",
            "last_symbols": ["btc", "eth"],
            "feedback_prefs": {"prefers_shorter": True},
        },
        chat_id=42,
        deps=deps,
    )

    assert reply == "fallback reply"
    prompt = hub.llm_client.reply.await_args.args[0]
    assert "Call them Ada." in prompt
    assert "Protect capital first." in prompt
    assert "They recently asked about: BTC, ETH." in prompt
    assert "keep it brief" in prompt
    assert cache.store["llm:history:42"] == [
        {"role": "user", "content": "how should i think about risk?"},
        {"role": "assistant", "content": "fallback reply"},
    ]


@pytest.mark.asyncio
async def test_llm_market_chat_reply_uses_definition_short_circuit() -> None:
    hub = _hub()
    deps = _deps(
        hub=hub,
        try_answer_definition=lambda _text: "smc means smart money concepts",
    )

    reply = await llm_reply_flow.llm_market_chat_reply(
        "what is smc",
        settings={},
        chat_id=42,
        deps=deps,
    )

    assert reply == "smc means smart money concepts"
    hub.llm_client.reply.assert_not_awaited()
    hub.analysis_service.get_market_context.assert_not_awaited()
    hub.news_service.get_digest.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_market_chat_reply_includes_requested_symbol_and_fundamentals() -> None:
    cache = _Cache({"llm:history:42": []})
    hub = _hub(cache=cache, llm_reply="market reply")
    hub.news_service.get_digest = AsyncMock(
        return_value={"headlines": [{"title": "ETF flows rise", "source": "CoinDesk"}]}
    )
    hub.market_router.get_price = AsyncMock(
        return_value={"price": 0.1234, "source_line": "Bybit PEPEUSDT"}
    )
    hub.coin_info_service.get_coin_info = AsyncMock(
        return_value={
            "name": "Pepe",
            "symbol": "PEPE",
            "market_cap": 1000000,
            "website": "https://pepe.example",
        }
    )
    hub.coin_info_service.get_fear_greed = AsyncMock(
        return_value={"value": 70, "classification": "Greed"}
    )
    deps = _deps(hub=hub)

    reply = await llm_reply_flow.llm_market_chat_reply(
        "give me pepe market cap and website",
        settings={"last_symbols": ["PEPE"]},
        chat_id=42,
        deps=deps,
    )

    assert reply == "market reply"
    prompt = hub.llm_client.reply.await_args.args[0]
    assert "Requested symbol PEPE: $0.12 (from Bybit PEPEUSDT)." in prompt
    assert "Coin fundamentals" in prompt
    assert "Market cap: $1.00M" in prompt
    assert "Website: https://pepe.example" in prompt
    assert "ETF flows rise (CoinDesk)" in prompt
    assert cache.store["llm:history:42"] == [
        {"role": "user", "content": "give me pepe market cap and website"},
        {"role": "assistant", "content": "market reply"},
    ]
