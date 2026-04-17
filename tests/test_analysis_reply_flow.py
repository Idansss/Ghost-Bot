from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.enums import ChatAction

from app.bot import analysis_reply_flow


class _Cache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}
        self.get_json = AsyncMock(side_effect=self._get_json)
        self.set_json = AsyncMock(side_effect=self._set_json)

    async def _get_json(self, key: str):
        return self.store.get(key)

    async def _set_json(self, key: str, value, ttl: int | None = None) -> None:
        self.store[key] = value


def _deps(**overrides) -> analysis_reply_flow.AnalysisReplyFlowDependencies:
    defaults = {
        "format_as_ghost": AsyncMock(return_value="ghost text"),
        "llm_analysis_reply": AsyncMock(return_value=None),
        "trade_plan_template": lambda payload, settings, detailed=False: (
            f"template:{payload.get('summary')}:{settings.get('tone')}:{detailed}"
        ),
        "analysis_progressive_menu": lambda symbol, direction: {"symbol": symbol, "direction": direction},
        "pause": AsyncMock(),
    }
    defaults.update(overrides)
    return analysis_reply_flow.AnalysisReplyFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_remember_and_read_recent_analysis_context() -> None:
    cache = _Cache()

    await analysis_reply_flow.remember_analysis_context(
        cache=cache,
        chat_id=42,
        symbol="btc",
        direction="long",
        payload={"summary": "trend up", "entry": "100", "market_context_text": "btc strong"},
    )

    recent = await analysis_reply_flow.recent_analysis_context(cache=cache, chat_id=42)

    assert recent == {
        "symbol": "BTC",
        "direction": "long",
        "analysis_summary": "trend up",
        "key_levels": {
            "entry": "100",
            "tp1": "",
            "tp2": "",
            "sl": "",
            "price": None,
        },
        "market_context": {},
        "market_context_text": "btc strong",
    }
    assert cache.store["last_analysis_context:42:BTC"] == recent


def test_looks_like_analysis_followup_matches_values_and_followup_language() -> None:
    context = {"symbol": "BTC"}

    assert analysis_reply_flow.looks_like_analysis_followup("btc 91000 good entry?", context) is True
    assert analysis_reply_flow.looks_like_analysis_followup("what if i use 10x?", context) is True
    assert analysis_reply_flow.looks_like_analysis_followup("gm fren", context) is False


@pytest.mark.asyncio
async def test_render_analysis_text_prefers_ghost_formatter() -> None:
    deps = _deps()

    text = await analysis_reply_flow.render_analysis_text(
        payload={"summary": "trend up"},
        symbol="BTC",
        direction="long",
        settings={"tone": "ghost"},
        chat_id=42,
        deps=deps,
    )

    assert text == "ghost text"
    deps.llm_analysis_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_render_analysis_text_falls_back_to_llm_then_template() -> None:
    llm_analysis_reply = AsyncMock(return_value="llm text")
    deps = _deps(
        format_as_ghost=AsyncMock(side_effect=RuntimeError("formatter failed")),
        llm_analysis_reply=llm_analysis_reply,
    )

    text = await analysis_reply_flow.render_analysis_text(
        payload={"summary": "trend up"},
        symbol="BTC",
        direction="long",
        settings={"tone": "ghost"},
        chat_id=42,
        deps=deps,
    )

    assert text == "llm text"
    llm_analysis_reply.assert_awaited_once()

    template_deps = _deps(
        format_as_ghost=AsyncMock(side_effect=RuntimeError("formatter failed")),
        llm_analysis_reply=AsyncMock(return_value=None),
    )
    template_text = await analysis_reply_flow.render_analysis_text(
        payload={"summary": "trend up"},
        symbol="BTC",
        direction="long",
        settings={"tone": "ghost"},
        chat_id=42,
        detailed=True,
        deps=template_deps,
    )

    assert template_text == "template:trend up:ghost:True"


@pytest.mark.asyncio
async def test_send_ghost_analysis_sends_typing_delay_and_markup() -> None:
    pause = AsyncMock()
    deps = _deps(pause=pause)
    message = SimpleNamespace(
        chat=SimpleNamespace(id=42),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
        answer=AsyncMock(),
    )

    await analysis_reply_flow.send_ghost_analysis(
        message,
        "BTC",
        "analysis text",
        direction="long",
        deps=deps,
    )

    message.bot.send_chat_action.assert_awaited_once_with(42, ChatAction.TYPING)
    pause.assert_awaited_once_with(1.3)
    message.answer.assert_awaited_once_with(
        "analysis text",
        reply_markup={"symbol": "BTC", "direction": "long"},
    )
