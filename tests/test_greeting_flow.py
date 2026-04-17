from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import greeting_flow


def _deps(**overrides) -> greeting_flow.GreetingFlowDependencies:
    defaults = {
        "analysis_service": SimpleNamespace(get_market_context=AsyncMock(return_value={})),
        "choose_reply": lambda seq: seq[0],
        "gm_replies": ["fallback gm"],
        "weekend_warning_text": "<i>warning</i>",
        "now_utc": lambda: datetime(2026, 4, 18, tzinfo=UTC),
    }
    defaults.update(overrides)
    return greeting_flow.GreetingFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_market_aware_gm_reply_prefers_dynamic_market_context() -> None:
    deps = _deps(
        analysis_service=SimpleNamespace(
            get_market_context=AsyncMock(return_value={"btc_change_pct_1h": 2.4})
        )
    )

    reply = await greeting_flow.market_aware_gm_reply("fred", deps=deps)

    assert "btc up 2.4%" in reply


@pytest.mark.asyncio
async def test_market_aware_gm_reply_falls_back_to_static_pool() -> None:
    reply = await greeting_flow.market_aware_gm_reply("fred", deps=_deps())

    assert reply == "fallback gm"


@pytest.mark.asyncio
async def test_maybe_send_market_warning_only_on_weekends() -> None:
    weekend_message = SimpleNamespace(answer=AsyncMock())
    weekday_message = SimpleNamespace(answer=AsyncMock())

    await greeting_flow.maybe_send_market_warning(weekend_message, deps=_deps())
    await greeting_flow.maybe_send_market_warning(
        weekday_message,
        deps=_deps(now_utc=lambda: datetime(2026, 4, 17, tzinfo=UTC)),
    )

    weekend_message.answer.assert_awaited_once_with("<i>warning</i>")
    weekday_message.answer.assert_not_called()
