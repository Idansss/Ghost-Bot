from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class GreetingFlowDependencies:
    analysis_service: Any
    choose_reply: Callable[[Sequence[str]], str]
    gm_replies: Sequence[str]
    weekend_warning_text: str
    now_utc: Callable[[], datetime]


async def market_aware_gm_reply(name: str, *, deps: GreetingFlowDependencies) -> str:
    try:
        ctx = await asyncio.wait_for(deps.analysis_service.get_market_context(), timeout=3.0)
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
                    f"gm {name} btc {direction} {abs_pct:.1f}% {mood} session. drop a ticker or ask what's moving.",
                    f"gm {name} btc is {direction} {abs_pct:.1f}%. tape is {'hot' if abs_pct >= 2.0 else 'breathing'}. what are we hunting?",
                    f"gm btc {direction} {abs_pct:.1f}% this hour. {mood} market. send a coin.",
                ]
                return deps.choose_reply(pool)
    except Exception:
        pass
    return deps.choose_reply(deps.gm_replies)


async def maybe_send_market_warning(message, *, deps: GreetingFlowDependencies) -> None:
    now = deps.now_utc()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    if now.weekday() < 5:
        return
    with suppress(Exception):
        await message.answer(deps.weekend_warning_text)
