from __future__ import annotations

from app.bot.prefs import effective_timeframe


def test_effective_timeframe_pref_used_when_missing() -> None:
    tf = effective_timeframe(user_text="chart btc", settings={"preferred_timeframe": "4h"}, default="1h")
    assert tf == "4h"


def test_effective_timeframe_text_wins_over_pref() -> None:
    tf = effective_timeframe(user_text="chart btc 15m", settings={"preferred_timeframe": "4h"}, default="1h")
    assert tf == "15m"

