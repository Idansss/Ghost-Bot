from __future__ import annotations

from app.bot.handlers import _format_quality_summary


def test_format_quality_summary_renders_headline_and_examples() -> None:
    text = _format_quality_summary(
        {
            "hours": 24,
            "headline": {
                "negative_feedback": 4,
                "suggestions": 2,
                "reply_negative_rate_pct": 33.3,
                "top_reason": {"reason": "wrong", "count": 3},
                "worst_route": {"route": "market_chat", "negative": 2, "total": 5, "negative_rate_pct": 40.0},
            },
            "feedback": {
                "top_reasons": [("wrong", 3), ("long", 1)],
            },
            "replies": {
                "top_routes": [("market_chat", 5), ("analysis_followup", 3)],
            },
            "recent_negative_examples": [
                {
                    "created_at": "2026-04-15T10:00:00+00:00",
                    "reason": "wrong",
                    "preview": "btc call was stale",
                }
            ],
        }
    )

    assert "<b>quality last 24h</b>" in text
    assert "negative feedback <b>4</b>" in text
    assert "top complaint: <b>wrong</b> (3)" in text
    assert "worst route: <b>market_chat</b> (2/5 negative" in text
    assert "top reasons: wrong 3, long 1" in text
    assert "top routes: market_chat 5, analysis_followup 3" in text
    assert "<b>recent bad examples</b>" in text
    assert "btc call was stale" in text
