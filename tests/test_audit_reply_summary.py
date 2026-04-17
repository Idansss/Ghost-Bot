from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from app.bot.handlers import _format_reply_stats_summary
from app.services.audit import AuditService


def test_summarize_reply_analytics_rows_correlates_feedback() -> None:
    svc = AuditService(db_factory=None)
    reply_rows = [
        SimpleNamespace(
            created_at=datetime(2026, 4, 15, 10, 0, tzinfo=UTC),
            payload_json={
                "chat_id": 1,
                "reply_message_id": 100,
                "route": "market_chat",
                "reply_kind": "market_chat",
                "chat_mode": "hybrid",
                "reply_preview": "btc looks heavy",
            },
        ),
        SimpleNamespace(
            created_at=datetime(2026, 4, 15, 9, 45, tzinfo=UTC),
            payload_json={
                "chat_id": 1,
                "reply_message_id": 101,
                "route": "market_chat",
                "reply_kind": "market_chat",
                "chat_mode": "hybrid",
                "reply_preview": "eth holding up",
            },
        ),
        SimpleNamespace(
            created_at=datetime(2026, 4, 15, 9, 30, tzinfo=UTC),
            payload_json={
                "chat_id": 2,
                "reply_message_id": 200,
                "route": "analysis_followup",
                "reply_kind": "followup",
                "chat_mode": "chat_only",
                "reply_preview": "tighten the stop",
            },
        ),
    ]
    feedback_rows = [
        SimpleNamespace(
            created_at=datetime(2026, 4, 15, 10, 5, tzinfo=UTC),
            payload_json={
                "chat_id": 1,
                "reply_message_id": 100,
                "sentiment": "negative",
            },
        ),
        SimpleNamespace(
            created_at=datetime(2026, 4, 15, 10, 7, tzinfo=UTC),
            payload_json={
                "chat_id": 1,
                "reply_message_id": 101,
                "sentiment": "positive",
            },
        ),
    ]

    summary = svc.summarize_reply_analytics_rows(reply_rows, feedback_rows, hours=24)

    assert summary["total"] == 3
    assert summary["touched"] == 2
    assert summary["negative_feedback"] == 1
    assert summary["positive_feedback"] == 1
    assert summary["top_routes"][0] == ("market_chat", 2)
    assert summary["top_negative_routes"][0] == ("market_chat", 1)
    assert summary["negative_rate_pct"] == 50.0
    assert summary["worst_routes"][0] == ("market_chat", 1, 2, 50.0)


def test_format_reply_stats_summary_renders_sections() -> None:
    text = _format_reply_stats_summary(
        {
            "hours": 24,
            "total": 12,
            "touched": 5,
            "positive_feedback": 3,
            "negative_feedback": 2,
            "suggestion_feedback": 1,
            "negative_rate_pct": 40.0,
            "top_routes": [("market_chat", 7), ("analysis_followup", 3)],
            "top_reply_kinds": [("market_chat", 7), ("followup", 3)],
            "worst_routes": [("market_chat", 2, 7, 28.6)],
            "recent": [
                {
                    "created_at": "2026-04-15T10:00:00+00:00",
                    "route": "market_chat",
                    "reply_kind": "market_chat",
                    "reply_preview": "btc looks heavy",
                    "has_negative": True,
                    "has_positive": False,
                }
            ],
            "sampled": False,
        }
    )

    assert "<b>reply stats last 24h</b>" in text
    assert "replies <b>12</b>" in text
    assert "top routes: market_chat 7, analysis_followup 3" in text
    assert "<b>worst routes</b>" in text
    assert "market_chat · 2/7 negative" in text
    assert "<b>recent replies</b>" in text
    assert "btc looks heavy" in text


def test_quality_summary_builds_headline_from_feedback_and_replies(monkeypatch) -> None:
    svc = AuditService(db_factory=None)

    async def fake_feedback_summary(*, hours: int, limit: int = 500) -> dict:
        return {
            "hours": hours,
            "sentiments": {"negative": 2, "suggestion": 1},
            "top_reasons": [("wrong", 2)],
            "recent": [
                {
                    "sentiment": "negative",
                    "reason": "wrong",
                    "reply_preview": "btc call stale",
                    "created_at": "2026-04-15T10:00:00+00:00",
                }
            ],
        }

    async def fake_reply_summary(*, hours: int, limit: int = 1000) -> dict:
        return {
            "hours": hours,
            "negative_rate_pct": 25.0,
            "worst_routes": [("market_chat", 2, 8, 25.0)],
            "top_routes": [("market_chat", 8)],
        }

    monkeypatch.setattr(svc, "feedback_summary", fake_feedback_summary)
    monkeypatch.setattr(svc, "reply_analytics_summary", fake_reply_summary)

    import asyncio

    summary = asyncio.run(svc.quality_summary(hours=24))

    assert summary["headline"]["negative_feedback"] == 2
    assert summary["headline"]["suggestions"] == 1
    assert summary["headline"]["top_reason"] == {"reason": "wrong", "count": 2}
    assert summary["headline"]["worst_route"]["route"] == "market_chat"
    assert summary["recent_negative_examples"][0]["preview"] == "btc call stale"
