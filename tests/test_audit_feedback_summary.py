from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from app.bot.handlers import _format_feedback_summary
from app.services.audit import AuditService


def test_summarize_feedback_rows_aggregates_counts_and_suggestions() -> None:
    svc = AuditService(db_factory=None)
    rows = [
        SimpleNamespace(
            created_at=datetime(2026, 4, 15, 10, 0, tzinfo=UTC),
            payload_json={
                "sentiment": "negative",
                "reason": "wrong",
                "source": "button",
                "reply_preview": "btc read was off",
                "improvement_text": "",
                "from_username": "anon1",
            },
        ),
        SimpleNamespace(
            created_at=datetime(2026, 4, 15, 9, 30, tzinfo=UTC),
            payload_json={
                "sentiment": "positive",
                "reason": "thumbs_up",
                "source": "button",
                "reply_preview": "clean answer",
                "from_username": "anon2",
            },
        ),
        SimpleNamespace(
            created_at=datetime(2026, 4, 15, 9, 0, tzinfo=UTC),
            payload_json={
                "sentiment": "suggestion",
                "reason": "long",
                "source": "message",
                "reply_preview": "too much detail",
                "improvement_text": "keep it tighter",
                "from_username": "anon3",
            },
        ),
    ]

    summary = svc.summarize_feedback_rows(rows, hours=24)

    assert summary["total"] == 3
    assert summary["sentiments"]["negative"] == 1
    assert summary["sentiments"]["positive"] == 1
    assert summary["sentiments"]["suggestion"] == 1
    assert summary["top_reasons"][0] == ("wrong", 1)
    assert summary["top_sources"][0] == ("button", 2)
    assert summary["recent"][0]["reply_preview"] == "btc read was off"
    assert summary["suggestions"][0]["improvement_text"] == "keep it tighter"


def test_format_feedback_summary_renders_key_sections() -> None:
    text = _format_feedback_summary(
        {
            "hours": 24,
            "total": 4,
            "sentiments": {"positive": 1, "negative": 2, "suggestion": 1},
            "top_reasons": [("wrong", 2), ("long", 1)],
            "top_sources": [("button", 3), ("message", 1)],
            "recent": [
                {
                    "created_at": "2026-04-15T10:00:00+00:00",
                    "sentiment": "negative",
                    "reason": "wrong",
                    "reply_preview": "sol call was stale",
                }
            ],
            "suggestions": [
                {
                    "improvement_text": "add fresher prices",
                }
            ],
            "sampled": False,
        }
    )

    assert "<b>feedback last 24h</b>" in text
    assert "top reasons: wrong 2, long 1" in text
    assert "<b>recent feedback</b>" in text
    assert "sol call was stale" in text
    assert "<b>latest suggestions</b>" in text
    assert "add fresher prices" in text
