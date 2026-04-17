from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.db.models import AuditEvent


class AuditService:
    def __init__(self, db_factory) -> None:
        self.db_factory = db_factory

    async def log(self, event_type: str, payload: dict, success: bool = True, user_id: int | None = None, latency_ms: int | None = None) -> None:
        async with self.db_factory() as session:
            row = AuditEvent(
                user_id=user_id,
                event_type=event_type,
                payload_json=payload,
                success=success,
                latency_ms=latency_ms,
            )
            session.add(row)
            await session.commit()

    def summarize_feedback_rows(self, rows: list[AuditEvent], *, hours: int) -> dict:
        sentiments: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        recent: list[dict] = []
        suggestions: list[dict] = []

        for row in rows:
            payload = dict(getattr(row, "payload_json", {}) or {})
            sentiment = str(payload.get("sentiment") or "unknown").strip().lower() or "unknown"
            reason = str(payload.get("reason") or "other").strip().lower() or "other"
            source = str(payload.get("source") or "unknown").strip().lower() or "unknown"
            preview = str(payload.get("reply_preview") or "").strip()
            improvement = str(payload.get("improvement_text") or "").strip()
            from_username = str(payload.get("from_username") or "").strip()

            sentiments[sentiment] += 1
            reasons[reason] += 1
            sources[source] += 1

            created_at = getattr(row, "created_at", None)
            created_at_iso = created_at.astimezone(UTC).isoformat() if isinstance(created_at, datetime) else ""
            item = {
                "created_at": created_at_iso,
                "sentiment": sentiment,
                "reason": reason,
                "source": source,
                "from_username": from_username,
                "reply_preview": preview[:160],
                "improvement_text": improvement[:240] or None,
            }
            if len(recent) < 5:
                recent.append(item)
            if improvement and len(suggestions) < 3:
                suggestions.append(item)

        total = len(rows)
        return {
            "hours": int(hours),
            "total": total,
            "sentiments": dict(sentiments),
            "sources": dict(sources),
            "reasons": dict(reasons),
            "top_reasons": reasons.most_common(5),
            "top_sources": sources.most_common(5),
            "recent": recent,
            "suggestions": suggestions,
        }

    def summarize_reply_analytics_rows(
        self,
        reply_rows: list[AuditEvent],
        feedback_rows: list[AuditEvent],
        *,
        hours: int,
    ) -> dict:
        routes: Counter[str] = Counter()
        reply_kinds: Counter[str] = Counter()
        chat_modes: Counter[str] = Counter()
        negative_routes: Counter[str] = Counter()
        total_route_counts: Counter[str] = Counter()
        recent: list[dict] = []
        feedback_by_reply: dict[tuple[int, int], Counter[str]] = {}

        for row in feedback_rows:
            payload = dict(getattr(row, "payload_json", {}) or {})
            chat_id = payload.get("chat_id")
            reply_message_id = payload.get("reply_message_id")
            if chat_id is None or reply_message_id is None:
                continue
            try:
                key = (int(chat_id), int(reply_message_id))
            except Exception:
                continue
            sentiment = str(payload.get("sentiment") or "unknown").strip().lower() or "unknown"
            if key not in feedback_by_reply:
                feedback_by_reply[key] = Counter()
            feedback_by_reply[key][sentiment] += 1

        touched = 0
        positives = 0
        negatives = 0
        suggestions = 0

        for row in reply_rows:
            payload = dict(getattr(row, "payload_json", {}) or {})
            chat_id = payload.get("chat_id")
            reply_message_id = payload.get("reply_message_id")
            route = str(payload.get("route") or "unknown").strip().lower() or "unknown"
            reply_kind = str(payload.get("reply_kind") or "unknown").strip().lower() or "unknown"
            chat_mode = str(payload.get("chat_mode") or "unknown").strip().lower() or "unknown"

            routes[route] += 1
            reply_kinds[reply_kind] += 1
            chat_modes[chat_mode] += 1
            total_route_counts[route] += 1

            feedback_counts: Counter[str] = Counter()
            try:
                if chat_id is not None and reply_message_id is not None:
                    feedback_counts = feedback_by_reply.get((int(chat_id), int(reply_message_id)), Counter())
            except Exception:
                feedback_counts = Counter()

            if feedback_counts:
                touched += 1
                if feedback_counts.get("positive", 0) > 0:
                    positives += 1
                if feedback_counts.get("negative", 0) > 0:
                    negatives += 1
                    negative_routes[route] += 1
                if feedback_counts.get("suggestion", 0) > 0:
                    suggestions += 1

            created_at = getattr(row, "created_at", None)
            created_at_iso = created_at.astimezone(UTC).isoformat() if isinstance(created_at, datetime) else ""
            if len(recent) < 5:
                recent.append(
                    {
                        "created_at": created_at_iso,
                        "route": route,
                        "reply_kind": reply_kind,
                        "chat_mode": chat_mode,
                        "reply_preview": str(payload.get("reply_preview") or "")[:160],
                        "has_negative": feedback_counts.get("negative", 0) > 0,
                        "has_positive": feedback_counts.get("positive", 0) > 0,
                    }
                )

        total = len(reply_rows)
        eligible_negative_rates = []
        for route, count in total_route_counts.items():
            if count < 2:
                continue
            neg = negative_routes.get(route, 0)
            eligible_negative_rates.append((route, neg, count, round((neg / count) * 100.0, 1)))
        eligible_negative_rates.sort(key=lambda item: (-item[3], -item[2], item[0]))

        return {
            "hours": int(hours),
            "total": total,
            "touched": touched,
            "positive_feedback": positives,
            "negative_feedback": negatives,
            "suggestion_feedback": suggestions,
            "routes": dict(routes),
            "reply_kinds": dict(reply_kinds),
            "chat_modes": dict(chat_modes),
            "top_routes": routes.most_common(5),
            "top_reply_kinds": reply_kinds.most_common(5),
            "top_negative_routes": negative_routes.most_common(5),
            "negative_rate_pct": round((negatives / touched) * 100.0, 1) if touched else 0.0,
            "worst_routes": eligible_negative_rates[:5],
            "recent": recent,
        }

    async def feedback_summary(self, *, hours: int = 24, limit: int = 500) -> dict:
        hours = max(1, min(int(hours), 24 * 30))
        limit = max(1, min(int(limit), 1000))
        cutoff = datetime.now(UTC) - timedelta(hours=hours)

        async with self.db_factory() as session:
            result = await session.execute(
                select(AuditEvent)
                .where(
                    AuditEvent.event_type == "user_feedback",
                    AuditEvent.created_at >= cutoff,
                )
                .order_by(AuditEvent.created_at.desc())
                .limit(limit)
            )
            rows = list(result.scalars().all())

        summary = self.summarize_feedback_rows(rows, hours=hours)
        summary["limit"] = limit
        summary["sampled"] = len(rows) >= limit
        summary["from"] = cutoff.isoformat()
        summary["to"] = datetime.now(UTC).isoformat()
        return summary

    async def reply_analytics_summary(self, *, hours: int = 24, limit: int = 1000) -> dict:
        hours = max(1, min(int(hours), 24 * 30))
        limit = max(1, min(int(limit), 5000))
        cutoff = datetime.now(UTC) - timedelta(hours=hours)

        async with self.db_factory() as session:
            reply_result = await session.execute(
                select(AuditEvent)
                .where(
                    AuditEvent.event_type == "bot_reply",
                    AuditEvent.created_at >= cutoff,
                )
                .order_by(AuditEvent.created_at.desc())
                .limit(limit)
            )
            feedback_result = await session.execute(
                select(AuditEvent)
                .where(
                    AuditEvent.event_type == "user_feedback",
                    AuditEvent.created_at >= cutoff,
                )
                .order_by(AuditEvent.created_at.desc())
                .limit(limit)
            )
            reply_rows = list(reply_result.scalars().all())
            feedback_rows = list(feedback_result.scalars().all())

        summary = self.summarize_reply_analytics_rows(reply_rows, feedback_rows, hours=hours)
        summary["limit"] = limit
        summary["sampled"] = len(reply_rows) >= limit or len(feedback_rows) >= limit
        summary["from"] = cutoff.isoformat()
        summary["to"] = datetime.now(UTC).isoformat()
        return summary

    async def quality_summary(self, *, hours: int = 24) -> dict:
        hours = max(1, min(int(hours), 24 * 30))
        feedback = await self.feedback_summary(hours=hours)
        replies = await self.reply_analytics_summary(hours=hours)

        top_reason = None
        if feedback.get("top_reasons"):
            reason, count = feedback["top_reasons"][0]
            top_reason = {"reason": reason, "count": count}

        worst_route = None
        if replies.get("worst_routes"):
            route, neg, total, rate = replies["worst_routes"][0]
            worst_route = {
                "route": route,
                "negative": neg,
                "total": total,
                "negative_rate_pct": rate,
            }

        recent_negative_examples: list[dict] = []
        for item in list(feedback.get("recent") or []):
            if str(item.get("sentiment") or "").lower() != "negative":
                continue
            recent_negative_examples.append(
                {
                    "reason": str(item.get("reason") or "other"),
                    "preview": str(item.get("reply_preview") or "")[:160],
                    "created_at": str(item.get("created_at") or ""),
                }
            )
            if len(recent_negative_examples) >= 3:
                break

        return {
            "hours": hours,
            "feedback": feedback,
            "replies": replies,
            "headline": {
                "negative_feedback": int(feedback.get("sentiments", {}).get("negative", 0) or 0),
                "suggestions": int(feedback.get("sentiments", {}).get("suggestion", 0) or 0),
                "reply_negative_rate_pct": float(replies.get("negative_rate_pct") or 0.0),
                "top_reason": top_reason,
                "worst_route": worst_route,
            },
            "recent_negative_examples": recent_negative_examples,
            "generated_at": datetime.now(UTC).isoformat(),
        }
