"""Scheduled reports: subscribe to daily market summary at a given UTC time."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from app.db.models import ScheduledReport, User


class ScheduledReportService:
    def __init__(self, db_factory) -> None:
        self.db_factory = db_factory

    async def _user_id(self, chat_id: int) -> int | None:
        async with self.db_factory() as session:
            r = await session.execute(select(User.id).where(User.telegram_chat_id == chat_id))
            row = r.scalar_one_or_none()
            return int(row) if row is not None else None

    async def subscribe(
        self,
        chat_id: int,
        report_type: str = "market_summary",
        hour_utc: int = 9,
        minute_utc: int = 0,
        tz: str | None = None,
    ) -> ScheduledReport | None:
        user_id = await self._user_id(chat_id)
        if user_id is None:
            return None
        async with self.db_factory() as session:
            r = await session.execute(
                select(ScheduledReport).where(
                    ScheduledReport.user_id == user_id,
                    ScheduledReport.chat_id == chat_id,
                )
            )
            existing = r.scalar_one_or_none()
            if existing:
                existing.report_type = report_type
                existing.cron_hour_utc = max(0, min(23, hour_utc))
                existing.cron_minute_utc = max(0, min(59, minute_utc))
                existing.timezone = (tz or "UTC")[:48]
                existing.enabled = True
                await session.commit()
                await session.refresh(existing)
                return existing
            rec = ScheduledReport(
                user_id=user_id,
                chat_id=chat_id,
                report_type=report_type,
                cron_hour_utc=max(0, min(23, hour_utc)),
                cron_minute_utc=max(0, min(59, minute_utc)),
                timezone=(tz or "UTC")[:48],
                enabled=True,
            )
            session.add(rec)
            await session.commit()
            await session.refresh(rec)
            return rec

    async def unsubscribe(self, chat_id: int) -> bool:
        user_id = await self._user_id(chat_id)
        if user_id is None:
            return False
        async with self.db_factory() as session:
            r = await session.execute(
                select(ScheduledReport).where(
                    ScheduledReport.user_id == user_id,
                    ScheduledReport.chat_id == chat_id,
                )
            )
            rec = r.scalar_one_or_none()
            if rec:
                await session.delete(rec)
                await session.commit()
                return True
        return False

    async def list_reports(self, chat_id: int) -> list[dict]:
        user_id = await self._user_id(chat_id)
        if user_id is None:
            return []
        async with self.db_factory() as session:
            r = await session.execute(
                select(ScheduledReport).where(ScheduledReport.user_id == user_id)
            )
            rows = list(r.scalars().all())
        return [
            {
                "id": x.id,
                "chat_id": x.chat_id,
                "report_type": x.report_type,
                "cron_hour_utc": x.cron_hour_utc,
                "cron_minute_utc": x.cron_minute_utc,
                "timezone": x.timezone,
                "enabled": x.enabled,
            }
            for x in rows
        ]

    async def get_due_reports(self, now: datetime | None = None) -> list[tuple[int, str]]:
        """Return list of (chat_id, report_type) for subscriptions due at the given UTC time (default now)."""
        now = now or datetime.now(timezone.utc)
        h, m = now.hour, now.minute
        async with self.db_factory() as session:
            r = await session.execute(
                select(ScheduledReport.chat_id, ScheduledReport.report_type).where(
                    ScheduledReport.enabled.is_(True),
                    ScheduledReport.cron_hour_utc == h,
                    ScheduledReport.cron_minute_utc == m,
                )
            )
            return [(row[0], row[1]) for row in r.all()]
