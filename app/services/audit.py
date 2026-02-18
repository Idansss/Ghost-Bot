from __future__ import annotations

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
