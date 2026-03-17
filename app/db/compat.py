from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Columns that must exist on the alerts table, in order.
# Using ADD COLUMN IF NOT EXISTS (PostgreSQL 9.6+) so these are idempotent.
_ALERT_COLUMNS: list[tuple[str, str]] = [
    ("source",          "VARCHAR(50)"),
    ("source_exchange", "VARCHAR(20)"),
    ("instrument_id",   "VARCHAR(40)"),
    ("market_kind",     "VARCHAR(10)"),
    ("conditions_json", "JSONB"),
    ("idempotency_key", "VARCHAR(64)"),
]

_ALERT_INDEXES: list[tuple[str, str]] = [
    # (index_name, CREATE INDEX ... statement)
    (
        "ix_alerts_idempotency_key",
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_alerts_idempotency_key "
        "ON alerts (idempotency_key)",
    ),
]


def _collect_exc_text(exc: BaseException) -> tuple[str, str]:
    messages: list[str] = []
    type_names: list[str] = []
    stack: list[BaseException] = [exc]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        type_names.append(type(current).__name__)
        message = str(current).strip()
        if message:
            messages.append(message)
        for attr in ("orig", "__cause__", "__context__"):
            child = getattr(current, attr, None)
            if isinstance(child, BaseException):
                stack.append(child)

    return " | ".join(messages).lower(), " | ".join(type_names).lower()


def is_alert_schema_issue(exc: BaseException) -> bool:
    message, exc_types = _collect_exc_text(exc)
    combined = message + " " + exc_types
    return any(
        token in combined
        for token in (
            "undefinedcolumn",
            "undefinedtable",
            "nosuchtable",
            "no such column",
            "has no column named",
            "no such table",
            "unknown column",
            "invalid column name",
            "column does not exist",
            "relation does not exist",
            "table or view does not exist",
        )
    )


async def ensure_alert_schema_compat(
    db_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
) -> int:
    """Add any missing columns/indexes to the alerts table.

    Uses ADD COLUMN IF NOT EXISTS (PostgreSQL 9.6+) via raw SQL so it works
    with asyncpg and any other async driver — no SQLAlchemy reflection needed.
    Returns number of columns that were missing and added.
    """
    async with db_factory() as session:
        # Query information_schema directly — works with asyncpg, psycopg2, etc.
        result = await session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'alerts' AND table_schema = current_schema()"
            )
        )
        existing = {row[0] for row in result.fetchall()}

        if not existing:
            # Table doesn't exist yet — nothing to patch (migrations will create it)
            return 0

        missing = [col for col, _ in _ALERT_COLUMNS if col not in existing]

        for col_name, col_type in _ALERT_COLUMNS:
            if col_name not in existing:
                await session.execute(
                    text(f"ALTER TABLE alerts ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
                )

        for _idx_name, idx_stmt in _ALERT_INDEXES:
            try:
                await session.execute(text(idx_stmt))
            except Exception:
                # Index may already exist under a different name — non-fatal
                await session.rollback()

        await session.commit()
        return len(missing)
