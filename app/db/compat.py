from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from sqlalchemy import inspect, text
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.ext.asyncio import AsyncSession

_ALERT_COLUMN_DEFS: dict[str, dict[str, str]] = {
    "source": {
        "default": "VARCHAR(50)",
        "postgresql": "VARCHAR(50)",
        "sqlite": "TEXT",
    },
    "source_exchange": {
        "default": "VARCHAR(20)",
        "postgresql": "VARCHAR(20)",
        "sqlite": "TEXT",
    },
    "instrument_id": {
        "default": "VARCHAR(40)",
        "postgresql": "VARCHAR(40)",
        "sqlite": "TEXT",
    },
    "market_kind": {
        "default": "VARCHAR(10)",
        "postgresql": "VARCHAR(10)",
        "sqlite": "TEXT",
    },
    "conditions_json": {
        "default": "TEXT",
        "postgresql": "JSONB",
        "sqlite": "TEXT",
    },
    "idempotency_key": {
        "default": "VARCHAR(64)",
        "postgresql": "VARCHAR(64)",
        "sqlite": "TEXT",
    },
}


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
        orig = getattr(current, "orig", None)
        if isinstance(orig, BaseException):
            stack.append(orig)
        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            stack.append(cause)
        context = getattr(current, "__context__", None)
        if isinstance(context, BaseException):
            stack.append(context)

    return " | ".join(messages).lower(), " | ".join(type_names).lower()


def is_alert_schema_issue(exc: BaseException) -> bool:
    message, exc_types = _collect_exc_text(exc)
    return any(
        token in message or token in exc_types
        for token in (
            "undefinedcolumn",
            "undefinedtable",
            "nosuchtableerror",
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
    async with db_factory() as session:
        conn = await session.connection()
        dialect = (conn.dialect.name or "").lower()

        try:
            existing_columns = await conn.run_sync(
                lambda sync_conn: {
                    str(col["name"])
                    for col in inspect(sync_conn).get_columns("alerts")
                }
            )
        except NoSuchTableError:
            return 0

        stmts: list[str] = []
        for column, ddl_by_dialect in _ALERT_COLUMN_DEFS.items():
            if column in existing_columns:
                continue
            ddl_type = ddl_by_dialect.get(dialect, ddl_by_dialect["default"])
            stmts.append(f"ALTER TABLE alerts ADD COLUMN {column} {ddl_type}")

        for stmt in stmts:
            await session.execute(text(stmt))

        if stmts:
            await session.commit()

        return len(stmts)
