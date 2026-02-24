"""Trade journal: log trades and list with optional stats."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from app.db.models import TradeJournalEntry, User


class TradeJournalService:
    def __init__(self, db_factory) -> None:
        self.db_factory = db_factory

    async def _user_id(self, chat_id: int) -> int | None:
        async with self.db_factory() as session:
            r = await session.execute(select(User.id).where(User.telegram_chat_id == chat_id))
            row = r.scalar_one_or_none()
            return int(row) if row is not None else None

    async def log_trade(
        self,
        chat_id: int,
        symbol: str,
        side: str,
        entry: float,
        exit_price: float | None = None,
        stop: float | None = None,
        targets: list[float] | None = None,
        outcome: str | None = None,
        pnl_quote: float | None = None,
        notes: str | None = None,
    ) -> TradeJournalEntry | None:
        user_id = await self._user_id(chat_id)
        if user_id is None:
            return None
        async with self.db_factory() as session:
            entry_obj = TradeJournalEntry(
                user_id=user_id,
                symbol=symbol.upper(),
                side=(side or "long").strip().lower() or "long",
                entry=float(entry),
                exit_price=float(exit_price) if exit_price is not None else None,
                stop=float(stop) if stop is not None else None,
                targets_json=list(targets or []),
                outcome=(outcome or "")[:20] or None,
                pnl_quote=float(pnl_quote) if pnl_quote is not None else None,
                notes=notes,
            )
            session.add(entry_obj)
            await session.commit()
            await session.refresh(entry_obj)
            return entry_obj

    async def list_trades(self, chat_id: int, limit: int = 30) -> list[dict]:
        user_id = await self._user_id(chat_id)
        if user_id is None:
            return []
        async with self.db_factory() as session:
            r = await session.execute(
                select(TradeJournalEntry)
                .where(TradeJournalEntry.user_id == user_id)
                .order_by(TradeJournalEntry.created_at.desc())
                .limit(limit)
            )
            rows = list(r.scalars().all())
        return [
            {
                "id": e.id,
                "symbol": e.symbol,
                "side": e.side,
                "entry": e.entry,
                "exit_price": e.exit_price,
                "stop": e.stop,
                "targets": e.targets_json or [],
                "outcome": e.outcome,
                "pnl_quote": e.pnl_quote,
                "notes": e.notes,
                "created_at": e.created_at,
            }
            for e in rows
        ]

    async def get_stats(self, chat_id: int, days: int = 30) -> dict:
        user_id = await self._user_id(chat_id)
        if user_id is None:
            return {"trades": 0, "wins": 0, "win_rate": 0, "total_pnl": 0}
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        async with self.db_factory() as session:
            r = await session.execute(
                select(
                    func.count(TradeJournalEntry.id).label("n"),
                    func.sum(func.coalesce(TradeJournalEntry.pnl_quote, 0)).label("total"),
                ).where(
                    TradeJournalEntry.user_id == user_id,
                    TradeJournalEntry.created_at >= cutoff,
                    TradeJournalEntry.pnl_quote.isnot(None),
                )
            )
            row = r.one()
            n = int(row.n or 0)
            total = float(row.total or 0)
            r2 = await session.execute(
                select(func.count(TradeJournalEntry.id)).where(
                    TradeJournalEntry.user_id == user_id,
                    TradeJournalEntry.created_at >= cutoff,
                    TradeJournalEntry.pnl_quote > 0,
                )
            )
            wins = int(r2.scalar_one() or 0)
        return {
            "trades": n,
            "wins": wins,
            "win_rate": round(100 * wins / n, 1) if n else 0,
            "total_pnl": round(total, 2),
        }
