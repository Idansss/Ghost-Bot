"""GDPR data export and account deletion."""
from __future__ import annotations

from sqlalchemy import delete, select

from app.db.models import Alert, Position, TradeJournalEntry, User, Wallet


class GDPRService:
    def __init__(self, db_factory) -> None:
        self.db_factory = db_factory

    async def export_my_data(self, chat_id: int) -> dict | None:
        """Return all stored data for a user as a serialisable dict."""
        async with self.db_factory() as session:
            r = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
            user = r.scalar_one_or_none()
            if user is None:
                return None

            alerts_r = await session.execute(select(Alert).where(Alert.user_id == user.id))
            alerts = [
                {
                    "id": a.id,
                    "symbol": a.symbol,
                    "condition": a.condition,
                    "target_price": a.target_price,
                    "status": a.status,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in alerts_r.scalars().all()
            ]

            wallets_r = await session.execute(select(Wallet).where(Wallet.user_id == user.id))
            wallets = [
                {
                    "chain": w.chain,
                    "address": w.address,
                    "label": w.label,
                    "created_at": w.created_at.isoformat() if w.created_at else None,
                }
                for w in wallets_r.scalars().all()
            ]

            positions_r = await session.execute(select(Position).where(Position.user_id == user.id))
            positions = [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "size_quote": p.size_quote,
                    "leverage": p.leverage,
                    "notes": p.notes,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in positions_r.scalars().all()
            ]

            journal_r = await session.execute(
                select(TradeJournalEntry).where(TradeJournalEntry.user_id == user.id)
            )
            journal = [
                {
                    "id": e.id,
                    "symbol": e.symbol,
                    "side": e.side,
                    "entry": e.entry,
                    "exit_price": e.exit_price,
                    "outcome": e.outcome,
                    "pnl_quote": e.pnl_quote,
                    "notes": e.notes,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in journal_r.scalars().all()
            ]

        return {
            "telegram_chat_id": user.telegram_chat_id,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_seen_at": user.last_seen_at.isoformat() if user.last_seen_at else None,
            "settings": user.settings_json or {},
            "alerts": alerts,
            "wallets": wallets,
            "positions": positions,
            "trade_journal": journal,
        }

    async def delete_account(self, chat_id: int) -> bool:
        """Delete all data for a user. CASCADE handles child rows."""
        async with self.db_factory() as session:
            result = await session.execute(
                delete(User).where(User.telegram_chat_id == chat_id)
            )
            await session.commit()
            return (result.rowcount or 0) > 0
