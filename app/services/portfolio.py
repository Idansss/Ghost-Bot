# Portfolio / positions and unrealized PnL
from __future__ import annotations

from sqlalchemy import select

from app.adapters.prices import PriceAdapter
from app.db.models import Position, User


class PortfolioService:
    def __init__(self, db_factory, price_adapter: PriceAdapter, max_notional_warning_usd: float = 100_000.0) -> None:
        self.db_factory = db_factory
        self.price_adapter = price_adapter
        self.max_notional_warning_usd = max_notional_warning_usd

    async def _user_id(self, chat_id: int) -> int | None:
        async with self.db_factory() as session:
            r = await session.execute(select(User.id).where(User.telegram_chat_id == chat_id))
            row = r.scalar_one_or_none()
            return int(row) if row is not None else None

    async def add_position(
        self,
        chat_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        size_quote: float = 0.0,
        leverage: float = 1.0,
        notes: str | None = None,
    ) -> tuple[Position | None, str | None]:
        user_id = await self._user_id(chat_id)
        if user_id is None:
            return None, "User not found."
        side = (side or "long").strip().lower()
        if side not in ("long", "short"):
            side = "long"
        notional = size_quote * leverage if size_quote and leverage else 0.0
        warning = None
        if notional > self.max_notional_warning_usd:
            warning = f"Notional ${notional:,.0f} - consider sizing down."
        async with self.db_factory() as session:
            pos = Position(
                user_id=user_id,
                symbol=symbol.upper(),
                side=side,
                entry_price=float(entry_price),
                size_quote=float(size_quote or 0),
                leverage=float(leverage or 1),
                notes=(notes or "")[:255],
            )
            session.add(pos)
            await session.commit()
            await session.refresh(pos)
            return pos, warning

    async def list_positions(self, chat_id: int) -> list[dict]:
        user_id = await self._user_id(chat_id)
        if user_id is None:
            return []
        async with self.db_factory() as session:
            r = await session.execute(select(Position).where(Position.user_id == user_id).order_by(Position.created_at.desc()))
            positions = list(r.scalars().all())
        out = []
        for p in positions:
            try:
                price_data = await self.price_adapter.get_price(p.symbol)
                current = float(price_data.get("price") or p.entry_price)
            except Exception:
                current = p.entry_price
            entry = p.entry_price
            if p.side == "long":
                pnl_pct = ((current - entry) / entry * 100) if entry else 0
            else:
                pnl_pct = ((entry - current) / entry * 100) if entry else 0
            notional = (p.size_quote or 0) * (p.leverage or 1)
            pnl_quote = notional * (pnl_pct / 100) if notional else 0
            out.append({
                "id": p.id,
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": p.entry_price,
                "current_price": current,
                "size_quote": p.size_quote,
                "leverage": p.leverage,
                "pnl_pct": round(pnl_pct, 2),
                "pnl_quote": round(pnl_quote, 2),
                "notes": p.notes,
                "created_at": p.created_at,
            })
        return out

    async def delete_position(self, chat_id: int, position_id: int) -> bool:
        user_id = await self._user_id(chat_id)
        if user_id is None:
            return False
        async with self.db_factory() as session:
            r = await session.execute(select(Position).where(Position.id == position_id, Position.user_id == user_id))
            pos = r.scalar_one_or_none()
            if pos:
                await session.delete(pos)
                await session.commit()
                return True
        return False

    async def total_unrealized_pnl(self, chat_id: int) -> float:
        positions = await self.list_positions(chat_id)
        return sum(p["pnl_quote"] for p in positions)
