from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from app.db.models import User

DEFAULT_SETTINGS = {
    "risk_profile": "medium",
    "preferred_timeframe": "1h",
    "preferred_timeframes": ["1h"],
    "preferred_ema_periods": [20, 50, 200],
    "preferred_rsi_periods": [14],
    "preferred_exchange": "binance",
    "tone_mode": "wild",
    "anon_mode": True,
    "profanity_level": "light",
    "formal_mode": False,
}


class UserService:
    def __init__(self, db_factory) -> None:
        self.db_factory = db_factory

    async def ensure_user(self, chat_id: int) -> User:
        async with self.db_factory() as session:
            q = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
            user = q.scalar_one_or_none()
            if user:
                user.last_seen_at = datetime.utcnow()
                if not user.settings_json:
                    user.settings_json = DEFAULT_SETTINGS.copy()
                await session.commit()
                await session.refresh(user)
                return user

            user = User(
                telegram_chat_id=chat_id,
                settings_json=DEFAULT_SETTINGS.copy(),
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def get_settings(self, chat_id: int) -> dict:
        user = await self.ensure_user(chat_id)
        out = DEFAULT_SETTINGS.copy()
        out.update(user.settings_json or {})
        return out

    async def update_settings(self, chat_id: int, updates: dict) -> dict:
        async with self.db_factory() as session:
            q = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
            user = q.scalar_one_or_none()
            if not user:
                user = User(telegram_chat_id=chat_id, settings_json=DEFAULT_SETTINGS.copy())
                session.add(user)
                await session.flush()

            merged = DEFAULT_SETTINGS.copy()
            merged.update(user.settings_json or {})
            merged.update(updates)
            user.settings_json = merged
            user.last_seen_at = datetime.utcnow()
            await session.commit()
            return merged
