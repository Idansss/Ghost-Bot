from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.prices import PriceAdapter
from app.core.cache import RedisCache
from app.db.models import Alert, User


class AlertsService:
    def __init__(
        self,
        db_factory,
        cache: RedisCache,
        price_adapter: PriceAdapter,
        alerts_limit_per_day: int,
        cooldown_minutes: int,
    ) -> None:
        self.db_factory = db_factory
        self.cache = cache
        self.price_adapter = price_adapter
        self.alerts_limit_per_day = alerts_limit_per_day
        self.cooldown_minutes = cooldown_minutes

    async def _get_or_create_user(self, session: AsyncSession, chat_id: int) -> User:
        q = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
        user = q.scalar_one_or_none()
        if user:
            user.last_seen_at = datetime.utcnow()
            return user
        user = User(telegram_chat_id=chat_id, settings_json={})
        session.add(user)
        await session.flush()
        return user

    async def create_alert(self, chat_id: int, symbol: str, condition: str, target_price: float, source: str = "user") -> Alert:
        daily_key = f"rl:alerts:{chat_id}:{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        count = await self.cache.incr_with_expiry(daily_key, 86400)
        if count > self.alerts_limit_per_day:
            raise RuntimeError("Daily alert creation limit reached.")

        async with self.db_factory() as session:
            user = await self._get_or_create_user(session, chat_id)
            alert = Alert(
                user_id=user.id,
                symbol=symbol.upper(),
                condition=condition,
                target_price=target_price,
                status="active",
                source=source,
            )
            session.add(alert)
            await session.commit()
            await session.refresh(alert)
            return alert

    async def list_alerts(self, chat_id: int) -> list[Alert]:
        async with self.db_factory() as session:
            q = await session.execute(
                select(Alert)
                .join(User, User.id == Alert.user_id)
                .where(User.telegram_chat_id == chat_id)
                .where(Alert.status.in_(["active", "paused"]))
                .order_by(Alert.created_at.desc())
            )
            return list(q.scalars().all())

    async def delete_alert(self, chat_id: int, alert_id: int) -> bool:
        async with self.db_factory() as session:
            q = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
            user = q.scalar_one_or_none()
            if not user:
                return False

            alert_q = await session.execute(select(Alert).where(Alert.id == alert_id, Alert.user_id == user.id))
            alert = alert_q.scalar_one_or_none()
            if not alert:
                return False
            await session.delete(alert)
            await session.commit()
            return True

    def _condition_met(self, condition: str, target: float, prev_price: float, current_price: float) -> bool:
        if condition == "above":
            return prev_price < target <= current_price
        if condition == "below":
            return prev_price > target >= current_price
        crossed_up = prev_price < target <= current_price
        crossed_down = prev_price > target >= current_price
        return crossed_up or crossed_down

    async def process_alerts(self, notifier) -> int:
        triggered_count = 0
        async with self.db_factory() as session:
            q = await session.execute(select(Alert).where(Alert.status == "active"))
            alerts = list(q.scalars().all())
            if not alerts:
                return 0

            symbol_prices: dict[str, float] = {}
            for alert in alerts:
                if alert.symbol not in symbol_prices:
                    try:
                        symbol_prices[alert.symbol] = float((await self.price_adapter.get_price(alert.symbol))["price"])
                    except Exception:  # noqa: BLE001
                        continue

            now = datetime.utcnow()
            for alert in alerts:
                if alert.symbol not in symbol_prices:
                    continue

                current_price = symbol_prices[alert.symbol]
                prev_key = f"alert:lastprice:{alert.symbol}"
                prev_payload = await self.cache.get_json(prev_key)
                prev_price = float(prev_payload["price"]) if prev_payload else current_price

                await self.cache.set_json(prev_key, {"price": current_price}, ttl=7200)

                if alert.cooldown_until and alert.cooldown_until > now:
                    continue

                if not self._condition_met(alert.condition, alert.target_price, prev_price, current_price):
                    continue

                dedupe_key = f"alert:dedupe:{alert.id}:{now.strftime('%Y%m%d%H%M')}"
                if not await self.cache.set_if_absent(dedupe_key, ttl=120):
                    continue

                alert.status = "triggered"
                alert.triggered_at = now
                alert.cooldown_until = now + timedelta(minutes=self.cooldown_minutes)
                alert.last_triggered_price = current_price
                triggered_count += 1

                user_q = await session.execute(select(User).where(User.id == alert.user_id))
                user = user_q.scalar_one_or_none()
                if user:
                    await notifier(
                        user.telegram_chat_id,
                        (
                            f"Alert #{alert.id} hit: {alert.symbol} {alert.condition} {alert.target_price:.4f}\n"
                            f"Now: {current_price:.4f} (source: live)"
                        ),
                    )

            await session.commit()

        return triggered_count

    async def clear_user_alerts(self, chat_id: int) -> int:
        async with self.db_factory() as session:
            user_q = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
            user = user_q.scalar_one_or_none()
            if not user:
                return 0
            result = await session.execute(delete(Alert).where(Alert.user_id == user.id))
            await session.commit()
            return int(result.rowcount or 0)

    async def _set_user_alert_status(self, chat_id: int, status: str) -> int:
        async with self.db_factory() as session:
            user_q = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
            user = user_q.scalar_one_or_none()
            if not user:
                return 0
            q = await session.execute(select(Alert).where(Alert.user_id == user.id))
            alerts = list(q.scalars().all())
            count = 0
            for row in alerts:
                row.status = status
                count += 1
            await session.commit()
            return count

    async def pause_user_alerts(self, chat_id: int) -> int:
        return await self._set_user_alert_status(chat_id, "paused")

    async def resume_user_alerts(self, chat_id: int) -> int:
        return await self._set_user_alert_status(chat_id, "active")
