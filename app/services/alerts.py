from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.market_router import MarketDataRouter
from app.adapters.prices import PriceAdapter
from app.core.cache import RedisCache
from app.core.enums import AlertStatus
from app.db.models import Alert, IndicatorSnapshot, User
from app.domain.alerts import condition_met, extra_condition_met, parse_extra_conditions

logger = logging.getLogger(__name__)


class AlertsService:
    def __init__(
        self,
        db_factory,
        cache: RedisCache,
        price_adapter: PriceAdapter,
        market_router: MarketDataRouter | None,
        alerts_limit_per_day: int,
        cooldown_minutes: int,
        max_deviation_pct: float = 30.0,
    ) -> None:
        self.db_factory = db_factory
        self.cache = cache
        self.price_adapter = price_adapter
        self.market_router = market_router
        self.alerts_limit_per_day = alerts_limit_per_day
        self.cooldown_minutes = cooldown_minutes
        self.max_deviation_pct = max(1.0, float(max_deviation_pct))

    async def _get_or_create_user(self, session: AsyncSession, chat_id: int) -> User:
        q = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
        user = q.scalar_one_or_none()
        if user:
            user.last_seen_at = datetime.now(UTC)
            return user
        user = User(telegram_chat_id=chat_id, settings_json={})
        session.add(user)
        await session.flush()
        return user

    async def create_alert(self, chat_id: int, symbol: str, condition: str, target_price: float, source: str = "user", extra_conditions: list | None = None, idempotency_key: str | None = None) -> Alert:
        daily_key = f"rl:alerts:{chat_id}:{datetime.now(UTC).strftime('%Y%m%d')}"
        count = await self.cache.incr_with_expiry(daily_key, 86400)
        if count > self.alerts_limit_per_day:
            raise RuntimeError("Daily alert creation limit reached.")
        quote = None
        if source != "button":
            try:
                quote = await self.price_adapter.get_price(symbol)
                current = float(quote["price"])
                if current > 0:
                    deviation_pct = abs((float(target_price) - current) / current) * 100.0
                    if deviation_pct > self.max_deviation_pct:
                        raise RuntimeError(
                            f"Alert target is {deviation_pct:.1f}% away from current price. "
                            f"Use a closer level (<={self.max_deviation_pct:.0f}%) or confirm with a tighter target."
                        )
            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning("alert_price_fetch_failed", extra={"event": "alert_price_fetch_failed", "symbol": symbol, "error": str(exc)})

        async with self.db_factory() as session:
            # Return existing alert if an idempotency key is provided and already used
            if idempotency_key:
                existing_q = await session.execute(
                    select(Alert).where(Alert.idempotency_key == idempotency_key)
                )
                existing = existing_q.scalar_one_or_none()
                if existing is not None:
                    return existing

            user = await self._get_or_create_user(session, chat_id)
            alert = Alert(
                user_id=user.id,
                symbol=symbol.upper(),
                condition=condition,
                target_price=target_price,
                status=AlertStatus.ACTIVE,
                source=source,
                source_exchange=(quote or {}).get("exchange"),
                instrument_id=(quote or {}).get("instrument_id"),
                market_kind=(quote or {}).get("market_kind") or "spot",
                conditions_json=extra_conditions or None,
                idempotency_key=idempotency_key or None,
            )
            session.add(alert)
            await session.commit()
            await session.refresh(alert)
            return alert

    async def _get_alert_price(self, alert: Alert) -> tuple[float | None, dict]:
        if self.market_router and alert.source_exchange and alert.instrument_id:
            ex = str(alert.source_exchange).lower()
            adapter = self.market_router.adapters.get(ex)
            if adapter:
                try:
                    payload = await adapter.get_price(alert.instrument_id, market_kind=alert.market_kind or "spot")
                    return float(payload["price"]), {
                        "exchange": ex,
                        "instrument_id": alert.instrument_id,
                        "market_kind": alert.market_kind or "spot",
                    }
                except Exception as exc:
                    logger.debug("alert_direct_price_failed", extra={"alert_id": alert.id, "exchange": ex, "error": str(exc)})

        try:
            fallback = await self.price_adapter.get_price(alert.symbol)
            return float(fallback["price"]), {
                "exchange": fallback.get("exchange"),
                "instrument_id": fallback.get("instrument_id"),
                "market_kind": fallback.get("market_kind") or "spot",
            }
        except Exception as exc:
            logger.warning("alert_fallback_price_failed", extra={"alert_id": alert.id, "symbol": alert.symbol, "error": str(exc)})
            return None, {}

    async def list_alerts(self, chat_id: int) -> list[Alert]:
        async with self.db_factory() as session:
            q = await session.execute(
                select(Alert)
                .join(User, User.id == Alert.user_id)
                .where(User.telegram_chat_id == chat_id)
                .where(Alert.status.in_([AlertStatus.ACTIVE, AlertStatus.PAUSED]))
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

    async def delete_alerts_by_symbol(self, chat_id: int, symbol: str) -> int:
        target = symbol.upper()
        async with self.db_factory() as session:
            user_q = await session.execute(select(User).where(User.telegram_chat_id == chat_id))
            user = user_q.scalar_one_or_none()
            if not user:
                return 0
            q = await session.execute(
                select(Alert).where(
                    Alert.user_id == user.id,
                    Alert.symbol == target,
                )
            )
            alerts = list(q.scalars().all())
            for row in alerts:
                await session.delete(row)
            await session.commit()
            return len(alerts)

    async def process_alerts(self, notifier) -> int:
        triggered_count = 0
        async with self.db_factory() as session:
            q = await session.execute(select(Alert).where(Alert.status == AlertStatus.ACTIVE))
            alerts = list(q.scalars().all())
            if not alerts:
                return 0

            # Deduplicate price keys and fetch all in parallel
            unique_keys: dict[str, Alert] = {}
            for alert in alerts:
                key = f"{alert.symbol}:{alert.source_exchange}:{alert.instrument_id}:{alert.market_kind}"
                if key not in unique_keys:
                    unique_keys[key] = alert

            fetch_results = await asyncio.gather(
                *[self._get_alert_price(a) for a in unique_keys.values()],
                return_exceptions=True,
            )

            symbol_prices: dict[str, float] = {}
            symbol_source: dict[str, dict] = {}
            for key, result in zip(unique_keys.keys(), fetch_results, strict=False):
                if isinstance(result, Exception) or result is None:
                    logger.warning("price_fetch_failed", extra={"key": key, "error": str(result)})
                    continue
                price_value, source_info = result
                if price_value is not None:
                    symbol_prices[key] = float(price_value)
                    symbol_source[key] = source_info

            now = datetime.now(UTC)
            for alert in alerts:
                key = f"{alert.symbol}:{alert.source_exchange}:{alert.instrument_id}:{alert.market_kind}"
                if key not in symbol_prices:
                    continue

                current_price = symbol_prices[key]
                prev_key = (
                    f"alert:lastprice:{alert.symbol}:"
                    f"{alert.source_exchange or 'auto'}:{alert.instrument_id or 'na'}:{alert.market_kind or 'spot'}"
                )
                prev_payload = await self.cache.get_json(prev_key)
                prev_price = float(prev_payload.get("price", current_price)) if prev_payload else current_price

                await self.cache.set_json(prev_key, {"price": current_price}, ttl=7200)

                now_naive = now.replace(tzinfo=None)
                if alert.cooldown_until and alert.cooldown_until > now_naive:
                    continue

                if not condition_met(
                    condition=alert.condition,
                    target=float(alert.target_price),
                    prev_price=float(prev_price),
                    current_price=float(current_price),
                ):
                    continue

                # Evaluate extra AND-conditions (RSI, EMA, etc.)
                extras = parse_extra_conditions(alert.conditions_json)
                if extras:
                    all_met = True
                    for extra in extras:
                        tf = str(extra.get("timeframe", "1h"))
                        snap_q = await session.execute(
                            select(IndicatorSnapshot)
                            .where(
                                IndicatorSnapshot.symbol == alert.symbol,
                                IndicatorSnapshot.timeframe == tf,
                            )
                            .order_by(IndicatorSnapshot.computed_at.desc())
                            .limit(1)
                        )
                        snap = snap_q.scalar_one_or_none()
                        if not extra_condition_met(cond=extra, snapshot=snap, current_price=float(current_price)):
                            all_met = False
                            break
                    if not all_met:
                        continue

                dedupe_key = f"alert:dedupe:{alert.id}:{now.strftime('%Y%m%d%H%M')}"
                if not await self.cache.set_if_absent(dedupe_key, ttl=120):
                    continue

                user_q = await session.execute(select(User).where(User.id == alert.user_id))
                user = user_q.scalar_one_or_none()

                # Attempt delivery before marking triggered.
                # If delivery fails, keep alert active — the dedupe key (TTL=120s)
                # prevents immediate re-trigger; it will retry on the next cycle.
                notify_ok = False
                if user:
                    direction_word = "above" if alert.condition == "above" else "below"
                    msg = (
                        f"🔔 <b>{alert.symbol}</b> just crossed <b>${alert.target_price:,.2f}</b> {direction_word}, fren.\n"
                        f"trading at <b>${current_price:,.2f}</b> right now. you set this one — go check the chart."
                    )
                    try:
                        await notifier(user.telegram_chat_id, msg, symbol=alert.symbol)
                        notify_ok = True
                    except Exception as exc:
                        logger.warning(
                            "alert_notify_failed",
                            extra={"alert_id": alert.id, "chat_id": user.telegram_chat_id, "error": str(exc), "will_retry": True},
                        )

                if not notify_ok:
                    # Don't mark triggered — retry after dedupe key expires (~2 min)
                    continue

                alert.status = AlertStatus.TRIGGERED
                alert.triggered_at = now_naive
                alert.cooldown_until = now_naive + timedelta(minutes=self.cooldown_minutes)
                alert.last_triggered_price = current_price
                src = symbol_source.get(key, {})
                if src.get("exchange"):
                    alert.source_exchange = str(src.get("exchange"))
                if src.get("instrument_id"):
                    alert.instrument_id = str(src.get("instrument_id"))
                if src.get("market_kind"):
                    alert.market_kind = str(src.get("market_kind"))
                triggered_count += 1

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
        return await self._set_user_alert_status(chat_id, AlertStatus.PAUSED)

    async def resume_user_alerts(self, chat_id: int) -> int:
        return await self._set_user_alert_status(chat_id, AlertStatus.ACTIVE)
