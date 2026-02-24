from __future__ import annotations

import logging

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.config import get_settings
from app.core.container import ServiceHub

logger = logging.getLogger(__name__)


def _alert_triggered_menu(symbol: str) -> InlineKeyboardMarkup:
    """Quick follow-up actions shown right after an alert fires."""
    sym = symbol.upper()
    kb = InlineKeyboardBuilder()
    kb.button(text=f"Analyze {sym}", callback_data=f"quick:analysis_tf:{sym}:1h")
    kb.button(text=f"Chart {sym}", callback_data=f"quick:chart:{sym}:1h")
    kb.adjust(2)
    return kb.as_markup()


class WorkerScheduler:
    def __init__(self, hub: ServiceHub) -> None:
        self.hub = hub
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler(timezone="UTC")

    async def _notify(self, chat_id: int, text: str, **kwargs) -> None:
        symbol: str | None = kwargs.get("symbol")
        reply_markup = _alert_triggered_menu(symbol) if symbol else None
        await self.hub.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    async def _process_alerts(self) -> None:
        try:
            count = await self.hub.alerts_service.process_alerts(self._notify)
            logger.info("alerts_processed", extra={"event": "alerts_processed", "count": count})
        except Exception as exc:  # noqa: BLE001
            logger.exception("alerts_task_failed", extra={"event": "alerts_task_failed", "error": str(exc)})

    async def _refresh_news_cache(self) -> None:
        try:
            await self.hub.news_service.get_daily_brief(limit=10)
        except Exception as exc:  # noqa: BLE001
            logger.warning("news_cache_refresh_failed", extra={"event": "news_cache_refresh_failed", "error": str(exc)})

    async def _process_giveaways(self) -> None:
        try:
            count = await self.hub.giveaway_service.process_due_giveaways(self._notify)
            logger.info("giveaways_processed", extra={"event": "giveaways_processed", "count": count})
        except Exception as exc:  # noqa: BLE001
            logger.exception("giveaways_task_failed", extra={"event": "giveaways_task_failed", "error": str(exc)})

    async def _refresh_scan_universe(self) -> None:
        try:
            payload = await self.hub.rsi_scanner_service.refresh_universe(self.settings.rsi_scan_universe_size)
            logger.info("scan_universe_refreshed", extra={"event": "scan_universe_refreshed", **payload})
        except Exception as exc:  # noqa: BLE001
            logger.warning("scan_universe_failed", extra={"event": "scan_universe_failed", "error": str(exc)})

    async def _refresh_scan_indicators(self) -> None:
        try:
            payload = await self.hub.rsi_scanner_service.refresh_indicators(force=False)
            logger.info("scan_indicators_refreshed", extra={"event": "scan_indicators_refreshed", **payload})
        except Exception as exc:  # noqa: BLE001
            logger.warning("scan_indicators_failed", extra={"event": "scan_indicators_failed", "error": str(exc)})

    async def _process_market_broadcasts(self) -> None:
        try:
            channel_ids = self.settings.broadcast_channel_ids_list()
            if not self.settings.broadcast_enabled or not channel_ids or not self.hub.broadcast_service:
                return
            sent = await self.hub.broadcast_service.check_and_broadcast(self.hub.bot, channel_ids)
            logger.info(
                "market_broadcast_checked",
                extra={"event": "market_broadcast_checked", "sent": int(sent), "channels": len(channel_ids)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("broadcast_task_failed", extra={"event": "broadcast_task_failed", "error": str(exc)})

    async def _process_scheduled_reports(self) -> None:
        try:
            svc = getattr(self.hub, "scheduled_report_service", None)
            if not svc:
                return
            due = await svc.get_due_reports()
            if not due:
                return
            from app.services.market_context import format_market_context
            ctx = await self.hub.analysis_service.get_market_context()
            body = "📊 <b>Scheduled market summary</b>\n\n" + format_market_context(ctx)
            for chat_id, _ in due:
                try:
                    await self.hub.bot.send_message(chat_id=chat_id, text=body, parse_mode="HTML")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "scheduled_report_send_failed",
                        extra={"event": "scheduled_report_send_failed", "chat_id": chat_id, "error": str(exc)},
                    )
            logger.info(
                "scheduled_reports_sent",
                extra={"event": "scheduled_reports_sent", "count": len(due)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("scheduled_reports_task_failed", extra={"event": "scheduled_reports_task_failed", "error": str(exc)})

    def start(self) -> None:
        self.scheduler.add_job(self._process_alerts, "interval", seconds=self.settings.alert_check_interval_sec, max_instances=1)
        if getattr(self.hub, "scheduled_report_service", None):
            self.scheduler.add_job(self._process_scheduled_reports, "interval", minutes=1, max_instances=1)
        self.scheduler.add_job(self._refresh_news_cache, "interval", minutes=15, max_instances=1)
        self.scheduler.add_job(self._process_giveaways, "interval", seconds=20, max_instances=1)
        self.scheduler.add_job(self._refresh_scan_universe, "interval", minutes=30, max_instances=1)
        self.scheduler.add_job(self._refresh_scan_indicators, "interval", minutes=5, max_instances=1)
        if self.settings.broadcast_enabled and self.settings.broadcast_channel_ids_list():
            self.scheduler.add_job(
                self._process_market_broadcasts,
                "interval",
                minutes=max(5, int(self.settings.broadcast_interval_minutes)),
                max_instances=1,
            )
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
