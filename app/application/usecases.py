from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.metrics import FeatureTimer, record_feature

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsResult:
    payload: dict[str, Any]
    degraded: bool


class NewsUseCase:
    def __init__(self, *, cache, news_service) -> None:
        self._cache = cache
        self._news = news_service

    async def get_digest(self, *, chat_id: int, topic: str | None, mode: str, limit: int) -> NewsResult | None:
        """Fetch a digest with caching + graceful degradation.

        Returns None if a concurrent digest is already running (caller should short-circuit).
        """
        async with self._cache.distributed_lock(f"user:{chat_id}:news", ttl=25) as acquired:
            if not acquired:
                return None

        mode = (mode or "crypto").strip() or "crypto"
        limit = max(1, min(int(limit or 6), 20))
        cache_key = f"last_news:{chat_id}:{mode}:{str(topic or '')}:{limit}"
        with FeatureTimer("news"):
            try:
                payload = await self._news.get_digest(topic=topic, mode=mode, limit=limit)
                record_feature("news", ok=True)
                if isinstance(payload, dict):
                    await self._cache.set_json(cache_key, payload, ttl=900)
                return NewsResult(payload=payload if isinstance(payload, dict) else {"headlines": []}, degraded=False)
            except Exception as exc:
                logger.warning(
                    "news_digest_failed",
                    extra={"event": "news_digest_failed", "error": str(exc), "chat_id": chat_id},
                )
                cached = await self._cache.get_json(cache_key)
                if isinstance(cached, dict) and cached:
                    record_feature("news", ok=False)
                    return NewsResult(payload=cached, degraded=True)
                record_feature("news", ok=False)
                return NewsResult(payload={}, degraded=False)


@dataclass(frozen=True)
class AnalysisResult:
    kind: str  # ok|cached|unsupported|busy|error
    payload: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    error: str | None = None


class AnalysisUseCase:
    def __init__(self, *, cache, analysis_service) -> None:
        self._cache = cache
        self._analysis = analysis_service

    async def analyze(
        self,
        *,
        chat_id: int,
        symbol: str,
        direction: str | None,
        entities: dict[str, Any],
        settings_timeframes: list[str],
        settings_emas: list[int],
        settings_rsis: list[int],
    ) -> AnalysisResult:
        async with self._cache.distributed_lock(f"user:{chat_id}:analysis", ttl=45) as acquired:
            if not acquired:
                return AnalysisResult(kind="busy")

        with FeatureTimer("analysis"):
            try:
                payload = await self._analysis.analyze(
                    symbol,
                    direction=direction,
                    timeframe=entities.get("timeframe"),
                    timeframes=entities.get("timeframes") or settings_timeframes,
                    ema_periods=entities.get("ema_periods") or settings_emas,
                    rsi_periods=entities.get("rsi_periods") or settings_rsis,
                    all_timeframes=bool(entities.get("all_timeframes")),
                    all_emas=bool(entities.get("all_emas")),
                    all_rsis=bool(entities.get("all_rsis")),
                    include_derivatives=bool(entities.get("include_derivatives")),
                    include_news=bool(entities.get("include_news")),
                    notes=entities.get("notes", []),
                )
                if isinstance(payload, dict):
                    await self._cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
                record_feature("analysis", ok=True)
                return AnalysisResult(kind="ok", payload=payload if isinstance(payload, dict) else None)
            except Exception as exc:
                cached = await self._cache.get_json(f"last_analysis:{chat_id}:{symbol}")
                if isinstance(cached, dict) and cached:
                    logger.warning(
                        "analysis_degraded_served_cache",
                        extra={
                            "event": "analysis_degraded_served_cache",
                            "chat_id": chat_id,
                            "symbol": symbol,
                            "error": str(exc),
                        },
                    )
                    record_feature("analysis", ok=False)
                    return AnalysisResult(kind="cached", payload=cached, error=str(exc))

                err = str(exc).lower()
                if any(
                    marker in err
                    for marker in (
                        "price unavailable",
                        "no valid ohlcv",
                        "isn't supported",
                        "binance-only",
                        "unavailable",
                    )
                ):
                    with_reason = str(exc)
                    fallback = await self._analysis.fallback_asset_brief(symbol, reason=with_reason)
                    return AnalysisResult(
                        kind="unsupported",
                        fallback=fallback if isinstance(fallback, dict) else None,
                        error=with_reason,
                    )

                record_feature("analysis", ok=False)
                return AnalysisResult(kind="error", error=str(exc))


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RSIScanResult:
    payload: dict[str, Any] | None
    degraded: bool
    busy: bool = False


class RSIScanUseCase:
    def __init__(self, *, cache, rsi_scanner_service) -> None:
        self._cache = cache
        self._svc = rsi_scanner_service

    async def scan(self, *, chat_id: int, timeframe: str, mode: str, limit: int, rsi_length: int, symbol: str | None) -> RSIScanResult:
        async with self._cache.distributed_lock(f"user:{chat_id}:rsi", ttl=25) as acquired:
            if not acquired:
                return RSIScanResult(payload=None, degraded=False, busy=True)

        tf = str(timeframe or "1h")
        m = str(mode or "oversold")
        lim = max(1, min(int(limit or 10), 50))
        rsi_len = max(2, min(int(rsi_length or 14), 200))
        sym = (str(symbol).strip().upper() if symbol else None) or None
        cache_key = f"last_rsi:{chat_id}:{tf}:{m}:{rsi_len}:{str(sym or 'all').upper()}:{lim}"
        with FeatureTimer("rsi_scan"):
            try:
                payload = await self._svc.scan(timeframe=tf, mode=m, limit=lim, rsi_length=rsi_len, symbol=sym)
                record_feature("rsi_scan", ok=True)
                if isinstance(payload, dict):
                    await self._cache.set_json(cache_key, payload, ttl=900)
                return RSIScanResult(payload=payload if isinstance(payload, dict) else None, degraded=False)
            except Exception as exc:
                logger.warning("rsi_scan_intent_failed", extra={"event": "rsi_intent_error", "error": str(exc), "chat_id": chat_id})
                cached = await self._cache.get_json(cache_key)
                if isinstance(cached, dict) and cached:
                    record_feature("rsi_scan", ok=False)
                    return RSIScanResult(payload=cached, degraded=True)
                record_feature("rsi_scan", ok=False)
                return RSIScanResult(payload=None, degraded=False)


@dataclass(frozen=True)
class EMAScanResult:
    payload: dict[str, Any] | None
    degraded: bool
    busy: bool = False


class EMAScanUseCase:
    def __init__(self, *, cache, ema_scanner_service) -> None:
        self._cache = cache
        self._svc = ema_scanner_service

    async def scan(self, *, chat_id: int, timeframe: str, ema_length: int, mode: str, limit: int) -> EMAScanResult:
        async with self._cache.distributed_lock(f"user:{chat_id}:ema", ttl=25) as acquired:
            if not acquired:
                return EMAScanResult(payload=None, degraded=False, busy=True)

        tf = str(timeframe or "4h")
        ema_len = max(2, min(int(ema_length or 200), 5000))
        m = str(mode or "closest")
        lim = max(1, min(int(limit or 10), 50))
        cache_key = f"last_ema:{chat_id}:{tf}:{ema_len}:{m}:{lim}"
        with FeatureTimer("ema_scan"):
            try:
                payload = await self._svc.scan(timeframe=tf, ema_length=ema_len, mode=m, limit=lim)
                record_feature("ema_scan", ok=True)
                if isinstance(payload, dict):
                    await self._cache.set_json(cache_key, payload, ttl=900)
                return EMAScanResult(payload=payload if isinstance(payload, dict) else None, degraded=False)
            except Exception as exc:
                logger.warning("ema_scan_intent_failed", extra={"event": "ema_intent_error", "error": str(exc), "chat_id": chat_id})
                cached = await self._cache.get_json(cache_key)
                if isinstance(cached, dict) and cached:
                    record_feature("ema_scan", ok=False)
                    return EMAScanResult(payload=cached, degraded=True)
                record_feature("ema_scan", ok=False)
                return EMAScanResult(payload=None, degraded=False)


class AlertsUseCase:
    def __init__(self, *, alerts_service) -> None:
        self._svc = alerts_service

    async def create(
        self,
        *,
        chat_id: int,
        symbol: str,
        condition: str,
        target_price: float,
        source: str = "user",
        extra_conditions: list | None = None,
        idempotency_key: str | None = None,
    ):
        with FeatureTimer("alert_create"):
            try:
                alert = await self._svc.create_alert(
                    chat_id,
                    symbol,
                    condition,
                    float(target_price),
                    source=source,
                    extra_conditions=extra_conditions,
                    idempotency_key=idempotency_key,
                )
                record_feature("alert_create", ok=True)
                return alert
            except Exception:
                record_feature("alert_create", ok=False)
                raise

    async def list(self, *, chat_id: int):
        with FeatureTimer("alert_list"):
            alerts = await self._svc.list_alerts(chat_id)
            record_feature("alert_list", ok=True)
            return alerts

    async def delete(self, *, chat_id: int, alert_id: int) -> bool:
        with FeatureTimer("alert_delete"):
            ok = await self._svc.delete_alert(chat_id, int(alert_id))
            record_feature("alert_delete", ok=ok)
            return ok

    async def delete_by_symbol(self, *, chat_id: int, symbol: str) -> int:
        with FeatureTimer("alert_delete_by_symbol"):
            count = await self._svc.delete_alerts_by_symbol(chat_id, str(symbol))
            record_feature("alert_delete_by_symbol", ok=True)
            return int(count)

    async def pause(self, *, chat_id: int) -> int:
        with FeatureTimer("alert_pause"):
            count = await self._svc.pause_user_alerts(chat_id)
            record_feature("alert_pause", ok=True)
            return int(count)

    async def resume(self, *, chat_id: int) -> int:
        with FeatureTimer("alert_resume"):
            count = await self._svc.resume_user_alerts(chat_id)
            record_feature("alert_resume", ok=True)
            return int(count)

