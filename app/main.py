from __future__ import annotations

import asyncio
import contextlib
import logging
import platform
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand, Update
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import text

from app.adapters.derivatives import DerivativesAdapter
from app.adapters.llm import LLMClient
from app.adapters.market_router import MarketDataRouter
from app.adapters.news_sources import NewsSourcesAdapter
from app.adapters.ohlcv import OHLCVAdapter
from app.adapters.prices import PriceAdapter
from app.adapters.solana import SolanaAdapter
from app.adapters.tron import TronAdapter
from app.application.usecases import (
    AlertsUseCase,
    AnalysisUseCase,
    EMAScanUseCase,
    NewsUseCase,
    RSIScanUseCase,
)
from app.bot.handlers import init_handlers, router
from app.core.cache import RedisCache
from app.core.config import Settings, get_settings
from app.core.container import ServiceHub
from app.core.http import ResilientHTTPClient
from app.core.logging import setup_logging
from app.core.metrics import metrics_middleware_factory, metrics_response
from app.core.rate_limit import RateLimiter
from app.core.tracing import configure_tracing, instrument_app
from app.db.session import AsyncSessionLocal
from app.services.alerts import AlertsService
from app.services.audit import AuditService
from app.services.broadcast_service import BroadcastService
from app.services.charting import ChartService
from app.services.coin_info import CoinInfoService
from app.services.correlation import CorrelationService
from app.services.cycles import CyclesService
from app.services.discovery import DiscoveryService
from app.services.ema_scanner import EMAScannerService
from app.services.gdpr import GDPRService
from app.services.giveaway import GiveawayService
from app.services.market_analysis import MarketAnalysisService
from app.services.news import NewsService
from app.services.orderbook_heatmap import OrderbookHeatmapService
from app.services.portfolio import PortfolioService
from app.services.rsi_scanner import RSIScannerService
from app.services.scheduled_report import ScheduledReportService
from app.services.setup_review import SetupReviewService
from app.services.trade_journal import TradeJournalService
from app.services.trade_verify import TradeVerifyService
from app.services.users import UserService
from app.services.wallet_scan import WalletScanService
from app.services.watchlist import WatchlistService
from app.workers.scheduler import WorkerScheduler

logger = logging.getLogger(__name__)

_WEBHOOK_RATE_LIMIT = 300   # max requests per IP per minute
_WEBHOOK_RATE_WINDOW = 60   # seconds


async def _sync_bot_commands(bot: Bot) -> None:
    command_specs = [
        ("admins", "Show bot admins"),
        ("alert", "Create a price alert"),
        ("alertclear", "Clear alerts (all or by symbol)"),
        ("alertdel", "Delete one alert by ID"),
        ("alerts", "List your active alerts"),
        ("alpha", "Full market analysis (multi-timeframe)"),
        ("chart", "Send candlestick chart image"),
        ("cycle", "Cycle check"),
        ("ema", "EMA scan (near EMA levels)"),
        ("findpair", "Find coin by price / partial name"),
        ("giveaway", "Admin: start/end/reroll giveaways"),
        ("heatmap", "Orderbook heatmap snapshot"),
        ("help", "Show examples + what I can do"),
        ("id", "Show your user/chat id"),
        ("join", "Join active giveaway"),
        ("goals", "Set your trading goals (for memory)"),
        ("margin", "Position size + margin calculator"),
        ("name", "Set display name (for memory)"),
        ("news", "Crypto + macro + OpenAI news digest"),
        ("compare", "Compare prices for multiple symbols"),
        ("deleteaccount", "Delete all your data from the bot (GDPR)"),
        ("export", "Export alerts or journal"),
        ("mydata", "Export all your stored data (GDPR)"),
        ("journal", "Trade journal: log, list, stats"),
        ("pnl", "PnL calculator (entry/exit/size/lev)"),
        ("position", "Track positions and unrealized PnL"),
        ("price", "Latest price + 24h stats"),
        ("report", "Scheduled daily market summary"),
        ("rsi", "RSI scan (overbought/oversold)"),
        ("scan", "Wallet scan (solana/tron address)"),
        ("settings", "Preferences (default TF, risk, etc.)"),
        ("setup", "RR + PnL + margin from entry/SL/TP"),
        ("start", "Start the bot / show quick intro"),
        ("tradecheck", "Verify trade outcome from timestamp"),
        ("watch", "Quick levels + bias for a coin"),
        ("watchlist", "Coins to watch list"),
    ]
    commands = [
        BotCommand(command=command, description=description)
        for command, description in sorted(command_specs, key=lambda x: x[0])
    ]
    await bot.set_my_commands(commands)


async def _ensure_alert_schema_compat() -> None:
    """Hot-fix missing alert source columns on older DBs."""
    try:
        async with AsyncSessionLocal() as session:
            bind = session.get_bind()
            dialect = (bind.dialect.name or "").lower() if bind is not None else ""
            if dialect != "postgresql":
                return

            result = await session.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name='alerts' "
                    "AND column_name IN ('source_exchange','instrument_id','market_kind')"
                )
            )
            existing = {str(row[0]) for row in result.all()}
            stmts: list[str] = []
            if "source_exchange" not in existing:
                stmts.append("ALTER TABLE alerts ADD COLUMN source_exchange VARCHAR(20)")
            if "instrument_id" not in existing:
                stmts.append("ALTER TABLE alerts ADD COLUMN instrument_id VARCHAR(40)")
            if "market_kind" not in existing:
                stmts.append("ALTER TABLE alerts ADD COLUMN market_kind VARCHAR(10)")

            for stmt in stmts:
                await session.execute(text(stmt))
            if stmts:
                await session.execute(text("CREATE INDEX IF NOT EXISTS ix_alerts_source_exchange ON alerts (source_exchange)"))
                await session.commit()
                logger.info(
                    "alert_schema_hotfix_applied",
                    extra={"event": "alert_schema_hotfix_applied", "changes": len(stmts)},
                )
    except Exception as exc:
        logger.warning(
            "alert_schema_hotfix_failed",
            extra={"event": "alert_schema_hotfix_failed", "error": str(exc)},
        )


def build_hub(settings: Settings, bot: Bot, cache: RedisCache, http: ResilientHTTPClient) -> ServiceHub:
    rate_limiter = RateLimiter(cache)
    llm_client = None

    # Provider priority: Claude (paid) → Grok (paid) → OpenAI (last resort)
    if settings.anthropic_api_key:
        llm_client = LLMClient(
            api_key=settings.anthropic_api_key,
            model="anthropic/claude-3-5-haiku-20241022",
            router_model="anthropic/claude-3-5-haiku-20241022",
            max_output_tokens=settings.openai_max_output_tokens,
            temperature=settings.openai_temperature,
            fallback_model="xai/grok-3-fast-beta" if settings.xai_api_key else None,
            fallback_api_key=settings.xai_api_key or None,
            fallback_base_url="https://api.x.ai/v1" if settings.xai_api_key else None,
        )
    elif settings.xai_api_key:
        llm_client = LLMClient(
            api_key=settings.xai_api_key,
            model="xai/grok-3-fast-beta",
            router_model="xai/grok-3-fast-beta",
            max_output_tokens=settings.openai_max_output_tokens,
            temperature=settings.openai_temperature,
            fallback_base_url="https://api.x.ai/v1",
        )
    elif settings.openai_api_key:
        llm_client = LLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            router_model=settings.openai_router_model or None,
            max_output_tokens=settings.openai_max_output_tokens,
            temperature=settings.openai_temperature,
        )

    market_router = MarketDataRouter(
        http=http,
        cache=cache,
        binance_base_url=settings.binance_base_url,
        binance_futures_base_url=settings.binance_futures_base_url,
        bybit_base_url=settings.bybit_base_url,
        okx_base_url=settings.okx_base_url,
        mexc_base_url=settings.mexc_base_url,
        blofin_base_url=settings.blofin_base_url,
        enable_binance=settings.enable_binance,
        enable_bybit=settings.enable_bybit,
        enable_okx=settings.enable_okx,
        enable_mexc=settings.enable_mexc,
        enable_blofin=settings.enable_blofin,
        exchange_priority=settings.exchange_priority,
        market_prefer_spot=settings.market_prefer_spot,
        best_source_ttl_hours=settings.best_source_ttl_hours,
        instruments_ttl_min=settings.instruments_ttl_min,
    )

    price_adapter = PriceAdapter(
        http=http,
        cache=cache,
        binance_base=settings.binance_base_url,
        coingecko_base=settings.coingecko_base_url,
        test_mode=settings.test_mode,
        mock_prices=settings.mock_prices,
        market_router=market_router,
    )
    ohlcv_adapter = OHLCVAdapter(
        http=http,
        cache=cache,
        binance_base=settings.binance_base_url,
        coingecko_base=settings.coingecko_base_url,
        market_router=market_router,
    )
    deriv_adapter = DerivativesAdapter(
        http=http,
        cache=cache,
        futures_base=settings.binance_futures_base_url,
        market_router=market_router,
    )
    news_adapter = NewsSourcesAdapter(
        http=http,
        cache=cache,
        rss_feeds=settings.rss_feed_list(),
        cryptopanic_key=settings.cryptopanic_api_key,
        openai_rss_feeds=settings.openai_rss_feed_list(),
    )
    solana_adapter = SolanaAdapter(http=http, rpc_url=settings.solana_rpc_url)
    tron_adapter = TronAdapter(http=http, api_url=settings.tron_api_url, api_key=settings.trongrid_api_key)

    news_service = NewsService(news_adapter, llm_client=llm_client)
    rsi_scanner_service = RSIScannerService(
        http=http,
        cache=cache,
        ohlcv_adapter=ohlcv_adapter,
        market_router=market_router,
        coingecko_base=settings.coingecko_base_url,
        binance_base=settings.binance_base_url,
        db_factory=AsyncSessionLocal,
        universe_size=settings.rsi_scan_universe_size,
        scan_timeframes=settings.rsi_scan_timeframes_list(),
        concurrency=settings.rsi_scan_concurrency,
        freshness_minutes=settings.rsi_scan_freshness_minutes,
        live_fallback_universe=settings.rsi_scan_live_fallback_universe,
    )
    discovery_service = DiscoveryService(
        http=http,
        cache=cache,
        market_router=market_router,
        price_adapter=price_adapter,
        binance_base=settings.binance_base_url,
        coingecko_base=settings.coingecko_base_url,
    )
    giveaway_service = GiveawayService(
        db_factory=AsyncSessionLocal,
        admin_chat_ids=settings.admin_ids_list(),
        min_participants=settings.giveaway_min_participants,
    )
    ema_scanner_service = EMAScannerService(
        http=http,
        cache=cache,
        ohlcv_adapter=ohlcv_adapter,
        market_router=market_router,
        binance_base=settings.binance_base_url,
        db_factory=AsyncSessionLocal,
        freshness_minutes=settings.rsi_scan_freshness_minutes,
        live_fallback_universe=settings.rsi_scan_live_fallback_universe,
        concurrency=settings.rsi_scan_concurrency,
    )
    chart_service = ChartService(ohlcv_adapter=ohlcv_adapter)
    orderbook_heatmap_service = OrderbookHeatmapService(market_router=market_router)
    analysis_service = MarketAnalysisService(
        price_adapter,
        ohlcv_adapter,
        deriv_adapter,
        news_service,
        fast_mode=settings.analysis_fast_mode,
        default_timeframes=settings.analysis_default_timeframes_list(),
        include_derivatives_default=settings.analysis_include_derivatives_default,
        include_news_default=settings.analysis_include_news_default,
        request_timeout_sec=settings.analysis_request_timeout_sec,
    )
    broadcast_service = BroadcastService(
        analysis_service=analysis_service,
        llm_client=llm_client,
        cache=cache,
        rate_limit_minutes=settings.broadcast_rate_limit_minutes,
    )

    alerts_service = AlertsService(
        db_factory=AsyncSessionLocal,
        cache=cache,
        price_adapter=price_adapter,
        market_router=market_router,
        alerts_limit_per_day=settings.alerts_create_limit_per_day,
        cooldown_minutes=settings.alert_cooldown_min,
        max_deviation_pct=settings.alert_max_deviation_pct,
    )

    return ServiceHub(
        bot=bot,
        bot_username=None,
        llm_client=llm_client,
        market_router=market_router,
        cache=cache,
        rate_limiter=rate_limiter,
        analysis_uc=AnalysisUseCase(cache=cache, analysis_service=analysis_service),
        news_uc=NewsUseCase(cache=cache, news_service=news_service),
        rsi_scan_uc=RSIScanUseCase(cache=cache, rsi_scanner_service=rsi_scanner_service),
        ema_scan_uc=EMAScanUseCase(cache=cache, ema_scanner_service=ema_scanner_service),
        alerts_uc=AlertsUseCase(alerts_service=alerts_service),
        user_service=UserService(AsyncSessionLocal),
        audit_service=AuditService(AsyncSessionLocal),
        # analysis service defaults tuned for low latency; deep data can still be requested on demand
        analysis_service=analysis_service,
        alerts_service=alerts_service,
        wallet_service=WalletScanService(db_factory=AsyncSessionLocal, solana=solana_adapter, tron=tron_adapter, price=price_adapter),
        trade_verify_service=TradeVerifyService(ohlcv_adapter),
        setup_review_service=SetupReviewService(ohlcv_adapter),
        watchlist_service=WatchlistService(
            http=http,
            news_adapter=news_adapter,
            market_router=market_router,
            price_adapter=price_adapter,
            coingecko_base=settings.coingecko_base_url,
            include_btc_eth=settings.include_btc_eth_watchlist,
        ),
        news_service=news_service,
        cycles_service=CyclesService(ohlcv_adapter),
        correlation_service=CorrelationService(ohlcv_adapter, price_adapter),
        rsi_scanner_service=rsi_scanner_service,
        ema_scanner_service=ema_scanner_service,
        chart_service=chart_service,
        orderbook_heatmap_service=orderbook_heatmap_service,
        discovery_service=discovery_service,
        coin_info_service=CoinInfoService(
            http=http,
            cache=cache,
            coingecko_base=settings.coingecko_base_url,
        ),
        giveaway_service=giveaway_service,
        broadcast_service=broadcast_service,
        portfolio_service=PortfolioService(
            AsyncSessionLocal,
            price_adapter,
            max_notional_warning_usd=settings.max_position_notional_warning_usd,
        ),
        trade_journal_service=TradeJournalService(AsyncSessionLocal),
        scheduled_report_service=ScheduledReportService(AsyncSessionLocal),
        gdpr_service=GDPRService(AsyncSessionLocal),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)

    configure_tracing(
        enabled=bool(settings.otel_enabled),
        service_name=str(settings.otel_service_name or "ghost-bot"),
        otlp_endpoint=str(settings.otel_exporter_otlp_endpoint or ""),
        otlp_headers=str(settings.otel_exporter_otlp_headers or ""),
    )

    await _ensure_alert_schema_compat()

    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    cache = RedisCache(settings.redis_url)
    http = ResilientHTTPClient(
        timeout=float(settings.http_timeout_sec),
        retries=int(settings.http_retries),
        backoff_base=float(settings.http_backoff_base_sec),
        breaker_threshold=int(settings.http_breaker_threshold),
        breaker_cooldown=int(settings.http_breaker_cooldown_sec),
    )

    bot = Bot(
        token=settings.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    hub = build_hub(settings, bot, cache, http)
    try:
        me = await bot.get_me()
        hub.bot_username = me.username.lower() if me.username else None
    except Exception:
        hub.bot_username = None

    try:
        await _sync_bot_commands(bot)
    except Exception as exc:
        logger.warning("set_bot_commands_failed", extra={"event": "set_bot_commands_failed", "error": str(exc)})
    init_handlers(hub)
    dp.include_router(router)

    scheduler = None
    if not settings.serverless_mode:
        scheduler = WorkerScheduler(hub)
        scheduler.start()

    if not settings.cron_secret:
        logger.warning(
            "cron_secret_not_set",
            extra={
                "event": "cron_secret_not_set",
                "detail": "CRON_SECRET is not set — all /tasks/* endpoints will return 401. Set CRON_SECRET in your environment to enable scheduled task endpoints.",
            },
        )

    polling_task = None
    if settings.serverless_mode and not settings.telegram_use_webhook:
        logger.warning("serverless_mode_enabled_without_webhook", extra={"event": "serverless_warning"})

    if settings.telegram_use_webhook and settings.telegram_auto_set_webhook:
        webhook_url = settings.telegram_webhook_url.rstrip("/") + settings.telegram_webhook_path
        try:
            allowed = list(dp.resolve_used_update_types())
            if "message_reaction" not in allowed:
                allowed.append("message_reaction")
            await bot.set_webhook(
                webhook_url,
                secret_token=settings.telegram_webhook_secret or None,
                allowed_updates=allowed,
            )
            logger.info("webhook_configured", extra={"event": "webhook", "url": webhook_url})
        except Exception as exc:
            # Do not crash the whole API on webhook registration failures.
            logger.exception(
                "webhook_configure_failed",
                extra={"event": "webhook_error", "url": webhook_url, "error": str(exc)},
            )
    elif not settings.telegram_use_webhook and not settings.serverless_mode:
        polling_task = asyncio.create_task(dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types()))

    app.state.settings = settings
    app.state.hub = hub
    app.state.dp = dp
    app.state.bot = bot
    app.state.http = http
    app.state.cache = cache
    app.state.scheduler = scheduler
    app.state.polling_task = polling_task

    try:
        yield
    finally:
        if scheduler:
            scheduler.stop()
        if polling_task:
            polling_task.cancel()
            with contextlib.suppress(Exception):
                await polling_task
        await bot.session.close()
        await http.close()
        await cache.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Ghost Alpha Bot", version="1.0.0", lifespan=lifespan)

    if bool(settings.otel_enabled):
        instrument_app(app)

    app.middleware("http")(metrics_middleware_factory())

    def _cron_authorized(req: Request) -> bool:
        # Native Vercel cron invocations include this header.
        if req.headers.get("x-vercel-cron"):
            return True
        if not settings.cron_secret:
            # No secret configured — deny all external cron calls to prevent
            # unauthenticated triggering of alert/giveaway/scanner tasks.
            logger.warning(
                "cron_secret_not_set",
                extra={
                    "event": "cron_secret_not_set",
                    "detail": "CRON_SECRET is unset; all /tasks/* requests are denied. Set CRON_SECRET to enable cron endpoints.",
                },
            )
            return False
        auth = req.headers.get("authorization", "")
        if auth == f"Bearer {settings.cron_secret}":
            return True
        if req.headers.get("x-cron-secret", "") == settings.cron_secret:
            return True
        return False

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics(req: Request) -> Response:
        # Restrict metrics endpoint to cron/admin callers (same as admin stats)
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")
        body, code, headers = metrics_response()
        return Response(content=body, status_code=code, headers=headers)

    @app.get("/health/deep")
    async def health_deep() -> dict:
        """Check DB, Redis, and optional exchange ping."""
        out = {"status": "ok", "checks": {}}
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            out["checks"]["db"] = "ok"
        except Exception as e:
            out["checks"]["db"] = str(e)
            out["status"] = "degraded"
        try:
            await app.state.cache.redis.ping()
            out["checks"]["redis"] = "ok"
        except Exception as e:
            out["checks"]["redis"] = str(e)
            out["status"] = "degraded"
        if getattr(app.state, "settings", None) and getattr(app.state.settings, "binance_base_url", None):
            try:
                await app.state.http.get_json(f"{app.state.settings.binance_base_url}/api/v3/ping")
                out["checks"]["exchange"] = "ok"
            except Exception as e:
                out["checks"]["exchange"] = str(e)
        return out

    @app.get("/admin/stats")
    async def admin_stats(req: Request) -> dict:
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")
        try:
            async with AsyncSessionLocal() as session:
                r1 = await session.execute(
                    text(
                        "SELECT COUNT(DISTINCT telegram_chat_id) FROM users WHERE last_seen_at > NOW() - INTERVAL '1 day'"
                    )
                )
                r2 = await session.execute(text("SELECT COUNT(*) FROM alerts WHERE status = 'active'"))
            return {
                "dau": r1.scalar() or 0,
                "active_alerts": r2.scalar() or 0,
                "ts": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/admin/config")
    async def admin_config(req: Request) -> dict:
        """Safe runtime configuration summary for debugging misconfigurations.

        Never includes secrets (API keys/tokens/passwords).
        """
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")

        s = app.state.settings
        return {
            "ts": datetime.now(UTC).isoformat(),
            "env": str(getattr(s, "env", "dev")),
            "app_name": str(getattr(s, "app_name", "ghost")),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "serverless_mode": bool(getattr(s, "serverless_mode", False)),
                "telegram_use_webhook": bool(getattr(s, "telegram_use_webhook", False)),
                "telegram_webhook_path": str(getattr(s, "telegram_webhook_path", "/telegram/webhook")),
                "otel_enabled": bool(getattr(s, "otel_enabled", False)),
            },
            "llm": {
                "enabled": bool(getattr(app.state, "hub", None) and getattr(app.state.hub, "llm_client", None)),
                "primary_model": str(getattr(s, "openai_model", "")),
                "router_model": str(getattr(s, "openai_router_model", "")),
            },
            "features": {
                "flags": sorted({str(x).lower() for x in getattr(s, "feature_flags_set", lambda: set())()}),
                "broadcast_enabled": bool(getattr(s, "broadcast_enabled", False)),
            },
            "limits": {
                "request_rate_limit_per_minute": int(getattr(s, "request_rate_limit_per_minute", 0)),
                "wallet_scan_limit_per_hour": int(getattr(s, "wallet_scan_limit_per_hour", 0)),
                "alerts_create_limit_per_day": int(getattr(s, "alerts_create_limit_per_day", 0)),
                "abuse_strikes_to_block": int(getattr(s, "abuse_strikes_to_block", 0)),
                "abuse_strike_window_sec": int(getattr(s, "abuse_strike_window_sec", 0)),
                "abuse_block_ttl_sec": int(getattr(s, "abuse_block_ttl_sec", 0)),
            },
            "http_policy": {
                "timeout_sec": float(getattr(s, "http_timeout_sec", 0.0)),
                "retries": int(getattr(s, "http_retries", 0)),
                "backoff_base_sec": float(getattr(s, "http_backoff_base_sec", 0.0)),
                "breaker_threshold": int(getattr(s, "http_breaker_threshold", 0)),
                "breaker_cooldown_sec": int(getattr(s, "http_breaker_cooldown_sec", 0)),
            },
            "adapters": {
                "binance_enabled": bool(getattr(s, "enable_binance", False)),
                "bybit_enabled": bool(getattr(s, "enable_bybit", False)),
                "okx_enabled": bool(getattr(s, "enable_okx", False)),
                "mexc_enabled": bool(getattr(s, "enable_mexc", False)),
                "blofin_enabled": bool(getattr(s, "enable_blofin", False)),
                "exchange_priority": str(getattr(s, "exchange_priority", "")),
                "market_prefer_spot": bool(getattr(s, "market_prefer_spot", True)),
            },
        }

    @app.get("/ready")
    async def ready() -> dict:
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            pong = await app.state.cache.redis.ping()
            if not pong:
                raise RuntimeError("Redis ping failed")
            return {"status": "ready"}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(settings.telegram_webhook_path)
    async def telegram_webhook(req: Request) -> dict:
        app_settings = app.state.settings
        if not app_settings.telegram_use_webhook:
            raise HTTPException(status_code=400, detail="Webhook mode disabled")

        # Per-IP rate limiting — blocks floods while allowing normal Telegram traffic.
        client_ip = req.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
            req.client.host if req.client else "unknown"
        )
        rl = RateLimiter(app.state.cache)
        rl_result = await rl.check(f"webhook:ip:{client_ip}", _WEBHOOK_RATE_LIMIT, _WEBHOOK_RATE_WINDOW)
        if not rl_result.allowed:
            logger.warning(
                "webhook_rate_limited",
                extra={"event": "webhook_rate_limited", "ip": client_ip},
            )
            raise HTTPException(status_code=429, detail="Too many requests")

        if app_settings.telegram_webhook_secret:
            secret = req.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if secret != app_settings.telegram_webhook_secret:
                raise HTTPException(status_code=403, detail="Invalid secret")

        payload = await req.json()
        update = Update.model_validate(payload)
        try:
            await app.state.dp.feed_update(app.state.bot, update)
            return {"ok": True}
        except Exception as exc:
            logger.exception(
                "telegram_update_failed",
                extra={
                    "event": "telegram_update_failed",
                    "error": str(exc),
                    "update_id": payload.get("update_id"),
                },
            )
            # Always return 200-style payload so one bad update does not block the whole webhook queue.
            try:
                chat_id = None
                msg = payload.get("message") if isinstance(payload, dict) else None
                if isinstance(msg, dict):
                    chat = msg.get("chat") if isinstance(msg.get("chat"), dict) else {}
                    chat_id = chat.get("id")
                if chat_id is None and isinstance(payload, dict):
                    cq = payload.get("callback_query") if isinstance(payload.get("callback_query"), dict) else {}
                    cqm = cq.get("message") if isinstance(cq.get("message"), dict) else {}
                    cqchat = cqm.get("chat") if isinstance(cqm.get("chat"), dict) else {}
                    chat_id = cqchat.get("id")
                if chat_id is not None:
                    await app.state.bot.send_message(
                        chat_id=chat_id,
                        text="Request failed on my side. Try again in a few seconds.",
                    )
            except Exception:
                pass
            return {"ok": True, "error": "update_failed"}

    @app.post("/test/mock-price")
    async def mock_price(payload: dict) -> dict:
        settings = app.state.settings
        if not settings.test_mode:
            raise HTTPException(status_code=403, detail="TEST_MODE disabled")

        symbol = payload.get("symbol")
        price = payload.get("price")
        if not symbol or price is None:
            raise HTTPException(status_code=400, detail="symbol and price required")

        await app.state.hub.analysis_service.price_adapter.set_mock_price(symbol, float(price))
        return {"ok": True, "symbol": symbol.upper(), "price": float(price)}

    @app.api_route("/tasks/alerts/run", methods=["GET", "POST"])
    async def task_alerts(req: Request) -> dict:
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")

        async def _notify(chat_id: int, text: str) -> None:
            try:
                await app.state.bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:
                logger.warning("task_alert_notify_failed", extra={"event": "task_alert_notify_failed", "chat_id": chat_id, "error": str(exc)})

        try:
            count = await app.state.hub.alerts_service.process_alerts(_notify)
            return {"ok": True, "processed": count, "task": "alerts", "ts": datetime.now(UTC).isoformat()}
        except Exception as exc:
            logger.exception("task_alerts_failed", extra={"event": "task_alerts_failed", "error": str(exc)})
            return {"ok": False, "processed": 0, "task": "alerts", "error": str(exc), "ts": datetime.now(UTC).isoformat()}

    @app.api_route("/tasks/giveaways/run", methods=["GET", "POST"])
    async def task_giveaways(req: Request) -> dict:
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")

        async def _notify(chat_id: int, text: str) -> None:
            try:
                await app.state.bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:
                logger.warning("task_giveaway_notify_failed", extra={"event": "task_giveaway_notify_failed", "chat_id": chat_id, "error": str(exc)})

        try:
            count = await app.state.hub.giveaway_service.process_due_giveaways(_notify)
            return {"ok": True, "processed": count, "task": "giveaways", "ts": datetime.now(UTC).isoformat()}
        except Exception as exc:
            logger.exception("task_giveaways_failed", extra={"event": "task_giveaways_failed", "error": str(exc)})
            return {"ok": False, "processed": 0, "task": "giveaways", "error": str(exc), "ts": datetime.now(UTC).isoformat()}

    @app.api_route("/tasks/news/warm", methods=["GET", "POST"])
    async def task_news(req: Request) -> dict:
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")

        try:
            await app.state.hub.news_service.get_daily_brief(limit=10)
            return {"ok": True, "task": "news", "ts": datetime.now(UTC).isoformat()}
        except Exception as exc:
            logger.exception("task_news_failed", extra={"event": "task_news_failed", "error": str(exc)})
            return {"ok": False, "task": "news", "error": str(exc), "ts": datetime.now(UTC).isoformat()}

    @app.api_route("/tasks/rsi/refresh", methods=["GET", "POST"])
    async def task_rsi_refresh(req: Request) -> dict:
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")

        force = str(req.query_params.get("force", "")).lower() in {"1", "true", "yes"}
        try:
            universe = await app.state.hub.rsi_scanner_service.refresh_universe(app.state.settings.rsi_scan_universe_size)
            payload = await app.state.hub.rsi_scanner_service.refresh_indicators(force=force)
            return {"ok": True, "task": "rsi_refresh", "force": force, "universe": universe, **payload}
        except Exception as exc:
            logger.exception("task_rsi_refresh_failed", extra={"event": "task_rsi_refresh_failed", "error": str(exc)})
            return {
                "ok": False,
                "task": "rsi_refresh",
                "force": force,
                "updated": 0,
                "timeframes": [],
                "symbols": 0,
                "error": str(exc),
                "ts": datetime.now(UTC).isoformat(),
            }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
