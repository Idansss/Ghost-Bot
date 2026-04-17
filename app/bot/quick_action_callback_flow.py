from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from aiogram.types import BufferedInputFile


@dataclass(frozen=True)
class QuickActionCallbackDependencies:
    hub: Any
    run_with_typing_lock: Callable[[Any, int, Callable[[], Awaitable[None]]], Awaitable[None]]
    analysis_timeframes_from_settings: Callable[[dict], list[str]]
    parse_int_list: Callable[[Any, list[int]], list[int]]
    remember_analysis_context: Callable[[int, str, Any, dict], Awaitable[None]]
    remember_source_context: Callable[..., Awaitable[None]]
    render_analysis_text: Callable[..., Awaitable[str]]
    send_ghost_analysis: Callable[..., Awaitable[None]]
    set_pending_alert: Callable[[int, str], Awaitable[None]]
    as_int: Callable[[Any, int], int]
    rsi_scan_template: Callable[[dict], str]
    news_template: Callable[[dict], str]


async def handle_quick_analysis_callback(*, callback, deps: QuickActionCallbackDependencies) -> None:
    with suppress(Exception):
        await callback.answer("Analyzing...")
    chat_id = callback.message.chat.id

    async def _run() -> None:
        symbol = (callback.data or "").split(":")[-1]
        settings = await deps.hub.user_service.get_settings(chat_id)
        payload = await deps.hub.analysis_service.analyze(
            symbol,
            timeframes=deps.analysis_timeframes_from_settings(settings),
            ema_periods=deps.parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=deps.parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=False,
            include_news=False,
        )
        await deps.hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await deps.remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await deps.render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
        )
        await deps.send_ghost_analysis(callback.message, symbol, analysis_text, direction=payload.get("side"))
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_quick_analysis_tf_callback(*, callback, deps: QuickActionCallbackDependencies) -> None:
    with suppress(Exception):
        await callback.answer("Analyzing...")
    chat_id = callback.message.chat.id

    async def _run() -> None:
        _, _, symbol, timeframe = (callback.data or "").split(":", 3)
        symbol = symbol.upper()
        settings = await deps.hub.user_service.get_settings(chat_id)
        payload = await deps.hub.analysis_service.analyze(
            symbol,
            timeframe=timeframe,
            timeframes=[timeframe],
            ema_periods=deps.parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=deps.parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=False,
            include_news=False,
        )
        await deps.hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await deps.remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
        analysis_text = await deps.render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
        )
        await deps.send_ghost_analysis(callback.message, symbol, analysis_text, direction=payload.get("side"))
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_quick_chart_callback(*, callback, deps: QuickActionCallbackDependencies) -> None:
    chat_id = callback.message.chat.id

    async def _run() -> None:
        _, _, symbol, timeframe = (callback.data or "").split(":", 3)
        img, _ = await deps.hub.chart_service.render_chart(symbol=symbol.upper(), timeframe=timeframe)
        await callback.message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol.upper()}-{timeframe}.png"),
            caption=f"{symbol.upper()} {timeframe} chart.",
        )
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_quick_heatmap_callback(*, callback, deps: QuickActionCallbackDependencies) -> None:
    chat_id = callback.message.chat.id

    async def _run() -> None:
        _, _, symbol = (callback.data or "").split(":", 2)
        img, _ = await deps.hub.heatmap_service.render(symbol=symbol.upper())
        await callback.message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol.upper()}-heatmap.png"),
            caption=f"{symbol.upper()} order-book heatmap.",
        )
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_quick_rsi_callback(*, callback, deps: QuickActionCallbackDependencies) -> None:
    chat_id = callback.message.chat.id

    async def _run() -> None:
        _, _, mode, timeframe, limit_raw = (callback.data or "").split(":", 4)
        limit = max(1, min(deps.as_int(limit_raw, 5), 20))
        payload = await deps.hub.rsi_scanner_service.scan(
            timeframe=timeframe,
            mode="overbought" if mode == "overbought" else "oversold",
            limit=limit,
            rsi_length=14,
            symbol=None,
        )
        await callback.message.answer(deps.rsi_scan_template(payload))
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_quick_news_callback(*, callback, deps: QuickActionCallbackDependencies) -> None:
    chat_id = callback.message.chat.id

    async def _run() -> None:
        _, _, mode = (callback.data or "").split(":", 2)
        mode_norm = "openai" if mode == "openai" else "crypto"
        topic = "openai" if mode_norm == "openai" else "crypto"
        payload = await deps.hub.news_service.get_digest(topic=topic, mode=mode_norm, limit=6)
        await callback.message.answer(deps.news_template(payload), parse_mode="HTML")
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_define_easter_egg_callback(*, callback, deps: QuickActionCallbackDependencies) -> None:
    chat_id = callback.message.chat.id

    async def _run() -> None:
        settings = await deps.hub.user_service.get_settings(chat_id)
        parts = (callback.data or "").split(":")
        action = parts[1] if len(parts) > 1 else ""

        if action == "analyze":
            timeframe = parts[2] if len(parts) > 2 else "1h"
            symbol = "DEFINE"
            payload = await deps.hub.analysis_service.analyze(
                symbol,
                timeframe=timeframe,
                timeframes=[timeframe],
                ema_periods=deps.parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
                rsi_periods=deps.parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
                include_derivatives=True,
                include_news=True,
            )
            await deps.hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
            await deps.remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
            analysis_text = await deps.render_analysis_text(
                payload=payload,
                symbol=symbol,
                direction=payload.get("side"),
                settings=settings,
                chat_id=chat_id,
            )
            await deps.send_ghost_analysis(callback.message, symbol, analysis_text)
            await callback.answer()
            return

        if action == "chart":
            timeframe = parts[2] if len(parts) > 2 else "1h"
            try:
                img, _ = await deps.hub.chart_service.render_chart(symbol="DEFINE", timeframe=timeframe)
            except Exception:
                await callback.message.answer("Drop a real ticker for chart, e.g. `chart SOL 1h`.")
                await callback.answer()
                return
            await callback.message.answer_photo(
                BufferedInputFile(img, filename=f"DEFINE-{timeframe}.png"),
                caption=f"DEFINE {timeframe} chart.",
            )
            await callback.answer()
            return

        if action == "heatmap":
            symbol = "DEFINE"
            try:
                img, meta = await deps.hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
            except Exception:
                symbol = "BTC"
                img, meta = await deps.hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
            await callback.message.answer_photo(
                BufferedInputFile(img, filename=f"{symbol}_heatmap.png"),
                caption=(
                    f"{meta['pair']} orderbook heatmap\n"
                    f"Best bid: {meta['best_bid']:.6f} | Best ask: {meta['best_ask']:.6f}\n"
                    f"Bid wall: {meta['bid_wall']:.6f} | Ask wall: {meta['ask_wall']:.6f}"
                ),
            )
            await callback.answer()
            return

        if action == "alert":
            await deps.set_pending_alert(chat_id, "DEFINE")
            await callback.message.answer("Send alert level for DEFINE, e.g. DEFINE 0.50")
            await callback.answer()
            return

        if action == "news":
            payload = await deps.hub.news_service.get_digest(topic="DEFINE", mode="crypto", limit=6)
            await callback.message.answer(deps.news_template(payload), parse_mode="HTML")
            await callback.answer()
            return

        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_top_rsi_callback(*, callback, deps: QuickActionCallbackDependencies) -> None:
    chat_id = callback.message.chat.id

    async def _run() -> None:
        parts = (callback.data or "").split(":")
        mode = parts[1] if len(parts) > 1 else "oversold"
        timeframe = parts[2] if len(parts) > 2 else "1h"
        mode = "overbought" if mode == "overbought" else "oversold"
        payload = await deps.hub.rsi_scanner_service.scan(
            timeframe=timeframe,
            mode=mode,
            limit=10,
            rsi_length=14,
            symbol=None,
        )
        await callback.message.answer(deps.rsi_scan_template(payload))
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)
