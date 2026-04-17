from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class AnalysisDetailFlowDependencies:
    hub: Any
    set_pending_alert: Callable[[int, str], Awaitable[None]]
    run_with_typing_lock: Callable[[Any, int, Callable[[], Awaitable[None]]], Awaitable[None]]
    analysis_timeframes_from_settings: Callable[[dict], list[str]]
    parse_int_list: Callable[[Any, list[int]], list[int]]
    remember_analysis_context: Callable[[int, str, str | None, dict], Awaitable[None]]
    remember_source_context: Callable[..., Awaitable[None]]
    render_analysis_text: Callable[..., Awaitable[str]]
    send_ghost_analysis: Callable[..., Awaitable[None]]


async def handle_set_alert_callback(*, callback, deps: AnalysisDetailFlowDependencies) -> None:
    symbol = (callback.data or "").split(":", 1)[1]
    await deps.set_pending_alert(callback.message.chat.id, symbol)
    await callback.message.answer(
        f"send me the target price for <b>{symbol}</b>.\n"
        f"e.g. <code>{symbol} 100</code> or <code>alert {symbol} 100 above</code>"
    )
    await callback.answer()


async def handle_show_levels_callback(*, callback, deps: AnalysisDetailFlowDependencies) -> None:
    symbol = (callback.data or "").split(":", 1)[1]
    payload = await deps.hub.cache.get_json(f"last_analysis:{callback.message.chat.id}:{symbol}")
    if not payload:
        await callback.answer("No cached levels - run a fresh analysis first.", show_alert=True)
        return
    entry = payload.get("entry", "-")
    tp1 = payload.get("tp1", "-")
    tp2 = payload.get("tp2", "-")
    sl = payload.get("sl", "-")
    await callback.message.answer(
        f"<b>{symbol}</b> key levels\n\n"
        f"entry    <code>{entry}</code>\n"
        f"target 1  <code>{tp1}</code>\n"
        f"target 2  <code>{tp2}</code>\n"
        f"stop      <code>{sl}</code>"
    )
    await callback.answer()


async def handle_why_callback(*, callback, deps: AnalysisDetailFlowDependencies) -> None:
    symbol = (callback.data or "").split(":", 1)[1]
    payload = await deps.hub.cache.get_json(f"last_analysis:{callback.message.chat.id}:{symbol}")
    if not payload:
        await callback.answer("No context saved - run a fresh analysis first.", show_alert=True)
        return
    bullets = payload.get("why", [])
    if bullets:
        bullet_lines = "\n".join(f"- {item}" for item in bullets)
        text = f"<b>why {symbol}</b>\n\n{bullet_lines}"
    else:
        summary = payload.get("summary", "")
        text = f"<b>why {symbol}</b>\n\n{summary or 'no reasoning available for this setup.'}"
    await callback.message.answer(text)
    await callback.answer()


async def _analyze_and_send(
    *,
    callback,
    chat_id: int,
    symbol: str,
    include_derivatives: bool,
    include_news: bool,
    detailed: bool,
    success_message: str,
    deps: AnalysisDetailFlowDependencies,
) -> None:
    settings = await deps.hub.user_service.get_settings(chat_id)
    payload = await deps.hub.analysis_service.analyze(
        symbol,
        timeframes=deps.analysis_timeframes_from_settings(settings),
        ema_periods=deps.parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
        rsi_periods=deps.parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
        include_derivatives=include_derivatives,
        include_news=include_news,
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
        detailed=detailed,
    )
    await deps.send_ghost_analysis(callback.message, symbol, analysis_text, direction=payload.get("side"))
    await callback.answer(success_message)


async def handle_refresh_callback(*, callback, deps: AnalysisDetailFlowDependencies) -> None:
    chat_id = callback.message.chat.id
    symbol = (callback.data or "").split(":", 1)[1]

    async def _run() -> None:
        await _analyze_and_send(
            callback=callback,
            chat_id=chat_id,
            symbol=symbol,
            include_derivatives=False,
            include_news=False,
            detailed=False,
            success_message="Refreshed",
            deps=deps,
        )

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_details_callback(*, callback, deps: AnalysisDetailFlowDependencies) -> None:
    chat_id = callback.message.chat.id
    symbol = (callback.data or "").split(":", 1)[1]

    async def _run() -> None:
        await _analyze_and_send(
            callback=callback,
            chat_id=chat_id,
            symbol=symbol,
            include_derivatives=True,
            include_news=True,
            detailed=True,
            success_message="Detailed mode",
            deps=deps,
        )

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)
