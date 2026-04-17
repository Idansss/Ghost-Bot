from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class MarketDetailFlowDependencies:
    hub: Any
    feature_flags_set: Callable[[], set[str]]
    run_with_typing_lock: Callable[[Any, int, Callable[[], Awaitable[None]]], Awaitable[None]]
    remember_source_context: Callable[..., Awaitable[None]]


async def handle_derivatives_callback(*, callback, deps: MarketDetailFlowDependencies) -> None:
    chat_id = callback.message.chat.id
    symbol = (callback.data or "").split(":", 1)[1]

    async def _run() -> None:
        payload = await deps.hub.analysis_service.deriv_adapter.get_funding_and_oi(symbol)
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            exchange=str(payload.get("source") or ""),
            market_kind="perp",
            symbol=symbol,
            context="derivatives",
        )
        funding = payload.get("funding_rate")
        oi = payload.get("open_interest")
        source = payload.get("source") or payload.get("source_line") or "live"
        funding_str = f"{float(funding) * 100:.4f}%" if funding is not None else "n/a"
        oi_str = f"${float(oi) / 1_000_000:.2f}B" if oi is not None else "n/a"
        await callback.message.answer(
            f"<b>{symbol}</b> derivatives\n\n"
            f"funding rate  <code>{funding_str}</code>\n"
            f"open interest <code>{oi_str}</code>\n\n"
            f"<i>source: {source}</i>"
        )
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_catalysts_callback(*, callback, deps: MarketDetailFlowDependencies) -> None:
    chat_id = callback.message.chat.id
    symbol = (callback.data or "").split(":", 1)[1]

    async def _run() -> None:
        headlines = await deps.hub.news_service.get_asset_headlines(symbol, limit=3)
        if not headlines:
            await callback.message.answer(f"no fresh catalysts for <b>{symbol}</b> right now. check back later.")
            await callback.answer()
            return
        lines = [f"<b>{symbol} catalysts</b>\n"]
        for item in headlines[:3]:
            title = item.get("title", "")
            url = item.get("url", "")
            source = item.get("source", "")
            lines.append(f'<a href="{url}">{title}</a>' if url else title)
            if source:
                lines.append(f"  <i>{source}</i>")
            lines.append("")
        await callback.message.answer("\n".join(lines).strip(), disable_web_page_preview=True)
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def handle_backtest_callback(*, callback, deps: MarketDetailFlowDependencies) -> None:
    if "backtest" not in deps.feature_flags_set():
        await callback.answer("Backtest is disabled.", show_alert=True)
        return

    symbol = (callback.data or "").split(":", 1)[1]
    await callback.message.answer(
        f"drop your <b>{symbol}</b> trade details and i'll check it.\n\n"
        f"format: <code>{symbol} entry 2100 stop 2060 targets 2140 2180 2220 timeframe 1h</code>\n\n"
        "<i>or just paste it in natural language - i'll figure it out.</i>"
    )
    await callback.answer()


async def handle_save_wallet_callback(*, callback, deps: MarketDetailFlowDependencies) -> None:
    chat_id = callback.message.chat.id

    async def _run() -> None:
        _, chain, address = (callback.data or "").split(":", 2)
        await deps.hub.wallet_service.scan(chain, address, chat_id=chat_id, save=True)
        await callback.answer("Wallet saved", show_alert=True)

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)
