from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


@dataclass(frozen=True)
class CommandMenuFlowDependencies:
    hub: Any
    dispatch_command_text: Callable[[Any, str], Awaitable[bool]]
    run_with_typing_lock: Callable[[Any, int, Callable[[], Awaitable[None]]], Awaitable[None]]
    set_cmd_wizard: Callable[[int, dict], Awaitable[None]]
    set_wizard: Callable[[int, dict], Awaitable[None]]
    as_int: Callable[[Any, int], int]
    alpha_quick_menu: Callable[[], Any]
    watch_quick_menu: Callable[[], Any]
    chart_quick_menu: Callable[[], Any]
    heatmap_quick_menu: Callable[[], Any]
    rsi_quick_menu: Callable[[], Any]
    ema_quick_menu: Callable[[], Any]
    news_quick_menu: Callable[[], Any]
    alert_quick_menu: Callable[[], Any]
    findpair_quick_menu: Callable[[], Any]
    setup_quick_menu: Callable[[], Any]
    scan_quick_menu: Callable[[], Any]
    giveaway_menu: Callable[..., Any]
    rsi_scan_template: Callable[[dict], str]
    logger: Any


def _menu_for(name: str, *, callback, deps: CommandMenuFlowDependencies):
    user_id = getattr(callback.from_user, "id", 0)
    mapping = {
        "alpha": ("Pick analysis shortcut.", deps.alpha_quick_menu()),
        "watch": ("Pick watch shortcut.", deps.watch_quick_menu()),
        "chart": ("Pick chart shortcut.", deps.chart_quick_menu()),
        "heatmap": ("Pick heatmap symbol.", deps.heatmap_quick_menu()),
        "rsi": ("Pick RSI scanner preset.", deps.rsi_quick_menu()),
        "ema": ("Pick EMA scanner preset.", deps.ema_quick_menu()),
        "news": ("Pick news mode.", deps.news_quick_menu()),
        "alert": ("Pick alert action.", deps.alert_quick_menu()),
        "findpair": ("Pick find mode.", deps.findpair_quick_menu()),
        "setup": ("Choose setup input mode.", deps.setup_quick_menu()),
        "scan": ("Pick chain first.", deps.scan_quick_menu()),
        "giveaway": (
            "Pick giveaway action.",
            deps.giveaway_menu(is_admin=deps.hub.giveaway_service.is_admin(user_id)),
        ),
    }
    return mapping.get(name)


async def _dispatch_with_typing(*, callback, chat_id: int, synthetic_text: str, deps: CommandMenuFlowDependencies) -> None:
    async def _run() -> None:
        await deps.dispatch_command_text(callback.message, synthetic_text)
        await callback.answer()

    await deps.run_with_typing_lock(callback.bot, chat_id, _run)


async def _start_dispatch_wizard(
    *,
    callback,
    chat_id: int,
    prefix: str,
    prompt: str,
    deps: CommandMenuFlowDependencies,
) -> None:
    await deps.set_cmd_wizard(chat_id, {"step": "dispatch_text", "prefix": prefix})
    await callback.message.answer(prompt)
    await callback.answer()


async def handle_command_menu_callback(*, callback, deps: CommandMenuFlowDependencies) -> None:
    chat_id = callback.message.chat.id
    data = callback.data or ""
    parts = data.split(":")
    if len(parts) < 2:
        await callback.answer()
        return

    if parts[1] == "menu":
        menu = _menu_for(parts[2] if len(parts) > 2 else "", callback=callback, deps=deps)
        if menu:
            await callback.message.answer(menu[0], reply_markup=menu[1])
        await callback.answer()
        return

    action = parts[1]
    if action == "alpha":
        if len(parts) >= 3 and parts[2] == "custom":
            await _start_dispatch_wizard(
                callback=callback,
                chat_id=chat_id,
                prefix="",
                prompt="Send symbol and optional tf, e.g. `SOL 4h`.",
                deps=deps,
            )
            return
        if len(parts) >= 4:
            await _dispatch_with_typing(
                callback=callback,
                chat_id=chat_id,
                synthetic_text=f"{parts[2]} {parts[3]}",
                deps=deps,
            )
            return
    if action == "watch":
        if len(parts) >= 3 and parts[2] == "custom":
            await _start_dispatch_wizard(
                callback=callback,
                chat_id=chat_id,
                prefix="watch ",
                prompt="Send symbol and optional tf, e.g. `BTC 1h`.",
                deps=deps,
            )
            return
        if len(parts) >= 4:
            await _dispatch_with_typing(
                callback=callback,
                chat_id=chat_id,
                synthetic_text=f"watch {parts[2]} {parts[3]}",
                deps=deps,
            )
            return
    if action == "chart":
        if len(parts) >= 3 and parts[2] == "custom":
            await _start_dispatch_wizard(
                callback=callback,
                chat_id=chat_id,
                prefix="chart ",
                prompt="Send symbol and optional tf, e.g. `ETH 4h`.",
                deps=deps,
            )
            return
        if len(parts) >= 4:
            await _dispatch_with_typing(
                callback=callback,
                chat_id=chat_id,
                synthetic_text=f"chart {parts[2]} {parts[3]}",
                deps=deps,
            )
            return
    if action == "heatmap":
        if len(parts) >= 3 and parts[2] == "custom":
            await _start_dispatch_wizard(
                callback=callback,
                chat_id=chat_id,
                prefix="heatmap ",
                prompt="Send symbol, e.g. `BTC`.",
                deps=deps,
            )
            return
        if len(parts) >= 3:
            await _dispatch_with_typing(
                callback=callback,
                chat_id=chat_id,
                synthetic_text=f"heatmap {parts[2]}",
                deps=deps,
            )
            return
    if action == "rsi":
        if len(parts) >= 3 and parts[2] == "custom":
            await _start_dispatch_wizard(
                callback=callback,
                chat_id=chat_id,
                prefix="rsi ",
                prompt="Send format: <code>1h oversold top 10 rsi14</code>.",
                deps=deps,
            )
            return
        if len(parts) >= 6:
            await callback.answer("Scanning...")
            timeframe = parts[2]
            mode = "overbought" if parts[3] == "overbought" else "oversold"
            limit = max(1, min(deps.as_int(parts[4], 10), 20))
            rsi_length = max(2, min(deps.as_int(parts[5], 14), 50))

            async def _run_rsi() -> None:
                try:
                    payload = await deps.hub.rsi_scanner_service.scan(
                        timeframe=timeframe,
                        mode=mode,
                        limit=limit,
                        rsi_length=rsi_length,
                        symbol=None,
                    )
                    await callback.message.answer(deps.rsi_scan_template(payload))
                except Exception as exc:
                    deps.logger.warning(
                        "rsi_scan_button_failed",
                        extra={"event": "rsi_button_error", "error": str(exc)},
                    )
                    await callback.message.answer("rsi scan hit an error - try again in a moment.")

            await deps.run_with_typing_lock(callback.bot, chat_id, _run_rsi)
            return
    if action == "ema":
        if len(parts) >= 3 and parts[2] == "custom":
            await _start_dispatch_wizard(
                callback=callback,
                chat_id=chat_id,
                prefix="ema ",
                prompt="Send format: `200 4h top 10`.",
                deps=deps,
            )
            return
        if len(parts) >= 5:
            await _dispatch_with_typing(
                callback=callback,
                chat_id=chat_id,
                synthetic_text=f"ema {parts[2]} {parts[3]} top {parts[4]}",
                deps=deps,
            )
            return
    if action == "news" and len(parts) >= 4:
        await _dispatch_with_typing(
            callback=callback,
            chat_id=chat_id,
            synthetic_text=f"news {parts[2]} {parts[3]}",
            deps=deps,
        )
        return
    if action == "alert":
        if len(parts) >= 3 and parts[2] == "create":
            await deps.set_cmd_wizard(chat_id, {"step": "dispatch_text", "prefix": "alert "})
            await callback.message.answer(
                "send me the alert details:\n\n"
                "<code>SOL 100 above</code>\n"
                "<code>BTC 66000 below</code>\n"
                "<code>ETH 3200</code>  <- defaults to cross\n\n"
                "<i>format: symbol  price  [above | below | cross]</i>"
            )
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "list":
            await _dispatch_with_typing(
                callback=callback,
                chat_id=chat_id,
                synthetic_text="list my alerts",
                deps=deps,
            )
            return
        if len(parts) >= 3 and parts[2] == "clear":
            alerts = await deps.hub.alerts_service.list_alerts(chat_id)
            count = len(alerts)
            if count == 0:
                await callback.message.answer("No alerts to clear.")
                await callback.answer()
                return
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="Yes, clear all", callback_data=f"confirm:clear_alerts:{count}")],
                    [InlineKeyboardButton(text="Cancel", callback_data="confirm:clear_alerts:no")],
                ]
            )
            await callback.message.answer(
                f"Clear all <b>{count}</b> alerts? This can't be undone.",
                reply_markup=kb,
            )
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "pause":
            await _dispatch_with_typing(
                callback=callback,
                chat_id=chat_id,
                synthetic_text="pause alerts",
                deps=deps,
            )
            return
        if len(parts) >= 3 and parts[2] == "resume":
            await _dispatch_with_typing(
                callback=callback,
                chat_id=chat_id,
                synthetic_text="resume alerts",
                deps=deps,
            )
            return
        if len(parts) >= 3 and parts[2] == "delete":
            await deps.set_cmd_wizard(chat_id, {"step": "dispatch_text", "prefix": "delete alert "})
            await callback.message.answer("Send alert id, e.g. `12`.")
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "clear_symbol":
            await deps.set_cmd_wizard(chat_id, {"step": "alert_clear_symbol"})
            await callback.message.answer("Send symbol to clear, e.g. `SOL`.")
            await callback.answer()
            return
    if action == "findpair":
        if len(parts) >= 3 and parts[2] == "price":
            await deps.set_cmd_wizard(chat_id, {"step": "dispatch_text", "prefix": "coin around "})
            await callback.message.answer("Send target price, e.g. `0.155`.")
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "query":
            await deps.set_cmd_wizard(chat_id, {"step": "dispatch_text", "prefix": "find pair "})
            await callback.message.answer("Send name/ticker/context, e.g. `xion`.")
            await callback.answer()
            return
    if action == "setup":
        if len(parts) >= 3 and parts[2] == "wizard":
            await deps.set_wizard(chat_id, {"step": "symbol", "data": {}})
            await callback.message.answer("Tradecheck wizard: send symbol (e.g., ETH).")
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "freeform":
            await deps.set_cmd_wizard(chat_id, {"step": "dispatch_text", "prefix": ""})
            await callback.message.answer("Paste setup text, e.g. `entry 2100 stop 2165 targets 2043 2027 1991`.")
            await callback.answer()
            return
    if action == "scan" and len(parts) >= 3:
        chain = "solana" if parts[2] == "solana" else "tron"
        await deps.set_cmd_wizard(chat_id, {"step": "dispatch_text", "prefix": f"scan {chain} "})
        await callback.message.answer(f"Paste {chain} address.")
        await callback.answer()
        return
    if action == "alertdel" and len(parts) >= 3:
        await _dispatch_with_typing(
            callback=callback,
            chat_id=chat_id,
            synthetic_text=f"delete alert {parts[2]}",
            deps=deps,
        )
        return

    await callback.answer()
