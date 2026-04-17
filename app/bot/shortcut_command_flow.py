from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class ShortcutCommandFlowDependencies:
    hub: Any
    dispatch_command_text: Callable[[Any, str], Awaitable[bool]]
    alpha_quick_menu: Callable[[], Any]
    watch_quick_menu: Callable[[], Any]
    chart_quick_menu: Callable[[], Any]
    heatmap_quick_menu: Callable[[], Any]
    rsi_quick_menu: Callable[[], Any]
    ema_quick_menu: Callable[[], Any]
    as_int: Callable[[Any, int], int]


async def handle_alpha_command(*, message, deps: ShortcutCommandFlowDependencies) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick an analysis shortcut or tap Custom.", reply_markup=deps.alpha_quick_menu())
        return
    tokens = text.split()
    if len(tokens) == 1:
        text = f"watch {tokens[0]}"
    await deps.dispatch_command_text(message, text)


async def handle_watch_command(*, message, deps: ShortcutCommandFlowDependencies) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick a watch shortcut or tap Custom.", reply_markup=deps.watch_quick_menu())
        return
    await deps.dispatch_command_text(message, f"watch {text}")


async def handle_price_command(*, message, deps: ShortcutCommandFlowDependencies) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    symbol = args.strip().upper().lstrip("$") or ""
    if not symbol:
        await message.answer("send symbol for quick price.\nExample: <code>/price BTC</code> or <code>/price SOL</code>")
        return
    try:
        data = await deps.hub.market_router.get_price(symbol)
        price = float(data.get("price") or 0)
        change_24h = data.get("change_24h")
        if change_24h is not None:
            line = f"<b>{symbol}</b> ${price:,.2f} ({change_24h:+.2f}%)"
        else:
            line = f"<b>{symbol}</b> ${price:,.2f}"
        await message.answer(line, reply_to_message_id=message.message_id)
    except Exception:
        await deps.dispatch_command_text(message, f"watch {symbol}")


async def handle_chart_command(*, message, deps: ShortcutCommandFlowDependencies) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick a chart shortcut or tap Custom.", reply_markup=deps.chart_quick_menu())
        return
    await deps.dispatch_command_text(message, f"chart {text}")


async def handle_heatmap_command(*, message, deps: ShortcutCommandFlowDependencies) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick a symbol for heatmap or tap Custom.", reply_markup=deps.heatmap_quick_menu())
        return
    await deps.dispatch_command_text(message, f"heatmap {text}")


async def handle_rsi_command(*, message, deps: ShortcutCommandFlowDependencies) -> None:
    raw = (message.text or "").strip()
    parts = raw.split()
    if len(parts) < 3:
        await message.answer("Pick an RSI scanner preset or tap Custom.", reply_markup=deps.rsi_quick_menu())
        return
    timeframe = parts[1].lower()
    mode = parts[2].lower()
    if mode not in {"overbought", "oversold"}:
        await message.answer("Mode must be `overbought` or `oversold`.")
        return
    top_n = max(1, min(deps.as_int(parts[3], 10), 20)) if len(parts) >= 4 else 10
    rsi_len = max(2, min(deps.as_int(parts[4], 14), 50)) if len(parts) >= 5 else 14
    await deps.dispatch_command_text(message, f"rsi top {top_n} {timeframe} {mode} rsi{rsi_len}")


async def handle_ema_command(*, message, deps: ShortcutCommandFlowDependencies) -> None:
    raw = (message.text or "").strip()
    parts = raw.split()
    if len(parts) < 3:
        await message.answer("Pick an EMA scanner preset or tap Custom.", reply_markup=deps.ema_quick_menu())
        return
    ema_len = max(2, min(deps.as_int(parts[1], 200), 500))
    timeframe = parts[2].lower()
    top_n = max(1, min(deps.as_int(parts[3], 10), 20)) if len(parts) >= 4 else 10
    await deps.dispatch_command_text(message, f"ema {ema_len} {timeframe} top {top_n}")
