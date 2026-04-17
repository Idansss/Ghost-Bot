from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PositionCommandFlowDependencies:
    hub: Any
    feature_flags_set: Callable[[], set[str]]


def _command_args(message) -> list[str]:
    raw = (message.text or "").strip()
    text = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    return (text or "").strip().lower().split()


async def handle_position_command(*, message, deps: PositionCommandFlowDependencies) -> None:
    if "portfolio" not in deps.feature_flags_set():
        await message.answer("Portfolio feature is disabled.")
        return

    service = getattr(deps.hub, "portfolio_service", None)
    if not service:
        await message.answer("Portfolio is not available.")
        return

    args = _command_args(message)
    chat_id = message.chat.id

    if args and args[0] == "summary":
        summary = await service.get_portfolio_summary(chat_id)
        if not summary["positions"]:
            await message.answer("No positions. Add one: <code>/position add BTC long 50000 1000 2</code>")
            return
        pnl_marker = "UP" if summary["total_pnl_usd"] >= 0 else "DOWN"
        lines = [
            "<b>Portfolio Summary</b>",
            "",
            f"Total cost:  <b>${summary['total_cost_usd']:,.2f}</b>",
            f"Total value: <b>${summary['total_value_usd']:,.2f}</b>",
            f"Unrealized PnL: {pnl_marker} <b>${summary['total_pnl_usd']:+,.2f} ({summary['total_pnl_pct']:+.2f}%)</b>",
            "",
            "<b>Allocation</b>",
        ]
        for position in summary["positions"]:
            lines.append(
                f"  {position['symbol']} {position['side']} | {position['allocation_pct']}% | PnL {position['pnl_pct']:+.2f}%"
            )
        if summary.get("best"):
            lines.append(f"\nBest: <b>{summary['best']['symbol']}</b> {summary['best']['pnl_pct']:+.2f}%")
        if summary.get("worst"):
            lines.append(f"Worst: <b>{summary['worst']['symbol']}</b> {summary['worst']['pnl_pct']:+.2f}%")
        await message.answer("\n".join(lines))
        return

    if not args or args[0] == "list":
        positions = await service.list_positions(chat_id)
        if not positions:
            await message.answer("No positions. Add one: <code>/position add BTC long 50000 1000 2</code>")
            return
        lines = ["<b>Positions</b>", ""]
        total_pnl = 0.0
        for position in positions:
            pnl_quote = position.get("pnl_quote", 0) or 0
            total_pnl += pnl_quote
            lines.append(
                f"<code>#{position['id']}</code> <b>{position['symbol']}</b> {position['side']} | "
                f"entry ${position['entry_price']:,.2f} -> ${position['current_price']:,.2f} | "
                f"PnL {position['pnl_pct']:+.2f}% (${pnl_quote:+,.2f})"
            )
        lines.append(f"\n<b>Total unrealized PnL:</b> ${total_pnl:+,.2f}")
        await message.answer("\n".join(lines))
        return

    if args[0] == "delete" and len(args) >= 2:
        try:
            position_id = int(args[1])
        except ValueError:
            await message.answer("Usage: /position delete &lt;id&gt;")
            return
        ok = await service.delete_position(chat_id, position_id)
        await message.answer("Position removed." if ok else "Position not found.")
        return

    if args[0] == "add" and len(args) >= 5:
        symbol = (args[1] or "").upper()
        side = (args[2] or "long").lower()
        if side not in {"long", "short"}:
            side = "long"
        try:
            entry_price = float(args[3])
            size_quote = float(args[4])
        except (ValueError, IndexError):
            await message.answer("Usage: /position add SYMBOL long|short ENTRY_PRICE SIZE_QUOTE [leverage]")
            return

        leverage = 1.0
        notes = ""
        if len(args) >= 6:
            try:
                leverage = float(args[5])
            except ValueError:
                notes = " ".join(args[5:])[:255]
        if len(args) >= 7 and not notes:
            notes = " ".join(args[6:])[:255]

        position, warning = await service.add_position(
            chat_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size_quote=size_quote,
            leverage=leverage,
            notes=notes or None,
        )
        if position is None:
            await message.answer(warning or "Failed to add position.")
            return
        reply = f"Position added: <b>{symbol}</b> {side} @ ${entry_price:,.2f} size ${size_quote:,.0f} {leverage}x"
        if warning:
            reply += f"\nWarning: {warning}"
        await message.answer(reply)
        return

    await message.answer(
        "Usage:\n"
        "<code>/position summary</code> total value, allocation, best/worst\n"
        "<code>/position list</code> list positions with PnL\n"
        "<code>/position add SYMBOL long|short ENTRY SIZE [leverage]</code>\n"
        "<code>/position delete ID</code>"
    )
