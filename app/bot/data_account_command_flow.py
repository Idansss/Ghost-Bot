from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DataAccountCommandFlowDependencies:
    hub: Any
    feature_flags_set: Callable[[], set[str]]
    safe_exc: Callable[[Exception], str]
    buffered_input_file_cls: Any


def _command_text(message) -> str:
    raw = (message.text or "").strip()
    return raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""


async def handle_journal_command(*, message, deps: DataAccountCommandFlowDependencies) -> None:
    if "journal" not in deps.feature_flags_set():
        await message.answer("Journal feature is disabled.")
        return
    service = getattr(deps.hub, "trade_journal_service", None)
    if not service:
        await message.answer("Journal is not available.")
        return

    args = _command_text(message).strip().lower().split()
    chat_id = message.chat.id

    if not args or args[0] == "list":
        limit = 20
        if len(args) >= 2 and args[1].isdigit():
            limit = min(int(args[1]), 50)
        trades = await service.list_trades(chat_id, limit=limit)
        if not trades:
            await message.answer("No journal entries. Log one: <code>/journal log BTC long 50000 51000 100</code>")
            return
        lines = ["<b>Trade journal</b>", ""]
        for trade in trades:
            line = f"<code>#{trade['id']}</code> {trade['symbol']} {trade['side']} entry {trade['entry']} -> exit {trade['exit_price']}"
            if trade.get("pnl_quote") is not None:
                line += f" | PnL ${trade['pnl_quote']:+,.2f}"
            lines.append(line)
        await message.answer("\n".join(lines))
        return

    if args[0] == "stats":
        days = 30
        if len(args) >= 2 and args[1].isdigit():
            days = min(int(args[1]), 365)
        stats = await service.get_stats(chat_id, days=days)
        await message.answer(
            f"<b>Journal stats</b> (last {days}d)\n"
            f"Trades: {stats['trades']} | Wins: {stats['wins']} | Win rate: {stats['win_rate']}% | "
            f"Total PnL: ${stats['total_pnl']:+,.2f}"
        )
        return

    if args[0] == "update" and len(args) >= 3:
        try:
            entry_id = int(args[1])
        except (ValueError, IndexError):
            await message.answer("Usage: /journal update ID exit PRICE outcome win|loss|be pnl AMOUNT")
            return
        raw_args = " ".join(args[2:])
        exit_price = None
        outcome = None
        pnl_quote = None
        exit_match = re.search(r"\bexit\s+([\d.]+)", raw_args)
        outcome_match = re.search(r"\boutcome\s+(win|loss|be|partial)\b", raw_args)
        pnl_match = re.search(r"\bpnl\s+([+-]?[\d.]+)", raw_args)
        if exit_match:
            exit_price = float(exit_match.group(1))
        if outcome_match:
            outcome = outcome_match.group(1)
        if pnl_match:
            pnl_quote = float(pnl_match.group(1))
        ok = await service.update_trade(
            chat_id,
            entry_id=entry_id,
            exit_price=exit_price,
            outcome=outcome,
            pnl_quote=pnl_quote,
        )
        await message.answer(f"Trade #{entry_id} updated." if ok else f"Trade #{entry_id} not found.")
        return

    if args[0] == "log" and len(args) >= 5:
        symbol = (args[1] or "").upper()
        side = (args[2] or "long").lower()
        if side not in {"long", "short"}:
            side = "long"
        try:
            entry = float(args[3])
            exit_price = float(args[4])
        except (ValueError, IndexError):
            await message.answer("Usage: /journal log SYMBOL long|short ENTRY EXIT [pnl]")
            return
        pnl = None
        if len(args) >= 6:
            try:
                pnl = float(args[5])
            except ValueError:
                pass
        entry_obj = await service.log_trade(
            chat_id,
            symbol=symbol,
            side=side,
            entry=entry,
            exit_price=exit_price,
            pnl_quote=pnl,
        )
        if entry_obj:
            reply = f"Logged: <b>{symbol}</b> {side} {entry} -> {exit_price}"
            if pnl is not None:
                reply += f" (${pnl:+,.2f})"
            await message.answer(reply)
        else:
            await message.answer("Failed to log trade.")
        return

    await message.answer(
        "Usage:\n"
        "<code>/journal list [N]</code>\n"
        "<code>/journal stats [days]</code>\n"
        "<code>/journal log SYMBOL long|short ENTRY EXIT [pnl]</code>\n"
        "<code>/journal update ID exit PRICE outcome win|loss|be pnl AMOUNT</code>"
    )


async def handle_compare_command(*, message, deps: DataAccountCommandFlowDependencies) -> None:
    if "multi_compare" not in deps.feature_flags_set():
        await message.answer("Multi-symbol compare is disabled.")
        return
    text = _command_text(message)
    symbols = [item.upper().lstrip("$") for item in (text or "BTC ETH SOL").strip().split() if item][:8]
    if not symbols:
        await message.answer("Usage: /compare BTC ETH SOL [SYMBOL...]")
        return
    try:
        prices = await asyncio.gather(
            *[deps.hub.market_router.get_price(symbol) for symbol in symbols],
            return_exceptions=True,
        )
    except Exception as exc:
        await message.answer(f"Could not fetch prices: {deps.safe_exc(exc)}")
        return
    lines = ["<b>Compare</b>", ""]
    for symbol, payload in zip(symbols, prices, strict=False):
        if isinstance(payload, Exception):
            lines.append(f"<b>{symbol}</b> error")
            continue
        price = float((payload or {}).get("price") or 0)
        lines.append(f"<b>{symbol}</b> ${price:,.2f}" if price > 0 else f"<b>{symbol}</b> no data")
    await message.answer("\n".join(lines))


async def handle_report_command(*, message, deps: DataAccountCommandFlowDependencies) -> None:
    if "scheduled_report" not in deps.feature_flags_set():
        await message.answer("Scheduled reports are disabled.")
        return
    service = getattr(deps.hub, "scheduled_report_service", None)
    if not service:
        await message.answer("Scheduled reports not available.")
        return

    args = _command_text(message).strip().lower().split()
    chat_id = message.chat.id

    if not args or args[0] == "list":
        reports = await service.list_reports(chat_id)
        if not reports:
            await message.answer("No scheduled report. Subscribe: <code>/report on 9 0</code> (9:00 UTC daily).")
            return
        lines = ["<b>Scheduled reports</b>", ""]
        for report in reports:
            line = f"- {report['report_type']} @ {report['cron_hour_utc']:02d}:{report['cron_minute_utc']:02d} UTC"
            if report.get("timezone"):
                line += f" ({report['timezone']})"
            lines.append(line)
        await message.answer("\n".join(lines))
        return

    if args[0] in {"off", "unsubscribe"}:
        ok = await service.unsubscribe(chat_id)
        await message.answer("Scheduled report turned off." if ok else "No subscription found.")
        return

    if args[0] in {"on", "subscribe"} or args[0].isdigit():
        hour_utc, minute_utc = 9, 0
        if args[0].isdigit() and len(args) >= 2 and args[1].isdigit():
            hour_utc = max(0, min(23, int(args[0])))
            minute_utc = max(0, min(59, int(args[1])))
        elif len(args) >= 3 and args[1].isdigit() and args[2].isdigit():
            hour_utc = max(0, min(23, int(args[1])))
            minute_utc = max(0, min(59, int(args[2])))
        record = await service.subscribe(
            chat_id,
            report_type="market_summary",
            hour_utc=hour_utc,
            minute_utc=minute_utc,
        )
        if record:
            await message.answer(
                f"Scheduled report on. You'll get a market summary daily at {record.cron_hour_utc:02d}:{record.cron_minute_utc:02d} UTC."
            )
        else:
            await message.answer("Could not subscribe. Try /start first.")
        return

    await message.answer(
        "Usage:\n"
        "<code>/report on [HOUR] [MINUTE]</code>\n"
        "<code>/report off</code>\n"
        "<code>/report list</code>"
    )


async def handle_export_command(*, message, deps: DataAccountCommandFlowDependencies) -> None:
    if "export" not in deps.feature_flags_set():
        await message.answer("Export feature is disabled.")
        return

    kind = (_command_text(message).strip().lower() or "alerts")
    chat_id = message.chat.id

    if kind == "alerts":
        try:
            alerts = await deps.hub.alerts_service.list_alerts(chat_id)
        except Exception:
            await message.answer("Could not load alerts.")
            return
        if not alerts:
            await message.answer("No alerts to export.")
            return
        lines = ["# Alerts export", ""]
        for alert in alerts:
            lines.append(f"#{alert.id} {alert.symbol} {alert.condition} {alert.target_price} [{alert.status}]")
        body = "\n".join(lines)
    elif kind == "journal":
        service = getattr(deps.hub, "trade_journal_service", None)
        if not service:
            await message.answer("Journal not available.")
            return
        trades = await service.list_trades(chat_id, limit=200)
        if not trades:
            await message.answer("No journal entries to export.")
            return
        lines = ["# Trade journal export", ""]
        for trade in trades:
            line = f"#{trade['id']} {trade['symbol']} {trade['side']} entry={trade['entry']} exit={trade['exit_price']}"
            if trade.get("pnl_quote") is not None:
                line += f" pnl={trade['pnl_quote']}"
            lines.append(line)
        body = "\n".join(lines)
    else:
        await message.answer("Usage: /export alerts | /export journal")
        return

    if len(body) <= 4000:
        await message.answer(f"<pre>{body}</pre>")
        return

    try:
        await message.answer_document(
            deps.buffered_input_file_cls(body.encode("utf-8"), filename=f"ghost_export_{kind}.txt"),
            caption=f"Export: {kind}",
        )
    except Exception:
        await message.answer("Export too long; sending in parts.")
        for index in range(0, len(body), 4000):
            await message.answer(f"<pre>{body[index:index + 4000]}</pre>")


async def handle_mydata_command(*, message, deps: DataAccountCommandFlowDependencies) -> None:
    service = getattr(deps.hub, "gdpr_service", None)
    if not service:
        await message.answer("Data export is not available.")
        return
    data = await service.export_my_data(message.chat.id)
    if data is None:
        await message.answer("No data found for your account.")
        return
    body = json.dumps(data, indent=2, default=str)
    try:
        await message.answer_document(
            deps.buffered_input_file_cls(body.encode("utf-8"), filename="my_ghost_data.json"),
            caption="Here's all the data stored about you. To delete it, use /deleteaccount.",
        )
    except Exception:
        await message.answer("Could not send file. Try /export alerts or /export journal for partial exports.")


async def handle_deleteaccount_command(*, message, deps: DataAccountCommandFlowDependencies) -> None:
    service = getattr(deps.hub, "gdpr_service", None)
    if not service:
        await message.answer("Account deletion is not available.")
        return
    if "confirm" not in (message.text or "").strip().lower():
        await message.answer(
            "This will permanently delete all your data: alerts, wallets, positions, trade journal, and settings.\n\n"
            "To confirm, send: <code>/deleteaccount confirm</code>"
        )
        return
    ok = await service.delete_account(message.chat.id)
    await message.answer("All your data has been deleted. Goodbye." if ok else "No account found. Nothing to delete.")
