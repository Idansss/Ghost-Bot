from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlertCommandFlowDependencies:
    hub: Any
    wallet_scan_limit_per_hour: int
    remember_source_context: Callable[..., Awaitable[None]]
    watchlist_template: Callable[[dict], str]
    news_template: Callable[[dict], str]
    cycle_template: Callable[[dict, dict | None], str]
    wallet_scan_template: Callable[[dict], str]
    wallet_actions: Callable[[str, str], Any]
    scan_quick_menu: Callable[[], Any]
    news_quick_menu: Callable[[], Any]
    alert_quick_menu: Callable[[], Any]
    simple_followup: Callable[[list[tuple[str, str]]], Any]
    alert_created_menu: Callable[[str], Any]


def _render_alerts_lines(alerts: list[Any]) -> str:
    rows = ["<b>Active Alerts</b>", ""]
    for alert in alerts:
        rows.append(
            f"<code>#{alert.id}</code>  <b>{alert.symbol}</b>  {alert.condition}  {alert.target_price}  <i>[{alert.status}]</i>"
        )
    return "\n".join(rows)


async def handle_watchlist_command(*, message, deps: AlertCommandFlowDependencies) -> None:
    n_match = re.search(r"/watchlist\s+(\d+)", message.text or "")
    count = int(n_match.group(1)) if n_match else 5
    direction = None
    if re.search(r"\blong\b", message.text or "", re.IGNORECASE):
        direction = "long"
    elif re.search(r"\bshort\b", message.text or "", re.IGNORECASE):
        direction = "short"
    payload = await deps.hub.watchlist_service.build_watchlist(count=max(1, min(count, 20)), direction=direction)
    await deps.remember_source_context(
        message.chat.id,
        source_line=str(payload.get("source_line") or ""),
        context="watchlist",
    )
    await message.answer(deps.watchlist_template(payload))


async def handle_news_command(*, message, deps: AlertCommandFlowDependencies) -> None:
    text = (message.text or "").strip()
    topic: str | None = None
    mode = "crypto"
    limit = 6
    parts = text.split()
    if len(parts) == 1:
        await message.answer("Pick a news mode.", reply_markup=deps.news_quick_menu())
        return
    if len(parts) > 1:
        raw_topic = parts[1].strip()
        if raw_topic.isdigit():
            limit = max(3, min(int(raw_topic), 10))
        else:
            topic = raw_topic
    if len(parts) > 2 and parts[2].isdigit():
        limit = max(3, min(int(parts[2]), 10))

    if topic:
        lowered = topic.lower().strip()
        if lowered in {"crypto", "openai", "cpi", "fomc"}:
            topic = lowered
        if re.search(r"\b(openai|chatgpt|gpt|codex)\b", lowered):
            mode = "openai"
            topic = "openai"
        elif re.search(r"\b(cpi|inflation)\b", lowered):
            mode = "macro"
            topic = "cpi"
        elif re.search(r"\b(fomc|fed|powell|macro|rates?)\b", lowered):
            mode = "macro"
            topic = "macro"

    payload = await deps.hub.news_service.get_digest(topic=topic, mode=mode, limit=limit)
    headlines = payload.get("headlines") if isinstance(payload, dict) else None
    head = headlines[0] if isinstance(headlines, list) and headlines else {}
    await deps.remember_source_context(
        message.chat.id,
        source_line=f"{head.get('source', 'news feed')} | {head.get('url', '')}".strip(),
        context="news",
    )
    await message.answer(deps.news_template(payload), parse_mode="HTML")


async def handle_cycle_command(*, message, deps: AlertCommandFlowDependencies) -> None:
    settings = await deps.hub.user_service.get_settings(message.chat.id)
    payload = await deps.hub.cycles_service.cycle_check()
    await message.answer(deps.cycle_template(payload, settings))


async def handle_scan_command(*, message, deps: AlertCommandFlowDependencies) -> None:
    text = message.text or ""
    match = re.search(r"/scan\s+(solana|tron)\s+([A-Za-z0-9]+)", text, re.IGNORECASE)
    if not match:
        await message.answer("Pick chain first, then paste address.", reply_markup=deps.scan_quick_menu())
        return

    limiter = await deps.hub.rate_limiter.check(
        key=f"rl:scan:{message.chat.id}:{datetime.now(UTC).strftime('%Y%m%d%H')}",
        limit=deps.wallet_scan_limit_per_hour,
        window_seconds=3600,
    )
    if not limiter.allowed:
        await message.answer("Wallet scan limit reached for this hour.")
        return

    chain, address = match.group(1).lower(), match.group(2)
    result = await deps.hub.wallet_service.scan(chain, address, chat_id=message.chat.id)
    await message.answer(deps.wallet_scan_template(result), reply_markup=deps.wallet_actions(chain, address))


async def handle_alert_command(*, message, deps: AlertCommandFlowDependencies) -> None:
    text = (message.text or "").strip()

    if text.startswith("/alert list"):
        alerts = await deps.hub.alerts_service.list_alerts(message.chat.id)
        if not alerts:
            await message.answer("No active alerts.")
            return
        first = alerts[0]
        await deps.remember_source_context(
            message.chat.id,
            exchange=first.source_exchange,
            market_kind=first.market_kind,
            instrument_id=first.instrument_id,
            symbol=first.symbol,
            context="alerts list",
        )
        await message.answer(_render_alerts_lines(alerts))
        return

    if text.startswith("/alert clear"):
        count = await deps.hub.alerts_service.clear_user_alerts(message.chat.id)
        await message.answer(f"Cleared {count} alerts.")
        return

    if text.startswith("/alert pause"):
        count = await deps.hub.alerts_service.pause_user_alerts(message.chat.id)
        await message.answer(f"Paused {count} alerts.")
        return

    if text.startswith("/alert resume"):
        count = await deps.hub.alerts_service.resume_user_alerts(message.chat.id)
        await message.answer(f"Resumed {count} alerts.")
        return

    delete_match = re.search(r"/alert\s+delete\s+(\d+)", text)
    if delete_match:
        ok = await deps.hub.alerts_service.delete_alert(message.chat.id, int(delete_match.group(1)))
        await message.answer("Deleted." if ok else "Alert not found.")
        return

    add_match = re.search(r"/alert\s+add\s+([A-Za-z0-9]+)\s+(above|below|cross)\s+([0-9.]+)", text, re.IGNORECASE)
    if add_match:
        symbol = add_match.group(1).upper()
        condition = add_match.group(2).lower()
        price = float(add_match.group(3))
        alert = await deps.hub.alerts_service.create_alert(message.chat.id, symbol, condition, price, source="command")
        await deps.remember_source_context(
            message.chat.id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        cond_word = {"above": "crosses above", "below": "crosses below"}.get(condition, "crosses")
        await message.answer(
            f"alert set <b>{symbol}</b> {cond_word} <b>${price:,.2f}</b>.\n"
            "i'll ping you the moment it hits. don't get liquidated.",
            reply_markup=deps.alert_created_menu(symbol),
        )
        return

    simple_match = re.search(
        r"^/alert\s+([A-Za-z0-9$]{2,20})\s+([0-9]+(?:\.[0-9]+)?)(?:\s+(above|below|cross))?\s*$",
        text,
        re.IGNORECASE,
    )
    if simple_match:
        symbol = simple_match.group(1).upper().lstrip("$")
        price = float(simple_match.group(2))
        condition = (simple_match.group(3) or "cross").lower()
        alert = await deps.hub.alerts_service.create_alert(message.chat.id, symbol, condition, price, source="command")
        await deps.remember_source_context(
            message.chat.id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        cond_word = {"above": "crosses above", "below": "crosses below"}.get(condition, "crosses")
        await message.answer(
            f"alert set <b>{symbol}</b> {cond_word} <b>${price:,.2f}</b>.\n"
            "i'll ping you the moment it hits. don't get liquidated.",
            reply_markup=deps.alert_created_menu(symbol),
        )
        return

    await message.answer("Pick an alert action.", reply_markup=deps.alert_quick_menu())


async def handle_alerts_command(*, message, deps: AlertCommandFlowDependencies) -> None:
    try:
        alerts = await deps.hub.alerts_service.list_alerts(message.chat.id)
    except Exception as exc:
        logger.exception(
            "alerts_list_failed",
            extra={"event": "alerts_list_failed", "error": str(exc), "chat_id": message.chat.id},
        )
        await message.answer("Alerts are temporarily unavailable. Try again in a few seconds.")
        return
    if not alerts:
        await message.answer("No active alerts.")
        return
    first = alerts[0]
    await deps.remember_source_context(
        message.chat.id,
        exchange=first.source_exchange,
        market_kind=first.market_kind,
        instrument_id=first.instrument_id,
        symbol=first.symbol,
        context="alerts list",
    )
    await message.answer(_render_alerts_lines(alerts))


async def handle_alertdel_command(*, message, deps: AlertCommandFlowDependencies) -> None:
    text = (message.text or "").strip()
    match = re.search(r"^/alertdel\s+(\d+)\s*$", text, re.IGNORECASE)
    if not match:
        try:
            alerts = await deps.hub.alerts_service.list_alerts(message.chat.id)
        except Exception as exc:
            logger.exception(
                "alertdel_list_failed",
                extra={"event": "alertdel_list_failed", "error": str(exc), "chat_id": message.chat.id},
            )
            await message.answer("Alerts are temporarily unavailable. Try again in a few seconds.")
            return
        if not alerts:
            await message.answer("No active alerts.", reply_markup=deps.alert_quick_menu())
            return
        options = [(f"Delete #{alert.id}", f"cmd:alertdel:{alert.id}") for alert in alerts[:8]]
        await message.answer("Tap an alert to delete.", reply_markup=deps.simple_followup(options))
        return
    try:
        ok = await deps.hub.alerts_service.delete_alert(message.chat.id, int(match.group(1)))
    except Exception as exc:
        logger.exception(
            "alertdel_failed",
            extra={"event": "alertdel_failed", "error": str(exc), "chat_id": message.chat.id},
        )
        await message.answer("Delete failed on my side. Try again in a few seconds.")
        return
    await message.answer("Deleted." if ok else "Alert not found.")


async def handle_alertclear_command(*, message, deps: AlertCommandFlowDependencies) -> None:
    text = (message.text or "").strip()
    match = re.search(r"^/alertclear\s+([A-Za-z0-9$]{2,20})\s*$", text, re.IGNORECASE)
    if match:
        symbol = match.group(1).upper().lstrip("$")
        count = await deps.hub.alerts_service.delete_alerts_by_symbol(message.chat.id, symbol)
        await message.answer(f"Cleared {count} alerts for {symbol}.")
        return
    await message.answer(
        "Pick clear action.",
        reply_markup=deps.simple_followup(
            [("Clear all alerts", "cmd:alert:clear"), ("Clear by symbol", "cmd:alert:clear_symbol")]
        ),
    )
