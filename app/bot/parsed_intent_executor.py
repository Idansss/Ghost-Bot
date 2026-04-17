from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

from aiogram.types import BufferedInputFile, InlineKeyboardButton, InlineKeyboardMarkup

from app.bot.keyboards import alert_created_menu, settings_menu, wallet_actions
from app.bot.prefs import effective_timeframe
from app.bot.templates import (
    asset_unsupported_template,
    correlation_template,
    cycle_template,
    giveaway_status_template,
    help_text,
    news_template,
    pair_find_template,
    price_guess_template,
    rsi_scan_template,
    settings_text,
    setup_review_template,
    smalltalk_reply,
    trade_math_template,
    trade_verification_template,
    wallet_scan_template,
    watchlist_template,
)
from app.bot.ux import busy as ux_busy
from app.bot.ux import degraded as ux_degraded
from app.bot.ux import transient_error as ux_transient_error
from app.core.nlu import Intent


@dataclass(frozen=True)
class ParsedIntentDependencies:
    hub: Any
    wallet_scan_limit_per_hour: int
    maybe_send_market_warning: Callable[[Any], Awaitable[None]]
    analysis_timeframes_from_settings: Callable[[dict], list[str]]
    parse_int_list: Callable[[Any, list[int]], list[int]]
    append_last_symbol: Callable[[int, str], Awaitable[None]]
    remember_analysis_context: Callable[[int, str, str | None, dict], Awaitable[None]]
    remember_source_context: Callable[..., Awaitable[None]]
    render_analysis_text: Callable[..., Awaitable[str]]
    send_ghost_analysis: Callable[..., Awaitable[None]]
    as_float: Callable[[Any, float | None], float | None]
    trade_math_payload: Callable[..., dict]
    llm_fallback_reply: Callable[[str, dict | None, int | None], Awaitable[str | None]]
    safe_exc: Callable[[Exception], str]
    parse_duration_to_seconds: Callable[[str], int | None]
    save_trade_check: Callable[[int, dict, dict], Awaitable[None]]


async def execute_parsed_intent(
    *,
    message,
    parsed,
    settings: dict,
    deps: ParsedIntentDependencies,
) -> bool:
    hub = deps.hub
    chat_id = message.chat.id
    raw_text = message.text or ""

    if parsed.intent == Intent.ANALYSIS:
        symbol = parsed.entities["symbol"]
        direction = parsed.entities.get("direction")
        await deps.maybe_send_market_warning(message)
        if not hub.analysis_uc:
            raise RuntimeError("analysis use-case not configured")
        result = await hub.analysis_uc.analyze(
            chat_id=chat_id,
            symbol=symbol,
            direction=direction,
            entities=parsed.entities,
            settings_timeframes=deps.analysis_timeframes_from_settings(settings),
            settings_emas=deps.parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            settings_rsis=deps.parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
        )
        if result.kind == "busy":
            await message.answer(ux_busy("analysis"))
            return True
        if result.kind == "unsupported":
            await message.answer(asset_unsupported_template(result.fallback or {}, settings))
            return True
        if result.kind == "error" or not result.payload:
            raise RuntimeError(result.error or "analysis failed")
        payload = result.payload
        await deps.append_last_symbol(chat_id, symbol)
        await deps.remember_analysis_context(chat_id, symbol, direction, payload)
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await deps.render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=direction,
            settings=settings,
            chat_id=chat_id,
        )
        if result.kind == "cached":
            await message.answer(ux_degraded("analysis"))
        await deps.send_ghost_analysis(message, symbol, analysis_text, direction=direction)
        return True

    if parsed.intent == Intent.SETUP_REVIEW:
        timeframe = parsed.entities.get("timeframe", "1h")
        timeframes = parsed.entities.get("timeframes") or []
        if timeframes:
            timeframe = timeframes[0]
        symbol = parsed.entities.get("symbol")
        entry = deps.as_float(parsed.entities.get("entry"))
        stop = deps.as_float(parsed.entities.get("stop"))
        targets = [float(item) for item in (parsed.entities.get("targets") or [])]
        if entry is None and symbol and re.search(r"\bmarket\s*(?:price|order)?\b|\bmp\b|\bat\s+market\b", raw_text, re.IGNORECASE):
            with suppress(Exception):
                price_data = await hub.market_router.get_price(symbol)
                entry = deps.as_float(price_data.get("price") or price_data.get("last"))
        if not symbol or entry is None or stop is None or not targets:
            await message.answer(
                "need <b>symbol</b>, <b>entry</b>, <b>stop</b>, and at least one <b>target</b>.\n"
                "e.g. <code>SNXUSDT entry 0.028 stop 0.036 tp 0.022</code>"
            )
            return True
        payload = await hub.setup_review_service.review(
            symbol=symbol,
            timeframe=timeframe,
            entry=float(entry),
            stop=float(stop),
            targets=targets,
            direction=parsed.entities.get("direction"),
            amount_usd=parsed.entities.get("amount_usd"),
            leverage=parsed.entities.get("leverage"),
        )
        await message.answer(setup_review_template(payload, settings))
        return True

    if parsed.intent == Intent.TRADE_MATH:
        payload = deps.trade_math_payload(
            entry=float(parsed.entities["entry"]),
            stop=float(parsed.entities["stop"]),
            targets=[float(item) for item in parsed.entities["targets"]],
            direction=parsed.entities.get("direction"),
            margin_usd=parsed.entities.get("amount_usd"),
            leverage=parsed.entities.get("leverage"),
            symbol=parsed.entities.get("symbol"),
        )
        await message.answer(trade_math_template(payload, settings))
        return True

    if parsed.intent == Intent.RSI_SCAN:
        if not hub.rsi_scan_uc:
            raise RuntimeError("rsi scan use-case not configured")
        result = await hub.rsi_scan_uc.scan(
            chat_id=chat_id,
            timeframe=effective_timeframe(user_text=raw_text, settings=settings, default="1h"),
            mode=str(parsed.entities.get("mode", "oversold")),
            limit=int(parsed.entities.get("limit", 10)),
            rsi_length=int(parsed.entities.get("rsi_length", 14)),
            symbol=str(parsed.entities.get("symbol")).strip() if parsed.entities.get("symbol") else None,
        )
        if result.busy:
            await message.answer(ux_busy("rsi"))
            return True
        if not result.payload:
            await message.answer(ux_transient_error("rsi"))
            return True
        payload = result.payload
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            symbol=parsed.entities.get("symbol"),
            context="rsi scan",
        )
        if result.degraded:
            await message.answer(ux_degraded("rsi"))
        await message.answer(rsi_scan_template(payload))
        return True

    if parsed.intent == Intent.EMA_SCAN:
        if not hub.ema_scan_uc:
            raise RuntimeError("ema scan use-case not configured")
        result = await hub.ema_scan_uc.scan(
            chat_id=chat_id,
            timeframe=effective_timeframe(user_text=raw_text, settings=settings, default="4h"),
            ema_length=int(parsed.entities.get("ema_length", 200)),
            mode=str(parsed.entities.get("mode", "closest")),
            limit=int(parsed.entities.get("limit", 10)),
        )
        if result.busy:
            await message.answer(ux_busy("ema"))
            return True
        if not result.payload:
            await message.answer(ux_transient_error("ema"))
            return True
        payload = result.payload
        lines = [payload["summary"], ""]
        for idx, row in enumerate(payload.get("items", []), start=1):
            lines.append(
                f"{idx}. {row['symbol']} price {row['price']} | EMA{payload['ema_length']} {row['ema']} | "
                f"dist {row['distance_pct']}% ({row['side']})"
            )
        if not payload.get("items"):
            lines.append("No EMA matches right now.")
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="ema scan",
        )
        if result.degraded:
            await message.answer(ux_degraded("ema"))
        await message.answer("\n".join(lines))
        return True

    if parsed.intent == Intent.CHART:
        async with hub.cache.distributed_lock(f"user:{chat_id}:chart", ttl=25) as acquired:
            if not acquired:
                await message.answer("Chart render already running \u2014 try again in a few seconds.")
                return True
        timeframe = effective_timeframe(user_text=raw_text, settings=settings, default="1h")
        img, meta = await hub.chart_service.render_chart(
            symbol=parsed.entities["symbol"],
            timeframe=timeframe,
        )
        symbol = str(parsed.entities["symbol"]).upper()
        await deps.remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("instrument_id") or ""),
            updated_at=str(meta.get("updated_at") or ""),
            symbol=symbol,
            context="chart",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_{timeframe}_chart.png"),
            caption=f"{symbol} {timeframe} chart.",
        )
        return True

    if parsed.intent == Intent.HEATMAP:
        async with hub.cache.distributed_lock(f"user:{chat_id}:heatmap", ttl=25) as acquired:
            if not acquired:
                await message.answer("Heatmap render already running \u2014 try again in a few seconds.")
                return True
        symbol = str(parsed.entities.get("symbol", "BTC"))
        img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
        await deps.remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("pair") or ""),
            symbol=symbol,
            context="heatmap",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_heatmap.png"),
            caption=(
                f"{meta['pair']} orderbook heatmap\n"
                f"Best bid: {meta['best_bid']:.6f} | Best ask: {meta['best_ask']:.6f}\n"
                f"Bid wall: {meta['bid_wall']:.6f} | Ask wall: {meta['ask_wall']:.6f}"
            ),
        )
        return True

    if parsed.intent == Intent.PAIR_FIND:
        payload = await hub.discovery_service.find_pair(parsed.entities["query"])
        await message.answer(pair_find_template(payload))
        return True

    if parsed.intent == Intent.PRICE_GUESS:
        payload = await hub.discovery_service.guess_by_price(
            target_price=float(parsed.entities["target_price"]),
            limit=int(parsed.entities.get("limit", 10)),
        )
        await message.answer(price_guess_template(payload))
        return True

    if parsed.intent == Intent.SMALLTALK:
        llm_reply = await deps.llm_fallback_reply(raw_text, settings, chat_id=chat_id)
        await message.answer(llm_reply or smalltalk_reply(settings))
        return True

    if parsed.intent == Intent.ASSET_UNSUPPORTED:
        await message.answer("Send the ticker + context and I'll give a safe fallback brief.")
        return True

    if parsed.intent == Intent.ALERT_CREATE:
        symbol = parsed.entities["symbol"]
        price_val = float(parsed.entities["target_price"])
        condition = parsed.entities.get("condition", "cross")
        extra_conditions = parsed.entities.get("extra_conditions") or None
        try:
            if not hub.alerts_uc:
                raise RuntimeError("alerts use-case not configured")
            alert = await hub.alerts_uc.create(
                chat_id=chat_id,
                symbol=symbol,
                condition=condition,
                target_price=price_val,
                extra_conditions=extra_conditions,
            )
        except RuntimeError as exc:
            await message.answer(f"couldn't set that alert \u2014 {deps.safe_exc(exc)}")
            return True
        except Exception:
            await message.answer("alert creation failed. try again in a sec.")
            return True
        await deps.remember_source_context(
            chat_id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=alert.symbol,
            context="alert",
        )
        cond_word = {"above": "crosses above", "below": "crosses below"}.get(condition, "crosses")
        extra_note = ""
        if extra_conditions:
            parts: list[str] = []
            for item in extra_conditions:
                if item.get("type") == "rsi":
                    operator = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}.get(item["operator"], item["operator"])
                    parts.append(f"RSI({item.get('timeframe', '1h')}) {operator} {item['value']}")
                elif item.get("type") == "ema":
                    parts.append(
                        f"price {item.get('operator', 'above')} EMA{item.get('period', 200)}({item.get('timeframe', '1h')})"
                    )
            if parts:
                extra_note = "\n<i>extra conditions: " + " AND ".join(parts) + "</i>"
        await message.answer(
            f"\U0001F514 alert set \u2014 <b>{alert.symbol}</b> {cond_word} <b>${price_val:,.2f}</b>.{extra_note}\n"
            "i'll ping you the moment it hits. don't get liquidated.",
            reply_markup=alert_created_menu(alert.symbol),
        )
        return True

    if parsed.intent == Intent.ALERT_LIST:
        if not hub.alerts_uc:
            raise RuntimeError("alerts use-case not configured")
        alerts = await hub.alerts_uc.list(chat_id=chat_id)
        if not alerts:
            await message.answer("No active alerts.")
        else:
            lines = ["<b>Active Alerts</b>", ""]
            for alert in alerts:
                lines.append(
                    f"<code>#{alert.id}</code>  <b>{alert.symbol}</b>  {alert.condition}  {alert.target_price}  <i>[{alert.status}]</i>"
                )
            first = alerts[0]
            await deps.remember_source_context(
                chat_id,
                exchange=first.source_exchange,
                market_kind=first.market_kind,
                instrument_id=first.instrument_id,
                symbol=first.symbol,
                context="alerts list",
            )
            await message.answer("\n".join(lines))
        return True

    if parsed.intent == Intent.ALERT_CLEAR:
        alerts = await hub.alerts_service.list_alerts(chat_id)
        count = len(alerts)
        if count == 0:
            await message.answer("No alerts to clear.")
            return True
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Yes, clear all", callback_data=f"confirm:clear_alerts:{count}")],
                [InlineKeyboardButton(text="Cancel", callback_data="confirm:clear_alerts:no")],
            ]
        )
        await message.answer(
            f"Clear all <b>{count}</b> alerts? This can't be undone.",
            reply_markup=keyboard,
            reply_to_message_id=message.message_id,
        )
        return True

    if parsed.intent == Intent.ALERT_PAUSE:
        if not hub.alerts_uc:
            raise RuntimeError("alerts use-case not configured")
        count = await hub.alerts_uc.pause(chat_id=chat_id)
        await message.answer(f"Paused {count} alerts.")
        return True

    if parsed.intent == Intent.ALERT_RESUME:
        if not hub.alerts_uc:
            raise RuntimeError("alerts use-case not configured")
        count = await hub.alerts_uc.resume(chat_id=chat_id)
        await message.answer(f"Resumed {count} alerts.")
        return True

    if parsed.intent == Intent.ALERT_DELETE:
        symbol = parsed.entities.get("symbol")
        if symbol:
            if not hub.alerts_uc:
                raise RuntimeError("alerts use-case not configured")
            count = await hub.alerts_uc.delete_by_symbol(chat_id=chat_id, symbol=str(symbol))
            await message.answer(f"Removed {count} alert(s) for {str(symbol).upper()}.")
            return True
        if not hub.alerts_uc:
            raise RuntimeError("alerts use-case not configured")
        ok = await hub.alerts_uc.delete(chat_id=chat_id, alert_id=int(parsed.entities["alert_id"]))
        await message.answer("Deleted." if ok else "Alert not found.")
        return True

    if parsed.intent == Intent.GIVEAWAY_JOIN:
        if not message.from_user:
            await message.answer("Could not identify user for join.")
            return True
        payload = await hub.giveaway_service.join_active(chat_id, message.from_user.id)
        await message.answer(f"Joined giveaway #{payload['giveaway_id']}. Participants: {payload['participants']}")
        return True

    if parsed.intent == Intent.GIVEAWAY_STATUS:
        payload = await hub.giveaway_service.status(chat_id)
        await message.answer(giveaway_status_template(payload))
        return True

    if parsed.intent == Intent.GIVEAWAY_START:
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        duration_seconds = deps.parse_duration_to_seconds(str(parsed.entities.get("duration", "10m")))
        if duration_seconds is None:
            await message.answer("Duration format should look like 10m, 1h, or 1d.")
            return True
        payload = await hub.giveaway_service.start_giveaway(
            group_chat_id=chat_id,
            admin_chat_id=message.from_user.id,
            duration_seconds=duration_seconds,
            prize=str(parsed.entities.get("prize", "Prize")),
        )
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\nEnds at: {payload['end_time']}\nUsers enter with /join"
        )
        return True

    if parsed.intent == Intent.GIVEAWAY_CANCEL:
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.end_giveaway(chat_id, message.from_user.id)
        if payload.get("winner_user_id"):
            await message.answer(
                f"Giveaway #{payload['giveaway_id']} ended.\nWinner: {payload['winner_user_id']}\nPrize: {payload['prize']}"
            )
        else:
            await message.answer(f"Giveaway ended with no winner. {payload.get('note')}")
        return True

    if parsed.intent == Intent.GIVEAWAY_REROLL:
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.reroll(chat_id, message.from_user.id)
        await message.answer(
            f"Reroll complete for giveaway #{payload['giveaway_id']}.\n"
            f"New winner: {payload['winner_user_id']} (prev: {payload.get('previous_winner_user_id')})"
        )
        return True

    if parsed.intent == Intent.WATCHLIST:
        payload = await hub.watchlist_service.build_watchlist(
            count=parsed.entities.get("count", 5),
            direction=parsed.entities.get("direction"),
        )
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="watchlist",
        )
        await message.answer(watchlist_template(payload))
        return True

    if parsed.intent == Intent.NEWS:
        if not hub.news_uc:
            raise RuntimeError("news use-case not configured")
        result = await hub.news_uc.get_digest(
            chat_id=chat_id,
            topic=parsed.entities.get("topic"),
            mode=str(parsed.entities.get("mode", "crypto")),
            limit=int(parsed.entities.get("limit", 6)),
        )
        if result is None:
            await message.answer(ux_busy("news"))
            return True
        if not result.payload:
            await message.answer(ux_transient_error("news"))
            return True
        payload = result.payload
        if result.degraded:
            await message.answer(ux_degraded("news"))
        headlines = payload.get("headlines") if isinstance(payload, dict) else None
        headline = headlines[0] if isinstance(headlines, list) and headlines else {}
        await deps.remember_source_context(
            chat_id,
            source_line=f"{headline.get('source', 'news feed')} | {headline.get('url', '')}".strip(),
            context="news",
        )
        await message.answer(news_template(payload), parse_mode="HTML")
        return True

    if parsed.intent == Intent.SCAN_WALLET:
        async with hub.cache.distributed_lock(f"user:{chat_id}:wallet_scan", ttl=40) as acquired:
            if not acquired:
                await message.answer("Wallet scan already running \u2014 try again in a few seconds.")
                return True
        limiter = await hub.rate_limiter.check(
            key=f"rl:scan:{chat_id}:{datetime.now(UTC).strftime('%Y%m%d%H')}",
            limit=deps.wallet_scan_limit_per_hour,
            window_seconds=3600,
        )
        if not limiter.allowed:
            await message.answer("Wallet scan limit reached for this hour.")
            return True
        result = await hub.wallet_service.scan(parsed.entities["chain"], parsed.entities["address"], chat_id=chat_id)
        await message.answer(
            wallet_scan_template(result),
            reply_markup=wallet_actions(parsed.entities["chain"], parsed.entities["address"]),
        )
        return True

    if parsed.intent == Intent.CYCLE:
        payload = await hub.cycles_service.cycle_check()
        await message.answer(cycle_template(payload, settings))
        return True

    if parsed.intent == Intent.TRADECHECK:
        async with hub.cache.distributed_lock(f"user:{chat_id}:tradecheck", ttl=40) as acquired:
            if not acquired:
                await message.answer("Trade check already running \u2014 try again in a few seconds.")
                return True
        timestamp = parsed.entities["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        data = {
            "symbol": parsed.entities["symbol"],
            "timeframe": parsed.entities.get("timeframe", "1h"),
            "timestamp": timestamp,
            "entry": float(parsed.entities["entry"]),
            "stop": float(parsed.entities["stop"]),
            "targets": [float(item) for item in parsed.entities["targets"]],
            "mode": "ambiguous",
        }
        result = await hub.trade_verify_service.verify(**data)
        await deps.save_trade_check(chat_id, data, result)
        await deps.remember_source_context(
            chat_id,
            source_line=str(result.get("source_line") or ""),
            symbol=data["symbol"],
            context="trade check",
        )
        await message.answer(trade_verification_template(result, settings))
        return True

    if parsed.intent == Intent.CORRELATION:
        payload = await hub.correlation_service.check_following(
            parsed.entities["symbol"],
            benchmark=parsed.entities.get("benchmark", "BTC"),
        )
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            symbol=parsed.entities["symbol"],
            context="correlation",
        )
        await message.answer(correlation_template(payload, settings))
        return True

    if parsed.intent == Intent.SETTINGS:
        user_settings = await hub.user_service.get_settings(chat_id)
        await message.answer(settings_text(user_settings), reply_markup=settings_menu(user_settings))
        return True

    if parsed.intent == Intent.HELP:
        await message.answer(help_text())
        return True

    return False
