from __future__ import annotations

import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from aiogram.enums import ChatAction
from aiogram.types import BufferedInputFile

from app.bot.keyboards import alert_created_menu
from app.bot.templates import (
    giveaway_status_template,
    news_template,
    pair_find_template,
    price_guess_template,
    rsi_scan_template,
    setup_review_template,
    smalltalk_reply,
    trade_math_template,
    watchlist_template,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutedIntentDependencies:
    hub: Any
    openai_router_min_confidence: float
    bot_meta_re: re.Pattern[str]
    try_answer_howto: Callable[[str], str | None]
    llm_fallback_reply: Callable[[str, dict | None, int | None], Awaitable[str | None]]
    llm_market_chat_reply: Callable[[str, dict | None, int | None], Awaitable[str | None]]
    send_llm_reply: Callable[..., Awaitable[None]]
    as_int: Callable[[Any, int], int]
    as_float: Callable[[Any, float | None], float | None]
    as_float_list: Callable[[Any], list[float]]
    extract_symbol: Callable[[dict], str | None]
    normalize_symbol_value: Callable[[Any], str | None]
    analysis_timeframes_from_settings: Callable[[dict], list[str]]
    parse_int_list: Callable[[Any, list[int]], list[int]]
    append_last_symbol: Callable[[int, str], Awaitable[None]]
    remember_analysis_context: Callable[[int, str, str | None, dict], Awaitable[None]]
    remember_source_context: Callable[..., Awaitable[None]]
    render_analysis_text: Callable[..., Awaitable[str]]
    send_ghost_analysis: Callable[..., Awaitable[None]]
    safe_exc: Callable[[Exception], str]
    parse_duration_to_seconds: Callable[[str], int | None]
    trade_math_payload: Callable[..., dict]
    feature_flags_set: Callable[[], set[str]]


async def execute_routed_intent(
    *,
    message,
    settings: dict,
    route: dict,
    deps: RoutedIntentDependencies,
) -> bool:
    intent = str(route.get("intent", "")).strip().lower()
    try:
        confidence = float(route.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    params = route.get("params") if isinstance(route.get("params"), dict) else {}
    chat_id = message.chat.id
    raw_text = message.text or ""
    hub = deps.hub

    if confidence < deps.openai_router_min_confidence:
        return False

    if intent in {"smalltalk", "market_chat", "general_chat"}:
        with suppress(Exception):
            await message.bot.send_chat_action(chat_id, ChatAction.TYPING)
        if deps.bot_meta_re.search(raw_text):
            howto = deps.try_answer_howto(raw_text)
            if howto:
                await message.answer(howto)
                return True
            llm_reply = await deps.llm_fallback_reply(raw_text, settings, chat_id=chat_id)
            await deps.send_llm_reply(
                message,
                llm_reply or smalltalk_reply(settings),
                settings,
                user_message=raw_text,
                analytics={
                    "route": intent,
                    "reply_kind": "bot_meta",
                    "router_intent": intent,
                    "router_confidence": confidence,
                },
            )
            return True

        llm_reply = await deps.llm_market_chat_reply(raw_text, settings, chat_id=chat_id)
        if llm_reply:
            await deps.send_llm_reply(
                message,
                llm_reply,
                settings,
                user_message=raw_text,
                analytics={
                    "route": intent,
                    "reply_kind": "market_chat",
                    "router_intent": intent,
                    "router_confidence": confidence,
                },
            )
        return True

    if intent == "news_digest":
        limit = max(3, min(deps.as_int(params.get("limit"), 6), 10))
        topic = params.get("topic")
        symbol_param = params.get("symbol") or topic
        if isinstance(symbol_param, str) and 2 <= len(symbol_param.strip()) <= 10 and symbol_param.strip().isalnum():
            ticker = symbol_param.strip().upper()
            headlines = await hub.news_service.get_asset_headlines(ticker, limit=limit)
            if headlines:
                lines = [f"<b>News for {ticker}</b>", ""]
                for item in headlines:
                    title = (item.get("title") or "")[:120]
                    url = item.get("url") or ""
                    source = item.get("source") or ""
                    lines.append(f"• {title}" + (f" ({source})" if source else "") + (f"\n  {url}" if url else ""))
                await message.answer("\n".join(lines), disable_web_page_preview=True)
            else:
                await message.answer(f"No recent headlines for <b>{ticker}</b>. Try general /news.")
            return True
        mode = str(params.get("mode") or "crypto").strip().lower() or "crypto"
        if isinstance(topic, str) and topic.strip().lower() == "openai" and mode == "crypto":
            mode = "openai"
        payload = await hub.news_service.get_digest(
            topic=topic if isinstance(topic, str) else None,
            mode=mode,
            limit=limit,
        )
        headlines = payload.get("headlines") if isinstance(payload, dict) else None
        headline = headlines[0] if isinstance(headlines, list) and headlines else {}
        await deps.remember_source_context(
            chat_id,
            source_line=f"{headline.get('source', 'news feed')} | {headline.get('url', '')}".strip(),
            context="news",
        )
        await message.answer(news_template(payload), parse_mode="HTML")
        return True

    if intent in {"watch_asset", "market_analysis"}:
        symbol = deps.extract_symbol(params)
        if not symbol:
            await message.answer("Which coin should I analyze?")
            return True
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        side = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        payload = await hub.analysis_service.analyze(
            symbol,
            direction=side if side in {"long", "short"} else None,
            timeframe=timeframe,
            timeframes=[timeframe] if timeframe else deps.analysis_timeframes_from_settings(settings),
            ema_periods=deps.parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=deps.parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=bool(params.get("include_derivatives") or params.get("derivatives")),
            include_news=bool(params.get("include_news") or params.get("news") or params.get("catalysts")),
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await deps.append_last_symbol(chat_id, symbol)
        await deps.remember_analysis_context(chat_id, symbol, side, payload)
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await deps.render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=side,
            settings=settings,
            chat_id=chat_id,
        )
        await deps.send_ghost_analysis(message, symbol, analysis_text, direction=side)
        return True

    if intent == "rsi_scan":
        symbol = deps.extract_symbol(params)
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        mode_raw = str(params.get("mode", "oversold")).strip().lower()
        mode = "overbought" if mode_raw == "overbought" else "oversold"
        limit = max(1, min(deps.as_int(params.get("limit"), 10), 20))
        rsi_length = max(2, min(deps.as_int(params.get("rsi_length"), 14), 50))
        payload = await hub.rsi_scanner_service.scan(
            timeframe=timeframe,
            mode=mode,
            limit=limit,
            rsi_length=rsi_length,
            symbol=symbol,
        )
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            symbol=symbol,
            context="rsi scan",
        )
        await message.answer(rsi_scan_template(payload))
        return True

    if intent == "ema_scan":
        timeframe = str(params.get("timeframe", "4h")).strip() or "4h"
        ema_length = max(2, min(deps.as_int(params.get("ema_length"), 200), 500))
        mode_raw = str(params.get("mode", "closest")).strip().lower()
        mode = mode_raw if mode_raw in {"closest", "above", "below"} else "closest"
        limit = max(1, min(deps.as_int(params.get("limit"), 10), 20))
        payload = await hub.ema_scanner_service.scan(
            timeframe=timeframe,
            ema_length=ema_length,
            mode=mode,
            limit=limit,
        )
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
        await message.answer("\n".join(lines))
        return True

    if intent == "chart":
        symbol = deps.extract_symbol(params)
        if not symbol:
            await message.answer("Which symbol should I chart?")
            return True
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        img, meta = await hub.chart_service.render_chart(symbol=symbol, timeframe=timeframe)
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

    if intent == "heatmap":
        symbol = deps.extract_symbol(params) or "BTC"
        img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
        caption = (
            f"{meta['pair']} orderbook heatmap\n"
            f"Best bid: {meta['best_bid']:.6f} | Best ask: {meta['best_ask']:.6f}\n"
            f"Bid wall: {meta['bid_wall']:.6f} | Ask wall: {meta['ask_wall']:.6f}"
        )
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
            caption=caption,
        )
        return True

    if intent == "watchlist":
        count = max(1, min(deps.as_int(params.get("count"), 5), 20))
        direction_raw = str(params.get("direction") or "").strip().lower()
        direction = direction_raw if direction_raw in {"long", "short"} else None
        payload = await hub.watchlist_service.build_watchlist(count=count, direction=direction)
        await deps.remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="watchlist",
        )
        await message.answer(watchlist_template(payload))
        return True

    if intent == "alert_create":
        symbol = deps.normalize_symbol_value(deps.extract_symbol(params) or params.get("symbol") or params.get("asset"))
        price = deps.as_float(params.get("price") or params.get("target_price"))
        if not symbol or price is None:
            from app.core.nlu import _extract_prices, _extract_symbols

            if not symbol:
                symbols = _extract_symbols(raw_text)
                symbol = deps.normalize_symbol_value(symbols[0]) if symbols else None
            if price is None:
                prices = _extract_prices(raw_text)
                price = prices[0] if prices else None

        if not symbol or price is None:
            await message.answer("Need symbol and price — e.g. <code>alert BTC 66k</code> or <code>set alert for SOL 200</code>.")
            return True
        operator = str(params.get("operator") or params.get("condition") or "cross").strip().lower()
        if operator in {">", ">=", "above", "gt", "gte", "crosses above", "cross above"}:
            condition = "above"
        elif operator in {"<", "<=", "below", "lt", "lte", "crosses below", "cross below"}:
            condition = "below"
        else:
            condition = "cross"
        try:
            alert = await hub.alerts_service.create_alert(chat_id, symbol, condition, float(price))
        except RuntimeError as exc:
            await message.answer(f"couldn't set that alert — {deps.safe_exc(exc)}")
            return True
        except Exception:
            logger.exception("alert_create_failed", extra={"chat_id": chat_id, "symbol": symbol, "price": price})
            await message.answer("alert creation failed. try again in a sec.")
            return True
        await deps.remember_source_context(
            chat_id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        cond_word = {"above": "crosses above", "below": "crosses below"}.get(condition, "crosses")
        await message.answer(
            f"🔔 alert set — <b>{alert.symbol}</b> {cond_word} <b>${float(price):,.2f}</b>.\n"
            "i'll ping you the moment it hits. don't get liquidated.",
            reply_markup=alert_created_menu(alert.symbol),
        )
        return True

    if intent == "alert_list":
        alerts = await hub.alerts_service.list_alerts(chat_id)
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

    if intent == "alert_delete":
        symbol = deps.extract_symbol(params)
        if symbol:
            count = await hub.alerts_service.delete_alerts_by_symbol(chat_id, symbol)
            await message.answer(f"Removed {count} alert(s) for {symbol}.")
            return True
        alert_id = deps.as_int(params.get("id") or params.get("alert_id"), 0)
        if alert_id <= 0:
            await message.answer("Which alert id should I delete?")
            return True
        ok = await hub.alerts_service.delete_alert(chat_id, alert_id)
        await message.answer("Deleted." if ok else "Alert not found.")
        return True

    if intent == "alert_clear":
        count = await hub.alerts_service.clear_user_alerts(chat_id)
        await message.answer(f"Cleared {count} alerts.")
        return True

    if intent == "pair_find":
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            query = deps.extract_symbol(params)
        if not isinstance(query, str) or not query.strip():
            await message.answer("Which coin should I resolve to a pair?")
            return True
        payload = await hub.discovery_service.find_pair(query.strip())
        await message.answer(pair_find_template(payload))
        return True

    if intent == "price_guess":
        price = deps.as_float(params.get("price") or params.get("target_price"))
        if price is None:
            await message.answer("What price should I search around?")
            return True
        limit = max(1, min(deps.as_int(params.get("limit"), 10), 20))
        payload = await hub.discovery_service.guess_by_price(price, limit=limit)
        await message.answer(price_guess_template(payload))
        return True

    if intent == "setup_review":
        symbol = deps.extract_symbol(params)
        entry = deps.as_float(params.get("entry"))
        stop = deps.as_float(params.get("stop") or params.get("sl"))
        targets = deps.as_float_list(params.get("targets") or params.get("tp"))
        leverage = deps.as_float(params.get("leverage"))
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
        direction = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        amount_usd = deps.as_float(params.get("amount") or params.get("amount_usd") or params.get("margin"))
        payload = await hub.setup_review_service.review(
            symbol=symbol,
            timeframe=timeframe,
            entry=float(entry),
            stop=float(stop),
            targets=[float(x) for x in targets],
            direction=direction if direction in {"long", "short"} else None,
            amount_usd=amount_usd,
            leverage=leverage,
        )
        await message.answer(setup_review_template(payload, settings))
        service = getattr(hub, "trade_journal_service", None)
        if service and "journal" in deps.feature_flags_set():
            with suppress(Exception):
                await service.log_trade(
                    chat_id=chat_id,
                    symbol=symbol,
                    side=direction or "long",
                    entry=float(entry),
                    stop=float(stop),
                    targets=[float(x) for x in targets],
                    outcome="pending",
                    notes=f"auto-logged from /setup tf={timeframe}",
                )
        return True

    if intent == "trade_math":
        entry = deps.as_float(params.get("entry"))
        stop = deps.as_float(params.get("stop") or params.get("sl"))
        targets = deps.as_float_list(params.get("targets") or params.get("tp"))
        if entry is None or stop is None or not targets:
            await message.answer("Send entry, stop, and target(s), e.g. `entry 100 sl 95 tp 110`.")
            return True
        symbol = deps.extract_symbol(params)
        side = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        margin_usd = deps.as_float(params.get("amount") or params.get("amount_usd") or params.get("margin"))
        leverage = deps.as_float(params.get("leverage"))
        payload = deps.trade_math_payload(
            entry=float(entry),
            stop=float(stop),
            targets=[float(x) for x in targets],
            direction=side,
            margin_usd=margin_usd,
            leverage=leverage,
            symbol=symbol,
        )
        await message.answer(trade_math_template(payload, settings))
        return True

    if intent == "giveaway_join":
        if not message.from_user:
            await message.answer("Could not identify user for giveaway join.")
            return True
        payload = await hub.giveaway_service.join_active(chat_id, message.from_user.id)
        await message.answer(f"Joined giveaway #{payload['giveaway_id']}. Participants: {payload['participants']}")
        return True

    if intent == "giveaway_status":
        payload = await hub.giveaway_service.status(chat_id)
        await message.answer(giveaway_status_template(payload))
        return True

    if intent == "giveaway_end":
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

    if intent == "giveaway_reroll":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.reroll(chat_id, message.from_user.id)
        await message.answer(
            f"Reroll complete for giveaway #{payload['giveaway_id']}.\nNew winner: {payload['winner_user_id']}"
        )
        return True

    if intent == "giveaway_cancel":
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

    if intent == "giveaway_start":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        duration = params.get("duration") or params.get("duration_text") or "10m"
        duration_seconds = None
        if isinstance(duration, (int, float)):
            duration_seconds = max(30, int(duration))
        elif isinstance(duration, str):
            duration_seconds = deps.parse_duration_to_seconds(duration)
        if duration_seconds is None:
            await message.answer("Give a duration like 10m or 1h for giveaway start.")
            return True
        prize = str(params.get("prize") or "Prize").strip() or "Prize"
        payload = await hub.giveaway_service.start_giveaway(
            group_chat_id=chat_id,
            admin_chat_id=message.from_user.id,
            duration_seconds=duration_seconds,
            prize=prize,
        )
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\nEnds at: {payload['end_time']}\nUsers enter with /join"
        )
        return True

    return False
