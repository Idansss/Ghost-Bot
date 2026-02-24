from __future__ import annotations

import random
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from app.core.fmt import fmt_price, safe_html, strip_html_tags

# ---------------------------------------------------------------------------
# Personality pools
# ---------------------------------------------------------------------------

SMALLTALK_REPLIES = [
    "Ghost Alpha online. Scanning the tape and filtering noise.\nDrop a ticker, ask for news, or run an RSI scan.",
    "All systems green. Market is choppy but clean levels still print.\nSend a coin and direction.",
    "Quiet session. I'm watching liquidity and breakout zones.\nWhat are we hunting?",
    "Operational. Alerts armed, charts loaded.\nSend me a coin and direction.",
    "Lurking where clean entries live.\nWant analysis, watchlist, or news?",
    "Stable and tracking. If it moves, I'll see it.\nScalp or swing today?",
    "Locked in. Market is noisy, signal is not.\nTry <code>SOL long</code> or <code>ETH short</code>.",
    "Signal check complete. Give me a ticker and I'll map entry/targets fast.",
    "Ghost Alpha active. Watching BTC lead and alt reactions.\nWhat do you want first?",
    "Scanner is hot. Trade plans, alerts, wallet scans — your call.",
]

WILD_CLOSERS = [
    "drop the next ticker and i'll map it fast.",
    "want me to set the alert level now?",
    "keep it clean. one setup at a time.",
    "send the next chart if structure changes.",
    "if btc flips, reassess everything.",
    "stop is non-negotiable. don't move it.",
    "size it right — the best trade means nothing if you're overleveraged.",
    "partial out at tp1 and let the rest run.",
    "if it doesn't set up clean, skip it. there's always another trade.",
]

_LONG_CLOSERS = [
    "partial out at tp1 and let the rest run — don't give it all back.",
    "stop is non-negotiable. don't move it down.",
    "if btc loses the daily ema, your long thesis is over. reassess.",
    "size it correctly. the best setup means nothing overleveraged.",
]

_SHORT_CLOSERS = [
    "stop is above the wick — not above your feelings.",
    "shorts live and die by timing. if it doesn't reject cleanly, flat is a position.",
    "don't chase the short. wait for confirmation or you're just guessing.",
    "cover at least half at tp1. shorts can snap back hard.",
]

_THIN_RR_CLOSERS = [
    "r/r is tight on this. size it small or tighten the entry before you click.",
    "marginal r/r — only take it if the setup is textbook clean.",
]

STANDARD_CLOSERS = [
    "send the next chart if you want a second read.",
    "set the alert and let the market come to you.",
    "size correctly and manage the trade — that's the edge.",
    "re-evaluate if macro conditions change.",
]

UNKNOWN_FOLLOWUPS = [
    "Give me a clear target: <code>SOL long</code>, <code>cpi news</code>, <code>chart BTC 1h</code>, or <code>alert me when BTC hits 70000</code>.",
    "I can route free text. Try <code>ETH short 4h</code>, <code>rsi top 10 1h oversold</code>, or <code>list my alerts</code>.",
    "Send the intent directly: <code>coins to watch 5</code>, <code>openai updates</code>, or <code>scan solana &lt;address&gt;</code>.",
    "Short question? I'll keep it brief. Need a full read? Send a ticker or <code>BTC 4h</code>.",
    "Not sure what you need — try <code>BTC long</code>, <code>latest news</code>, or <code>coins to watch</code>. I'll match the depth of your ask.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tone_mode(settings: dict) -> str:
    if settings.get("formal_mode"):
        return "formal"
    tone = str(settings.get("tone_mode", "wild")).lower()
    return "wild" if tone not in {"formal", "standard", "wild"} else tone


def _pick(options: list[str]) -> str:
    return random.choice(options)


def _contextual_closer(direction: str, rr_first: float, tone: str) -> str:
    """Pick a closer that matches the trade direction and R/R quality."""
    if tone not in {"wild", "standard"}:
        return ""
    if tone == "standard":
        return _pick(STANDARD_CLOSERS)
    direction = (direction or "").strip().lower()
    pool: list[str]
    if direction == "short":
        pool = list(_SHORT_CLOSERS)
        if rr_first < 1.5:
            pool.extend(_THIN_RR_CLOSERS)
    elif direction == "long":
        pool = list(_LONG_CLOSERS)
        if rr_first < 1.5:
            pool.extend(_THIN_RR_CLOSERS)
        elif rr_first >= 2.5:
            pool.append("clean r/r on this long — let it breathe and don't cut early.")
    else:
        pool = list(WILD_CLOSERS)
    return _pick(pool)


def market_condition_warning(*, is_weekend: bool, btc_vol_pct: float | None = None) -> str | None:
    """Return a one-line warning string if market conditions are notable, else None."""
    parts: list[str] = []
    if is_weekend:
        parts.append("weekend session — liquidity is thin, spreads are wide, wicks are random. size accordingly.")
    if btc_vol_pct is not None and abs(btc_vol_pct) >= 4.0:
        direction = "pumping" if btc_vol_pct > 0 else "dumping"
        parts.append(f"btc is {direction} hard ({btc_vol_pct:+.1f}%) — alts will follow or won't. confirm before trading.")
    if not parts:
        return None
    return "⚠ " + " ".join(parts)


def _render_summary(summary: str, settings: dict) -> str:
    text = safe_html(summary.strip())
    if not settings.get("formal_mode"):
        prof = settings.get("profanity_level", "light")
        if prof == "medium":
            text = text.replace("volatile", "wild")
    return text


def relative_updated(ts_iso: str | None) -> str:
    if not ts_iso:
        return ""
    try:
        then = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - then.astimezone(timezone.utc)
        minutes = int(delta.total_seconds() // 60)
        return f"updated {minutes}m ago"
    except Exception:  # noqa: BLE001
        return ts_iso


def _clean_url(raw: str) -> str:
    if not raw:
        return "n/a"
    try:
        parts = urlsplit(raw)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:  # noqa: BLE001
        return raw


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def trade_plan_template(plan: dict, settings: dict, detailed: bool = False) -> str:
    symbol = safe_html(str(plan.get("symbol") or "asset").upper())
    side = str(plan.get("side") or "").strip().lower()
    price = plan.get("price")
    summary = _render_summary(plan.get("summary", "").strip(), settings)
    context = safe_html(str(plan.get("market_context_text") or "").strip())

    side_label = f" · {side}" if side in {"long", "short"} else ""
    header = f"<b>{symbol}{side_label}</b>"

    if isinstance(price, (int, float)):
        lead = f"{safe_html(fmt_price(float(price)))} — {summary}"
    else:
        lead = summary or f"{symbol} setup mapped."

    lines = [header, "", lead]
    if context:
        lines.append(f"<i>backdrop: {context}</i>")

    lines += [
        "",
        f"• <b>entry</b>   {safe_html(plan['entry'])}",
        f"• <b>target</b>  {safe_html(plan['tp1'])} → {safe_html(plan['tp2'])}",
        f"• <b>stop</b>    {safe_html(plan['sl'])}",
    ]

    why_items = plan.get("why", [])
    if why_items:
        lines.append("")
        for bullet in (why_items if detailed else why_items[:2]):
            lines.append(f"  — {safe_html(bullet)}")

    if detailed and plan.get("mtf_snapshot"):
        lines += ["", "<b>mtf:</b>"]
        for row in plan["mtf_snapshot"][:4]:
            lines.append(f"  {safe_html(row)}")

    if detailed and plan.get("input_notes"):
        lines.append("")
        for note in plan["input_notes"][:3]:
            lines.append(f"<i>note: {safe_html(note)}</i>")

    if detailed:
        updated = plan.get("updated_at")
        if updated:
            lines += ["", f"<i>{relative_updated(updated) or updated}</i>"]

    lines += ["", f"<i>{safe_html(plan.get('risk', 'Stay nimble and cut it if structure breaks.'))}</i>"]
    tone = _tone_mode(settings)
    closer = _contextual_closer(side, rr_first=0.0, tone=tone)
    if closer:
        lines.append(closer)
    return "\n".join(lines)


def watchlist_template(payload: dict) -> str:
    summary = safe_html(payload.get("summary", ""))
    lines = [
        "<b>Watchlist</b>",
        "",
        summary,
        "",
    ]
    for idx, item in enumerate(payload.get("items", []), start=1):
        lines.append(f"<b>{idx}.</b> {safe_html(item)}")
    return "\n".join(lines)


def news_template(payload: dict) -> str:
    # Strip LLM HTML so user never sees literal <b> or </b>; then escape for safety
    summary = safe_html(strip_html_tags(payload.get("summary", "")))
    vibe = safe_html(strip_html_tags(payload.get("vibe", "")))
    headlines = payload.get("headlines", [])[:5]

    lines = [
        "<b>News Digest</b>",
        "",
        summary,
    ]

    if not headlines:
        lines += ["", "No fresh items from configured feeds.", "", f"<i>{vibe}</i>"]
        return "\n".join(lines)

    lines.append("")
    for idx, item in enumerate(headlines, start=1):
        title = safe_html(item.get("title", ""))
        source = safe_html(item.get("source", ""))
        url = _clean_url(item.get("url", ""))
        lines.append(f"<b>{idx}. {title}</b>")
        if url and url != "n/a":
            lines.append(f'<i>{source}</i> · <a href="{url}">read →</a>')
        else:
            lines.append(f"<i>{source}</i>")
        lines.append("")

    updated = relative_updated(payload.get("updated_at")) or payload.get("updated_at", "")
    if updated:
        lines.append(f"<i>{safe_html(updated)}</i>")
    if vibe:
        lines.append(f"<i>{vibe}</i>")
    return "\n".join(lines)


def wallet_scan_template(payload: dict) -> str:
    chain = safe_html(str(payload.get("chain", "")).upper())
    address = safe_html(str(payload.get("address", "")))
    balance_line = f"{payload['native_balance']:.6f} {safe_html(payload['native_symbol'])}"
    if payload.get("native_usd") is not None:
        balance_line += f" (~${payload['native_usd']:.2f})"

    lines = [
        f"<b>{chain} Wallet Scan</b>",
        "",
        f"<code>{address}</code>",
        "",
        f"<b>Native balance</b>  {balance_line}",
    ]

    tokens = payload.get("tokens", [])[:8]
    if tokens:
        lines.append("")
        lines.append("<b>Top tokens:</b>")
        for token in tokens:
            sym = safe_html(token.get("symbol", "UNK"))
            amt = token.get("amount", 0)
            lines.append(f"  {sym}  {amt}")
    else:
        lines += ["", "No non-native holdings detected."]

    if payload.get("resources"):
        lines += ["", "<b>Resources:</b>"]
        for k, v in payload["resources"].items():
            lines.append(f"  {safe_html(k)}: {safe_html(str(v))}")

    tx_count = len(payload.get("recent_transactions", []))
    lines += ["", f"<b>Recent tx count</b>  {tx_count}"]

    warnings = payload.get("warnings", [])[:4]
    if warnings:
        lines += ["", "<b>Warnings:</b>"]
        for warn in warnings:
            lines.append(f"  ⚠ {safe_html(warn)}")
    return "\n".join(lines)


def cycle_template(payload: dict, settings: dict | None = None) -> str:
    tone = _tone_mode(settings or {})
    summary = safe_html(payload.get("summary", ""))
    confidence = payload.get("confidence", 0)
    if tone == "wild":
        header = "<b>Cycle Check</b>"
        conf_line = f"<i>conviction: {confidence:.0%}</i>"
    elif tone == "standard":
        header = "<b>Market Cycle Analysis</b>"
        conf_line = f"<i>Confidence: {confidence:.0%}</i>"
    else:
        header = "<b>Market Cycle Analysis</b>"
        conf_line = f"Confidence: {confidence:.0%}"
    lines = [header, "", summary, conf_line, ""]
    for b in payload.get("bullets", []):
        lines.append(f"— {safe_html(b)}")
    if tone == "wild":
        lines += ["", "<i>cycles repeat but timing is never identical. use as context, not as a signal.</i>"]
    return "\n".join(lines)


def trade_verification_template(payload: dict, settings: dict | None = None) -> str:
    tone = _tone_mode(settings or {})
    symbol = safe_html(str(payload.get("symbol", "")))
    if payload.get("result") == "not_filled":
        note_line = safe_html(payload.get("note", ""))
        if tone == "wild":
            return (
                f"<b>Trade Check · {symbol}</b>\n\n"
                f"result: <b>not filled</b> — entry never triggered.\n"
                f"{note_line}"
            )
        return (
            f"<b>Trade Check · {symbol}</b>\n\n"
            f"Result: <b>not filled</b>\n"
            f"{note_line}"
        )

    direction = safe_html(str(payload.get("direction", "")))
    result = safe_html(str(payload.get("result", "")))
    mode = safe_html(str(payload.get("mode", "")))

    lines = [
        f"<b>Trade Check · {symbol} {direction}</b>",
        f"Result: <b>{result}</b> · mode: {mode}",
        "",
        f"Filled at:   {safe_html(str(payload.get('filled_at', 'n/a')))}",
        f"First hit:   {safe_html(str(payload.get('first_hit', 'n/a')))}",
        f"MFE:         {safe_html(str(payload.get('mfe', 'n/a')))}",
        f"MAE:         {safe_html(str(payload.get('mae', 'n/a')))}",
        f"R multiple:  {safe_html(str(payload.get('r_multiple', 'n/a')))}",
    ]
    if tone == "wild":
        r_mult = payload.get("r_multiple")
        if r_mult is not None:
            try:
                r_float = float(str(r_mult).replace("R", "").strip())
                if r_float >= 2.0:
                    lines += ["", "<i>clean exit. that's how you build an account.</i>"]
                elif r_float >= 1.0:
                    lines += ["", "<i>took profit. that's the job.</i>"]
                elif r_float < 0:
                    lines += ["", "<i>it happens. review the entry, not just the result.</i>"]
            except (TypeError, ValueError):
                pass
    return "\n".join(lines)


def correlation_template(payload: dict, settings: dict | None = None) -> str:
    tone = _tone_mode(settings or {})
    summary = safe_html(payload.get("summary", ""))
    header = "<b>Correlation</b>" if tone == "wild" else "<b>Correlation Analysis</b>"
    lines = [header, "", summary, ""]
    for b in payload.get("bullets", []):
        lines.append(f"— {safe_html(b)}")
    if tone == "wild":
        lines += ["", "<i>correlation isn't causation. watch structure, not just the line.</i>"]
    elif tone == "standard":
        lines += ["", "<i>Use as context alongside your own analysis.</i>"]
    return "\n".join(lines)


def rsi_scan_template(payload: dict) -> str:
    tf = safe_html(str(payload.get("timeframe", "")))
    rsi_len = safe_html(str(payload.get("rsi_length", 14)))
    summary = safe_html(payload.get("summary", ""))
    # Never show "precomputed" or similar to the user
    for phrase in ("[precomputed]", "(precomputed)", "precomputed", "[live]", "pre-computed"):
        summary = summary.replace(phrase, "").strip()
    summary = summary.strip()

    items = payload.get("items", [])
    if not items:
        return (
            f"<b>RSI Scan · {tf} · RSI({rsi_len})</b>\n\n"
            f"{summary}\n\n"
            "No results for that request. Try another timeframe or symbol."
        )

    # Derive direction label from first item's note
    first_note = str(items[0].get("note", "")).lower()
    direction = "Overbought" if "overbought" in first_note else "Oversold" if "oversold" in first_note else "Scan"

    lines = [
        f"<b>RSI Scan · {tf} · RSI({rsi_len}) · {direction}</b>",
        "",
        summary,
        "",
    ]
    for idx, row in enumerate(items, start=1):
        sym = safe_html(str(row.get("symbol", "")))
        rsi_val = row.get("rsi", "—")
        note = safe_html(str(row.get("note", "")))
        try:
            rsi_fmt = f"{float(rsi_val):.1f}"
        except (TypeError, ValueError):
            rsi_fmt = safe_html(str(rsi_val))
        lines.append(f"<b>{idx}. {sym}</b>   RSI {rsi_fmt} — <i>{note}</i>")
    return "\n".join(lines)


def pair_find_template(payload: dict) -> str:
    query = safe_html(str(payload.get("query", "")))
    summary = safe_html(payload.get("summary", ""))
    matches = payload.get("matches", [])

    lines = [
        "<b>Pair Finder</b>",
        f"<i>query: {query}</i>",
        "",
        summary,
        "",
    ]
    if not matches:
        lines.append("No direct match. Try ticker, full name, or contract address.")
    else:
        for idx, row in enumerate(matches, start=1):
            sym = safe_html(str(row.get("symbol", "")))
            name = safe_html(str(row.get("name", "")))
            pair = safe_html(str(row.get("pair") or "n/a"))
            tradable = "✓" if row.get("tradable_binance") else "✗"
            price = row.get("price")
            price_txt = f"${price}" if price is not None else "n/a"
            lines.append(f"<b>{idx}. {sym}</b> <i>({name})</i>")
            lines.append(f"   Pair: <code>{pair}</code> · Tradable: {tradable} · ${safe_html(str(price_txt))}")
            lines.append("")
    lines.append("<i>Missing candles? I can still give narrative + execution rules.</i>")
    return "\n".join(lines)


def price_guess_template(payload: dict) -> str:
    summary = safe_html(payload.get("summary", ""))
    matches = payload.get("matches", [])

    lines = ["<b>Price Search</b>", "", summary, ""]
    if not matches:
        lines.append("No close candidates found. Try a wider hint or exact range.")
        return "\n".join(lines)

    for idx, row in enumerate(matches[:10], start=1):
        sym = safe_html(str(row.get("symbol", "")))
        name = safe_html(str(row.get("name", "")))
        tradable = "✓" if row.get("tradable_binance") else "✗"
        price = safe_html(str(row.get("price", "")))
        lines.append(f"<b>{idx}. {sym}</b> <i>({name})</i>  ${price} · Tradable: {tradable}")
    return "\n".join(lines)


def _fmt_price(v: object) -> str:
    """Format a price value cleanly (strip trailing zeros)."""
    try:
        f = float(str(v))
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1:
        return f"{f:.4f}".rstrip("0").rstrip(".")
    return f"{f:.8f}".rstrip("0").rstrip(".")


def setup_review_template(payload: dict, settings: dict) -> str:
    tone      = _tone_mode(settings)
    symbol    = safe_html(str(payload.get("symbol", "")))
    direction = safe_html(str(payload.get("direction", "")))
    verdict   = safe_html(str(payload.get("verdict", "")).upper())
    tf        = safe_html(str(payload.get("timeframe", "1h")))

    entry_val = _fmt_price(payload.get("entry", ""))
    stop_val  = _fmt_price(payload.get("stop", ""))
    targets   = payload.get("targets", [])
    rr_first  = float(payload.get("rr_first", 0) or 0)
    rr_best   = float(payload.get("rr_best", 0) or 0)
    stop_atr  = float(payload.get("stop_atr", 0) or 0)
    entry_ctx = safe_html(str(payload.get("entry_context", "")))
    stop_note = safe_html(str(payload.get("stop_note", "")))
    support   = payload.get("support")
    resistance = payload.get("resistance")

    # --- Opinionated one-liner about the setup ---
    if rr_first < 1.0:
        rr_comment = f"<b>your first TP gives sub-1:1 R/R ({rr_first:.2f}R) — that's a bad trade.</b> move it out or don't take it."
    elif rr_first < 1.5:
        rr_comment = f"first TP is marginal at {rr_first:.2f}R. acceptable if you partial and trail, but don't hold the whole bag."
    else:
        rr_comment = f"solid {rr_first:.2f}R to first target — this setup pays."

    if stop_atr < 0.8:
        stop_comment = f"stop is dangerously tight ({stop_atr:.2f} ATR) — market noise will hunt it."
    elif stop_atr > 4.0:
        stop_comment = f"stop is very wide ({stop_atr:.2f} ATR) — you're risking a lot of capital."
    else:
        stop_comment = ""

    sug = payload.get("suggested", {})

    def _sug_line(key: str, label: str) -> str:
        val = _fmt_price(sug.get(key, ""))
        why = safe_html(str(sug.get(f"{key}_why", "")))
        if why:
            return f"  {label}:  <b>${val}</b>  <i>({why})</i>"
        return f"  {label}:  <b>${val}</b>"

    lines = [
        f"<b>{symbol} {direction} — {verdict}</b>",
        "",
        f"entry:  <b>${safe_html(entry_val)}</b>",
        f"stop:   <b>${safe_html(stop_val)}</b>",
    ]
    for i, t in enumerate(targets, 1):
        lines.append(f"tp{i}:   <b>${safe_html(_fmt_price(t))}</b>")

    lines += [
        "",
        f"R/R:  first <b>{rr_first:.2f}R</b>  ·  best <b>{rr_best:.2f}R</b>",
        f"ATR ({tf}):  {stop_atr:.2f}",
        f"<i>{entry_ctx}</i>",
    ]
    if stop_comment:
        lines.append(f"<i>{stop_comment}</i>")

    lines += [
        "",
        f"<b>ghost's take:</b>  {rr_comment}",
    ]

    # Key levels context
    if support and resistance:
        lines.append(
            f"support <b>${safe_html(_fmt_price(support))}</b>  ·  resistance <b>${safe_html(_fmt_price(resistance))}</b>"
        )

    lines += [
        "",
        "<b>ghost's levels:</b>",
        _sug_line("entry", "entry"),
        _sug_line("stop",  "sl"),
        _sug_line("tp1",   "tp1"),
        _sug_line("tp2",   "tp2"),
    ]

    position = payload.get("position")
    if position:
        lines += [
            "",
            "<b>position sizing:</b>",
            f"  margin:    <b>${safe_html(str(position.get('margin_usd', '')))}</b>",
            f"  leverage:  <b>{safe_html(str(position.get('leverage', '')))}x</b>",
            f"  notional:  <b>${safe_html(str(position.get('notional_usd', '')))}</b>",
            f"  qty:       {safe_html(str(position.get('qty', '')))}",
            f"  stop PnL:  <b>${safe_html(str(position.get('stop_pnl_usd', '')))}</b>",
        ]
        for row in position.get("tp_pnls", [])[:3]:
            lines.append(
                f"  tp {safe_html(_fmt_price(row.get('tp', '')))}: "
                f"<b>${safe_html(str(row.get('pnl_usd', '')))}</b>"
            )

    size_note = safe_html(str(payload.get("size_note", "")).strip())
    if size_note:
        lines += ["", f"<i>{size_note}</i>"]

    lines += ["", "<i>use as a risk-planning map — execute only if structure still holds</i>"]

    closer = _contextual_closer(direction, rr_first=rr_first, tone=tone)
    if closer:
        lines.append(closer)
    return "\n".join(lines)


def trade_math_template(payload: dict, settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "wild":
        opener = "Trade math locked. Here's the risk map."
    elif tone == "standard":
        opener = "Trade math summary."
    else:
        opener = "Trade risk/reward summary."

    direction = safe_html(str(payload.get("direction", "")))
    entry = safe_html(str(payload.get("entry", "")))
    stop = safe_html(str(payload.get("stop", "")))
    targets = ", ".join(safe_html(str(x)) for x in payload.get("targets", []))
    risk_pu = safe_html(str(payload.get("risk_per_unit", "")))
    best_r = safe_html(str(payload.get("best_r", "")))

    lines = [
        "<b>Trade Math</b>",
        f"<i>{opener}</i>",
        "",
        f"Direction: {direction}",
        f"Entry:     {entry}",
        f"Stop:      {stop}",
        f"Targets:   {targets}",
        "",
        f"Risk/unit: {risk_pu}",
        f"Best R:    {best_r}",
    ]
    for row in payload.get("rows", [])[:4]:
        lines.append(f"  TP {safe_html(str(row.get('tp', '')))}: {safe_html(str(row.get('r_multiple', '')))}R")

    position = payload.get("position")
    if position:
        lines += [
            "",
            "<b>Position sizing:</b>",
            f"  Margin:    ${safe_html(str(position.get('margin_usd', '')))}",
            f"  Leverage:  {safe_html(str(position.get('leverage', '')))}x",
            f"  Notional:  ${safe_html(str(position.get('notional_usd', '')))}",
            f"  Qty:       {safe_html(str(position.get('qty', '')))}",
            f"  Stop PnL:  ${safe_html(str(position.get('stop_pnl_usd', '')))}",
        ]
        for row in position.get("tp_pnls", [])[:4]:
            lines.append(f"  TP {safe_html(str(row.get('tp', '')))}: ${safe_html(str(row.get('pnl_usd', '')))}")
    return "\n".join(lines)


def asset_unsupported_template(payload: dict, settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "wild":
        opener = "Can't chart this one clean right now."
        closer = "Send a chart link or contract address and I'll work with that."
    elif tone == "standard":
        opener = "Can't fetch full technical data for this asset right now."
        closer = "Try one of the alternatives above, or share chart/contract details."
    else:
        opener = "Data is currently unavailable for this asset."
        closer = "Please share additional context or another symbol."

    sym = safe_html(str(payload.get("symbol", "")))
    reason = safe_html(str(payload.get("reason", "")))
    narrative = safe_html(str(payload.get("narrative", "")))
    safe_action = safe_html(str(payload.get("safe_action", "")))

    lines = [
        f"<b>{sym}</b>",
        f"<i>{opener}</i>",
        "",
        f"Reason: {reason}",
        "",
        narrative,
        f"<i>{safe_action}</i>",
    ]
    if payload.get("headlines"):
        lines += ["", "<b>Recent context:</b>"]
        for item in payload["headlines"][:2]:
            lines.append(f"  — {safe_html(item.get('title', ''))}")
    if payload.get("alternatives"):
        alts = ", ".join(safe_html(a) for a in payload["alternatives"])
        lines += ["", f"Alternatives I can map now: {alts}"]
    lines += ["", closer]
    return "\n".join(lines)


def giveaway_status_template(payload: dict) -> str:
    if payload.get("active"):
        return (
            f"<b>Giveaway #{payload['id']} — ACTIVE</b>\n\n"
            f"Prize: {safe_html(str(payload['prize']))}\n"
            f"Participants: {payload['participants']}\n"
            f"Ends in: {payload['seconds_left']}s\n\n"
            "Use /join to enter."
        )
    if payload.get("message"):
        return safe_html(str(payload["message"]))
    winner = payload.get("winner_user_id")
    winner_txt = str(winner) if winner else "none"
    return (
        f"<b>Giveaway #{payload.get('id')} — {safe_html(str(payload.get('status', '')))}</b>\n\n"
        f"Prize: {safe_html(str(payload.get('prize', '')))}\n"
        f"Winner: {winner_txt}"
    )


def help_text() -> str:
    lines = [
        "<b>Ghost Alpha — Quick Reference</b>",
        "",
        "<b>Free-talk examples</b>",
        "<code>SOL long</code>",
        "<code>SOL 4h</code>",
        "<code>chart BTC 1h</code>",
        "<code>rsi top 10 1h oversold</code>",
        "<code>ema 200 4h top 10</code>",
        "<code>latest crypto news</code>",
        "<code>cpi news</code>",
        "<code>alert me when SOL hits 100</code>",
        "<code>list my alerts</code>",
        "<code>clear my alerts</code>",
        "<code>find pair xion</code>",
        "<code>coin around 0.155</code>",
        "<code>scan solana &lt;address&gt;</code>",
        "<code>check this trade from yesterday: ETH entry 2100 stop 2165 targets 2043 2027 1991 timeframe 1h</code>",
        "",
        "<b>Slash commands</b>",
        "<code>/alpha &lt;symbol&gt; [tf] [ema=..] [rsi=..]</code>",
        "<code>/watch &lt;symbol&gt; [tf]</code>",
        "<code>/chart &lt;symbol&gt; [tf]</code>",
        "<code>/heatmap &lt;symbol&gt;</code>",
        "<code>/rsi &lt;tf&gt; &lt;overbought|oversold&gt; [topN] [len]</code>",
        "<code>/ema &lt;ema_len&gt; &lt;tf&gt; [topN]</code>",
        "<code>/news [crypto|openai|cpi|fomc] [limit]</code>",
        "<code>/alert &lt;symbol&gt; &lt;price&gt; [above|below|cross]</code>",
        "<code>/alerts  /alertdel &lt;id&gt;  /alertclear [symbol]</code>",
        "<code>/findpair &lt;price_or_query&gt;</code>",
        "<code>/setup &lt;freeform setup text&gt;</code>",
        "<code>/scan &lt;chain&gt; &lt;address&gt;</code>",
        "<code>/tradecheck  /cycle</code>",
        "<code>/giveaway  /join</code>",
        "<code>/position ...</code> — track positions & unrealized PnL",
        "<code>/journal ...</code> — log trades, stats, export",
        "<code>/compare BTC ETH SOL</code> — quick price compare",
        "<code>/report on [HOUR] [MINUTE]</code> — daily market summary",
        "<code>/export alerts|journal</code>",
        "",
        "<b>How Ghost thinks about trading</b>",
        "Market structure, levels/zones, risk management, psychology, volatility/regimes, execution, top-down timeframes, trade planning, journaling & review, backtesting/forward testing, fundamentals & catalysts, on-chain & liquidity/flow, correlations, and basic security/ops.",
        "",
        "<i>Tip: free-talk works — commands are optional. Short question → short answer; send a ticker for a full read.</i>",
    ]
    return "\n".join(lines)


def settings_text(settings: dict) -> str:
    lines = [
        "<b>Settings</b>",
        "",
        f"risk_profile:          {safe_html(str(settings.get('risk_profile', '')))}",
        f"preferred_timeframe:   {safe_html(str(settings.get('preferred_timeframe', '')))}",
        f"preferred_timeframes:  {safe_html(str(settings.get('preferred_timeframes', '')))}",
        f"preferred_ema_periods: {safe_html(str(settings.get('preferred_ema_periods', '')))}",
        f"preferred_rsi_periods: {safe_html(str(settings.get('preferred_rsi_periods', '')))}",
        f"preferred_exchange:    {safe_html(str(settings.get('preferred_exchange', '')))}",
        f"tone_mode:             {safe_html(str(settings.get('tone_mode', '')))}",
        f"anon_mode:             {safe_html(str(settings.get('anon_mode', '')))}",
        f"profanity_level:       {safe_html(str(settings.get('profanity_level', '')))}",
        f"formal_mode:           {safe_html(str(settings.get('formal_mode', '')))}",
        f"timezone:              {safe_html(str(settings.get('timezone', 'UTC')))}",
        f"reply_in_dm:           {safe_html(str(settings.get('reply_in_dm', False)))}",
        f"ultra_brief:           {safe_html(str(settings.get('ultra_brief', False)))}",
        f"communication_style:   {safe_html(str(settings.get('communication_style', 'friendly')))}",
        f"display_name:          {safe_html(str(settings.get('display_name') or ''))}",
        f"trading_goals:         {safe_html(str(settings.get('trading_goals') or '')[:80])}",
    ]
    return "\n".join(lines)


def smalltalk_reply(settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "formal":
        return "Ghost Alpha is running normally.\nWhat would you like: analysis, alerts, wallet scan, or market news?"
    if tone == "standard":
        return "Ghost Alpha online.\nWant analysis, alerts, wallet scan, or market news?"
    return random.choice(SMALLTALK_REPLIES)


def unknown_prompt() -> str:
    return random.choice(UNKNOWN_FOLLOWUPS)


def clarifying_question(symbol_hint: str | None = None) -> str:
    """One clarifying question when intent is unclear; avoids guessing."""
    if symbol_hint:
        return f"Not sure what you need for <b>{symbol_hint}</b> — analysis, chart, or alert? Pick one or say which."
    return "What do you want — analysis (send a ticker), chart, alert, or news? One is enough."
