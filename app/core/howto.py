from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class HowToAnswer:
    key: str
    patterns: tuple[re.Pattern[str], ...]
    answer_html: str


def _re(p: str) -> re.Pattern[str]:
    return re.compile(p, re.IGNORECASE)


_ANSWERS: tuple[HowToAnswer, ...] = (
    HowToAnswer(
        key="help_overview",
        patterns=(
            _re(r"\b(help|what can you do|what commands|commands)\b"),
        ),
        answer_html=(
            "<b>quick commands</b>\n"
            "• <b>analysis</b>: \"BTC long\" / \"ETH short 4h\"\n"
            "• <b>alerts</b>: \"alert btc 100000 above\" | \"alerts\" | \"alertdel 12\"\n"
            "• <b>news</b>: \"latest crypto news\" / \"openai news\"\n"
            "• <b>price</b>: /price BTC\n"
            "• <b>chart</b>: /chart BTC 1h\n"
            "• <b>scans</b>: /rsi 1h | /ema 200 4h\n"
            "• <b>watchlist</b>: /watchlist\n\n"
            "tell me what you want (trade idea, levels, news, alerts) and i’ll route it.\n"
        ),
    ),
    HowToAnswer(
        key="alerts_create",
        patterns=(
            _re(r"\b(how do i|how to)\b.*\b(alert|alerts|set alert|create alert)\b"),
            _re(r"\bcreate alert\b"),
        ),
        answer_html=(
            "<b>set an alert</b>\n"
            "type it like this:\n"
            "• <code>alert btc 100000 above</code>\n"
            "• <code>alert eth 2500 below</code>\n\n"
            "<b>manage</b>\n"
            "• <code>alerts</code> = list\n"
            "• <code>alertdel 12</code> = delete by id\n"
            "• <code>alertclear</code> = clear all (or <code>alertclear BTC</code>)\n"
        ),
    ),
    HowToAnswer(
        key="not_responding",
        patterns=(
            _re(r"\b(not responding|bot down|not working|doesn'?t work|broken)\b"),
            _re(r"\bwhy\b.*\b(not responding|not working|broken|failing)\b"),
        ),
        answer_html=(
            "<b>if i’m not responding</b>\n"
            "• check your message sent (not “failed”)\n"
            "• try <code>/start</code>\n"
            "• try a simple command: <code>/price BTC</code>\n"
            "• if buttons don’t work, resend the command (telegram can drop callbacks)\n\n"
            "if it’s still dead, tell me: what command you tried + what chat (dm vs group) + any error text.\n"
        ),
    ),
    HowToAnswer(
        key="webhook_mode",
        patterns=(
            _re(r"\b(webhook|serverless|vercel)\b"),
        ),
        answer_html=(
            "<b>serverless mode (vercel)</b>\n"
            "you need webhook mode enabled.\n"
            "set: <code>SERVERLESS_MODE=true</code>, <code>TELEGRAM_USE_WEBHOOK=true</code>, "
            "<code>TELEGRAM_WEBHOOK_URL</code> to your deployed domain.\n"
        ),
    ),
    HowToAnswer(
        key="privacy_data",
        patterns=(
            _re(r"\b(my data|privacy|delete my data|delete account|gdpr)\b"),
        ),
        answer_html=(
            "<b>your data</b>\n"
            "• <code>/mydata</code> export what i have\n"
            "• <code>/deleteaccount</code> deletes your stored data\n"
        ),
    ),
)


def try_answer_howto(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    for a in _ANSWERS:
        if any(p.search(t) for p in a.patterns):
            return a.answer_html
    return None

