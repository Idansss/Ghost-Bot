from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def analysis_actions(symbol: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Set alert", callback_data=f"set_alert:{symbol}")
    kb.button(text="Show levels", callback_data=f"show_levels:{symbol}")
    kb.button(text="Why?", callback_data=f"why:{symbol}")
    kb.button(text="Refresh", callback_data=f"refresh:{symbol}")
    kb.button(text="More detail", callback_data=f"details:{symbol}")
    kb.button(text="Derivatives", callback_data=f"derivatives:{symbol}")
    kb.button(text="News", callback_data=f"catalysts:{symbol}")
    kb.button(text="Backtest this", callback_data=f"backtest:{symbol}")
    kb.adjust(2, 2, 2, 1, 1)
    return kb.as_markup()


def wallet_actions(chain: str, address: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Save wallet", callback_data=f"save_wallet:{chain}:{address}")
    kb.adjust(1)
    return kb.as_markup()


def settings_menu(settings: dict) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()

    anon = "ON" if settings.get("anon_mode") else "OFF"
    formal = "ON" if settings.get("formal_mode") else "OFF"
    profanity = settings.get("profanity_level", "light")

    kb.button(text=f"Anon mode: {anon}", callback_data="settings:toggle:anon_mode")
    kb.button(text=f"Formal mode: {formal}", callback_data="settings:toggle:formal_mode")

    for level in ("none", "light", "medium"):
        kb.button(text=f"Profanity: {level}", callback_data=f"settings:set:profanity_level:{level}")

    for risk in ("low", "medium", "high"):
        kb.button(text=f"Risk: {risk}", callback_data=f"settings:set:risk_profile:{risk}")

    for tf in ("15m", "1h", "4h"):
        kb.button(text=f"TF: {tf}", callback_data=f"settings:set:preferred_timeframe:{tf}")

    for ex in ("binance", "bybit", "okx"):
        kb.button(text=f"EX: {ex}", callback_data=f"settings:set:preferred_exchange:{ex}")

    for tone in ("wild", "standard"):
        kb.button(text=f"Tone: {tone}", callback_data=f"settings:set:tone_mode:{tone}")

    kb.adjust(2, 3, 3, 3, 2)
    return kb.as_markup()


def simple_followup(options: list[tuple[str, str]]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for text, value in options:
        kb.button(text=text, callback_data=value)
    kb.adjust(len(options))
    return kb.as_markup()


def smart_action_menu(symbol: str | None = None) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    if symbol:
        sym = symbol.upper()
        kb.button(text=f"{sym} Analyze 1h", callback_data=f"quick:analysis_tf:{sym}:1h")
        kb.button(text=f"{sym} Analyze 4h", callback_data=f"quick:analysis_tf:{sym}:4h")
        kb.button(text=f"{sym} Chart 1h", callback_data=f"quick:chart:{sym}:1h")
        kb.button(text=f"{sym} Heatmap", callback_data=f"quick:heatmap:{sym}")
        kb.button(text=f"{sym} Set Alert", callback_data=f"set_alert:{sym}")
        kb.button(text="Top Overbought 1h", callback_data="quick:rsi:overbought:1h:5")
        kb.button(text="Top Oversold 1h", callback_data="quick:rsi:oversold:1h:5")
        kb.button(text=f"{sym} News", callback_data=f"catalysts:{sym}")
        kb.adjust(2, 2, 2, 2)
        return kb.as_markup()

    kb.button(text="BTC Analyze 1h", callback_data="quick:analysis_tf:BTC:1h")
    kb.button(text="ETH Analyze 1h", callback_data="quick:analysis_tf:ETH:1h")
    kb.button(text="SOL Analyze 1h", callback_data="quick:analysis_tf:SOL:1h")
    kb.button(text="Top Overbought 1h", callback_data="quick:rsi:overbought:1h:5")
    kb.button(text="Top Oversold 1h", callback_data="quick:rsi:oversold:1h:5")
    kb.button(text="Crypto News", callback_data="quick:news:crypto")
    kb.button(text="OpenAI Updates", callback_data="quick:news:openai")
    kb.adjust(2, 2, 1, 2)
    return kb.as_markup()
