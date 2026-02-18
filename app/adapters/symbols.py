from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SymbolMeta:
    input_symbol: str
    base: str
    quote: str = "USDT"

    @property
    def pair(self) -> str:
        return f"{self.base}{self.quote}"


COINGECKO_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "TRX": "tron",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
}


def normalize_symbol(symbol: str) -> SymbolMeta:
    s = symbol.strip().upper()
    aliases = {
        "XBT": "BTC",
        "BITCOIN": "BTC",
        "ETHEREUM": "ETH",
        "SOLANA": "SOL",
    }
    s = aliases.get(s, s)
    if s.endswith("USDT"):
        s = s[:-4]
    return SymbolMeta(input_symbol=symbol, base=s)


def coingecko_id_for(symbol: str) -> str | None:
    return COINGECKO_MAP.get(normalize_symbol(symbol).base)
