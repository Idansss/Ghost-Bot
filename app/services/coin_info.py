"""Fetch coin fundamentals (24h high/low, ATH/ATL, cap, supply, FDV, links, about, Fear & Greed) from CoinGecko + Alternative.me."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.adapters.symbols import coingecko_id_for, normalize_symbol
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/"
FEAR_GREED_CACHE_TTL = 3600  # 1h
COIN_DETAILS_CACHE_TTL = 300   # 5 min


def _safe_float(obj: Any, default: float | None = None) -> float | None:
    if obj is None:
        return default
    try:
        return float(obj)
    except (TypeError, ValueError):
        return default


def _safe_str(obj: Any, max_len: int = 0) -> str:
    s = str(obj).strip() if obj is not None else ""
    if max_len and len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


class CoinInfoService:
    def __init__(
        self,
        http: ResilientHTTPClient,
        cache: RedisCache,
        coingecko_base: str,
    ) -> None:
        self.http = http
        self.cache = cache
        self.coingecko_base = (coingecko_base or "").rstrip("/")

    async def _resolve_coingecko_id(self, symbol: str) -> str | None:
        base = normalize_symbol(symbol).base
        cg_id = coingecko_id_for(base)
        if cg_id:
            return cg_id
        try:
            search = await self.http.get_json(
                f"{self.coingecko_base}/search",
                params={"query": base},
            )
            for item in search.get("coins", []):
                if (item.get("symbol") or "").upper() == base:
                    return item.get("id")
        except Exception as exc:  # noqa: BLE001
            logger.warning("coin_info_search_failed", extra={"symbol": base, "error": str(exc)})
        return None

    async def get_fear_greed(self) -> dict[str, Any]:
        """Crypto Fear & Greed Index (0–100, classification)."""
        cache_key = "coin_info:fear_greed"
        cached = await self.cache.get_json(cache_key)
        if cached:
            return cached
        try:
            data = await self.http.get_json(f"{FEAR_GREED_URL}?format=json")
            arr = data.get("data") or []
            if not arr:
                return {"value": None, "classification": None}
            item = arr[0]
            value = _safe_float(item.get("value"))
            classification = _safe_str(item.get("value_classification"))
            out = {"value": value, "classification": classification or None}
            await self.cache.set_json(cache_key, out, ttl=FEAR_GREED_CACHE_TTL)
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("fear_greed_fetch_failed", extra={"error": str(exc)})
            return {"value": None, "classification": None}

    async def get_coin_info(self, symbol: str) -> dict[str, Any] | None:
        """
        Fetch fundamentals for a coin. Returns dict with:
        name, symbol, 24h_high, 24h_low, ath, atl, market_cap, circulating_supply,
        total_supply, max_supply, fdv, market_cap_fdv_ratio, total_volume,
        website, explorers, social (twitter, telegram, reddit, etc.), about.
        """
        base = normalize_symbol(symbol).base
        cache_key = f"coin_info:details:{base}"
        cached = await self.cache.get_json(cache_key)
        if cached:
            return cached

        cg_id = await self._resolve_coingecko_id(base)
        if not cg_id:
            return None

        try:
            payload = await self.http.get_json(
                f"{self.coingecko_base}/coins/{cg_id}",
                params={"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("coin_info_fetch_failed", extra={"symbol": base, "cg_id": cg_id, "error": str(exc)})
            return None

        md = payload.get("market_data") or {}
        links = payload.get("links") or {}
        raw_desc = (payload.get("description") or {}).get("en") or ""
        desc = re.sub(r"<[^>]+>", " ", raw_desc)
        desc = re.sub(r"\s+", " ", desc).strip()

        def usd(key: str) -> float | None:
            return _safe_float((md.get(key) or {}).get("usd"))

        ath = usd("ath")
        atl = usd("atl")
        high_24h = usd("high_24h")
        low_24h = usd("low_24h")
        market_cap = usd("market_cap")
        fdv = _safe_float(md.get("fully_diluted_valuation", {}).get("usd"))
        total_volume = usd("total_volume")
        circ = _safe_float(md.get("circulating_supply"))
        total_supply = _safe_float(md.get("total_supply"))
        max_supply = _safe_float(md.get("max_supply"))

        cap_fdv_ratio = None
        if market_cap and fdv and fdv > 0:
            cap_fdv_ratio = round(market_cap / fdv, 4)

        homepage = _safe_str(links.get("homepage") or "", 500)
        if isinstance(links.get("homepage"), list):
            homepage = _safe_str((links.get("homepage") or [None])[0], 500)

        blockchain_sites = links.get("blockchain_site") or []
        explorers = [x for x in blockchain_sites if x][:5]

        social: dict[str, str] = {}
        if links.get("twitter_screen_name"):
            social["twitter"] = f"https://twitter.com/{links['twitter_screen_name']}"
        if links.get("telegram_channel_identifier"):
            social["telegram"] = f"https://t.me/{links['telegram_channel_identifier']}"
        if links.get("subreddit_url"):
            social["reddit"] = _safe_str(links["subreddit_url"], 200)
        if links.get("facebook_username"):
            social["facebook"] = f"https://facebook.com/{links['facebook_username']}"
        if links.get("discord"):
            social["discord"] = _safe_str(links["discord"], 200)

        # Treasury: CoinGecko doesn't have a standard field; some coins have it in description or we leave null
        treasury_holdings = None  # optional future: scrape or another API

        out = {
            "name": _safe_str(payload.get("name"), 100),
            "symbol": base,
            "coingecko_id": cg_id,
            "high_24h": high_24h,
            "low_24h": low_24h,
            "ath": ath,
            "atl": atl,
            "market_cap": market_cap,
            "circulating_supply": circ,
            "total_supply": total_supply,
            "max_supply": max_supply,
            "fdv": fdv,
            "market_cap_fdv_ratio": cap_fdv_ratio,
            "total_volume": total_volume,
            "website": homepage or None,
            "explorers": explorers[:5] if explorers else [],
            "social": social,
            "about": _safe_str(desc, 2000) if desc else None,
            "treasury_holdings": treasury_holdings,
        }
        await self.cache.set_json(cache_key, out, ttl=COIN_DETAILS_CACHE_TTL)
        return out
