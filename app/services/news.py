from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.news_sources import NewsSourcesAdapter


class NewsService:
    def __init__(self, adapter: NewsSourcesAdapter) -> None:
        self.adapter = adapter

    async def get_daily_brief(self, limit: int = 7) -> dict:
        stories = await self.adapter.get_latest_news(limit=limit)
        themes = []
        if any("etf" in s["title"].lower() for s in stories):
            themes.append("ETF flow chatter is still steering sentiment")
        if any("fed" in s["title"].lower() or "rates" in s["title"].lower() for s in stories):
            themes.append("Macro rates narrative remains a volatility driver")
        if any("hack" in s["title"].lower() or "exploit" in s["title"].lower() for s in stories):
            themes.append("Security headlines are raising risk premium")
        if not themes:
            themes.append("Risk-on tone is mixed, watch BTC reaction levels")

        bullets = []
        for item in stories[:limit]:
            bullets.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "source": item["source"],
                    "published_at": item["published_at"],
                }
            )

        return {
            "summary": " | ".join(themes[:2]),
            "headlines": bullets,
            "vibe": "Vibe: reactive tape, momentum still headline-driven.",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def get_asset_headlines(self, symbol: str, limit: int = 3) -> list[dict]:
        symbol_u = symbol.upper()
        stories = await self.adapter.get_latest_news(limit=25)
        filtered = [s for s in stories if symbol_u in s["title"].upper()]
        if not filtered:
            filtered = stories[:limit]
        return filtered[:limit]
