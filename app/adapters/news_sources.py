from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser

from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


class NewsSourcesAdapter:
    def __init__(self, http: ResilientHTTPClient, cache: RedisCache, rss_feeds: list[str], cryptopanic_key: str) -> None:
        self.http = http
        self.cache = cache
        self.rss_feeds = rss_feeds
        self.cryptopanic_key = cryptopanic_key

    async def _fetch_rss_feed(self, url: str) -> list[dict]:
        parsed = await asyncio.to_thread(feedparser.parse, url)
        items: list[dict] = []
        for entry in parsed.entries[:12]:
            published = entry.get("published") or entry.get("updated")
            dt = None
            if published:
                try:
                    dt = parsedate_to_datetime(published).astimezone(timezone.utc)
                except Exception:  # noqa: BLE001
                    dt = None
            items.append(
                {
                    "title": entry.get("title", "Untitled"),
                    "url": entry.get("link", ""),
                    "source": parsed.feed.get("title", "rss"),
                    "published_at": (dt or datetime.now(timezone.utc)).isoformat(),
                }
            )
        return items

    async def _fetch_cryptopanic(self) -> list[dict]:
        if not self.cryptopanic_key:
            return []
        data = await self.http.get_json(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": self.cryptopanic_key, "public": "true"},
        )
        rows = []
        for item in data.get("results", [])[:10]:
            rows.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "source": "CryptoPanic",
                    "published_at": item.get("published_at", datetime.now(timezone.utc).isoformat()),
                }
            )
        return rows

    async def get_latest_news(self, limit: int = 10) -> list[dict]:
        cache_key = "news:today"
        cached = await self.cache.get_json(cache_key)
        if cached:
            return cached[:limit]

        tasks = [asyncio.wait_for(self._fetch_rss_feed(url), timeout=5.0) for url in self.rss_feeds]
        rss_results = await asyncio.gather(*tasks, return_exceptions=True)

        stories: list[dict] = []
        for result in rss_results:
            if isinstance(result, Exception):
                continue
            stories.extend(result)

        try:
            stories.extend(await asyncio.wait_for(self._fetch_cryptopanic(), timeout=5.0))
        except Exception:  # noqa: BLE001
            pass

        deduped: list[dict] = []
        seen = set()
        for story in sorted(stories, key=lambda x: x.get("published_at", ""), reverse=True):
            k = story["title"].strip().lower()
            if k in seen:
                continue
            seen.add(k)
            deduped.append(story)

        await self.cache.set_json(cache_key, deduped, ttl=300)
        return deduped[:limit]
