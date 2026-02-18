from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable

import feedparser

from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


class NewsSourcesAdapter:
    def __init__(
        self,
        http: ResilientHTTPClient,
        cache: RedisCache,
        rss_feeds: list[str],
        cryptopanic_key: str,
        openai_rss_feeds: list[str] | None = None,
    ) -> None:
        self.http = http
        self.cache = cache
        self.rss_feeds = rss_feeds
        self.cryptopanic_key = cryptopanic_key
        self.openai_rss_feeds = openai_rss_feeds or []

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
                    "summary": entry.get("summary", "") or entry.get("description", ""),
                    "source": parsed.feed.get("title", "rss"),
                    "published_at": (dt or datetime.now(timezone.utc)).isoformat(),
                }
            )
        return items

    async def _fetch_cryptopanic(self, limit: int = 20) -> list[dict]:
        if not self.cryptopanic_key:
            return []
        data = await self.http.get_json(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": self.cryptopanic_key, "public": "true"},
        )
        rows = []
        for item in data.get("results", [])[:limit]:
            rows.append(
                {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "summary": "",
                    "source": "CryptoPanic",
                    "published_at": item.get("published_at", datetime.now(timezone.utc).isoformat()),
                }
            )
        return rows

    def _dedupe_stories(self, stories: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen = set()
        for story in sorted(stories, key=lambda x: x.get("published_at", ""), reverse=True):
            url_key = (story.get("url") or "").strip().lower()
            title_key = (story.get("title") or "").strip().lower()
            key = url_key or title_key
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(story)
        return deduped

    async def _fetch_rss_group(self, feeds: Iterable[str]) -> list[dict]:
        tasks = [asyncio.wait_for(self._fetch_rss_feed(url), timeout=5.0) for url in feeds]
        if not tasks:
            return []
        rss_results = await asyncio.gather(*tasks, return_exceptions=True)
        stories: list[dict] = []
        for result in rss_results:
            if isinstance(result, Exception):
                continue
            stories.extend(result)
        return stories

    async def _collect_news(
        self,
        *,
        cache_key: str,
        feeds: list[str],
        include_cryptopanic: bool,
        limit: int,
    ) -> list[dict]:
        cached = await self.cache.get_json(cache_key)
        if cached:
            return cached[:limit]

        stories = await self._fetch_rss_group(feeds)
        if include_cryptopanic:
            try:
                stories.extend(await asyncio.wait_for(self._fetch_cryptopanic(limit=max(limit, 20)), timeout=5.0))
            except Exception:  # noqa: BLE001
                pass

        deduped = self._dedupe_stories(stories)
        await self.cache.set_json(cache_key, deduped, ttl=300)
        return deduped[:limit]

    async def get_crypto_news(self, limit: int = 30) -> list[dict]:
        return await self._collect_news(
            cache_key="news:crypto",
            feeds=self.rss_feeds,
            include_cryptopanic=True,
            limit=limit,
        )

    async def get_openai_news(self, limit: int = 30) -> list[dict]:
        if not self.openai_rss_feeds:
            return []
        return await self._collect_news(
            cache_key="news:openai",
            feeds=self.openai_rss_feeds,
            include_cryptopanic=False,
            limit=limit,
        )

    async def get_latest_news(self, limit: int = 10, mode: str = "crypto") -> list[dict]:
        if mode == "openai":
            return await self.get_openai_news(limit=limit)
        if mode == "all":
            crypto, openai = await asyncio.gather(
                self.get_crypto_news(limit=max(limit, 20)),
                self.get_openai_news(limit=max(limit, 20)),
            )
            return self._dedupe_stories(crypto + openai)[:limit]
        return await self.get_crypto_news(limit=limit)
