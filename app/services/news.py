from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

from app.adapters.llm import LLMClient
from app.adapters.news_sources import NewsSourcesAdapter


class NewsService:
    TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
        "cpi": ("cpi", "inflation", "consumer price", "core cpi", "bls"),
        "macro": ("macro", "cpi", "inflation", "fomc", "fed", "powell", "rates", "jobs", "nfp", "ppi"),
        "openai": ("openai", "chatgpt", "gpt", "codex", "responses api", "api"),
    }

    def __init__(self, adapter: NewsSourcesAdapter, llm_client: LLMClient | None = None) -> None:
        self.adapter = adapter
        self.llm_client = llm_client

    def _clean_story(self, story: dict) -> dict:
        return {
            "title": story.get("title", "Untitled"),
            "url": story.get("url", ""),
            "source": story.get("source", "unknown"),
            "published_at": story.get("published_at", datetime.now(timezone.utc).isoformat()),
            "summary": story.get("summary", ""),
        }

    async def _fetch_stories(self, mode: str, limit: int) -> list[dict]:
        normalized_mode = mode.lower().strip()
        if normalized_mode not in {"crypto", "openai", "all", "macro"}:
            normalized_mode = "crypto"
        adapter_mode = "crypto" if normalized_mode == "macro" else normalized_mode
        stories = await self.adapter.get_latest_news(limit=max(limit, 10), mode=adapter_mode)
        return [self._clean_story(s) for s in stories]

    def _filter_by_topic(self, stories: Iterable[dict], topic: str | None) -> list[dict]:
        if not topic:
            return list(stories)

        topic_key = topic.lower().strip()
        if not topic_key:
            return list(stories)
        if topic_key in {"crypto", "general", "market", "all"}:
            return list(stories)

        keywords = list(self.TOPIC_KEYWORDS.get(topic_key, (topic_key,)))
        pattern = re.compile("|".join(re.escape(k) for k in keywords if k), flags=re.IGNORECASE)
        filtered: list[dict] = []
        for story in stories:
            haystack = f"{story.get('title', '')} {story.get('summary', '')}"
            if pattern.search(haystack):
                filtered.append(story)
        return filtered

    async def _summarize_with_llm(self, stories: list[dict], topic: str, mode: str) -> str | None:
        if not self.llm_client or not stories:
            return None
        bullet_lines = []
        for item in stories[:8]:
            bullet_lines.append(f"- {item['title']} ({item['source']})")

        prompt = (
            "Summarize these headlines for a crypto trader in 1-2 lines.\n"
            f"Topic: {topic}\n"
            f"Mode: {mode}\n"
            "Tone: concise, sharp, no hype.\n"
            "If topic is macro/cpi, mention BTC/ETH risk context briefly.\n\n"
            "Headlines:\n"
            + "\n".join(bullet_lines)
        )
        try:
            text = await self.llm_client.reply(prompt)
        except Exception:  # noqa: BLE001
            return None
        cleaned = (text or "").strip()
        return cleaned if cleaned else None

    def _heuristic_summary(self, stories: list[dict], topic: str | None, mode: str) -> str:
        topic_name = (topic or ("openai" if mode == "openai" else "crypto")).upper()
        themes = []
        if any("etf" in s["title"].lower() for s in stories):
            themes.append("ETF flow is still steering sentiment")
        if any(k in s["title"].lower() for s in stories for k in ("fed", "cpi", "inflation", "fomc", "rates")):
            themes.append("Macro prints remain the volatility trigger")
        if any(k in s["title"].lower() for s in stories for k in ("hack", "exploit", "breach")):
            themes.append("Security headlines are lifting risk premium")
        if not themes:
            themes.append("Tape is headline-driven, stay selective")
        return f"{topic_name} brief: {' | '.join(themes[:2])}."

    def _vibe_line(self, topic: str | None, mode: str) -> str:
        key = (topic or mode or "crypto").lower()
        if key in {"cpi", "macro"}:
            return "Vibe: macro-sensitive tape, expect whips around data prints."
        if key == "openai":
            return "Vibe: AI narrative is active, watch secondary correlation trades."
        return "Vibe: reactive tape, momentum still headline-driven."

    async def get_daily_brief(self, limit: int = 7) -> dict:
        return await self.get_digest(limit=limit)

    async def get_digest(self, topic: str | None = None, mode: str = "crypto", limit: int = 7) -> dict:
        capped_limit = max(3, min(limit, 10))
        stories = await self._fetch_stories(mode=mode, limit=max(capped_limit * 4, 24))
        filtered = self._filter_by_topic(stories, topic)

        fallback_used = False
        if topic and not filtered:
            fallback_used = True
            filtered = stories

        selected = filtered[:capped_limit]
        normalized_topic = (topic or "").strip().lower() or None

        if not selected:
            topic_name = normalized_topic.upper() if normalized_topic else "CRYPTO"
            return {
                "summary": f"No fresh {topic_name} headlines from configured feeds right now.",
                "headlines": [],
                "vibe": "Enable RSS/API feeds in .env for live news sources.",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        summary = await self._summarize_with_llm(selected, normalized_topic or "crypto", mode)
        if not summary:
            summary = self._heuristic_summary(selected, normalized_topic, mode)
        if fallback_used and normalized_topic:
            summary = f"No strict {normalized_topic.upper()} match found. Showing latest flow. {summary}"

        bullets = []
        for item in selected:
            bullets.append(
                {
                    "title": item["title"],
                    "url": item["url"],
                    "source": item["source"],
                    "published_at": item["published_at"],
                }
            )

        return {
            "summary": summary,
            "headlines": bullets,
            "vibe": self._vibe_line(normalized_topic, mode),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def get_asset_headlines(self, symbol: str, limit: int = 3) -> list[dict]:
        symbol_u = symbol.upper()
        stories = await self._fetch_stories("crypto", limit=25)
        filtered = [s for s in stories if symbol_u in s["title"].upper() or symbol_u in s.get("summary", "").upper()]
        if not filtered:
            filtered = stories[:limit]
        return filtered[:limit]
