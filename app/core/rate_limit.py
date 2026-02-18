from __future__ import annotations

from dataclasses import dataclass

from app.core.cache import RedisCache


@dataclass
class LimitResult:
    allowed: bool
    remaining: int
    reset_seconds: int


class RateLimiter:
    def __init__(self, cache: RedisCache) -> None:
        self.cache = cache

    async def check(self, key: str, limit: int, window_seconds: int) -> LimitResult:
        count = await self.cache.incr_with_expiry(key, window_seconds)
        remaining = max(limit - count, 0)
        return LimitResult(
            allowed=count <= limit,
            remaining=remaining,
            reset_seconds=window_seconds,
        )
