from __future__ import annotations

from typing import Any

import orjson
from redis.asyncio import Redis


class RedisCache:
    def __init__(self, redis_url: str) -> None:
        self.redis = Redis.from_url(redis_url, decode_responses=False)

    async def close(self) -> None:
        if hasattr(self.redis, "aclose"):
            await self.redis.aclose()  # type: ignore[attr-defined]
            return
        await self.redis.close()  # type: ignore[func-returns-value]

    async def get_json(self, key: str) -> Any | None:
        raw = await self.redis.get(key)
        if not raw:
            return None
        return orjson.loads(raw)

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        await self.redis.set(key, orjson.dumps(value), ex=ttl)

    async def incr_with_expiry(self, key: str, ttl: int) -> int:
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl)
        count, _ = await pipe.execute()
        return int(count)

    async def set_if_absent(self, key: str, ttl: int, value: str = "1") -> bool:
        return bool(await self.redis.set(key, value.encode("utf-8"), nx=True, ex=ttl))
