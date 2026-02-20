from __future__ import annotations

import logging
from typing import Any

import orjson
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, redis_url: str) -> None:
        self.redis = Redis.from_url(redis_url, decode_responses=False)

    async def close(self) -> None:
        if hasattr(self.redis, "aclose"):
            await self.redis.aclose()  # type: ignore[attr-defined]
            return
        await self.redis.close()  # type: ignore[func-returns-value]

    async def get_json(self, key: str) -> Any | None:
        try:
            raw = await self.redis.get(key)
            if not raw:
                return None
            return orjson.loads(raw)
        except orjson.JSONDecodeError:
            logger.warning("cache_json_decode_error", extra={"key": key})
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_get_error", extra={"key": key, "error": str(exc)})
            return None

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        try:
            await self.redis.set(key, orjson.dumps(value), ex=ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_set_error", extra={"key": key, "error": str(exc)})

    async def incr_with_expiry(self, key: str, ttl: int) -> int:
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl)
            count, _ = await pipe.execute()
            return int(count)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_incr_error", extra={"key": key, "error": str(exc)})
            return 0

    async def set_if_absent(self, key: str, ttl: int, value: str = "1") -> bool:
        try:
            return bool(await self.redis.set(key, value.encode("utf-8"), nx=True, ex=ttl))
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_set_if_absent_error", extra={"key": key, "error": str(exc)})
            return False

    async def delete(self, *keys: str) -> int:
        """Delete one or more keys. Returns number of keys removed."""
        if not keys:
            return 0
        try:
            return int(await self.redis.delete(*keys))
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_delete_error", extra={"keys": keys, "error": str(exc)})
            return 0

    async def get_ttl(self, key: str) -> int:
        """Returns remaining TTL in seconds. -1 = no expiry, -2 = key missing."""
        try:
            return int(await self.redis.ttl(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache_ttl_error", extra={"key": key, "error": str(exc)})
            return -2
