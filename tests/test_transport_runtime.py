from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.bot import transport_runtime


class _Cache:
    def __init__(self) -> None:
        self.redis = SimpleNamespace(get=AsyncMock(return_value=None))
        self.set_if_absent = AsyncMock(return_value=True)
        self.incr_with_expiry = AsyncMock(return_value=0)
        self.get_ttl = AsyncMock(return_value=600)


def _deps(**overrides) -> transport_runtime.TransportRuntimeDependencies:
    defaults = {
        "cache": _Cache(),
        "rate_limiter": SimpleNamespace(check=AsyncMock(return_value=SimpleNamespace(allowed=True))),
        "logger": Mock(),
        "request_rate_limit_per_minute": 5,
        "abuse_strike_window_sec": 60,
        "abuse_strikes_to_block": 3,
        "abuse_block_ttl_sec": 600,
        "record_abuse": Mock(),
        "chat_lock": lambda _chat_id: asyncio.Lock(),
    }
    defaults.update(overrides)
    return transport_runtime.TransportRuntimeDependencies(**defaults)


@pytest.mark.asyncio
async def test_record_strike_and_maybe_block_sets_temporary_block() -> None:
    cache = _Cache()
    cache.incr_with_expiry = AsyncMock(return_value=3)
    deps = _deps(cache=cache)

    blocked = await transport_runtime.record_strike_and_maybe_block(42, deps=deps)

    assert blocked is True
    cache.set_if_absent.assert_awaited_once_with("abuse:block:42", ttl=600)
    deps.record_abuse.assert_called_once_with("auto_block")


@pytest.mark.asyncio
async def test_run_with_typing_lock_skips_blocked_subject() -> None:
    deps = _deps()
    deps.cache.redis.get = AsyncMock(return_value=b"1")
    runner = AsyncMock()
    bot = SimpleNamespace(send_chat_action=AsyncMock())

    await transport_runtime.run_with_typing_lock(bot, 42, runner, deps=deps)

    runner.assert_not_called()
    deps.record_abuse.assert_called_once_with("blocked_callback")


@pytest.mark.asyncio
async def test_acquire_message_once_uses_message_dedupe_key() -> None:
    deps = _deps()
    message = SimpleNamespace(chat=SimpleNamespace(id=7), message_id=9)

    allowed = await transport_runtime.acquire_message_once(message, deps=deps)

    assert allowed is True
    deps.cache.set_if_absent.assert_awaited_once_with("seen:message:7:9", ttl=60 * 60 * 6)
