from __future__ import annotations

from contextlib import suppress
from typing import Any, Awaitable, Callable


async def run_deduped_callback(
    callback: Any,
    *,
    acquire_callback_once: Callable[[Any], Awaitable[bool]],
    runner: Callable[[], Awaitable[None]],
) -> bool:
    if not await acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return False
    await runner()
    return True
