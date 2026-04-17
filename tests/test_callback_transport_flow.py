from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import callback_transport_flow


@pytest.mark.asyncio
async def test_run_deduped_callback_skips_duplicate_and_answers() -> None:
    callback = SimpleNamespace(answer=AsyncMock())
    runner = AsyncMock()

    handled = await callback_transport_flow.run_deduped_callback(
        callback,
        acquire_callback_once=AsyncMock(return_value=False),
        runner=runner,
    )

    assert handled is False
    callback.answer.assert_awaited_once()
    runner.assert_not_called()


@pytest.mark.asyncio
async def test_run_deduped_callback_runs_handler_once() -> None:
    callback = SimpleNamespace(answer=AsyncMock())
    runner = AsyncMock()

    handled = await callback_transport_flow.run_deduped_callback(
        callback,
        acquire_callback_once=AsyncMock(return_value=True),
        runner=runner,
    )

    assert handled is True
    runner.assert_awaited_once()
    callback.answer.assert_not_called()
