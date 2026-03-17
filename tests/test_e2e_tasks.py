from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.mark.asyncio
async def test_tasks_alerts_run_calls_service_and_notifier() -> None:
    app = create_app()

    async def _process_alerts(notifier):
        await notifier(123, "ping")
        return 1

    hub = SimpleNamespace()
    hub.alerts_service = SimpleNamespace(process_alerts=_process_alerts)
    app.state.hub = hub
    app.state.bot = SimpleNamespace(send_message=AsyncMock())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/tasks/alerts/run", headers={"x-vercel-cron": "1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["processed"] == 1

    app.state.bot.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_tasks_alerts_run_rejects_unauthorized() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/tasks/alerts/run")
        assert resp.status_code == 401

