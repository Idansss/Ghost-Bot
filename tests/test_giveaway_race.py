"""Tests for the atomic giveaway winner selection (race-condition guard)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.giveaway import GiveawayService


def _make_giveaway(id: int = 1, status: str = "active", winner: int | None = None) -> MagicMock:
    g = MagicMock()
    g.id = id
    g.chat_id = 100
    g.status = status
    g.winner_user_id = winner
    g.prize = "Test Prize"
    return g


@pytest.mark.asyncio
async def test_finalize_skips_already_finalized_giveaway() -> None:
    """If another process already set status to 'completed', _finalize should not re-draw."""
    svc = GiveawayService(db_factory=None, admin_chat_ids=[1])

    giveaway = _make_giveaway(status="active")

    async def _refresh(obj):
        # Simulate another process having already finalized
        obj.status = "completed"
        obj.winner_user_id = 777

    session = AsyncMock()
    session.refresh = _refresh
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    result = await svc._finalize(session, giveaway)

    assert result["note"] == "already_finalized"
    assert result["winner_user_id"] == 777
    # commit should NOT have been called (no new winner was chosen)
    session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_draws_winner_when_active() -> None:
    """Normal flow: active giveaway with 3 participants → winner chosen."""
    svc = GiveawayService(db_factory=None, admin_chat_ids=[1], min_participants=2)

    giveaway = _make_giveaway(status="active")

    async def _refresh(obj):
        pass  # Status stays active

    async def _participants(_session, _giveaway_id):
        return [10, 20, 30]

    async def _last_winner(_session, _chat_id):
        return None

    session = AsyncMock()
    session.refresh = _refresh
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.add = MagicMock()

    svc._participant_ids = _participants  # type: ignore[method-assign]
    svc._last_winner = _last_winner  # type: ignore[method-assign]

    result = await svc._finalize(session, giveaway)

    assert result["winner_user_id"] in (10, 20, 30)
    assert result["note"] == "ok"
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_no_winner_when_too_few_participants() -> None:
    """Giveaway with 1 participant (min=2) → no winner, status ended_no_winner."""
    svc = GiveawayService(db_factory=None, admin_chat_ids=[1], min_participants=2)
    giveaway = _make_giveaway(status="active")

    async def _refresh(obj):
        pass

    async def _participants(_session, _giveaway_id):
        return [42]

    async def _last_winner(_session, _chat_id):
        return None

    session = AsyncMock()
    session.refresh = _refresh
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    svc._participant_ids = _participants  # type: ignore[method-assign]
    svc._last_winner = _last_winner  # type: ignore[method-assign]

    result = await svc._finalize(session, giveaway)

    assert result["winner_user_id"] is None
    assert giveaway.status == "ended_no_winner"
