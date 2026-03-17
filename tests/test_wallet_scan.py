"""Tests for WalletScanService."""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.wallet_scan import WalletScanService


def _make_solana_result() -> dict:
    return {
        "chain": "solana",
        "address": "ABC123",
        "native_symbol": "SOL",
        "native_balance": 5.0,
        "tokens": [
            {"symbol": "USDC", "amount": 1000.0},
            {"symbol": "BONK", "amount": 0.0000001},  # dust
        ],
    }


@asynccontextmanager
async def _empty_db():
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    yield session


@pytest.mark.asyncio
async def test_solana_scan_returns_native_balance() -> None:
    solana = AsyncMock()
    solana.scan_wallet = AsyncMock(return_value=_make_solana_result())

    price = AsyncMock()
    price.get_price = AsyncMock(return_value={"price": 150.0})

    svc = WalletScanService(
        db_factory=_empty_db,
        solana=solana,
        tron=AsyncMock(),
        price=price,
    )

    result = await svc.scan("solana", "ABC123")

    assert result["chain"] == "solana"
    assert result["native_balance"] == 5.0
    assert result["native_usd"] == pytest.approx(750.0)


@pytest.mark.asyncio
async def test_dust_token_flagged_as_suspicious() -> None:
    solana = AsyncMock()
    solana.scan_wallet = AsyncMock(return_value=_make_solana_result())

    price = AsyncMock()
    price.get_price = AsyncMock(return_value={"price": 150.0})

    svc = WalletScanService(
        db_factory=_empty_db,
        solana=solana,
        tron=AsyncMock(),
        price=price,
    )

    result = await svc.scan("solana", "ABC123")

    warnings = result.get("warnings", [])
    dust_warnings = [w for w in warnings if "dust" in w.lower()]
    assert len(dust_warnings) > 0


@pytest.mark.asyncio
async def test_unsupported_chain_raises() -> None:
    svc = WalletScanService(
        db_factory=_empty_db,
        solana=AsyncMock(),
        tron=AsyncMock(),
        price=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="Unsupported chain"):
        await svc.scan("ethereum", "0xDEAD")


@pytest.mark.asyncio
async def test_price_fetch_failure_does_not_crash_scan() -> None:
    """native_usd should be None (not crash) if price lookup fails."""
    solana = AsyncMock()
    solana.scan_wallet = AsyncMock(return_value=_make_solana_result())

    price = AsyncMock()
    price.get_price = AsyncMock(side_effect=Exception("price down"))

    svc = WalletScanService(
        db_factory=_empty_db,
        solana=solana,
        tron=AsyncMock(),
        price=price,
    )

    result = await svc.scan("solana", "ABC123")

    assert result["native_usd"] is None
    assert result["chain"] == "solana"
