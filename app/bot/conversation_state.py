from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.db.models import TradeCheck
from app.db.session import AsyncSessionLocal


def _pending_alert_key(chat_id: int) -> str:
    return f"pending_alert:{chat_id}"


def _tradecheck_wizard_key(chat_id: int) -> str:
    return f"wizard:tradecheck:{chat_id}"


def _command_wizard_key(chat_id: int) -> str:
    return f"wizard:cmd:{chat_id}"


async def get_pending_alert(cache, chat_id: int) -> str | None:
    payload = await cache.get_json(_pending_alert_key(chat_id))
    if not isinstance(payload, dict):
        return None
    symbol = payload.get("symbol")
    return str(symbol) if symbol else None


async def set_pending_alert(cache, chat_id: int, symbol: str, ttl: int = 300) -> None:
    await cache.set_json(_pending_alert_key(chat_id), {"symbol": symbol.upper()}, ttl=ttl)


async def clear_pending_alert(cache, chat_id: int) -> None:
    await cache.delete(_pending_alert_key(chat_id))


async def get_tradecheck_wizard(cache, chat_id: int) -> dict | None:
    payload = await cache.get_json(_tradecheck_wizard_key(chat_id))
    return payload if isinstance(payload, dict) else None


async def set_tradecheck_wizard(cache, chat_id: int, payload: dict, ttl: int = 900) -> None:
    await cache.set_json(_tradecheck_wizard_key(chat_id), payload, ttl=ttl)


async def clear_tradecheck_wizard(cache, chat_id: int) -> None:
    await cache.delete(_tradecheck_wizard_key(chat_id))


async def get_command_wizard(cache, chat_id: int) -> dict | None:
    payload = await cache.get_json(_command_wizard_key(chat_id))
    return payload if isinstance(payload, dict) else None


async def set_command_wizard(cache, chat_id: int, payload: dict, ttl: int = 900) -> None:
    await cache.set_json(_command_wizard_key(chat_id), payload, ttl=ttl)


async def clear_command_wizard(cache, chat_id: int) -> None:
    await cache.delete(_command_wizard_key(chat_id))


async def save_trade_check(
    *,
    ensure_user: Callable[[int], Awaitable[Any]],
    chat_id: int,
    data: dict,
    result: dict,
    session_factory: Callable[[], Any] = AsyncSessionLocal,
    tradecheck_model=TradeCheck,
) -> None:
    user = await ensure_user(chat_id)
    async with session_factory() as session:
        row = tradecheck_model(
            user_id=user.id,
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            timestamp=data["timestamp"],
            entry=float(data["entry"]),
            stop=float(data["stop"]),
            targets_json=[float(item) for item in data["targets"]],
            mode=data.get("mode", "ambiguous"),
            result_json=result,
        )
        session.add(row)
        await session.commit()
