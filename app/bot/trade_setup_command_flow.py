from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class TradeSetupCommandFlowDependencies:
    set_wizard: Callable[[int, dict], Awaitable[None]]
    dispatch_command_text: Callable[[Any, str], Awaitable[bool]]
    findpair_quick_menu: Callable[[], Any]
    setup_quick_menu: Callable[[], Any]


def _command_text(message) -> str:
    raw = (message.text or "").strip()
    return raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""


async def handle_tradecheck_command(*, message, deps: TradeSetupCommandFlowDependencies) -> None:
    await deps.set_wizard(message.chat.id, {"step": "symbol", "data": {}})
    await message.answer("Tradecheck wizard: send symbol (e.g., ETH).")


async def handle_findpair_command(*, message, deps: TradeSetupCommandFlowDependencies) -> None:
    query = _command_text(message).strip()
    if not query:
        await message.answer("Pick find mode.", reply_markup=deps.findpair_quick_menu())
        return
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", query):
        await deps.dispatch_command_text(message, f"coin around {query}")
        return
    await deps.dispatch_command_text(message, f"find pair {query}")


async def handle_setup_command(*, message, deps: TradeSetupCommandFlowDependencies) -> None:
    text = _command_text(message).strip()
    if not text:
        await message.answer("Choose setup input mode.", reply_markup=deps.setup_quick_menu())
        return
    await deps.dispatch_command_text(message, text)


async def handle_margin_command(*, message, deps: TradeSetupCommandFlowDependencies) -> None:
    text = _command_text(message).strip()
    if not text:
        await message.answer("Choose setup input mode.", reply_markup=deps.setup_quick_menu())
        return
    await deps.dispatch_command_text(message, text)


async def handle_pnl_command(*, message, deps: TradeSetupCommandFlowDependencies) -> None:
    text = _command_text(message).strip()
    if not text:
        await message.answer("Choose setup input mode.", reply_markup=deps.setup_quick_menu())
        return
    await deps.dispatch_command_text(message, text)
