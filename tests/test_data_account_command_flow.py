from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.bot import data_account_command_flow


class _BufferedFile:
    def __init__(self, data: bytes, filename: str) -> None:
        self.data = data
        self.filename = filename


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=42),
        answer=AsyncMock(),
        answer_document=AsyncMock(),
    )


def _journal_service() -> SimpleNamespace:
    return SimpleNamespace(
        list_trades=AsyncMock(return_value=[]),
        get_stats=AsyncMock(return_value={"trades": 3, "wins": 2, "win_rate": 66.7, "total_pnl": 120.5}),
        update_trade=AsyncMock(return_value=True),
        log_trade=AsyncMock(return_value={"id": 9}),
    )


def _deps(**overrides) -> data_account_command_flow.DataAccountCommandFlowDependencies:
    defaults = {
        "hub": SimpleNamespace(
            trade_journal_service=_journal_service(),
            market_router=SimpleNamespace(get_price=AsyncMock(side_effect=[{"price": 1.0}, {"price": 2.0}, {"price": 3.0}])),
            scheduled_report_service=SimpleNamespace(
                list_reports=AsyncMock(return_value=[]),
                unsubscribe=AsyncMock(return_value=True),
                subscribe=AsyncMock(return_value=SimpleNamespace(cron_hour_utc=13, cron_minute_utc=45)),
            ),
            alerts_service=SimpleNamespace(
                list_alerts=AsyncMock(return_value=[SimpleNamespace(id=1, symbol="BTC", condition="above", target_price=100.0, status="active")])
            ),
            gdpr_service=SimpleNamespace(
                export_my_data=AsyncMock(return_value={"user": 42}),
                delete_account=AsyncMock(return_value=True),
            ),
        ),
        "feature_flags_set": lambda: {"journal", "multi_compare", "scheduled_report", "export"},
        "safe_exc": lambda exc: f"safe:{exc}",
        "buffered_input_file_cls": _BufferedFile,
    }
    defaults.update(overrides)
    return data_account_command_flow.DataAccountCommandFlowDependencies(**defaults)


@pytest.mark.asyncio
async def test_journal_log_normalizes_symbol_and_side() -> None:
    message = _message("/journal log eth sideways 2500 2600 55")
    service = _journal_service()
    deps = _deps(hub=SimpleNamespace(
        trade_journal_service=service,
        market_router=SimpleNamespace(get_price=AsyncMock()),
        scheduled_report_service=SimpleNamespace(),
        alerts_service=SimpleNamespace(),
        gdpr_service=SimpleNamespace(),
    ))

    await data_account_command_flow.handle_journal_command(message=message, deps=deps)

    service.log_trade.assert_awaited_once_with(
        42,
        symbol="ETH",
        side="long",
        entry=2500.0,
        exit_price=2600.0,
        pnl_quote=55.0,
    )
    assert "Logged:" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_compare_uses_default_symbols() -> None:
    message = _message("/compare")
    get_price = AsyncMock(side_effect=[{"price": 100.0}, {"price": 200.0}, {"price": 300.0}])
    deps = _deps(hub=SimpleNamespace(
        trade_journal_service=_journal_service(),
        market_router=SimpleNamespace(get_price=get_price),
        scheduled_report_service=SimpleNamespace(),
        alerts_service=SimpleNamespace(),
        gdpr_service=SimpleNamespace(),
    ))

    await data_account_command_flow.handle_compare_command(message=message, deps=deps)

    assert get_price.await_count == 3
    reply = message.answer.await_args.args[0]
    assert "BTC" in reply
    assert "ETH" in reply
    assert "SOL" in reply


@pytest.mark.asyncio
async def test_report_subscribe_parses_numeric_prefix() -> None:
    message = _message("/report 13 45")
    service = SimpleNamespace(
        list_reports=AsyncMock(return_value=[]),
        unsubscribe=AsyncMock(return_value=True),
        subscribe=AsyncMock(return_value=SimpleNamespace(cron_hour_utc=13, cron_minute_utc=45)),
    )
    deps = _deps(hub=SimpleNamespace(
        trade_journal_service=_journal_service(),
        market_router=SimpleNamespace(get_price=AsyncMock()),
        scheduled_report_service=service,
        alerts_service=SimpleNamespace(),
        gdpr_service=SimpleNamespace(),
    ))

    await data_account_command_flow.handle_report_command(message=message, deps=deps)

    service.subscribe.assert_awaited_once_with(42, report_type="market_summary", hour_utc=13, minute_utc=45)
    assert "13:45 UTC" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_export_alerts_renders_inline_pre_block() -> None:
    message = _message("/export alerts")
    deps = _deps()

    await data_account_command_flow.handle_export_command(message=message, deps=deps)

    reply = message.answer.await_args.args[0]
    assert reply.startswith("<pre># Alerts export")
    message.answer_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_mydata_sends_document() -> None:
    message = _message("/mydata")
    deps = _deps()

    await data_account_command_flow.handle_mydata_command(message=message, deps=deps)

    document = message.answer_document.await_args.args[0]
    assert document.filename == "my_ghost_data.json"
    assert b'"user": 42' in document.data


@pytest.mark.asyncio
async def test_deleteaccount_requires_confirm() -> None:
    message = _message("/deleteaccount")
    deps = _deps()

    await data_account_command_flow.handle_deleteaccount_command(message=message, deps=deps)

    assert "delete all your data" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_deleteaccount_confirm_deletes_account() -> None:
    message = _message("/deleteaccount confirm")
    gdpr_service = SimpleNamespace(
        export_my_data=AsyncMock(return_value=None),
        delete_account=AsyncMock(return_value=True),
    )
    deps = _deps(hub=SimpleNamespace(
        trade_journal_service=_journal_service(),
        market_router=SimpleNamespace(get_price=AsyncMock()),
        scheduled_report_service=SimpleNamespace(),
        alerts_service=SimpleNamespace(),
        gdpr_service=gdpr_service,
    ))

    await data_account_command_flow.handle_deleteaccount_command(message=message, deps=deps)

    gdpr_service.delete_account.assert_awaited_once_with(42)
    message.answer.assert_awaited_once_with("All your data has been deleted. Goodbye.")
