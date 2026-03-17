# ruff: noqa
from __future__ import annotations

from alembic import op


revision = "0008_perf_indexes"
down_revision = "0007_alert_idempotency_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Alerts: common access patterns are "my active alerts ordered by newest"
    op.create_index(
        "ix_alerts_user_id_status_created_at",
        "alerts",
        ["user_id", "status", "created_at"],
        unique=False,
    )

    # Trade checks: "my checks ordered by newest"
    op.create_index(
        "ix_trade_checks_user_id_created_at",
        "trade_checks",
        ["user_id", "created_at"],
        unique=False,
    )

    # Positions: "my positions ordered by newest"
    op.create_index(
        "ix_positions_user_id_created_at",
        "positions",
        ["user_id", "created_at"],
        unique=False,
    )

    # Trade journal: "my journal ordered by newest"
    op.create_index(
        "ix_trade_journal_user_id_created_at",
        "trade_journal",
        ["user_id", "created_at"],
        unique=False,
    )

    # Scheduled reports: lookups by chat + enabled
    op.create_index(
        "ix_scheduled_reports_chat_id_enabled",
        "scheduled_reports",
        ["chat_id", "enabled"],
        unique=False,
    )

    # Wallets: common usage is "my saved wallets"
    op.create_index(
        "ix_wallets_user_id_is_saved",
        "wallets",
        ["user_id", "is_saved"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_wallets_user_id_is_saved", table_name="wallets")
    op.drop_index("ix_scheduled_reports_chat_id_enabled", table_name="scheduled_reports")
    op.drop_index("ix_trade_journal_user_id_created_at", table_name="trade_journal")
    op.drop_index("ix_positions_user_id_created_at", table_name="positions")
    op.drop_index("ix_trade_checks_user_id_created_at", table_name="trade_checks")
    op.drop_index("ix_alerts_user_id_status_created_at", table_name="alerts")

