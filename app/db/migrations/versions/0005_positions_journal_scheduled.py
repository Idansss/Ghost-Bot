# ruff: noqa
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005_positions_journal_scheduled"
down_revision = "0004_alert_source_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "positions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), index=True),
        sa.Column("symbol", sa.String(20), index=True),
        sa.Column("side", sa.String(8), default="long"),
        sa.Column("entry_price", sa.Float()),
        sa.Column("size_quote", sa.Float(), default=0.0),
        sa.Column("leverage", sa.Float(), default=1.0),
        sa.Column("notes", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "trade_journal",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), index=True),
        sa.Column("symbol", sa.String(20), index=True),
        sa.Column("side", sa.String(8), default="long"),
        sa.Column("entry", sa.Float()),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("stop", sa.Float(), nullable=True),
        sa.Column("targets_json", JSONB, default=list),
        sa.Column("outcome", sa.String(20), nullable=True),
        sa.Column("pnl_quote", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), index=True),
    )
    op.create_table(
        "scheduled_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), index=True),
        sa.Column("chat_id", sa.BigInteger(), index=True),
        sa.Column("report_type", sa.String(32), default="market_summary"),
        sa.Column("cron_hour_utc", sa.Integer(), default=9),
        sa.Column("cron_minute_utc", sa.Integer(), default=0),
        sa.Column("timezone", sa.String(48), nullable=True),
        sa.Column("enabled", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("scheduled_reports")
    op.drop_table("trade_journal")
    op.drop_table("positions")
