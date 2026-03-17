# ruff: noqa
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006_alert_multi_conditions"
down_revision = "0005_positions_journal_scheduled"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "alerts",
        sa.Column("conditions_json", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("alerts", "conditions_json")
