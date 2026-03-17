# ruff: noqa
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0007_alert_idempotency_key"
down_revision = "0006_alert_multi_conditions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {str(col["name"]) for col in inspector.get_columns("alerts")}
    indexes = {str(idx["name"]) for idx in inspector.get_indexes("alerts")}
    if "idempotency_key" not in columns:
        op.add_column(
            "alerts",
            sa.Column("idempotency_key", sa.String(64), nullable=True),
        )
    if "ix_alerts_idempotency_key" not in indexes:
        op.create_index("ix_alerts_idempotency_key", "alerts", ["idempotency_key"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_alerts_idempotency_key", table_name="alerts")
    op.drop_column("alerts", "idempotency_key")
