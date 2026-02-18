# ruff: noqa
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_giveaways"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "giveaways",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("prize", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=False),
        sa.Column("end_time", sa.DateTime(), nullable=False),
        sa.Column("created_by_chat_id", sa.BigInteger(), nullable=False),
        sa.Column("winner_user_id", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_giveaways_chat_id", "giveaways", ["chat_id"], unique=False)
    op.create_index("ix_giveaways_status", "giveaways", ["status"], unique=False)
    op.create_index("ix_giveaways_end_time", "giveaways", ["end_time"], unique=False)
    op.create_index("ix_giveaways_created_by_chat_id", "giveaways", ["created_by_chat_id"], unique=False)
    op.create_index("ix_giveaways_winner_user_id", "giveaways", ["winner_user_id"], unique=False)

    op.create_table(
        "giveaway_participants",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("giveaway_id", sa.Integer(), sa.ForeignKey("giveaways.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_chat_id", sa.BigInteger(), nullable=False),
        sa.Column("joined_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("giveaway_id", "user_chat_id", name="uq_giveaway_participant"),
    )
    op.create_index("ix_giveaway_participants_giveaway_id", "giveaway_participants", ["giveaway_id"], unique=False)
    op.create_index("ix_giveaway_participants_user_chat_id", "giveaway_participants", ["user_chat_id"], unique=False)

    op.create_table(
        "giveaway_winners",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("giveaway_id", sa.Integer(), sa.ForeignKey("giveaways.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chat_id", sa.BigInteger(), nullable=False),
        sa.Column("user_chat_id", sa.BigInteger(), nullable=False),
        sa.Column("won_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_giveaway_winners_giveaway_id", "giveaway_winners", ["giveaway_id"], unique=False)
    op.create_index("ix_giveaway_winners_chat_id", "giveaway_winners", ["chat_id"], unique=False)
    op.create_index("ix_giveaway_winners_user_chat_id", "giveaway_winners", ["user_chat_id"], unique=False)


def downgrade() -> None:
    op.drop_table("giveaway_winners")
    op.drop_table("giveaway_participants")
    op.drop_table("giveaways")
