from __future__ import annotations

from enum import StrEnum


class AlertStatus(StrEnum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    PAUSED = "paused"
    ENDED_NO_WINNER = "ended_no_winner"  # reused in giveaways context but kept for completeness
