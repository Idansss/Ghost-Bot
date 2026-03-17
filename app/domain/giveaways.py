from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GiveawayPolicy:
    min_participants: int = 2
    prevent_back_to_back_winners: bool = True

    def eligible_participants(self, *, participants: Sequence[int], last_winner: int | None) -> list[int]:
        ids = [int(x) for x in participants]
        if len(ids) < max(1, int(self.min_participants)):
            return []
        if not self.prevent_back_to_back_winners:
            return ids
        if last_winner is None:
            return ids
        eligible = [uid for uid in ids if uid != int(last_winner)]
        return eligible or ids

