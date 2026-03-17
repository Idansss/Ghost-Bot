from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanRequest:
    symbols: list[str]
    timeframe: str

