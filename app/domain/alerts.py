from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AlertCondition = Literal["above", "below", "cross"]


@dataclass(frozen=True)
class AlertSpec:
    symbol: str
    condition: AlertCondition
    target_price: float


def condition_met(*, condition: str, target: float, prev_price: float, current_price: float) -> bool:
    """Pure trigger logic used by alert processing."""
    cond = (condition or "cross").strip().lower()
    if cond == "above":
        return prev_price < target <= current_price
    if cond == "below":
        return prev_price > target >= current_price
    # cross (either direction)
    crossed_up = prev_price < target <= current_price
    crossed_down = prev_price > target >= current_price
    return crossed_up or crossed_down


def parse_extra_conditions(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def extra_condition_met(*, cond: dict[str, Any], snapshot: Any, current_price: float) -> bool:
    """Evaluate a single extra AND-condition against an IndicatorSnapshot-like object."""
    if snapshot is None:
        return False
    ctype = str(cond.get("type", "")).lower()
    op = str(cond.get("operator", "")).lower()

    if ctype == "rsi":
        rsi = getattr(snapshot, "rsi14", None)
        if rsi is None:
            return False
        val = float(cond.get("value", 0))
        if op == "gt":
            return float(rsi) > val
        if op == "lt":
            return float(rsi) < val
        if op == "gte":
            return float(rsi) >= val
        if op == "lte":
            return float(rsi) <= val
        return False

    if ctype == "ema":
        period = int(cond.get("period", 20))
        ema_val = getattr(snapshot, f"ema{period}", None)
        if ema_val is None:
            return False
        if op == "above":
            return current_price > float(ema_val)
        if op == "below":
            return current_price < float(ema_val)
        return False

    return False

