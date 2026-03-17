from __future__ import annotations

from app.core.nlu import parse_timeframe


def preferred_timeframe_from_settings(settings: dict, fallback: str = "1h") -> str:
    raw = settings.get("preferred_timeframe")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return fallback


def effective_timeframe(*, user_text: str, settings: dict, default: str = "1h") -> str:
    """Use timeframe from user text if present, else user preference, else default."""
    tf = parse_timeframe(user_text or "")
    if tf:
        return tf
    return preferred_timeframe_from_settings(settings, fallback=default)

