from __future__ import annotations

import re


def strip_html_tags(text: str) -> str:
    """Remove HTML tags so user never sees literal <b> or </b>. Keeps inner text."""
    if not text:
        return text
    return re.sub(r"<[^>]+>", "", str(text))


def safe_html(text: str) -> str:
    """Escape special HTML characters so dynamic content is safe in HTML parse_mode."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_price(v: float) -> str:
    """Human-readable price for summary text: $68.1k, $1.23M, $142.3"""
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    if v >= 10_000:
        return f"${v / 1_000:.1f}k"
    if v >= 1_000:
        return f"${v / 1_000:.2f}k"
    if v >= 100:
        return f"${v:.1f}"
    if v >= 10:
        return f"${v:.2f}"
    if v >= 1:
        return f"${v:.3f}"
    if v >= 0.01:
        return f"${v:.4f}"
    return f"${v:.6f}"


def fmt_level(v: float) -> str:
    """Clean price level for trade plans: no excess decimals for entry/TP/SL"""
    if v >= 10_000:
        return f"{v:,.0f}"
    if v >= 1_000:
        return f"{v:,.1f}"
    if v >= 100:
        return f"{v:.2f}"
    if v >= 10:
        return f"{v:.3f}"
    if v >= 1:
        return f"{v:.4f}"
    if v >= 0.1:
        return f"{v:.5f}"
    return f"{v:.6f}"
