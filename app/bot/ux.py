def busy(feature: str) -> str:
    feature = (feature or "that").strip().lower()
    if feature in {"analysis", "news", "rsi", "ema", "chart", "heatmap", "tradecheck"}:
        return f"{feature} already running — try again in a few seconds."
    return "still working — try again in a few seconds."


def degraded(feature: str) -> str:
    feature = (feature or "").strip().lower()
    if feature == "analysis":
        return "⚠ live data hiccup — showing your last saved snapshot for this symbol."
    if feature == "news":
        return "⚠ news is degraded — showing your last cached digest."
    if feature in {"rsi", "rsi_scan"}:
        return "⚠ RSI scan is degraded — showing your last cached result."
    if feature in {"ema", "ema_scan"}:
        return "⚠ EMA scan is degraded — showing your last cached result."
    return "⚠ degraded — showing cached results."


def transient_error(feature: str) -> str:
    feature = (feature or "").strip().lower()
    if feature == "news":
        return "News digest failed — try again in a moment."
    if feature in {"rsi", "rsi_scan"}:
        return "RSI scan hit an error — try again in a moment."
    if feature in {"ema", "ema_scan"}:
        return "EMA scan hit an error — try again in a moment."
    return "Request failed — try again in a moment."

