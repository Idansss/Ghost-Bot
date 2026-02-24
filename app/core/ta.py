from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(~((loss == 0) & (gain > 0)), 100.0)
    out = out.where(~((loss == 0) & (gain == 0)), 50.0)
    return out


def bollinger_mid(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Upper, middle, lower Bollinger Bands."""
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    histogram = line - signal
    return line, signal, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def pivot_levels(df: pd.DataFrame, lookback: int = 60) -> tuple[float, float]:
    chunk = df.tail(lookback)
    support = float(chunk["low"].min())
    resistance = float(chunk["high"].max())
    return support, resistance


def consolidation_zone(df: pd.DataFrame, window: int = 12) -> tuple[float, float]:
    chunk = df.tail(window)
    return float(chunk["low"].min()), float(chunk["high"].max())


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price. Requires columns: high, low, close, volume."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0).astype(float)
    return (typical * vol).cumsum() / vol.cumsum().replace(0, np.nan)


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    low_n = df["low"].rolling(window=k_period).min()
    high_n = df["high"].rolling(window=k_period).max()
    k = 100.0 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    k = k.clip(0, 100)
    d = k.rolling(window=d_period).mean()
    return k, d


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume.fillna(0)).cumsum()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (trend strength)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = atr(df, period=1)
    atr_ser = atr(df, period)
    plus_di = 100.0 * (plus_dm.rolling(period).mean() / atr_ser.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.rolling(period).mean() / atr_ser.replace(0, np.nan))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def fibonacci_retracements(swing_high: float, swing_low: float) -> dict[str, float]:
    """Key Fibonacci levels between swing_high and swing_low (high assumed > low)."""
    diff = swing_high - swing_low
    return {
        "0": swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1": swing_low,
    }


def ichimoku(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Ichimoku Cloud: Tenkan (9), Kijun (26), Senkou A, Senkou B (52), Chikou Span.
    Returns (tenkan, kijun, senkou_a, senkou_b, chikou).
    """
    high, low, close = df["high"], df["low"], df["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    return tenkan, kijun, senkou_a, senkou_b, chikou


def candlestick_patterns(df: pd.DataFrame, last_n: int = 3) -> list[str]:
    """
    Detect simple candlestick patterns on the last_n candles.
    Returns list of pattern names, e.g. ['doji', 'bullish_engulfing'].
    """
    if df.empty or len(df) < last_n + 1:
        return []
    out: list[str] = []
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l_ = df["low"].astype(float)
    c = df["close"].astype(float)
    body = (c - o).abs()
    full_range = (h - l_).replace(0, np.nan)
    for i in range(-last_n, 0):
        if full_range.iloc[i] == 0:
            continue
        body_ratio = body.iloc[i] / full_range.iloc[i]
        # Doji: very small body
        if body_ratio < 0.1:
            out.append("doji")
        if i > -last_n:
            # Engulfing: current body fully contains prior body
            prev_o, prev_c = o.iloc[i - 1], c.iloc[i - 1]
            curr_o, curr_c = o.iloc[i], c.iloc[i]
            curr_lo, curr_hi = min(curr_o, curr_c), max(curr_o, curr_c)
            prev_lo, prev_hi = min(prev_o, prev_c), max(prev_o, prev_c)
            if curr_lo <= prev_lo and curr_hi >= prev_hi:
                if curr_c > curr_o:
                    out.append("bullish_engulfing")
                else:
                    out.append("bearish_engulfing")
    return list(dict.fromkeys(out))  # preserve order, dedupe
