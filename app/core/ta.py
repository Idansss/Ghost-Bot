from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


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


def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal


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
