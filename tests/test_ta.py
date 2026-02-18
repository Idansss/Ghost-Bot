from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.ta import bollinger_mid, ema, macd, rsi


def test_ema_monotonic_uptrend() -> None:
    s = pd.Series(np.arange(1, 101, dtype=float))
    out = ema(s, 20)
    assert out.iloc[-1] > out.iloc[-2]


def test_rsi_high_on_uptrend() -> None:
    s = pd.Series(np.linspace(1, 200, 200))
    out = rsi(s, 14)
    assert out.iloc[-1] > 70


def test_macd_positive_on_uptrend() -> None:
    s = pd.Series(np.linspace(10, 200, 150))
    m_line, m_signal = macd(s)
    assert m_line.iloc[-1] > 0
    assert m_line.iloc[-1] >= m_signal.iloc[-1]


def test_bollinger_mid_is_mean() -> None:
    s = pd.Series(np.arange(1, 41, dtype=float))
    mid = bollinger_mid(s, 20)
    expected = s.tail(20).mean()
    assert round(float(mid.iloc[-1]), 8) == round(float(expected), 8)
