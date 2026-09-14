from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp").reset_index(drop=True)

    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")

    out["return_1"] = close.pct_change(fill_method=None)
    out["return_5"] = close.pct_change(5, fill_method=None)
    out["return_20"] = close.pct_change(20, fill_method=None)
    out["range"] = high - low
    out["range_pct"] = (high - low) / close.replace(0, np.nan)
    out["body_pct"] = (close - open_) / open_.replace(0, np.nan)

    previous = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - previous).abs(),
            (low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["true_range"] = tr
    out["atr_14"] = tr.rolling(14, min_periods=14).mean()
    out["atr_pct"] = out["atr_14"] / close.replace(0, np.nan)

    out["sma_20"] = close.rolling(20, min_periods=20).mean()
    out["sma_50"] = close.rolling(50, min_periods=50).mean()

    out["ema_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["hour_sin"] = np.sin(2 * np.pi * out["timestamp"].dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["timestamp"].dt.hour / 24)

    return out.replace([np.inf, -np.inf], np.nan)
