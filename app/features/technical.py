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
    out["return_3"] = close.pct_change(3, fill_method=None)
    out["return_5"] = close.pct_change(5, fill_method=None)
    out["return_10"] = close.pct_change(10, fill_method=None)
    out["return_20"] = close.pct_change(20, fill_method=None)
    out["log_return_1"] = np.log(close / close.shift(1))
    out["log_return_5"] = np.log(close / close.shift(5))
    out["log_return_20"] = np.log(close / close.shift(20))
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
    out["atr_50"] = tr.rolling(50, min_periods=50).mean()
    out["atr_pct"] = out["atr_14"] / close.replace(0, np.nan)
    out["atr_ratio_14_50"] = out["atr_14"] / out["atr_50"].replace(0, np.nan)
    out["range_to_atr"] = out["range"] / out["atr_14"].replace(0, np.nan)

    out["sma_20"] = close.rolling(20, min_periods=20).mean()
    out["sma_50"] = close.rolling(50, min_periods=50).mean()
    out["sma_20_distance"] = (close - out["sma_20"]) / out["atr_14"].replace(0, np.nan)
    out["sma_50_distance"] = (close - out["sma_50"]) / out["atr_14"].replace(0, np.nan)
    out["trend_strength"] = (out["sma_20"] - out["sma_50"]) / out["atr_14"].replace(0, np.nan)

    out["ema_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["ema_12_distance"] = (close - out["ema_12"]) / out["atr_14"].replace(0, np.nan)
    out["ema_26_distance"] = (close - out["ema_26"]) / out["atr_14"].replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    rolling_mean = close.rolling(20, min_periods=20).mean()
    rolling_std = close.rolling(20, min_periods=20).std()
    out["bollinger_z"] = (close - rolling_mean) / rolling_std.replace(0, np.nan)
    out["bollinger_width"] = 4.0 * rolling_std / rolling_mean.replace(0, np.nan)

    lowest_14 = low.rolling(14, min_periods=14).min()
    highest_14 = high.rolling(14, min_periods=14).max()
    out["stochastic_k_14"] = 100.0 * (close - lowest_14) / (highest_14 - lowest_14).replace(0, np.nan)

    out["hour_sin"] = np.sin(2 * np.pi * out["timestamp"].dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["timestamp"].dt.hour / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * out["timestamp"].dt.dayofweek / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["timestamp"].dt.dayofweek / 7)

    return out.replace([np.inf, -np.inf], np.nan)
