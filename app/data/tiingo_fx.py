from __future__ import annotations

from datetime import datetime, timedelta, timezone
import threading
import time

import httpx
import pandas as pd

BASE_URL = "https://api.tiingo.com"


class ProviderError(RuntimeError):
    pass


_cache_lock = threading.Lock()
_cache: dict[tuple[str, int], tuple[float, pd.DataFrame]] = {}


def _get_cached(key: tuple[str, int], ttl_seconds: int) -> pd.DataFrame | None:
    if ttl_seconds <= 0:
        return None
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        created_at, frame = entry
        if time.monotonic() - created_at > ttl_seconds:
            _cache.pop(key, None)
            return None
        return frame.copy(deep=True)


def _put_cached(key: tuple[str, int], frame: pd.DataFrame, ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        return
    with _cache_lock:
        _cache[key] = (time.monotonic(), frame.copy(deep=True))
        if len(_cache) > 64:
            oldest = min(_cache, key=lambda k: _cache[k][0])
            _cache.pop(oldest, None)


def _ticker(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace("_", "")


def _fetch_hourly(
    token: str,
    symbol: str,
    history_days: int,
    cache_ttl_seconds: int,
) -> pd.DataFrame:
    if not token:
        raise ProviderError("DATA UNAVAILABLE: TIINGO_API_TOKEN is not configured.")

    days = max(30, int(history_days))
    key = (_ticker(symbol), days)
    cached = _get_cached(key, cache_ttl_seconds)
    if cached is not None:
        return cached

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    params = {
        "startDate": start_date.isoformat(),
        "endDate": end_date.isoformat(),
        "resampleFreq": "1hour",
        "columns": "date,open,high,low,close",
    }
    response = httpx.get(
        f"{BASE_URL}/tiingo/fx/{_ticker(symbol)}/prices",
        params=params,
        headers={"Content-Type": "application/json", "Authorization": f"Token {token}"},
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ProviderError("PROVIDER ERROR: unexpected Tiingo FX response format.")
    if not payload:
        raise ProviderError("DATA UNAVAILABLE: Tiingo FX returned no hourly values.")

    frame = pd.DataFrame(payload).rename(columns={"date": "timestamp", "last": "close"})
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ProviderError(f"PROVIDER ERROR: Tiingo FX missing OHLC fields: {missing}")

    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = (
        frame.dropna(subset=required)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ProviderError("DATA UNAVAILABLE: Tiingo FX returned no valid hourly OHLC rows.")

    frame["provider"] = "Tiingo FX"
    frame["instrument"] = symbol.upper().replace("_", "/")
    frame["timeframe"] = "1h"
    frame["data_status"] = "REAL_DATA"
    frame["source_timeframe"] = "1h"
    frame["calculation_status"] = "SOURCE"

    _put_cached(key, frame, cache_ttl_seconds)
    return frame.copy(deep=True)


def _aggregate(frame: pd.DataFrame, rule: str, target_timeframe: str) -> pd.DataFrame:
    source = frame.set_index("timestamp").sort_index()
    grouped = source.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    counts = source["close"].resample(rule, label="left", closed="left").count()
    expected = 4 if target_timeframe == "4h" else 24
    grouped["source_bar_count"] = counts
    grouped = grouped[grouped["source_bar_count"] == expected]
    grouped = grouped.drop(columns=["source_bar_count"]).dropna(subset=["open", "high", "low", "close"])
    grouped = grouped.reset_index()

    grouped["provider"] = "Tiingo FX"
    grouped["instrument"] = frame["instrument"].iloc[0]
    grouped["timeframe"] = target_timeframe
    grouped["data_status"] = "REAL_DATA"
    grouped["source_timeframe"] = "1h"
    grouped["calculation_status"] = "CALCULATED"
    return grouped.reset_index(drop=True)


def fetch_time_series(
    api_key: str,
    symbol: str = "EUR/USD",
    interval: str = "1h",
    history_days: int = 365,
    cache_ttl_seconds: int = 300,
) -> pd.DataFrame:
    interval = interval.strip().lower()
    hourly = _fetch_hourly(api_key, symbol, history_days, cache_ttl_seconds)

    if interval == "1h":
        return hourly
    if interval == "4h":
        return _aggregate(hourly, "4h", "4h")
    if interval == "1day":
        return _aggregate(hourly, "1D", "1day")
    raise ProviderError(f"DATA UNAVAILABLE: unsupported Tiingo interval '{interval}'.")


def inspect_time_series(
    api_key: str,
    symbol: str = "EUR/USD",
    history_days: int = 7,
) -> dict:
    frame = _fetch_hourly(api_key, symbol, history_days, cache_ttl_seconds=0)
    now = datetime.now(timezone.utc).isoformat()
    return {
        "provider": "Tiingo FX",
        "symbol": symbol.upper().replace("_", "/"),
        "source_interval": "1h",
        "returned_row_count": int(len(frame)),
        "first_timestamp": str(frame["timestamp"].min()),
        "last_timestamp": str(frame["timestamp"].max()),
        "request_timestamp_utc": now,
        "provider_status": "ok",
        "data_available": True,
        "derived_intervals": ["4h", "1day"],
        "data_status": "REAL_DATA",
        "calculation_status_for_derived": "CALCULATED",
    }
