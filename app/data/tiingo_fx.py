from __future__ import annotations

from datetime import datetime, timedelta, timezone
import threading
import time
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from app.config import settings

BASE_URL = "https://api.tiingo.com"


class ProviderError(RuntimeError):
    pass


_cache_lock = threading.Lock()
_cache: dict[tuple[str, str, int], tuple[float, pd.DataFrame]] = {}


def _get_cached(key: tuple[str, str, int], ttl_seconds: int) -> pd.DataFrame | None:
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


def _put_cached(key: tuple[str, str, int], frame: pd.DataFrame, ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        return
    with _cache_lock:
        _cache[key] = (time.monotonic(), frame.copy(deep=True))
        if len(_cache) > 128:
            oldest = min(_cache, key=lambda k: _cache[k][0])
            _cache.pop(oldest, None)


def _ticker(symbol: str) -> str:
    return symbol.upper().replace("/", "").replace("_", "")


def _get(
    token: str,
    ticker: str,
    params: dict[str, str],
) -> list[dict]:
    if not token:
        raise ProviderError("DATA UNAVAILABLE: TIINGO_API_TOKEN is not configured.")

    response = httpx.get(
        f"{BASE_URL}/tiingo/fx/{ticker}/prices",
        params=params,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {token}",
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ProviderError("PROVIDER ERROR: unexpected Tiingo FX response format.")
    if not payload:
        raise ProviderError("DATA UNAVAILABLE: Tiingo FX returned no values.")
    return payload


def _normalize(payload: list[dict], symbol: str, timeframe: str) -> pd.DataFrame:
    frame = pd.DataFrame(payload).rename(columns={"date": "timestamp", "last": "close"})
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ProviderError(f"PROVIDER ERROR: Tiingo FX missing OHLC fields: {missing}")

    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    raw_timestamp = frame["timestamp"].astype(str)
    frame["timestamp"] = pd.to_datetime(raw_timestamp, utc=True, errors="coerce")
    frame["provider_timestamp"] = raw_timestamp
    frame = (
        frame.dropna(subset=required)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ProviderError("DATA UNAVAILABLE: no valid Tiingo FX OHLC rows remain.")

    frame["provider"] = "Tiingo FX"
    frame["instrument"] = symbol.upper().replace("_", "/")
    frame["timeframe"] = timeframe
    frame["data_status"] = "REAL_DATA"
    frame["calculation_status"] = "SOURCE"
    return frame


def _fetch_intraday(
    token: str,
    symbol: str,
    interval: str,
    resample_freq: str,
    history_days: int,
    cache_ttl_seconds: int,
) -> pd.DataFrame:
    days = max(30, int(history_days))
    key = (_ticker(symbol), interval, days)
    cached = _get_cached(key, cache_ttl_seconds)
    if cached is not None:
        return cached

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    payload = _get(
        token,
        _ticker(symbol),
        {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "resampleFreq": resample_freq,
            "columns": "date,open,high,low,close",
        },
    )
    frame = _normalize(payload, symbol, interval)
    _put_cached(key, frame, cache_ttl_seconds)
    return frame.copy(deep=True)


def _fetch_hourly(
    token: str,
    symbol: str,
    history_days: int,
    cache_ttl_seconds: int,
) -> pd.DataFrame:
    return _fetch_intraday(
        token,
        symbol,
        "1h",
        "1hour",
        history_days,
        cache_ttl_seconds,
    )


def _fetch_daily_native(
    token: str,
    symbol: str,
    history_days: int,
    cache_ttl_seconds: int,
) -> pd.DataFrame:
    days = max(365, int(history_days))
    key = (_ticker(symbol), "1day", days)
    cached = _get_cached(key, cache_ttl_seconds)
    if cached is not None:
        return cached

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    payload = _get(
        token,
        _ticker(symbol),
        {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "resampleFreq": "1day",
            "columns": "date,open,high,low,close",
        },
    )
    frame = _normalize(payload, symbol, "1day")

    # Canonical FX session metadata. We retain the provider's daily OHLC values
    # and use the documented 17:00 America/New_York session close as the
    # observation/close time for freshness calculations.
    tz = ZoneInfo(settings.fx_daily_boundary_timezone)
    session_dates = frame["provider_timestamp"].str.slice(0, 10)
    session_date = pd.to_datetime(session_dates, errors="coerce").dt.date
    frame["session_date"] = session_date
    frame["session_close_local"] = [
        datetime.combine(d, datetime.min.time()).replace(
            hour=settings.fx_daily_boundary_hour_local,
            minute=0,
            tzinfo=tz,
        )
        if pd.notna(d)
        else None
        for d in session_date
    ]
    frame["timestamp"] = pd.to_datetime(frame["session_close_local"], utc=True, errors="coerce")
    frame["source_timeframe"] = "1day"
    frame["calculation_status"] = "SOURCE"

    # Never admit a future or currently forming FX session.
    now_utc = datetime.now(timezone.utc)
    frame = frame.loc[frame["timestamp"].map(lambda x: pd.notna(x) and x.to_pydatetime() <= now_utc)].copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ProviderError("DATA UNAVAILABLE: no completed Tiingo daily FX sessions remain.")

    _put_cached(key, frame, cache_ttl_seconds)
    return frame.copy(deep=True)


def _aggregate_hourly(frame: pd.DataFrame, rule: str, target_timeframe: str, expected_bars: int) -> pd.DataFrame:
    source = frame.set_index("timestamp").sort_index()
    grouped = source.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    counts = source["close"].resample(rule, label="left", closed="left").count()
    grouped["source_row_count"] = counts
    grouped = grouped[grouped["source_row_count"] == expected_bars]
    grouped = grouped.drop(columns=["source_row_count"]).dropna(subset=["open", "high", "low", "close"])
    grouped = grouped.reset_index()
    grouped["provider"] = "Tiingo FX"
    grouped["instrument"] = frame["instrument"].iloc[0]
    grouped["timeframe"] = target_timeframe
    grouped["data_status"] = "REAL_DATA"
    grouped["source_timeframe"] = "1h"
    grouped["calculation_status"] = "CALCULATED"
    grouped["source_row_count"] = expected_bars
    return grouped.reset_index(drop=True)


def fetch_time_series(
    api_key: str,
    symbol: str = "EUR/USD",
    interval: str = "1h",
    history_days: int = 365,
    cache_ttl_seconds: int = 300,
) -> pd.DataFrame:
    interval = interval.strip().lower()
    if interval == "15m":
        return _fetch_intraday(api_key, symbol, "15m", "15min", history_days, cache_ttl_seconds)
    if interval == "30m":
        return _fetch_intraday(api_key, symbol, "30m", "30min", history_days, cache_ttl_seconds)
    if interval == "1h":
        return _fetch_hourly(api_key, symbol, history_days, cache_ttl_seconds)
    if interval == "4h":
        hourly = _fetch_hourly(api_key, symbol, history_days, cache_ttl_seconds)
        return _aggregate_hourly(hourly, "4h", "4h", expected_bars=4)
    if interval == "1day":
        return _fetch_daily_native(api_key, symbol, history_days, cache_ttl_seconds)
    raise ProviderError(f"DATA UNAVAILABLE: unsupported Tiingo interval '{interval}'.")


def inspect_time_series(
    api_key: str,
    symbol: str = "EUR/USD",
    history_days: int = 7,
) -> dict:
    frame = _fetch_hourly(api_key, symbol, history_days, cache_ttl_seconds=0)
    return {
        "provider": "Tiingo FX",
        "symbol": symbol.upper().replace("_", "/"),
        "source_interval": "1h",
        "returned_row_count": int(len(frame)),
        "first_timestamp": str(frame["timestamp"].min()),
        "last_timestamp": str(frame["timestamp"].max()),
        "request_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider_status": "ok",
        "data_available": True,
        "derived_intervals": ["4h"],
        "supported_intraday_intervals": ["15m", "30m", "1h", "4h"],
        "daily_source": "Tiingo native 1day",
        "data_status": "REAL_DATA",
    }
