from __future__ import annotations

from datetime import datetime, timedelta, timezone
import threading
import time
from collections import deque

import httpx
import pandas as pd

from app.config import settings

BASE_URL = "https://api.twelvedata.com"


class ProviderError(RuntimeError):
    pass


_rate_lock = threading.Lock()
_request_times: deque[float] = deque()
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str, int, str | None], tuple[float, pd.DataFrame]] = {}


def _wait_for_rate_slot() -> None:
    limit = max(1, int(settings.twelve_data_requests_per_minute))
    while True:
        with _rate_lock:
            now = time.monotonic()
            cutoff = now - 60.0
            while _request_times and _request_times[0] <= cutoff:
                _request_times.popleft()
            if len(_request_times) < limit:
                _request_times.append(now)
                return
            wait_seconds = max(0.1, 60.0 - (now - _request_times[0]))
        time.sleep(wait_seconds)


def _get_cached(key: tuple[str, str, int, str | None]) -> pd.DataFrame | None:
    ttl = max(0, int(settings.twelve_data_cache_ttl_seconds))
    if ttl == 0:
        return None
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        created_at, frame = entry
        if time.monotonic() - created_at > ttl:
            _cache.pop(key, None)
            return None
        return frame.copy(deep=True)


def _put_cached(key: tuple[str, str, int, str | None], frame: pd.DataFrame) -> None:
    ttl = max(0, int(settings.twelve_data_cache_ttl_seconds))
    if ttl == 0:
        return
    with _cache_lock:
        _cache[key] = (time.monotonic(), frame.copy(deep=True))
        if len(_cache) > 256:
            oldest_key = min(_cache, key=lambda k: _cache[k][0])
            _cache.pop(oldest_key, None)


def _daily_end_date() -> str:
    """Return yesterday's UTC calendar date so the current daily bar is excluded."""
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    return yesterday.isoformat()


def _request_time_series(
    api_key: str,
    symbol: str,
    interval: str,
    outputsize: int,
    end_date: str | None,
) -> dict:
    retries = max(0, int(settings.twelve_data_max_429_retries))
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "order": "asc",
        "format": "JSON",
        "apikey": api_key,
    }
    if interval != "1day":
        params["timezone"] = "UTC"
    if end_date is not None:
        params["end_date"] = end_date

    for attempt in range(retries + 1):
        _wait_for_rate_slot()
        response = httpx.get(
            f"{BASE_URL}/time_series",
            params=params,
            timeout=45,
        )
        if response.status_code == 429:
            if attempt >= retries:
                raise ProviderError(
                    "DATA UNAVAILABLE: Twelve Data rate limit reached after retry."
                )
            time.sleep(max(1, int(settings.twelve_data_retry_wait_seconds)))
            continue
        response.raise_for_status()
        return response.json()

    raise ProviderError("DATA UNAVAILABLE: Twelve Data request could not be completed.")


def fetch_time_series(
    api_key: str,
    symbol: str = "EUR/USD",
    interval: str = "1h",
    outputsize: int = 500,
) -> pd.DataFrame:
    if not api_key:
        raise ProviderError("DATA UNAVAILABLE: TWELVE_DATA_API_KEY is not configured.")

    symbol = symbol.upper().replace("_", "/")
    interval = interval.strip().lower()
    outputsize = int(outputsize)
    end_date = _daily_end_date() if interval == "1day" else None
    cache_key = (symbol, interval, outputsize, end_date)

    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    payload = _request_time_series(
        api_key,
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
        end_date=end_date,
    )

    if payload.get("status") == "error":
        raise ProviderError(
            f"PROVIDER ERROR: {payload.get('message', 'Unknown Twelve Data error')}"
        )

    values = payload.get("values", [])
    if not values:
        raise ProviderError("DATA UNAVAILABLE: Twelve Data returned no values.")

    meta = payload.get("meta") or {}
    exchange_timezone = meta.get("exchange_timezone") or "DATA_UNAVAILABLE"

    df = pd.DataFrame(values).rename(columns={"datetime": "timestamp"})
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ProviderError(f"PROVIDER ERROR: missing OHLC fields: {missing}")

    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Twelve Data documents timestamps as candle-open timestamps. Intraday
    # values are requested explicitly in UTC. Daily timestamps are intentionally
    # retained as the provider's exchange-local date label because timezone is
    # ignored for 1day bars.
    raw_timestamp = df["timestamp"].astype(str)
    df["timestamp"] = pd.to_datetime(raw_timestamp, utc=True, errors="coerce")
    df["provider_timestamp_local"] = raw_timestamp
    df["provider_exchange_timezone"] = exchange_timezone

    df = (
        df.dropna(subset=required)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ProviderError(
            "DATA UNAVAILABLE: no valid OHLC rows remain after validation."
        )

    df["provider"] = "Twelve Data"
    df["instrument"] = symbol
    df["timeframe"] = interval
    df["data_status"] = "REAL_DATA"

    _put_cached(cache_key, df)
    return df.copy(deep=True)
