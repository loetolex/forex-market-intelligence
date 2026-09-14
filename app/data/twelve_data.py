from __future__ import annotations

import threading
import time
from collections import deque

import httpx
import pandas as pd

from app.config import settings

BASE_URL = "https://api.twelvedata.com"


class ProviderError(RuntimeError):
    pass


# Process-local protection. Railway currently runs this API as a single service
# process, so this prevents bursts from one service instance from exhausting
# the provider quota. The cache also prevents repeated identical requests.
_rate_lock = threading.Lock()
_request_times: deque[float] = deque()
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str, int], tuple[float, pd.DataFrame]] = {}


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


def _get_cached(key: tuple[str, str, int]) -> pd.DataFrame | None:
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

        # Never expose the mutable cached object to callers.
        return frame.copy(deep=True)


def _put_cached(key: tuple[str, str, int], frame: pd.DataFrame) -> None:
    ttl = max(0, int(settings.twelve_data_cache_ttl_seconds))
    if ttl == 0:
        return

    with _cache_lock:
        _cache[key] = (time.monotonic(), frame.copy(deep=True))

        # Keep the process cache bounded.
        if len(_cache) > 256:
            oldest_key = min(_cache, key=lambda k: _cache[k][0])
            _cache.pop(oldest_key, None)


def _request_time_series(
    api_key: str,
    symbol: str,
    interval: str,
    outputsize: int,
) -> dict:
    retries = max(0, int(settings.twelve_data_max_429_retries))

    for attempt in range(retries + 1):
        _wait_for_rate_slot()

        response = httpx.get(
            f"{BASE_URL}/time_series",
            params={
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "order": "asc",
                "timezone": "UTC",
                "format": "JSON",
                "apikey": api_key,
            },
            timeout=45,
        )

        if response.status_code == 429:
            if attempt >= retries:
                raise ProviderError(
                    "DATA UNAVAILABLE: Twelve Data rate limit reached after retry."
                )

            # Do not immediately hammer the provider again. The next attempt
            # goes through the limiter as well, preserving the global quota.
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
    cache_key = (symbol, interval, outputsize)

    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    payload = _request_time_series(
        api_key,
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
    )

    if payload.get("status") == "error":
        message = payload.get("message", "Unknown Twelve Data error")
        raise ProviderError(f"PROVIDER ERROR: {message}")

    values = payload.get("values", [])
    if not values:
        raise ProviderError("DATA UNAVAILABLE: Twelve Data returned no values.")

    df = pd.DataFrame(values).rename(columns={"datetime": "timestamp"})
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ProviderError(f"PROVIDER ERROR: missing OHLC fields: {missing}")

    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    )

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
