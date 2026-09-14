from __future__ import annotations

import httpx
import pandas as pd

BASE_URL = "https://api.twelvedata.com"


class ProviderError(RuntimeError):
    pass


def fetch_time_series(
    api_key: str,
    symbol: str = "EUR/USD",
    interval: str = "1h",
    outputsize: int = 500,
) -> pd.DataFrame:
    if not api_key:
        raise ProviderError("DATA UNAVAILABLE: TWELVE_DATA_API_KEY is not configured.")

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

    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == "error":
        raise ProviderError(
            f"PROVIDER ERROR: {payload.get('message', 'Unknown Twelve Data error')}"
        )

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
        raise ProviderError("DATA UNAVAILABLE: no valid OHLC rows remain after validation.")

    df["provider"] = "Twelve Data"
    df["instrument"] = symbol
    df["timeframe"] = interval
    df["data_status"] = "REAL_DATA"

    return df
