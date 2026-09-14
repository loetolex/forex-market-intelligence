from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from app.config import settings
from app.data.twelve_data import fetch_time_series
from app.features.technical import add_features
from app.models.baseline import direction_probability_from_momentum
from app.risk.gate import evaluate_signal


def validate_freshness(df: pd.DataFrame) -> None:
    latest = pd.to_datetime(df["timestamp"].max(), utc=True, errors="coerce")
    if pd.isna(latest):
        raise RuntimeError("DATA UNAVAILABLE: invalid latest timestamp.")
    age_minutes = (datetime.now(timezone.utc) - latest.to_pydatetime()).total_seconds() / 60
    if age_minutes > settings.max_stale_minutes:
        raise RuntimeError(f"STALE_DATA: latest market observation is {age_minutes:.1f} minutes old.")


def run_market_cycle(symbol: str) -> dict:
    raw = fetch_time_series(
        settings.twelve_data_api_key,
        symbol=symbol,
        interval=settings.primary_interval,
        outputsize=500,
    )
    if raw["data_status"].ne("REAL_DATA").any():
        raise RuntimeError("DATA UNAVAILABLE: provider provenance is not REAL_DATA.")
    validate_freshness(raw)
    features = add_features(raw)
    forecast = direction_probability_from_momentum(features)
    risk = evaluate_signal(forecast["probability_up"])
    signal = "NO TRADE"
    if risk.approved:
        signal = "LONG" if forecast["probability_up"] >= 0.5 else "SHORT"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instrument": symbol,
        "timeframe": settings.primary_interval,
        "market_rows": len(raw),
        "last_market_timestamp": str(raw["timestamp"].max()),
        "forecast": forecast,
        "signal": signal,
        "risk": {"approved": risk.approved, "reason": risk.reason},
        "execution": {
            "mode": settings.trading_mode,
            "live_enabled": settings.live_trading_enabled,
            "order_placement_enabled": settings.order_placement_enabled,
            "status": "LOCKED",
        },
    }
