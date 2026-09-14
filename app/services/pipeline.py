from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from app.config import settings
from app.data.twelve_data import fetch_time_series
from app.models.multi_timeframe import combine_timeframes, forecast_timeframe
from app.risk.gate import evaluate_signal


def validate_freshness(df: pd.DataFrame, max_stale_minutes: int) -> None:
    latest = pd.to_datetime(df["timestamp"].max(), utc=True, errors="coerce")
    if pd.isna(latest):
        raise RuntimeError("DATA UNAVAILABLE: invalid latest timestamp.")
    age_minutes = (datetime.now(timezone.utc) - latest.to_pydatetime()).total_seconds() / 60
    if age_minutes > max_stale_minutes:
        raise RuntimeError(f"STALE_DATA: latest market observation is {age_minutes:.1f} minutes old.")


def run_market_cycle(symbol: str) -> dict:
    frames = {}
    per_tf = []
    for interval in settings.forecast_intervals:
        raw = fetch_time_series(
            settings.twelve_data_api_key,
            symbol=symbol,
            interval=interval,
            outputsize=settings.forecast_outputsize,
        )
        if raw["data_status"].ne("REAL_DATA").any():
            raise RuntimeError("DATA UNAVAILABLE: provider provenance is not REAL_DATA.")
        freshness_limit = settings.max_stale_minutes if interval == settings.primary_interval else settings.max_stale_minutes * 8
        validate_freshness(raw, freshness_limit)
        frames[interval] = raw
        per_tf.append(forecast_timeframe(raw, symbol, interval))

    combined = combine_timeframes(per_tf)
    p = combined.get("probability_up")
    risk = evaluate_signal(p)
    signal = "NO TRADE"
    if risk.approved and combined.get("validation_status") == "RESEARCH_CANDIDATE":
        signal = "LONG" if p >= 0.5 else "SHORT"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instrument": symbol,
        "forecast_engine": "MULTI_TIMEFRAME",
        "timeframes": settings.forecast_intervals,
        "market_rows": {k: len(v) for k, v in frames.items()},
        "last_market_timestamp": {k: str(v["timestamp"].max()) for k, v in frames.items()},
        "forecasts": per_tf,
        "combined_forecast": combined,
        "signal": signal,
        "risk": {"approved": risk.approved, "reason": risk.reason},
        "execution": {
            "mode": settings.trading_mode,
            "live_enabled": settings.live_trading_enabled,
            "order_placement_enabled": settings.order_placement_enabled,
            "status": "LOCKED",
        },
    }
