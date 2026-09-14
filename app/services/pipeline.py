from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd

from app.config import settings
from app.data.twelve_data import fetch_time_series
from app.execution.ibkr_readonly import read_only_status
from app.models.multi_timeframe import combine_timeframes, forecast_90d, forecast_timeframe
from app.risk.gate import evaluate_signal


def validate_freshness(df: pd.DataFrame, max_stale_minutes: int) -> None:
    latest = pd.to_datetime(df["timestamp"].max(), utc=True, errors="coerce")
    if pd.isna(latest):
        raise RuntimeError("DATA UNAVAILABLE: invalid latest timestamp.")
    age_minutes = (datetime.now(timezone.utc) - latest.to_pydatetime()).total_seconds() / 60
    if age_minutes > max_stale_minutes:
        raise RuntimeError(f"STALE_DATA: latest market observation is {age_minutes:.1f} minutes old.")


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().replace("_", "/")
    if "/" not in symbol and len(symbol) == 6:
        symbol = f"{symbol[:3]}/{symbol[3:]}"
    return symbol


def run_market_cycle(symbol: str) -> dict:
    symbol = normalize_symbol(symbol)
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
    daily = frames.get("1day")
    long_horizon = forecast_90d(daily, symbol) if daily is not None else {"status": "DATA UNAVAILABLE", "reason": "DAILY_DATA_MISSING", "horizon": "90D"}

    p = combined.get("probability_up")
    agreement = float(combined.get("agreement", 0.0))
    candidate_signal = "NO TRADE"
    if p is not None and abs(float(p) - 0.50) >= 0.10 and agreement >= 0.67:
        candidate_signal = "LONG CANDIDATE" if float(p) > 0.50 else "SHORT CANDIDATE"

    risk = evaluate_signal(p)
    final_decision = candidate_signal if risk.approved else "NO TRADE"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instrument": symbol,
        "forecast_engine": "MULTI_TIMEFRAME_RESEARCH_CANDIDATE",
        "provenance": "REAL_DATA",
        "timeframes": settings.forecast_intervals,
        "market_rows": {k: len(v) for k, v in frames.items()},
        "last_market_timestamp": {k: str(v["timestamp"].max()) for k, v in frames.items()},
        "forecasts": per_tf,
        "combined_forecast": combined,
        "forecast_90d": long_horizon,
        "signal_engine": {
            "candidate_signal": candidate_signal,
            "rule": "edge >= 10 percentage points and timeframe agreement >= 67%",
        },
        "risk": {"approved": risk.approved, "reason": risk.reason},
        "decision": final_decision,
        "execution": {
            "mode": settings.trading_mode,
            "broker": read_only_status().__dict__,
            "live_enabled": settings.live_trading_enabled,
            "order_placement_enabled": settings.order_placement_enabled,
            "status": "LOCKED",
        },
    }
