from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd

from app.config import settings
from app.data.twelve_data import fetch_time_series
from app.execution.ibkr_readonly import read_only_status
from app.models.multi_timeframe import combine_timeframes, forecast_90d, forecast_timeframe
from app.risk.gate import evaluate_signal


CANDLE_DELTAS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}


def validate_freshness(df: pd.DataFrame, max_stale_minutes: int, interval: str) -> None:
    if df.empty:
        raise RuntimeError("DATA UNAVAILABLE: no market observations remain after validation.")

    if interval == "1day":
        # Daily bars are returned by Twelve Data using the provider/exchange
        # local date label; timezone=UTC is ignored for 1day. Compare calendar
        # dates in that provider timezone instead of treating the label as UTC.
        raw_label = str(df["provider_timestamp_local"].iloc[-1]).split(" ")[0]
        try:
            bar_date = datetime.fromisoformat(raw_label).date()
        except ValueError as exc:
            raise RuntimeError("DATA UNAVAILABLE: invalid daily provider date label.") from exc

        provider_tz = str(df["provider_exchange_timezone"].iloc[-1])
        if provider_tz == "DATA_UNAVAILABLE":
            raise RuntimeError("DATA UNAVAILABLE: daily provider timezone metadata missing.")

        try:
            current_provider_date = datetime.now(timezone.utc).astimezone(ZoneInfo(provider_tz)).date()
        except Exception as exc:
            raise RuntimeError(
                f"DATA UNAVAILABLE: unsupported provider timezone '{provider_tz}'."
            ) from exc

        # The daily request already excludes the current provider-incomplete
        # calendar day. The latest returned date must therefore be either today
        # only if the provider has explicitly completed it, or the immediately
        # preceding completed calendar date. Do not reject a valid completed bar
        # solely because its exchange-local label is ahead of UTC.
        day_age = (current_provider_date - bar_date).days
        if day_age < 0:
            raise RuntimeError("DATA UNAVAILABLE: daily provider date is in the future.")
        max_daily_days = max(1, int((max_stale_minutes * 8 + 1439) // 1440))
        if day_age > max_daily_days:
            raise RuntimeError(
                f"STALE_DATA: latest daily market date is {day_age} calendar days old."
            )
        return

    latest = pd.to_datetime(df["timestamp"].max(), utc=True, errors="coerce")
    if pd.isna(latest):
        raise RuntimeError("DATA UNAVAILABLE: invalid latest timestamp.")
    age_minutes = (datetime.now(timezone.utc) - latest.to_pydatetime()).total_seconds() / 60
    if age_minutes > max_stale_minutes:
        raise RuntimeError(f"STALE_DATA: latest market observation is {age_minutes:.1f} minutes old.")


def keep_closed_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Admit only closed intraday candles; daily bars use provider date boundaries."""
    if interval not in CANDLE_DELTAS:
        return df.copy()

    now = pd.Timestamp.now(tz="UTC")
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    close_times = timestamps + CANDLE_DELTAS[interval]
    mask = timestamps.notna() & (timestamps <= now) & (close_times <= now)
    closed = df.loc[mask].copy()

    if closed.empty:
        raise RuntimeError(f"DATA UNAVAILABLE: no closed {interval} candles are available.")

    if len(closed) < 20:
        raise RuntimeError(
            f"DATA UNAVAILABLE: only {len(closed)} closed {interval} candles remain after completeness validation."
        )

    return closed.reset_index(drop=True)


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
        raw = keep_closed_candles(raw, interval)
        freshness_limit = settings.max_stale_minutes if interval == settings.primary_interval else settings.max_stale_minutes * 8
        validate_freshness(raw, freshness_limit, interval)
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
