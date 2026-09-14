from __future__ import annotations

from datetime import datetime, timezone
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

    latest = pd.to_datetime(df["timestamp"].max(), utc=True, errors="coerce")
    if pd.isna(latest):
        raise RuntimeError("DATA UNAVAILABLE: invalid latest timestamp.")

    if interval == "1day":
        # Internally calculated daily candles are timestamped at the UTC day start.
        # Their actual close occurs 24 hours later. Freshness is therefore measured
        # from the calculated candle close, not from the midnight label.
        latest_close = latest + pd.Timedelta(days=1)
        age_minutes = (
            datetime.now(timezone.utc) - latest_close.to_pydatetime()
        ).total_seconds() / 60
    else:
        age_minutes = (
            datetime.now(timezone.utc) - latest.to_pydatetime()
        ).total_seconds() / 60

    if age_minutes > max_stale_minutes:
        raise RuntimeError(
            f"STALE_DATA: latest {interval} market observation is {age_minutes:.1f} minutes old."
        )


def keep_closed_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Admit only closed intraday candles.

    The provider timestamps are interpreted as candle-open timestamps. Current
    incomplete bars are excluded before they can reach features or models.
    """
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


def aggregate_daily_from_hourly(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Build complete UTC 1-day OHLC candles from verified closed 1H REAL_DATA.

    This never invents market values: open/high/low/close/volume are aggregated
    only from provider observations. A UTC day is admitted only when all 24
    hourly candles are present and consecutive.
    """
    if df_1h.empty:
        raise RuntimeError("DATA UNAVAILABLE: no hourly source data for daily aggregation.")

    frame = df_1h.copy()
    if frame["data_status"].ne("REAL_DATA").any():
        raise RuntimeError("DATA UNAVAILABLE: daily aggregation requires REAL_DATA hourly observations.")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[timestamps.notna()].copy()
    frame["timestamp"] = timestamps.loc[frame.index]
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    frame["day"] = frame["timestamp"].dt.floor("D")
    records: list[dict] = []

    for day, group in frame.groupby("day", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        timestamps = group["timestamp"]

        # A complete UTC day must contain exactly 24 hourly candles spanning
        # 00:00 through 23:00 with no gaps or duplicates.
        if len(group) != 24:
            continue
        expected = pd.date_range(day, day + pd.Timedelta(hours=23), freq="1h", tz="UTC")
        if not timestamps.equals(pd.Series(expected)):
            continue

        record = {
            "timestamp": day,
            "open": float(group.iloc[0]["open"]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group.iloc[-1]["close"]),
            "provider": "Twelve Data",
            "instrument": str(group.iloc[0]["instrument"]),
            "timeframe": "1day",
            "data_status": "REAL_DATA",
            "calculation_status": "CALCULATED",
            "source_timeframe": "1h",
            "source_row_count": 24,
        }
        if "volume" in group.columns:
            volume = pd.to_numeric(group["volume"], errors="coerce").dropna()
            if not volume.empty:
                record["volume"] = float(volume.sum())
        records.append(record)

    daily = pd.DataFrame(records)
    if daily.empty:
        raise RuntimeError(
            "DATA UNAVAILABLE: no complete UTC daily candles could be calculated from closed 1H data."
        )

    return daily.reset_index(drop=True)


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().replace("_", "/")
    if "/" not in symbol and len(symbol) == 6:
        symbol = f"{symbol[:3]}/{symbol[3:]}"
    return symbol


def run_market_cycle(symbol: str) -> dict:
    symbol = normalize_symbol(symbol)
    frames: dict[str, pd.DataFrame] = {}
    per_tf = []

    # Pull the provider's verified intraday source first. This is the only source
    # used to construct our internally calculated 1D candles.
    hourly_raw = fetch_time_series(
        settings.twelve_data_api_key,
        symbol=symbol,
        interval="1h",
        outputsize=settings.forecast_outputsize,
    )
    if hourly_raw["data_status"].ne("REAL_DATA").any():
        raise RuntimeError("DATA UNAVAILABLE: provider provenance is not REAL_DATA.")
    hourly_closed = keep_closed_candles(hourly_raw, "1h")
    validate_freshness(hourly_closed, settings.max_stale_minutes, "1h")
    frames["1h"] = hourly_closed
    per_tf.append(forecast_timeframe(hourly_closed, symbol, "1h"))

    # Keep the existing 4H provider timeframe.
    if "4h" in settings.forecast_intervals:
        raw_4h = fetch_time_series(
            settings.twelve_data_api_key,
            symbol=symbol,
            interval="4h",
            outputsize=settings.forecast_outputsize,
        )
        if raw_4h["data_status"].ne("REAL_DATA").any():
            raise RuntimeError("DATA UNAVAILABLE: provider provenance is not REAL_DATA.")
        closed_4h = keep_closed_candles(raw_4h, "4h")
        validate_freshness(
            closed_4h,
            settings.max_stale_minutes * 8,
            "4h",
        )
        frames["4h"] = closed_4h
        per_tf.append(forecast_timeframe(closed_4h, symbol, "4h"))

    # Option A: calculate the 1D timeframe from verified closed 1H REAL_DATA.
    if "1day" in settings.forecast_intervals:
        daily = aggregate_daily_from_hourly(hourly_closed)
        validate_freshness(daily, settings.max_stale_minutes * 8, "1day")
        frames["1day"] = daily
        daily_forecast = forecast_timeframe(daily, symbol, "1day")
        daily_forecast["calculation_status"] = "CALCULATED"
        daily_forecast["source_timeframe"] = "1h"
        per_tf.append(daily_forecast)

    combined = combine_timeframes(per_tf)
    daily = frames.get("1day")
    long_horizon = (
        forecast_90d(daily, symbol)
        if daily is not None
        else {"status": "DATA UNAVAILABLE", "reason": "DAILY_DATA_MISSING", "horizon": "90D"}
    )
    if daily is not None:
        long_horizon["calculation_status"] = "CALCULATED"
        long_horizon["source_timeframe"] = "1h"

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
        "daily_aggregation": {
            "status": "CALCULATED" if "1day" in frames else "DATA UNAVAILABLE",
            "source": "Twelve Data 1H REAL_DATA",
            "complete_utc_days": int(len(frames["1day"])) if "1day" in frames else 0,
        },
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
