from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import settings
from app.data.tiingo_fx import fetch_time_series
from app.execution.ibkr_readonly import read_only_status
from app.models.multi_timeframe import combine_timeframes, forecast_90d, forecast_timeframe
from app.risk.gate import evaluate_signal


CANDLE_DELTAS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1day": pd.Timedelta(days=1),
}


def validate_freshness(df: pd.DataFrame, max_stale_minutes: int, interval: str) -> None:
    if df.empty:
        raise RuntimeError("DATA UNAVAILABLE: no market observations remain after validation.")

    if "bar_end_timestamp" in df.columns:
        latest_close = pd.to_datetime(df["bar_end_timestamp"].max(), utc=True, errors="coerce")
    else:
        latest = pd.to_datetime(df["timestamp"].max(), utc=True, errors="coerce")
        if pd.isna(latest):
            raise RuntimeError("DATA UNAVAILABLE: invalid latest timestamp.")
        latest_close = latest + CANDLE_DELTAS.get(interval, pd.Timedelta(0))

    if pd.isna(latest_close):
        raise RuntimeError("DATA UNAVAILABLE: invalid latest candle close timestamp.")

    age_minutes = (
        datetime.now(timezone.utc) - latest_close.to_pydatetime()
    ).total_seconds() / 60

    if age_minutes < 0:
        raise RuntimeError(f"DATA UNAVAILABLE: latest {interval} candle has a future close timestamp.")

    if age_minutes > max_stale_minutes:
        raise RuntimeError(
            f"STALE_DATA: latest {interval} market observation is {age_minutes:.1f} minutes old."
        )


def keep_closed_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Admit only fully closed candles for provider-supplied intervals."""
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

    closed["bar_end_timestamp"] = close_times.loc[closed.index]
    return closed.reset_index(drop=True)


def _fx_session_window(session_date: date, boundary_tz: ZoneInfo, boundary_hour: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_local = datetime.combine(session_date, time(hour=boundary_hour), tzinfo=boundary_tz)
    end_local = start_local + timedelta(days=1)
    return (
        pd.Timestamp(start_local).tz_convert("UTC"),
        pd.Timestamp(end_local).tz_convert("UTC"),
    )


def aggregate_fx_daily_from_hourly(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Build canonical FX trading-day candles from verified closed 1H REAL_DATA.

    The canonical FX day is defined by a DST-aware local boundary (default
    17:00 America/New_York). The expected hourly bar count is calculated from
    the actual UTC session duration, so DST transition sessions can contain
    23 or 25 hourly bars without being incorrectly rejected.
    """
    if df_1h.empty:
        raise RuntimeError("DATA UNAVAILABLE: no hourly source data for daily aggregation.")
    if df_1h["data_status"].ne("REAL_DATA").any():
        raise RuntimeError("DATA UNAVAILABLE: aggregation requires REAL_DATA hourly observations.")

    boundary_tz = ZoneInfo(settings.fx_daily_boundary_timezone)
    boundary_hour = int(settings.fx_daily_boundary_hour_local)
    if not 0 <= boundary_hour <= 23:
        raise RuntimeError("DATA UNAVAILABLE: invalid FX daily boundary hour configuration.")

    frame = df_1h.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    local_ts = frame["timestamp"].dt.tz_convert(boundary_tz)
    # Session date changes at the configured local boundary, not at UTC midnight.
    session_date = (local_ts - pd.Timedelta(hours=boundary_hour)).dt.date
    frame["session_date"] = session_date

    records: list[dict] = []
    for session_day, group in frame.groupby("session_date", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        start_utc, end_utc = _fx_session_window(session_day, boundary_tz, boundary_hour)
        expected_index = pd.date_range(
            start=start_utc,
            end=end_utc - pd.Timedelta(hours=1),
            freq="1h",
            tz="UTC",
        )
        expected_count = len(expected_index)

        timestamps = group["timestamp"].reset_index(drop=True)
        if len(group) != expected_count:
            continue
        if not timestamps.equals(pd.Series(expected_index, name=None)):
            continue

        record = {
            "timestamp": start_utc,
            "bar_end_timestamp": end_utc,
            "open": float(group.iloc[0]["open"]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group.iloc[-1]["close"]),
            "provider": "Tiingo FX",
            "instrument": str(group.iloc[0]["instrument"]),
            "timeframe": "1day",
            "data_status": "REAL_DATA",
            "source_timeframe": "1h",
            "calculation_status": "CALCULATED",
            "source_row_count": expected_count,
            "fx_daily_boundary": f"{boundary_hour:02d}:00 {settings.fx_daily_boundary_timezone}",
        }
        if "volume" in group.columns:
            volume = pd.to_numeric(group["volume"], errors="coerce").dropna()
            if not volume.empty:
                record["volume"] = float(volume.sum())
        records.append(record)

    daily = pd.DataFrame(records)
    if daily.empty:
        raise RuntimeError(
            "DATA UNAVAILABLE: no complete canonical FX daily candles could be calculated from closed 1H data."
        )

    return daily.reset_index(drop=True)


def aggregate_from_hourly(
    df_1h: pd.DataFrame,
    rule: str,
    target_timeframe: str,
    expected_bars: int,
) -> pd.DataFrame:
    """Aggregate complete fixed-duration candles from verified closed 1H data."""
    if target_timeframe == "1day":
        return aggregate_fx_daily_from_hourly(df_1h)

    if df_1h.empty:
        raise RuntimeError("DATA UNAVAILABLE: no hourly source data for aggregation.")
    if df_1h["data_status"].ne("REAL_DATA").any():
        raise RuntimeError("DATA UNAVAILABLE: aggregation requires REAL_DATA hourly observations.")

    frame = df_1h.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
        .set_index("timestamp")
    )

    grouped = frame.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    counts = frame["close"].resample(rule, label="left", closed="left").count()
    grouped["source_row_count"] = counts
    grouped = grouped[grouped["source_row_count"] == expected_bars]
    grouped = grouped.drop(columns=["source_row_count"]).dropna(
        subset=["open", "high", "low", "close"]
    )
    grouped = grouped.reset_index()
    grouped["bar_end_timestamp"] = grouped["timestamp"] + pd.Timedelta(hours=4)
    grouped["provider"] = "Tiingo FX"
    grouped["instrument"] = df_1h["instrument"].iloc[0]
    grouped["timeframe"] = target_timeframe
    grouped["data_status"] = "REAL_DATA"
    grouped["source_timeframe"] = "1h"
    grouped["calculation_status"] = "CALCULATED"
    grouped["source_row_count"] = expected_bars
    return grouped.reset_index(drop=True)


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().replace("_", "/")
    if "/" not in symbol and len(symbol) == 6:
        symbol = f"{symbol[:3]}/{symbol[3:]}"
    return symbol


def run_market_cycle(symbol: str) -> dict:
    symbol = normalize_symbol(symbol)

    hourly_raw = fetch_time_series(
        settings.tiingo_api_token,
        symbol=symbol,
        interval="1h",
        history_days=settings.tiingo_history_days,
        cache_ttl_seconds=settings.tiingo_cache_ttl_seconds,
    )
    hourly_closed = keep_closed_candles(hourly_raw, "1h")
    validate_freshness(hourly_closed, settings.max_stale_minutes, "1h")

    frames: dict[str, pd.DataFrame] = {"1h": hourly_closed}
    per_tf: list[dict] = [forecast_timeframe(hourly_closed, symbol, "1h")]

    if "4h" in settings.forecast_intervals:
        four_hour = aggregate_from_hourly(
            hourly_closed,
            rule="4h",
            target_timeframe="4h",
            expected_bars=4,
        )
        validate_freshness(four_hour, settings.max_stale_minutes * 8, "4h")
        frames["4h"] = four_hour
        four_hour_forecast = forecast_timeframe(four_hour, symbol, "4h")
        four_hour_forecast["calculation_status"] = "CALCULATED"
        four_hour_forecast["source_timeframe"] = "1h"
        per_tf.append(four_hour_forecast)

    if "1day" in settings.forecast_intervals:
        daily = aggregate_from_hourly(
            hourly_closed,
            rule="1D",
            target_timeframe="1day",
            expected_bars=24,
        )
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
        "market_data_provider": "Tiingo FX",
        "forecast_engine": "MULTI_TIMEFRAME_RESEARCH_CANDIDATE",
        "provenance": "REAL_DATA",
        "timeframes": settings.forecast_intervals,
        "market_rows": {k: len(v) for k, v in frames.items()},
        "last_market_timestamp": {k: str(v["timestamp"].max()) for k, v in frames.items()},
        "daily_aggregation": {
            "status": "CALCULATED" if "1day" in frames else "DATA UNAVAILABLE",
            "source": "Tiingo FX 1H REAL_DATA",
            "complete_fx_sessions": int(len(frames["1day"])) if "1day" in frames else 0,
            "boundary": f"{settings.fx_daily_boundary_hour_local:02d}:00 {settings.fx_daily_boundary_timezone}",
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
