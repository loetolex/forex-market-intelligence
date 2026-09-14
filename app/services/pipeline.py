from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from app.config import settings
from app.data.tiingo_fx import fetch_time_series
from app.execution.ibkr_readonly import read_only_status
from app.models.multi_timeframe import combine_timeframes, forecast_90d, forecast_timeframe
from app.risk.gate import evaluate_signal


CANDLE_DELTAS = {
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}


def _fx_boundary_tz() -> ZoneInfo:
    return ZoneInfo(settings.fx_daily_boundary_timezone)


def _daily_session_end_from_label(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Map a Tiingo daily date label to the canonical FX session close."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(_fx_boundary_tz())
    end_local = datetime.combine(
        local.date(),
        time(hour=settings.fx_daily_boundary_hour_local),
        tzinfo=_fx_boundary_tz(),
    )
    return pd.Timestamp(end_local).tz_convert("UTC")


def normalize_native_daily_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize native Tiingo 1D date labels to canonical FX session closes."""
    if df.empty:
        raise RuntimeError("DATA UNAVAILABLE: no daily observations returned by Tiingo.")

    frame = df.copy()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[timestamps.notna()].copy()
    frame["timestamp"] = timestamps.loc[frame.index]
    frame["bar_end_timestamp"] = frame["timestamp"].map(_daily_session_end_from_label)

    now = pd.Timestamp.now(tz="UTC")
    frame = frame.loc[frame["bar_end_timestamp"] <= now].copy()
    if frame.empty:
        raise RuntimeError("DATA UNAVAILABLE: no completed Tiingo daily FX sessions are available.")

    frame = (
        frame.sort_values("bar_end_timestamp")
        .drop_duplicates("bar_end_timestamp")
        .reset_index(drop=True)
    )
    frame["timestamp"] = frame["bar_end_timestamp"]
    frame["timeframe"] = "1day"
    frame["source_timeframe"] = "1day"
    frame["calculation_status"] = "SOURCE"
    frame["data_status"] = "REAL_DATA"
    frame["fx_daily_boundary"] = (
        f"{settings.fx_daily_boundary_hour_local:02d}:00 {settings.fx_daily_boundary_timezone}"
    )
    return frame


def validate_freshness(df: pd.DataFrame, max_stale_minutes: int, interval: str) -> None:
    if df.empty:
        raise RuntimeError("DATA UNAVAILABLE: no market observations remain after validation.")

    latest_close = pd.to_datetime(
        df["bar_end_timestamp"].max()
        if "bar_end_timestamp" in df.columns
        else df["timestamp"].max(),
        utc=True,
        errors="coerce",
    )
    if pd.isna(latest_close):
        raise RuntimeError("DATA UNAVAILABLE: invalid latest candle close timestamp.")

    if "bar_end_timestamp" not in df.columns:
        latest_close = latest_close + CANDLE_DELTAS.get(interval, pd.Timedelta(0))

    age_minutes = (
        datetime.now(timezone.utc) - latest_close.to_pydatetime()
    ).total_seconds() / 60

    if age_minutes < 0:
        raise RuntimeError(
            f"DATA UNAVAILABLE: latest {interval} candle has a future close timestamp."
        )

    if age_minutes > max_stale_minutes:
        raise RuntimeError(
            f"STALE_DATA: latest {interval} market observation is {age_minutes:.1f} minutes old."
        )


def keep_closed_candles(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Admit only fully closed candles for provider-supplied intraday intervals."""
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


def aggregate_from_frame(
    source_frame: pd.DataFrame,
    rule: str,
    target_timeframe: str,
    expected_bars: int,
) -> pd.DataFrame:
    if source_frame.empty:
        raise RuntimeError("DATA UNAVAILABLE: no source data for aggregation.")
    if source_frame["data_status"].ne("REAL_DATA").any():
        raise RuntimeError("DATA UNAVAILABLE: aggregation requires REAL_DATA observations.")

    frame = source_frame.copy()
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
    grouped = grouped.drop(columns=["source_row_count"]).dropna(subset=["open", "high", "low", "close"])
    grouped = grouped.reset_index()
    grouped["bar_end_timestamp"] = grouped["timestamp"] + CANDLE_DELTAS[target_timeframe]
    grouped["provider"] = "Tiingo FX"
    grouped["instrument"] = source_frame["instrument"].iloc[0]
    grouped["timeframe"] = target_timeframe
    grouped["data_status"] = "REAL_DATA"
    grouped["source_timeframe"] = str(source_frame["timeframe"].iloc[0])
    grouped["calculation_status"] = "CALCULATED"
    grouped["source_row_count"] = expected_bars
    return grouped.reset_index(drop=True)


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().replace("_", "/")
    if "/" not in symbol and len(symbol) == 6:
        symbol = f"{symbol[:3]}/{symbol[3:]}"
    return symbol


def _freshness_limit(interval: str) -> int:
    return int(settings.freshness_limits_minutes.get(interval, settings.max_stale_minutes))


def run_market_cycle(symbol: str) -> dict:
    symbol = normalize_symbol(symbol)

    frames: dict[str, pd.DataFrame] = {}
    per_tf: list[dict] = []

    # Each operational timeframe is fetched/derived sequentially. This avoids
    # parallel provider pressure while keeping the five-timeframe stack explicit.
    if "15m" in settings.forecast_intervals:
        fifteen = fetch_time_series(
            settings.tiingo_api_token,
            symbol=symbol,
            interval="15m",
            history_days=settings.tiingo_intraday_history_days,
            cache_ttl_seconds=settings.tiingo_cache_ttl_seconds,
        )
        fifteen = keep_closed_candles(fifteen, "15m")
        validate_freshness(fifteen, _freshness_limit("15m"), "15m")
        frames["15m"] = fifteen
        per_tf.append(forecast_timeframe(fifteen, symbol, "15m"))

    if "30m" in settings.forecast_intervals:
        thirty = fetch_time_series(
            settings.tiingo_api_token,
            symbol=symbol,
            interval="30m",
            history_days=settings.tiingo_intraday_history_days,
            cache_ttl_seconds=settings.tiingo_cache_ttl_seconds,
        )
        thirty = keep_closed_candles(thirty, "30m")
        validate_freshness(thirty, _freshness_limit("30m"), "30m")
        frames["30m"] = thirty
        per_tf.append(forecast_timeframe(thirty, symbol, "30m"))

    if "1h" in settings.forecast_intervals:
        hourly = fetch_time_series(
            settings.tiingo_api_token,
            symbol=symbol,
            interval="1h",
            history_days=settings.tiingo_intraday_history_days,
            cache_ttl_seconds=settings.tiingo_cache_ttl_seconds,
        )
        hourly = keep_closed_candles(hourly, "1h")
        validate_freshness(hourly, _freshness_limit("1h"), "1h")
        frames["1h"] = hourly
        per_tf.append(forecast_timeframe(hourly, symbol, "1h"))

    if "4h" in settings.forecast_intervals:
        hourly_source = frames.get("1h")
        if hourly_source is None:
            hourly_source = fetch_time_series(
                settings.tiingo_api_token,
                symbol=symbol,
                interval="1h",
                history_days=settings.tiingo_intraday_history_days,
                cache_ttl_seconds=settings.tiingo_cache_ttl_seconds,
            )
            hourly_source = keep_closed_candles(hourly_source, "1h")
        four_hour = aggregate_from_frame(hourly_source, "4h", "4h", expected_bars=4)
        validate_freshness(four_hour, _freshness_limit("4h"), "4h")
        frames["4h"] = four_hour
        forecast_4h = forecast_timeframe(four_hour, symbol, "4h")
        forecast_4h["calculation_status"] = "CALCULATED"
        forecast_4h["source_timeframe"] = "1h"
        per_tf.append(forecast_4h)

    if "1day" in settings.forecast_intervals:
        daily = fetch_time_series(
            settings.tiingo_api_token,
            symbol=symbol,
            interval="1day",
            history_days=settings.tiingo_daily_history_days,
            cache_ttl_seconds=settings.tiingo_cache_ttl_seconds,
        )
        daily = daily.loc[daily["data_status"].eq("REAL_DATA")].copy()
        daily = normalize_native_daily_sessions(daily)
        validate_freshness(daily, _freshness_limit("1day"), "1day")
        frames["1day"] = daily
        daily_forecast = forecast_timeframe(daily, symbol, "1day")
        daily_forecast["calculation_status"] = "SOURCE"
        daily_forecast["source_timeframe"] = "1day"
        daily_forecast["fx_daily_boundary"] = f"{settings.fx_daily_boundary_hour_local:02d}:00 {settings.fx_daily_boundary_timezone}"
        per_tf.append(daily_forecast)

    combined = combine_timeframes(per_tf)
    daily = frames.get("1day")
    long_horizon = forecast_90d(daily, symbol) if daily is not None else {
        "status": "DATA UNAVAILABLE", "reason": "DAILY_DATA_MISSING", "horizon": "90D"
    }
    if daily is not None:
        long_horizon["calculation_status"] = "SOURCE"
        long_horizon["source_timeframe"] = "1day"
        long_horizon["fx_daily_boundary"] = f"{settings.fx_daily_boundary_hour_local:02d}:00 {settings.fx_daily_boundary_timezone}"

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
            "status": "SOURCE" if "1day" in frames else "DATA UNAVAILABLE",
            "source": "Tiingo FX native 1day REAL_DATA" if "1day" in frames else "DATA UNAVAILABLE",
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
