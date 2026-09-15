from __future__ import annotations

import pandas as pd

from app.models.timeframe_signals import build_timeframe_opportunities


def _frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-09-15 00:00:00", periods=40, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.10] * 40,
            "high": [1.101] * 40,
            "low": [1.099] * 40,
            "close": [1.10] * 40,
            "provider": ["test"] * 40,
            "instrument": ["EUR/USD"] * 40,
            "timeframe": ["15m"] * 40,
            "data_status": ["REAL_DATA"] * 40,
        }
    )


def _forecasts() -> list[dict]:
    probabilities = {"15m": 0.71, "30m": 0.59, "1h": 0.45, "4h": 0.22, "1day": 0.19}
    return [
        {
            "instrument": "EUR/USD",
            "timeframe": timeframe,
            "status": "MODEL OUTPUT",
            "validation_status": "NOT_EVALUATED",
            "probability_up": probability,
            "probability_down": 1.0 - probability,
            "confidence": abs(2.0 * (probability - 0.5)),
            "expected_return": 0.001,
            "expected_abs_move": 0.002,
            "uncertainty": 0.001,
            "regime": {"regime": "TREND_DOWN"},
            "prediction_timestamp": "2026-09-15T20:45:00+00:00",
        }
        for timeframe, probability in probabilities.items()
    ]


def test_timeframes_remain_independent_when_directions_conflict():
    frame = _frame()
    frames = {timeframe: frame.copy() for timeframe in ("15m", "30m", "1h", "4h", "1day")}
    opportunities = build_timeframe_opportunities(_forecasts(), frames)

    assert opportunities["15m"]["prediction"] == "LONG"
    assert opportunities["1h"]["prediction"] == "SHORT"
    assert opportunities["4h"]["prediction"] == "SHORT"
    assert opportunities["1day"]["prediction"] == "SHORT"
    assert opportunities["15m"]["entry_signal"] is True
    assert opportunities["15m"]["context"]["classification"] == "COUNTER_TREND"
    assert opportunities["30m"]["context"]["classification"] == "COUNTER_TREND"
    assert opportunities["1h"]["context"]["classification"] == "TREND_ALIGNED"


def test_each_timeframe_has_distinct_strategy_identity_and_horizon():
    frame = _frame()
    frames = {timeframe: frame.copy() for timeframe in ("15m", "30m", "1h", "4h", "1day")}
    opportunities = build_timeframe_opportunities(_forecasts(), frames)

    strategy_ids = {opportunities[tf]["strategy_id"] for tf in opportunities}
    horizons = {opportunities[tf]["holding_horizon"] for tf in opportunities}
    assert len(strategy_ids) == 5
    assert len(horizons) == 5


def test_insufficient_edge_is_timeframe_no_trade_not_global_no_trade():
    forecasts = _forecasts()
    for item in forecasts:
        if item["timeframe"] == "30m":
            item["probability_up"] = 0.59
            item["probability_down"] = 0.41
    frame = _frame()
    frames = {timeframe: frame.copy() for timeframe in ("15m", "30m", "1h", "4h", "1day")}
    opportunities = build_timeframe_opportunities(forecasts, frames)

    assert opportunities["30m"]["entry_signal"] is False
    assert opportunities["30m"]["entry_status"] == "NO TRADE"
    assert opportunities["15m"]["entry_signal"] is True
    assert opportunities["4h"]["entry_signal"] is True


def test_real_data_levels_are_calculated_without_synthetic_prices():
    frame = _frame()
    frames = {timeframe: frame.copy() for timeframe in ("15m", "30m", "1h", "4h", "1day")}
    opportunities = build_timeframe_opportunities(_forecasts(), frames)
    fifteen = opportunities["15m"]

    assert fifteen["exit_framework"]["status"] == "CALCULATED"
    assert fifteen["reference_price"] == 1.10
    assert fifteen["stop_loss"] < fifteen["reference_price"] < fifteen["take_profit"]
    assert fifteen["reference_timestamp"] is not None


def test_missing_model_output_is_data_unavailable():
    forecasts = _forecasts()
    forecasts[0]["status"] = "DATA UNAVAILABLE"
    forecasts[0]["probability_up"] = None
    forecasts[0]["probability_down"] = None
    frame = _frame()
    frames = {timeframe: frame.copy() for timeframe in ("15m", "30m", "1h", "4h", "1day")}
    opportunities = build_timeframe_opportunities(forecasts, frames)

    assert opportunities["15m"]["status"] == "DATA UNAVAILABLE"
    assert opportunities["15m"]["entry_status"] == "DATA_UNAVAILABLE"
    assert opportunities["15m"]["entry_signal"] is False
