from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from app.features.technical import add_features

TIMEFRAME_ORDER = ("1day", "4h", "1h", "30m", "15m")

TIMEFRAME_STRATEGIES: dict[str, dict[str, Any]] = {
    "15m": {"strategy_id": "forex-15m-micro-v1", "holding_horizon": "15m-2h", "stop_atr": 1.00, "target_atr": 1.50},
    "30m": {"strategy_id": "forex-30m-intraday-v1", "holding_horizon": "30m-4h", "stop_atr": 1.10, "target_atr": 1.60},
    "1h": {"strategy_id": "forex-1h-intraday-v1", "holding_horizon": "1h-8h", "stop_atr": 1.25, "target_atr": 1.80},
    "4h": {"strategy_id": "forex-4h-swing-v1", "holding_horizon": "4h-2d", "stop_atr": 1.50, "target_atr": 2.20},
    "1day": {"strategy_id": "forex-1d-position-v1", "holding_horizon": "1d-10d", "stop_atr": 2.00, "target_atr": 3.00},
}

MIN_SIGNAL_EDGE_PERCENTAGE_POINTS = 10.0
NEUTRAL_BAND = 0.02


def _direction(probability_up: float | None) -> str:
    if probability_up is None:
        return "UNKNOWN"
    edge = float(probability_up) - 0.50
    if edge > NEUTRAL_BAND:
        return "LONG"
    if edge < -NEUTRAL_BAND:
        return "SHORT"
    return "NEUTRAL"


def _signal_id(instrument: str, timeframe: str, prediction_timestamp: Any, direction: str) -> str:
    raw = f"{instrument}|{timeframe}|{prediction_timestamp}|{direction}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"sig-{timeframe}-{digest}"


def _context(timeframe: str, direction: str, directions: dict[str, str]) -> dict[str, Any]:
    if direction not in {"LONG", "SHORT"}:
        return {"classification": "NEUTRAL", "higher_timeframes": {}}
    index = TIMEFRAME_ORDER.index(timeframe)
    higher = {tf: directions.get(tf, "UNKNOWN") for tf in TIMEFRAME_ORDER[:index]}
    available = {tf: value for tf, value in higher.items() if value in {"LONG", "SHORT"}}
    if not available:
        classification = "PRIMARY_BIAS"
    elif all(value == direction for value in available.values()):
        classification = "TREND_ALIGNED"
    elif all(value != direction for value in available.values()):
        classification = "COUNTER_TREND"
    else:
        classification = "MIXED"
    return {"classification": classification, "higher_timeframes": higher}


def _levels(frame: pd.DataFrame, direction: str, config: dict[str, Any]) -> dict[str, Any]:
    if frame.empty:
        return {"status": "DATA UNAVAILABLE"}
    enriched = add_features(frame.copy())
    latest = enriched.iloc[-1]
    close = pd.to_numeric(latest.get("close"), errors="coerce")
    atr = pd.to_numeric(latest.get("atr_14"), errors="coerce")
    if pd.isna(close) or pd.isna(atr) or float(close) <= 0 or float(atr) <= 0:
        return {"status": "DATA UNAVAILABLE"}
    close_f = float(close)
    atr_f = float(atr)
    if direction == "LONG":
        stop = close_f - config["stop_atr"] * atr_f
        target = close_f + config["target_atr"] * atr_f
    elif direction == "SHORT":
        stop = close_f + config["stop_atr"] * atr_f
        target = close_f - config["target_atr"] * atr_f
    else:
        return {"status": "DATA UNAVAILABLE"}
    if stop <= 0 or target <= 0:
        return {"status": "DATA UNAVAILABLE"}
    return {
        "status": "CALCULATED",
        "reference_price": close_f,
        "atr_14": atr_f,
        "stop_loss": float(stop),
        "take_profit": float(target),
        "stop_atr_multiple": config["stop_atr"],
        "target_atr_multiple": config["target_atr"],
        "reference_timestamp": str(pd.to_datetime(latest["timestamp"], utc=True)),
    }


def build_timeframe_opportunities(
    forecasts: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    by_tf = {item.get("timeframe"): item for item in forecasts}
    directions = {tf: _direction((by_tf.get(tf) or {}).get("probability_up")) for tf in TIMEFRAME_ORDER}
    opportunities: dict[str, dict[str, Any]] = {}
    for timeframe in TIMEFRAME_ORDER:
        forecast = by_tf.get(timeframe) or {}
        config = TIMEFRAME_STRATEGIES[timeframe]
        probability_up = forecast.get("probability_up")
        probability_down = forecast.get("probability_down")
        direction = directions[timeframe]
        status = forecast.get("status", "DATA UNAVAILABLE")
        valid_model = status == "MODEL OUTPUT" and probability_up is not None and probability_down is not None
        edge = abs(float(probability_up) - 0.50) * 100.0 if valid_model else None
        entry_signal = bool(valid_model and direction in {"LONG", "SHORT"} and edge is not None and edge >= MIN_SIGNAL_EDGE_PERCENTAGE_POINTS)
        levels = _levels(frames.get(timeframe, pd.DataFrame()), direction, config) if entry_signal else {"status": "NOT_REQUIRED"}
        context = _context(timeframe, direction, directions)
        prediction_timestamp = forecast.get("prediction_timestamp")
        signal_id = _signal_id(str(forecast.get("instrument") or "UNKNOWN"), timeframe, prediction_timestamp, direction)
        if not valid_model:
            entry_status = "DATA_UNAVAILABLE"
            entry_reason = str(forecast.get("reason") or "MODEL_OUTPUT_UNAVAILABLE")
        elif direction == "NEUTRAL":
            entry_status = "NO TRADE"
            entry_reason = "PROBABILITY_WITHIN_NEUTRAL_BAND"
        elif not entry_signal:
            entry_status = "NO TRADE"
            entry_reason = "INSUFFICIENT_TIMEFRAME_EDGE"
        elif levels.get("status") != "CALCULATED":
            entry_signal = False
            entry_status = "DATA_UNAVAILABLE"
            entry_reason = "ATR_LEVELS_UNAVAILABLE"
        else:
            entry_status = "SIGNAL GENERATED"
            entry_reason = "TIMEFRAME_EDGE_THRESHOLD_MET"
        opportunities[timeframe] = {
            "strategy_id": config["strategy_id"],
            "signal_id": signal_id,
            "instrument": forecast.get("instrument"),
            "timeframe": timeframe,
            "prediction": direction,
            "probability_up": probability_up,
            "probability_down": probability_down,
            "confidence": forecast.get("confidence"),
            "expected_return": forecast.get("expected_return"),
            "expected_abs_move": forecast.get("expected_abs_move"),
            "uncertainty": forecast.get("uncertainty"),
            "regime": (forecast.get("regime") or {}).get("regime", "UNKNOWN"),
            "entry_signal": entry_signal,
            "entry_status": entry_status,
            "entry_reason": entry_reason,
            "holding_horizon": config["holding_horizon"],
            "exit_framework": {
                "method": "ATR_14_REFERENCE",
                "stop_atr_multiple": config["stop_atr"],
                "target_atr_multiple": config["target_atr"],
                "status": levels.get("status", "DATA UNAVAILABLE"),
            },
            "reference_price": levels.get("reference_price"),
            "stop_loss": levels.get("stop_loss"),
            "take_profit": levels.get("take_profit"),
            "reference_timestamp": levels.get("reference_timestamp"),
            "context": context,
            "status": status if valid_model else "DATA UNAVAILABLE",
            "validation_status": forecast.get("validation_status", "NOT_EVALUATED"),
            "data_status": "REAL_DATA" if valid_model else "DATA_UNAVAILABLE",
            "execution_authorized": False,
        }
    return opportunities
