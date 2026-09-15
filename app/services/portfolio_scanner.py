from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from app.config import settings
from app.data.tiingo_fx import fetch_time_series as fetch_tiingo_time_series
from app.data.twelve_data import fetch_time_series as fetch_twelve_time_series
from app.features.technical import add_features
from app.models.hierarchy import build_hierarchical_decision
from app.risk.gate import evaluate_signal
from app.services.deep_analysis_router import (
    DEFAULT_TOP_N,
    ROUTER_VERSION,
    run_deep_analysis_for_candidates,
    select_deep_analysis_candidates,
)
from app.services.portfolio_ranking import RANKING_VERSION, rank_scanner_results, ranking_summary
from app.services.pipeline import (
    _run_shadow_learning,
    _freshness_limit,
    aggregate_from_frame,
    keep_closed_candles,
    normalize_native_daily_sessions,
    validate_freshness,
)

DEFAULT_PORTFOLIO_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "NZD/JPY",
]

SCAN_TIMEFRAMES = ["15m", "30m", "1h", "4h", "1day"]
SCAN_MIN_ROWS = 180
SCAN_TRAIN_ROWS = 240
SCAN_15M_OUTPUTSIZE = 420
SCAN_1H_HISTORY_DAYS = 60
SCAN_DAILY_HISTORY_DAYS = 365

_STATE_LOCK = Lock()
_STATE: dict[str, Any] = {
    "status": "IDLE",
    "pairs": [],
    "started_at_utc": None,
    "completed_at_utc": None,
    "pairs_total": len(DEFAULT_PORTFOLIO_PAIRS),
    "pairs_completed": 0,
    "results": [],
    "ranking": [],
    "ranking_summary": {},
    "top_candidates": [],
    "deep_analysis_results": [],
    "routing_status": "IDLE",
    "last_error": None,
}

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="portfolio-scan")
_LAST_TWELVE_CALLS: list[float] = []
_TWELVE_LOCK = Lock()


def _twelve_rate_limit() -> None:
    limit = max(1, int(settings.twelve_data_requests_per_minute))
    while True:
        with _TWELVE_LOCK:
            now = time.monotonic()
            _LAST_TWELVE_CALLS[:] = [t for t in _LAST_TWELVE_CALLS if now - t < 60.0]
            if len(_LAST_TWELVE_CALLS) < limit:
                _LAST_TWELVE_CALLS.append(now)
                return
            wait_for = max(0.1, 60.0 - (now - _LAST_TWELVE_CALLS[0]))
        time.sleep(wait_for)


def _classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=60,
            max_leaf_nodes=8,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])


def _regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=60,
            max_leaf_nodes=8,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])


def _fast_forecast(df: pd.DataFrame, instrument: str, timeframe: str) -> dict[str, Any]:
    frame = add_features(df.copy())
    close = pd.to_numeric(frame["close"], errors="coerce")
    y_return = np.log(close.shift(-1) / close)
    y_direction = (y_return > 0).astype(int)
    ids = {"timestamp", "provider", "instrument", "timeframe", "data_status"}
    numeric = [c for c in frame.columns if c not in ids and c not in {"target_return", "target_direction"}]
    numeric = [c for c in numeric if pd.api.types.is_numeric_dtype(frame[c])]
    X = frame[numeric].replace([np.inf, -np.inf], np.nan)
    valid = y_return.notna()
    X = X.loc[valid].reset_index(drop=True)
    y_return = y_return.loc[valid].reset_index(drop=True)
    y_direction = y_direction.loc[valid].reset_index(drop=True)

    base: dict[str, Any] = {
        "instrument": instrument,
        "timeframe": timeframe,
        "horizon_bars": 1,
        "model_version": "portfolio-fast-hgb-v1",
        "status": "DATA UNAVAILABLE",
        "validation_status": "NOT_EVALUATED",
        "validation": {
            "status": "NOT_EVALUATED",
            "reason": "PORTFOLIO_FAST_SCAN",
            "data_role": "SCANNER_ONLY",
        },
        "regime": {"status": "CALCULATED", "regime": "UNKNOWN"},
    }
    if len(X) < SCAN_MIN_ROWS or y_direction.nunique() < 2:
        base["reason"] = "INSUFFICIENT_SCANNER_HISTORY"
        return base

    X_train = X.tail(min(SCAN_TRAIN_ROWS, len(X)))
    y_r = y_return.tail(len(X_train))
    y_d = y_direction.tail(len(X_train))
    if y_d.nunique() < 2:
        base["reason"] = "SINGLE_CLASS_SCANNER_HISTORY"
        return base

    clf = _classifier()
    reg = _regressor()
    clf.fit(X_train, y_d)
    reg.fit(X_train, y_r)
    latest = X.tail(1)
    p = float(np.clip(clf.predict_proba(latest)[:, 1][0], 0.0, 1.0))
    expected_return = float(reg.predict(latest)[0])
    residuals = y_r - reg.predict(X_train)
    uncertainty = float(np.nanstd(residuals)) if len(residuals) > 5 else 0.0
    signed = 2.0 * (p - 0.5)
    atr = frame["atr_pct"].dropna()
    atr_last = float(frame.iloc[-1]["atr_pct"]) if pd.notna(frame.iloc[-1]["atr_pct"]) else None
    if atr.empty or atr_last is None:
        regime = "UNKNOWN"
    else:
        low, high = float(atr.quantile(0.20)), float(atr.quantile(0.80))
        if atr_last >= high:
            regime = "HIGH_VOLATILITY"
        elif atr_last <= low:
            regime = "LOW_VOLATILITY"
        else:
            sma20 = frame.iloc[-1].get("sma_20")
            sma50 = frame.iloc[-1].get("sma_50")
            regime = "TREND_UP" if pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50 else "TREND_DOWN" if pd.notna(sma20) and pd.notna(sma50) and sma20 < sma50 else "RANGE"

    return {
        **base,
        "status": "MODEL OUTPUT",
        "probability_up": p,
        "probability_down": 1.0 - p,
        "expected_return": expected_return,
        "expected_abs_move": abs(expected_return) + uncertainty,
        "uncertainty": uncertainty,
        "confidence": float(np.clip(abs(signed), 0.0, 1.0)),
        "regime": {"status": "CALCULATED", "regime": regime, "atr_pct": atr_last},
        "training_rows": int(len(X_train)),
        "training_data_role": "SCANNER_ONLY",
        "prediction_timestamp": str(pd.to_datetime(df["timestamp"].max(), utc=True)),
    }


def _scan_pair(symbol: str) -> dict[str, Any]:
    symbol = symbol.upper().replace("_", "/")
    if "/" not in symbol and len(symbol) == 6:
        symbol = f"{symbol[:3]}/{symbol[3:]}"

    frames: dict[str, pd.DataFrame] = {}
    forecasts: list[dict[str, Any]] = []

    _twelve_rate_limit()
    fifteen = fetch_twelve_time_series(
        settings.twelve_data_api_key,
        symbol=symbol,
        interval="15m",
        outputsize=SCAN_15M_OUTPUTSIZE,
    )
    fifteen = keep_closed_candles(fifteen, "15m")
    validate_freshness(fifteen, _freshness_limit("15m"), "15m")
    frames["15m"] = fifteen
    forecasts.append(_fast_forecast(fifteen, symbol, "15m"))

    thirty = aggregate_from_frame(fifteen, "30min", "30m", expected_bars=2)
    validate_freshness(thirty, _freshness_limit("30m"), "30m")
    frames["30m"] = thirty
    forecasts.append(_fast_forecast(thirty, symbol, "30m"))

    hourly = fetch_tiingo_time_series(
        settings.tiingo_api_token,
        symbol=symbol,
        interval="1h",
        history_days=SCAN_1H_HISTORY_DAYS,
        cache_ttl_seconds=settings.tiingo_cache_ttl_seconds,
    )
    hourly = keep_closed_candles(hourly, "1h")
    validate_freshness(hourly, _freshness_limit("1h"), "1h")
    frames["1h"] = hourly
    forecasts.append(_fast_forecast(hourly, symbol, "1h"))

    four_hour = aggregate_from_frame(hourly, "4h", "4h", expected_bars=4)
    validate_freshness(four_hour, _freshness_limit("4h"), "4h")
    frames["4h"] = four_hour
    forecasts.append(_fast_forecast(four_hour, symbol, "4h"))

    daily = fetch_tiingo_time_series(
        settings.tiingo_api_token,
        symbol=symbol,
        interval="1day",
        history_days=SCAN_DAILY_HISTORY_DAYS,
        cache_ttl_seconds=settings.tiingo_cache_ttl_seconds,
    )
    daily = daily.loc[daily["data_status"].eq("REAL_DATA")].copy()
    daily = normalize_native_daily_sessions(daily)
    validate_freshness(daily, _freshness_limit("1day"), "1day")
    frames["1day"] = daily
    forecasts.append(_fast_forecast(daily, symbol, "1day"))

    hierarchical = build_hierarchical_decision(forecasts)
    risk = evaluate_signal(hierarchical.get("probability_up"))
    shadow = _run_shadow_learning(symbol, hierarchical, frames["15m"])

    stage_summary = {}
    for item in forecasts:
        p = item.get("probability_up")
        stage_summary[item["timeframe"]] = {
            "direction": "LONG" if (p if p is not None else 0.5) > 0.52 else "SHORT" if (p if p is not None else 0.5) < 0.48 else "NEUTRAL",
            "probability_up": p,
            "regime": (item.get("regime") or {}).get("regime"),
            "data_status": "REAL_DATA",
            "last_timestamp": str(frames[item["timeframe"]]["timestamp"].max()),
            "validation_status": item.get("validation_status", "NOT_EVALUATED"),
            "forecast_status": item.get("status", "DATA UNAVAILABLE"),
        }

    stats = shadow.get("stats", {})
    return {
        "instrument": symbol,
        "status": "MODEL OUTPUT",
        "provenance": "REAL_DATA",
        "scanner_model": "portfolio-fast-hgb-v1",
        "training_data_role": "SCANNER_ONLY",
        "timeframes": SCAN_TIMEFRAMES,
        "hierarchical_forecast": {
            "method": hierarchical.get("method"),
            "primary_bias": hierarchical.get("primary_bias"),
            "higher_timeframe_alignment": hierarchical.get("higher_timeframe_alignment"),
            "setup_alignment": hierarchical.get("setup_alignment"),
            "refinement_alignment": hierarchical.get("refinement_alignment"),
            "entry_trigger": hierarchical.get("entry_trigger"),
            "aligned_count": hierarchical.get("aligned_count"),
            "non_neutral_count": hierarchical.get("non_neutral_count"),
            "hierarchy_score": hierarchical.get("hierarchy_score"),
            "hierarchy_edge_percentage_points": hierarchical.get("hierarchy_edge_percentage_points"),
            "probability_up": hierarchical.get("probability_up"),
            "probability_down": hierarchical.get("probability_down"),
            "candidate_signal": hierarchical.get("candidate_signal"),
            "baseline_action": hierarchical.get("baseline_action"),
        },
        "stages": stage_summary,
        "reinforcement_learning": {
            "status": shadow.get("status"),
            "policy_action": shadow.get("policy_action"),
            "baseline_action": shadow.get("baseline_action"),
            "pending_experiences": shadow.get("pending_experiences"),
            "transitions_learned": shadow.get("transitions_learned", stats.get("transitions_learned", 0)),
            "advisory_only": True,
            "execution_authorized": False,
        },
        "risk": {"approved": risk.approved, "reason": risk.reason},
        "decision": hierarchical.get("candidate_signal") if risk.approved else "NO TRADE",
        "execution_authorized": False,
    }


def _run_scan(pairs: list[str]) -> None:
    results: list[dict[str, Any]] = []
    for raw_symbol in pairs:
        symbol = raw_symbol.upper().replace("_", "/")
        try:
            result = _scan_pair(symbol)
        except Exception as exc:
            result = {
                "instrument": symbol,
                "status": "DATA UNAVAILABLE",
                "provenance": "UNKNOWN",
                "error": str(exc),
                "execution_authorized": False,
            }
        results.append(result)
        with _STATE_LOCK:
            _STATE["pairs_completed"] = len(results)
            _STATE["results"] = list(results)

    ranked = rank_scanner_results(results)
    selected = select_deep_analysis_candidates(ranked, top_n=DEFAULT_TOP_N)
    with _STATE_LOCK:
        _STATE["ranking"] = list(ranked)
        _STATE["ranking_summary"] = ranking_summary(ranked)
        _STATE["top_candidates"] = list(selected)
        _STATE["routing_status"] = "DEEP_ANALYSIS_RUNNING" if selected else "NO_CANDIDATES_READY"

    # This is deliberately limited to selected candidates, not the full portfolio.
    deep_results = run_deep_analysis_for_candidates(selected)
    with _STATE_LOCK:
        _STATE["deep_analysis_results"] = list(deep_results)
        _STATE["routing_status"] = "COMPLETE" if selected else "NO_CANDIDATES_READY"
        _STATE["status"] = "COMPLETE"
        _STATE["completed_at_utc"] = datetime.now(timezone.utc).isoformat()


def _normalized_pairs(pairs: list[str] | None) -> list[str]:
    requested = pairs or DEFAULT_PORTFOLIO_PAIRS
    normalized = [p.upper().replace("_", "/") for p in requested]
    if len(normalized) > len(DEFAULT_PORTFOLIO_PAIRS):
        raise ValueError(f"Maximum {len(DEFAULT_PORTFOLIO_PAIRS)} pairs per portfolio request in this stage.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Portfolio symbols must be unique.")
    unsupported = sorted(set(normalized) - set(DEFAULT_PORTFOLIO_PAIRS))
    if unsupported:
        raise ValueError(f"Unsupported portfolio symbols: {', '.join(unsupported)}")
    return normalized


def _snapshot(include_results: bool, include_ranking: bool = False) -> dict[str, Any]:
    with _STATE_LOCK:
        snapshot = dict(_STATE)
        snapshot["pairs"] = list(_STATE["pairs"])
        snapshot["results"] = list(_STATE["results"]) if include_results else []
        snapshot["ranking"] = list(_STATE["ranking"])
        snapshot["ranking_summary"] = dict(_STATE["ranking_summary"])
        snapshot["top_candidates"] = list(_STATE["top_candidates"])
        snapshot["deep_analysis_results"] = list(_STATE["deep_analysis_results"])
    response = {
        "status": snapshot["status"],
        "scanner": "LIGHTWEIGHT_ASYNC_5TF_V1",
        "model": "portfolio-fast-hgb-v1",
        "training_data_role": "SCANNER_ONLY",
        "ranking_engine": RANKING_VERSION,
        "deep_analysis_router": ROUTER_VERSION,
        "pairs_total": snapshot["pairs_total"],
        "pairs_completed": snapshot["pairs_completed"],
        "started_at_utc": snapshot["started_at_utc"],
        "completed_at_utc": snapshot["completed_at_utc"],
        "execution_authorized": False,
        "execution_mode": settings.trading_mode,
        "routing_status": snapshot["routing_status"],
        "portfolio_summary": snapshot["ranking_summary"],
        "top_candidates": snapshot["top_candidates"],
    }
    if include_results:
        response["results"] = snapshot["results"]
    if include_ranking:
        response["ranking"] = snapshot["ranking"]
    return response


def _start_scan(pairs: list[str]) -> None:
    """Reserve the sole worker before submitting it, preventing duplicate scans."""
    with _STATE_LOCK:
        _STATE.update({
            "status": "RUNNING",
            "pairs": list(pairs),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_at_utc": None,
            "pairs_total": len(pairs),
            "pairs_completed": 0,
            "results": [],
            "ranking": [],
            "ranking_summary": {},
            "top_candidates": [],
            "deep_analysis_results": [],
            "routing_status": "SCANNING",
            "last_error": None,
        })
    try:
        _EXECUTOR.submit(_run_scan, pairs)
    except Exception:
        with _STATE_LOCK:
            _STATE["status"] = "FAILED"
            _STATE["last_error"] = "Unable to start portfolio scanner worker."
            _STATE["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        raise


def request_portfolio_scan(pairs: list[str] | None = None, refresh: bool = False) -> dict[str, Any]:
    normalized = _normalized_pairs(pairs)
    with _STATE_LOCK:
        status = _STATE["status"]
        same_pairs = _STATE["pairs"] == normalized

    # A running scan always wins. A refresh is intentionally not queued behind it.
    if status != "RUNNING" and (refresh or status == "IDLE" or not same_pairs):
        _start_scan(normalized)

    snapshot = _snapshot(include_results=False, include_ranking=True)
    snapshot["refresh_in_background"] = snapshot["status"] == "RUNNING"
    return snapshot


def get_portfolio_status() -> dict[str, Any]:
    """Read scanner progress without fetching data or starting a scan."""
    snapshot = _snapshot(include_results=False)
    snapshot["refresh_in_background"] = snapshot["status"] == "RUNNING"
    return snapshot


def get_portfolio_results() -> dict[str, Any]:
    """Read compact completed/partial results without starting a scan."""
    snapshot = _snapshot(include_results=True, include_ranking=True)
    snapshot["refresh_in_background"] = snapshot["status"] == "RUNNING"
    return snapshot


def get_portfolio_ranking() -> dict[str, Any]:
    """Read all ranked scanner records without fetching or routing anything."""
    snapshot = _snapshot(include_results=False, include_ranking=True)
    snapshot["refresh_in_background"] = snapshot["status"] == "RUNNING"
    return snapshot


def get_portfolio_candidates() -> dict[str, Any]:
    """Read only selected candidates and their compact deep-analysis results."""
    snapshot = _snapshot(include_results=False)
    with _STATE_LOCK:
        snapshot["candidates"] = list(_STATE["deep_analysis_results"])
    snapshot["candidate_count"] = len(snapshot["top_candidates"])
    snapshot["deep_analyses_completed"] = len(snapshot["candidates"])
    snapshot["refresh_in_background"] = snapshot["status"] == "RUNNING"
    return snapshot
