from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from app.config import settings
from app.data.tiingo_fx import fetch_time_series as fetch_tiingo_time_series
from app.data.twelve_data import fetch_time_series_batch
from app.execution.ibkr_bridge_client import get_historical_15m_batch
from app.features.technical import add_features
from app.models.hierarchy import build_hierarchical_decision
from app.models.timeframe_signals import build_timeframe_opportunities
from app.risk.gate import evaluate_signal
from app.services.deep_analysis_router import DEFAULT_TOP_N, ROUTER_VERSION, run_deep_analysis_for_candidates, select_deep_analysis_candidates
from app.services.portfolio_ranking import RANKING_VERSION, rank_scanner_results, ranking_summary
from app.services.pipeline import _freshness_limit, _run_shadow_learning, aggregate_from_frame, keep_closed_candles, normalize_native_daily_sessions, validate_freshness

DEFAULT_PORTFOLIO_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "NZD/JPY",
]
SCAN_TIMEFRAMES = ["15m", "30m", "1h", "4h", "1day"]
SCAN_MIN_ROWS = 180
SCAN_TRAIN_ROWS = 240
SCAN_15M_OUTPUTSIZE = 420
SCAN_1H_HISTORY_DAYS = 60
SCAN_DAILY_HISTORY_DAYS = 365
SCAN_PAIR_WORKERS = 4

_STATE_LOCK = Lock()
_STATE: dict[str, Any] = {
    "status": "IDLE", "pairs": [], "started_at_utc": None, "completed_at_utc": None,
    "pairs_total": len(DEFAULT_PORTFOLIO_PAIRS), "pairs_completed": 0, "results": [],
    "ranking": [], "ranking_summary": {}, "top_candidates": [], "deep_analysis_results": [],
    "routing_status": "IDLE", "last_error": None,
}
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="portfolio-scan")


def _classifier() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(learning_rate=0.06, max_iter=60, max_leaf_nodes=8, l2_regularization=1.0, random_state=42))])


def _regressor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(learning_rate=0.06, max_iter=60, max_leaf_nodes=8, l2_regularization=1.0, random_state=42))])


def _fast_forecast(df: pd.DataFrame, instrument: str, timeframe: str) -> dict[str, Any]:
    frame = add_features(df.copy())
    close = pd.to_numeric(frame["close"], errors="coerce")
    y_return = np.log(close.shift(-1) / close)
    y_direction = (y_return > 0).astype(int)
    ids = {"timestamp", "provider", "instrument", "timeframe", "data_status"}
    numeric = [c for c in frame.columns if c not in ids and c not in {"target_return", "target_direction"} and pd.api.types.is_numeric_dtype(frame[c])]
    X = frame[numeric].replace([np.inf, -np.inf], np.nan)
    valid = y_return.notna()
    X = X.loc[valid].reset_index(drop=True)
    y_return = y_return.loc[valid].reset_index(drop=True)
    y_direction = y_direction.loc[valid].reset_index(drop=True)
    base: dict[str, Any] = {
        "instrument": instrument, "timeframe": timeframe, "horizon_bars": 1,
        "model_version": "portfolio-fast-hgb-v1", "status": "DATA UNAVAILABLE", "validation_status": "NOT_EVALUATED",
        "validation": {"status": "NOT_EVALUATED", "reason": "PORTFOLIO_FAST_SCAN", "data_role": "SCANNER_ONLY"},
        "regime": {"status": "CALCULATED", "regime": "UNKNOWN"},
    }
    if len(X) < SCAN_MIN_ROWS or y_direction.nunique() < 2:
        base["reason"] = "INSUFFICIENT_SCANNER_HISTORY"
        return base
    X_train = X.tail(min(SCAN_TRAIN_ROWS, len(X)))
    y_r, y_d = y_return.tail(len(X_train)), y_direction.tail(len(X_train))
    if y_d.nunique() < 2:
        base["reason"] = "SINGLE_CLASS_SCANNER_HISTORY"
        return base
    clf, reg = _classifier(), _regressor()
    clf.fit(X_train, y_d)
    reg.fit(X_train, y_r)
    latest = X.tail(1)
    p = float(np.clip(clf.predict_proba(latest)[:, 1][0], 0.0, 1.0))
    expected_return = float(reg.predict(latest)[0])
    residuals = y_r - reg.predict(X_train)
    uncertainty = float(np.nanstd(residuals)) if len(residuals) > 5 else 0.0
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
            sma20, sma50 = frame.iloc[-1].get("sma_20"), frame.iloc[-1].get("sma_50")
            regime = "TREND_UP" if pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50 else "TREND_DOWN" if pd.notna(sma20) and pd.notna(sma50) and sma20 < sma50 else "RANGE"
    return {
        **base, "status": "MODEL OUTPUT", "probability_up": p, "probability_down": 1.0 - p,
        "expected_return": expected_return, "expected_abs_move": abs(expected_return) + uncertainty,
        "uncertainty": uncertainty, "confidence": float(np.clip(abs(2.0 * (p - 0.5)), 0.0, 1.0)),
        "regime": {"status": "CALCULATED", "regime": regime, "atr_pct": atr_last},
        "training_rows": int(len(X_train)), "training_data_role": "SCANNER_ONLY",
        "prediction_timestamp": str(pd.to_datetime(df["timestamp"].max(), utc=True)),
    }


def _rows_to_frame(rows: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
    if not rows:
        raise RuntimeError(f"DATA UNAVAILABLE: no 15m rows returned for {symbol}.")
    frame = pd.DataFrame(rows).copy()
    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"PROVIDER ERROR: missing IBKR 15m fields for {symbol}: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in frame.columns:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=required).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"DATA UNAVAILABLE: no valid IBKR 15m rows remain for {symbol}.")
    frame["provider"], frame["instrument"], frame["timeframe"], frame["data_status"] = "IBKR", symbol, "15m", "REAL_DATA"
    return frame


def _fetch_15m_frames(pairs: list[str]) -> tuple[dict[str, pd.DataFrame], str]:
    if bool(str(settings.ibkr_bridge_url or "").strip()):
        try:
            raw = get_historical_15m_batch(pairs, outputsize=SCAN_15M_OUTPUTSIZE)
            frames = {symbol: _rows_to_frame(rows, symbol) for symbol, rows in raw.items()}
            for frame in frames.values():
                keep_closed_candles(frame, "15m")
            return frames, "IBKR_BRIDGE"
        except Exception:
            pass
    raw_frames = fetch_time_series_batch(settings.twelve_data_api_key, pairs, interval="15m", outputsize=SCAN_15M_OUTPUTSIZE)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in pairs:
        frame = raw_frames.get(symbol)
        if frame is None:
            raise RuntimeError(f"DATA UNAVAILABLE: missing Twelve Data 15m frame for {symbol}.")
        frame = keep_closed_candles(frame, "15m")
        validate_freshness(frame, _freshness_limit("15m"), "15m")
        frames[symbol] = frame
    return frames, "TWELVE_DATA_BATCH"


def _scan_pair(symbol: str, fifteen: pd.DataFrame, provider_route: str) -> dict[str, Any]:
    symbol = symbol.upper().replace("_", "/")
    frames: dict[str, pd.DataFrame] = {}
    forecasts: list[dict[str, Any]] = []
    fifteen = keep_closed_candles(fifteen.copy(deep=True), "15m")
    validate_freshness(fifteen, _freshness_limit("15m"), "15m")
    frames["15m"] = fifteen
    forecasts.append(_fast_forecast(fifteen, symbol, "15m"))
    thirty = aggregate_from_frame(fifteen, "30min", "30m", expected_bars=2)
    validate_freshness(thirty, _freshness_limit("30m"), "30m")
    frames["30m"] = thirty
    forecasts.append(_fast_forecast(thirty, symbol, "30m"))
    hourly = keep_closed_candles(fetch_tiingo_time_series(settings.tiingo_api_token, symbol=symbol, interval="1h", history_days=SCAN_1H_HISTORY_DAYS, cache_ttl_seconds=settings.tiingo_cache_ttl_seconds), "1h")
    validate_freshness(hourly, _freshness_limit("1h"), "1h")
    frames["1h"] = hourly
    forecasts.append(_fast_forecast(hourly, symbol, "1h"))
    four_hour = aggregate_from_frame(hourly, "4h", "4h", expected_bars=4)
    validate_freshness(four_hour, _freshness_limit("4h"), "4h")
    frames["4h"] = four_hour
    forecasts.append(_fast_forecast(four_hour, symbol, "4h"))
    daily = fetch_tiingo_time_series(settings.tiingo_api_token, symbol=symbol, interval="1day", history_days=SCAN_DAILY_HISTORY_DAYS, cache_ttl_seconds=settings.tiingo_cache_ttl_seconds)
    daily = normalize_native_daily_sessions(daily.loc[daily["data_status"].eq("REAL_DATA")].copy())
    validate_freshness(daily, _freshness_limit("1day"), "1day")
    frames["1day"] = daily
    forecasts.append(_fast_forecast(daily, symbol, "1day"))

    hierarchical = build_hierarchical_decision(forecasts)
    opportunities = build_timeframe_opportunities(forecasts, frames)
    per_tf_risk = {}
    for tf, opportunity in opportunities.items():
        decision = evaluate_signal(opportunity.get("probability_up"))
        per_tf_risk[tf] = {"approved": decision.approved, "reason": decision.reason}
    shadow = _run_shadow_learning(symbol, hierarchical, frames["15m"])
    stage_summary = {}
    for item in forecasts:
        p = item.get("probability_up")
        stage_summary[item["timeframe"]] = {
            "direction": "LONG" if (p if p is not None else 0.5) > 0.52 else "SHORT" if (p if p is not None else 0.5) < 0.48 else "NEUTRAL",
            "probability_up": p, "regime": (item.get("regime") or {}).get("regime"), "data_status": "REAL_DATA",
            "last_timestamp": str(frames[item["timeframe"]]["timestamp"].max()),
            "validation_status": item.get("validation_status", "NOT_EVALUATED"), "forecast_status": item.get("status", "DATA UNAVAILABLE"),
        }
    stats = shadow.get("stats", {})
    return {
        "instrument": symbol, "status": "MODEL OUTPUT", "provenance": "REAL_DATA", "market_data_route_15m": provider_route,
        "scanner_model": "portfolio-fast-hgb-v1", "training_data_role": "SCANNER_ONLY", "timeframes": SCAN_TIMEFRAMES,
        "hierarchical_forecast": {**{k: hierarchical.get(k) for k in ("method", "primary_bias", "higher_timeframe_alignment", "setup_alignment", "refinement_alignment", "entry_trigger", "aligned_count", "non_neutral_count", "hierarchy_score", "hierarchy_edge_percentage_points", "probability_up", "probability_down", "candidate_signal", "baseline_action")}, "role": "CONTEXT_AND_AGGREGATE", "conflict_policy": "CONTEXT_ONLY_NO_UNIVERSAL_VETO"},
        "timeframe_opportunities": opportunities,
        "stages": stage_summary,
        "timeframe_risk": per_tf_risk,
        "reinforcement_learning": {"status": shadow.get("status"), "policy_action": shadow.get("policy_action"), "baseline_action": shadow.get("baseline_action"), "pending_experiences": shadow.get("pending_experiences"), "transitions_learned": shadow.get("transitions_learned", stats.get("transitions_learned", 0)), "advisory_only": True, "execution_authorized": False},
        "risk": {"approved": False, "reason": "PAPER_ONLY_EXECUTION_LOCK"},
        "decision": "NO TRADE" if not any(op.get("entry_signal") for op in opportunities.values()) else "MULTI_TIMEFRAME_OPPORTUNITIES",
        "execution_authorized": False,
    }


def _run_scan(pairs: list[str]) -> None:
    results_by_symbol: dict[str, dict[str, Any]] = {}
    try:
        fifteen_frames, provider_route = _fetch_15m_frames(pairs)
    except Exception as exc:
        provider_route = "DATA_UNAVAILABLE"
        for symbol in pairs:
            results_by_symbol[symbol] = {"instrument": symbol, "status": "DATA UNAVAILABLE", "provenance": "UNKNOWN", "error": str(exc), "market_data_route_15m": provider_route, "execution_authorized": False}
        with _STATE_LOCK:
            _STATE["pairs_completed"] = len(results_by_symbol)
            _STATE["results"] = [results_by_symbol[s] for s in pairs]
    else:
        with ThreadPoolExecutor(max_workers=SCAN_PAIR_WORKERS, thread_name_prefix="pair-scan") as pool:
            future_map = {pool.submit(_scan_pair, symbol, fifteen_frames[symbol], provider_route): symbol for symbol in pairs if symbol in fifteen_frames}
            for future in as_completed(future_map):
                symbol = future_map[future]
                try:
                    results_by_symbol[symbol] = future.result()
                except Exception as exc:
                    results_by_symbol[symbol] = {"instrument": symbol, "status": "DATA UNAVAILABLE", "provenance": "UNKNOWN", "error": str(exc), "market_data_route_15m": provider_route, "execution_authorized": False}
                with _STATE_LOCK:
                    _STATE["pairs_completed"] = len(results_by_symbol)
                    _STATE["results"] = [results_by_symbol.get(s, {"instrument": s, "status": "PENDING"}) for s in pairs]
    results = [results_by_symbol.get(symbol, {"instrument": symbol, "status": "DATA UNAVAILABLE", "provenance": "UNKNOWN", "error": "15m frame unavailable.", "execution_authorized": False}) for symbol in pairs]
    ranked = rank_scanner_results(results)
    selected = select_deep_analysis_candidates(ranked, top_n=DEFAULT_TOP_N)
    with _STATE_LOCK:
        _STATE.update({"results": list(results), "pairs_completed": len(results), "ranking": list(ranked), "ranking_summary": ranking_summary(ranked), "top_candidates": list(selected), "routing_status": "DEEP_ANALYSIS_RUNNING" if selected else "NO_CANDIDATES_READY"})
    deep_results = run_deep_analysis_for_candidates(selected)
    with _STATE_LOCK:
        _STATE.update({"deep_analysis_results": list(deep_results), "routing_status": "COMPLETE" if selected else "NO_CANDIDATES_READY", "status": "COMPLETE", "completed_at_utc": datetime.now(timezone.utc).isoformat()})


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
        snapshot["pairs"] = list(_STATE["pairs"]); snapshot["results"] = list(_STATE["results"]) if include_results else []
        snapshot["ranking"] = list(_STATE["ranking"]); snapshot["ranking_summary"] = dict(_STATE["ranking_summary"])
        snapshot["top_candidates"] = list(_STATE["top_candidates"]); snapshot["deep_analysis_results"] = list(_STATE["deep_analysis_results"])
    response = {"status": snapshot["status"], "scanner": "LIGHTWEIGHT_ASYNC_5TF_V2", "model": "portfolio-fast-hgb-v1", "training_data_role": "SCANNER_ONLY", "ranking_engine": RANKING_VERSION, "deep_analysis_router": ROUTER_VERSION, "pairs_total": snapshot["pairs_total"], "pairs_completed": snapshot["pairs_completed"], "started_at_utc": snapshot["started_at_utc"], "completed_at_utc": snapshot["completed_at_utc"], "execution_authorized": False, "execution_mode": settings.trading_mode, "routing_status": snapshot["routing_status"], "portfolio_summary": snapshot["ranking_summary"], "top_candidates": snapshot["top_candidates"]}
    if include_results: response["results"] = snapshot["results"]
    if include_ranking: response["ranking"] = snapshot["ranking"]
    return response


def _start_scan(pairs: list[str]) -> None:
    with _STATE_LOCK:
        _STATE.update({"status": "RUNNING", "pairs": list(pairs), "started_at_utc": datetime.now(timezone.utc).isoformat(), "completed_at_utc": None, "pairs_total": len(pairs), "pairs_completed": 0, "results": [], "ranking": [], "ranking_summary": {}, "top_candidates": [], "deep_analysis_results": [], "routing_status": "SCANNING", "last_error": None})
    try:
        _EXECUTOR.submit(_run_scan, pairs)
    except Exception:
        with _STATE_LOCK:
            _STATE["status"] = "FAILED"; _STATE["last_error"] = "Unable to start portfolio scanner worker."; _STATE["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        raise


def request_portfolio_scan(pairs: list[str] | None = None, refresh: bool = False) -> dict[str, Any]:
    normalized = _normalized_pairs(pairs)
    with _STATE_LOCK:
        status, same_pairs = _STATE["status"], _STATE["pairs"] == normalized
    if status != "RUNNING" and (refresh or status == "IDLE" or not same_pairs): _start_scan(normalized)
    snapshot = _snapshot(include_results=False, include_ranking=True); snapshot["refresh_in_background"] = snapshot["status"] == "RUNNING"; return snapshot


def get_portfolio_status() -> dict[str, Any]:
    snapshot = _snapshot(include_results=False); snapshot["refresh_in_background"] = snapshot["status"] == "RUNNING"; return snapshot


def get_portfolio_results() -> dict[str, Any]:
    snapshot = _snapshot(include_results=True, include_ranking=True); snapshot["refresh_in_background"] = snapshot["status"] == "RUNNING"; return snapshot


def get_portfolio_ranking() -> dict[str, Any]:
    with _STATE_LOCK:
        ranking, summary, status = list(_STATE["ranking"]), dict(_STATE["ranking_summary"]), _STATE["status"]
    return {"status": status, "ranking_engine": RANKING_VERSION, "ranking": ranking, "portfolio_summary": summary, "execution_authorized": False}


def get_portfolio_candidates() -> dict[str, Any]:
    with _STATE_LOCK:
        candidates, deep, status, routing_status = list(_STATE["top_candidates"]), list(_STATE["deep_analysis_results"]), _STATE["status"], _STATE["routing_status"]
    return {"status": status, "routing_status": routing_status, "deep_analysis_router": ROUTER_VERSION, "top_candidates": candidates, "deep_analysis_results": deep, "execution_authorized": False}
