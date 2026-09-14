from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.features.technical import add_features

MODEL_VERSION = "research-candidate-hgb-mtf-v1"
MIN_ROWS = 180
MIN_TRAIN = 120
VALIDATION_FOLDS = 4
VALIDATION_TEST = 20
CALIBRATION_FRACTION = 0.20


def _feature_frame(df: pd.DataFrame, horizon_bars: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = add_features(df.copy())
    close = pd.to_numeric(x["close"], errors="coerce")
    y_return = np.log(close.shift(-horizon_bars) / close)
    y_direction = (y_return > 0).astype(int)
    ids = {"timestamp", "provider", "instrument", "timeframe", "data_status"}
    target_names = {"target_return", "target_direction"}
    numeric = [c for c in x.columns if c not in ids and c not in target_names]
    numeric = [c for c in numeric if pd.api.types.is_numeric_dtype(x[c])]
    features = x[numeric].replace([np.inf, -np.inf], np.nan)
    valid = y_return.notna()
    return (
        features.loc[valid].reset_index(drop=True),
        y_return.loc[valid].reset_index(drop=True),
        y_direction.loc[valid].reset_index(drop=True),
    )


def _classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])


def _regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])


def _walk_forward_validate(X: pd.DataFrame, y_return: pd.Series, y_direction: pd.Series) -> dict[str, Any]:
    n = len(X)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = MIN_TRAIN
    while len(splits) < VALIDATION_FOLDS:
        test_start = train_end
        test_end = test_start + VALIDATION_TEST
        if test_end > n:
            break
        splits.append((np.arange(train_end), np.arange(test_start, test_end)))
        train_end += VALIDATION_TEST

    records = []
    if not splits:
        return {"status": "HOLD", "reason": "INSUFFICIENT_WALK_FORWARD_HISTORY", "folds": []}

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        clf = _classifier()
        reg = _regressor()
        clf.fit(X.iloc[train_idx], y_direction.iloc[train_idx])
        reg.fit(X.iloc[train_idx], y_return.iloc[train_idx])
        p = np.clip(clf.predict_proba(X.iloc[test_idx])[:, 1], 0.0, 1.0)
        r = reg.predict(X.iloc[test_idx])
        yt_d = y_direction.iloc[test_idx].to_numpy()
        yt_r = y_return.iloc[test_idx].to_numpy()
        records.append({
            "fold": fold,
            "accuracy": float(accuracy_score(yt_d, p >= 0.5)),
            "balanced_accuracy": float(balanced_accuracy_score(yt_d, p >= 0.5)),
            "brier": float(brier_score_loss(yt_d, p)),
            "rmse_model": float(np.sqrt(mean_squared_error(yt_r, r))),
            "rmse_naive": float(np.sqrt(mean_squared_error(yt_r, np.zeros_like(yt_r)))),
        })

    frame = pd.DataFrame(records)
    beats_naive_rate = float((frame["rmse_model"] < frame["rmse_naive"]).mean())
    candidate = bool(
        len(frame) >= 3
        and beats_naive_rate >= 0.75
        and float(frame["balanced_accuracy"].mean()) > 0.50
        and float(frame["brier"].mean()) < 0.25
    )
    return {
        "status": "RESEARCH_CANDIDATE" if candidate else "NOT_ADMITTED",
        "folds": records,
        "fold_count": int(len(frame)),
        "beats_naive_rmse_rate": beats_naive_rate,
        "mean_balanced_accuracy": float(frame["balanced_accuracy"].mean()),
        "mean_brier": float(frame["brier"].mean()),
        "mean_rmse_model": float(frame["rmse_model"].mean()),
        "mean_rmse_naive": float(frame["rmse_naive"].mean()),
    }


def _calibrate_probability(raw_probability: float, X: pd.DataFrame, y_direction: pd.Series) -> tuple[float, dict[str, Any]]:
    n = len(X)
    cal_start = max(int(n * (1.0 - CALIBRATION_FRACTION)), MIN_TRAIN)
    if cal_start >= n - 10:
        return raw_probability, {"status": "NOT_CALIBRATED", "reason": "INSUFFICIENT_CALIBRATION_HISTORY"}
    train_idx = np.arange(cal_start)
    cal_idx = np.arange(cal_start, n)
    clf = _classifier()
    clf.fit(X.iloc[train_idx], y_direction.iloc[train_idx])
    p_cal = np.clip(clf.predict_proba(X.iloc[cal_idx])[:, 1], 1e-6, 1 - 1e-6)
    y_cal = y_direction.iloc[cal_idx].to_numpy()
    logits = np.log(p_cal / (1.0 - p_cal)).reshape(-1, 1)
    if len(np.unique(y_cal)) < 2:
        return raw_probability, {"status": "NOT_CALIBRATED", "reason": "CALIBRATION_SET_SINGLE_CLASS"}
    calibrator = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(random_state=42)),
    ])
    calibrator.fit(logits, y_cal)
    raw = float(np.clip(raw_probability, 1e-6, 1 - 1e-6))
    calibrated = float(calibrator.predict_proba(np.array([[np.log(raw / (1 - raw))]]))[:, 1][0])
    brier = float(brier_score_loss(y_cal, calibrator.predict_proba(logits)[:, 1]))
    return calibrated, {
        "status": "CALIBRATED",
        "method": "platt_scaling",
        "calibration_rows": int(len(cal_idx)),
        "brier": brier,
    }


def _regime(df: pd.DataFrame) -> dict[str, Any]:
    x = add_features(df.copy())
    atr = x["atr_pct"].dropna()
    if atr.empty:
        return {"status": "DATA UNAVAILABLE", "regime": "UNKNOWN"}
    last = x.iloc[-1]
    atr_pct = float(last["atr_pct"]) if pd.notna(last["atr_pct"]) else np.nan
    low = float(atr.quantile(0.20))
    high = float(atr.quantile(0.80))
    if np.isfinite(atr_pct) and atr_pct >= high:
        state = "HIGH_VOLATILITY"
    elif np.isfinite(atr_pct) and atr_pct <= low:
        state = "LOW_VOLATILITY"
    elif pd.notna(last["sma_20"]) and pd.notna(last["sma_50"]):
        state = "TREND_UP" if last["sma_20"] > last["sma_50"] else "TREND_DOWN" if last["sma_20"] < last["sma_50"] else "RANGE"
    else:
        state = "UNKNOWN"
    return {"status": "CALCULATED", "regime": state, "atr_pct": atr_pct}


def forecast_timeframe(df: pd.DataFrame, instrument: str, timeframe: str) -> dict[str, Any]:
    X, y_return, y_direction = _feature_frame(df, horizon_bars=1)
    base = {
        "instrument": instrument,
        "timeframe": timeframe,
        "model_version": MODEL_VERSION,
        "status": "DATA UNAVAILABLE",
        "probability_up": None,
        "probability_down": None,
        "expected_return": None,
        "expected_abs_move": None,
        "uncertainty": None,
        "confidence": None,
        "regime": _regime(df),
    }
    if len(X) < MIN_ROWS or y_direction.nunique() < 2:
        base["reason"] = "INSUFFICIENT_MODEL_HISTORY"
        return base

    validation = _walk_forward_validate(X, y_return, y_direction)
    clf = _classifier()
    reg = _regressor()
    fit_end = max(int(len(X) * (1.0 - CALIBRATION_FRACTION)), MIN_TRAIN)
    if fit_end >= len(X):
        fit_end = len(X) - 1
    clf.fit(X.iloc[:fit_end], y_direction.iloc[:fit_end])
    reg.fit(X.iloc[:fit_end], y_return.iloc[:fit_end])
    latest = X.tail(1)
    raw_p = float(np.clip(clf.predict_proba(latest)[:, 1][0], 0.0, 1.0))
    calibrated_p, calibration = _calibrate_probability(raw_p, X.iloc[:fit_end], y_direction.iloc[:fit_end])
    expected_return = float(reg.predict(latest)[0])
    residuals = y_return.iloc[:fit_end] - reg.predict(X.iloc[:fit_end])
    uncertainty = float(np.nanstd(residuals)) if len(residuals) > 5 else np.nan
    confidence = float(np.clip(2.0 * abs(calibrated_p - 0.5), 0.0, 1.0))
    return {
        **base,
        "status": "MODEL OUTPUT",
        "validation_status": validation["status"],
        "validation": validation,
        "calibration": calibration,
        "probability_up": calibrated_p,
        "probability_down": 1.0 - calibrated_p,
        "expected_return": expected_return,
        "expected_abs_move": float(abs(expected_return) + (uncertainty if np.isfinite(uncertainty) else 0.0)),
        "uncertainty": uncertainty,
        "confidence": confidence,
        "training_rows": int(fit_end),
        "prediction_timestamp": str(pd.to_datetime(df["timestamp"].max(), utc=True)),
    }


def combine_timeframes(forecasts: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [x for x in forecasts if x.get("status") == "MODEL OUTPUT" and x.get("probability_up") is not None]
    if not usable:
        return {
            "status": "DATA UNAVAILABLE",
            "probability_up": None,
            "probability_down": None,
            "agreement": 0.0,
            "timeframes": 0,
            "validation_status": "HOLD",
        }
    p = np.array([float(x["probability_up"]) for x in usable])
    combined = float(np.mean(p))
    directions = np.array([v >= 0.5 for v in p], dtype=int)
    agreement = float(max(directions.mean(), 1.0 - directions.mean()))
    all_admitted = all(x.get("validation_status") == "RESEARCH_CANDIDATE" for x in usable)
    return {
        "status": "MODEL OUTPUT",
        "probability_up": combined,
        "probability_down": 1.0 - combined,
        "agreement": agreement,
        "timeframes": len(usable),
        "validation_status": "RESEARCH_CANDIDATE" if all_admitted else "NOT_ADMITTED",
        "models": [x["timeframe"] for x in usable],
    }
