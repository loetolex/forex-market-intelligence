from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "research-candidate-hgb-mtf-v1"
MIN_ROWS = 180
MIN_TRAIN = 120
VALIDATION_FOLDS = 4
VALIDATION_TEST = 20
CALIBRATION_FRACTION = 0.20

FEATURES = ["return_1", "return_3", "return_6", "volatility_10", "sma_10_ratio", "sma_30_ratio", "range_pct"]


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["return_1"] = frame["close"].pct_change()
    frame["return_3"] = frame["close"].pct_change(3)
    frame["return_6"] = frame["close"].pct_change(6)
    frame["volatility_10"] = frame["return_1"].rolling(10).std()
    frame["sma_10_ratio"] = frame["close"] / frame["close"].rolling(10).mean() - 1.0
    frame["sma_30_ratio"] = frame["close"] / frame["close"].rolling(30).mean() - 1.0
    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"]
    return frame


def _prepare_supervised(df: pd.DataFrame, horizon_bars: int = 1):
    frame = _feature_frame(df)
    frame["future_return"] = frame["close"].shift(-horizon_bars) / frame["close"] - 1.0
    frame["target"] = (frame["future_return"] > 0).astype(int)
    frame = frame.dropna(subset=FEATURES + ["future_return"])
    return frame, frame[FEATURES], frame["target"]


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """Never calculate balanced accuracy on a fold missing a class.

    A single-class test fold is explicitly invalid evidence rather than being
    forced into a two-class confusion matrix.
    """
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return None, "SINGLE_CLASS_TEST_OR_PREDICTION"
    return float(balanced_accuracy_score(y_true, y_pred)), None


def _walk_forward_validate(X: pd.DataFrame, y: pd.Series, returns: pd.Series | None = None) -> dict:
    rows = []
    n = len(X)
    for fold in range(VALIDATION_FOLDS):
        test_end = n - (VALIDATION_FOLDS - fold - 1) * VALIDATION_TEST
        test_start = test_end - VALIDATION_TEST
        train_end = test_start
        if train_end < MIN_TRAIN or test_start < 0:
            continue

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            rows.append({"fold": fold + 1, "status": "INVALID_VALIDATION_FOLD", "reason": "SINGLE_CLASS_TRAIN_OR_TEST_SET"})
            continue

        model = HistGradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        p_model = model.predict_proba(X_test)[:, 1]
        y_pred = (p_model >= 0.5).astype(int)
        bal_acc, reason = _safe_balanced_accuracy(y_test.to_numpy(), y_pred)
        if reason:
            rows.append({"fold": fold + 1, "status": "INVALID_VALIDATION_FOLD", "reason": reason})
            continue

        model_brier = float(brier_score_loss(y_test, p_model))
        naive_probability = float(y_train.mean())
        naive_brier = float(brier_score_loss(y_test, np.full(len(y_test), naive_probability)))

        model_rmse = None
        naive_rmse = None
        if returns is not None:
            y_ret = returns.iloc[test_start:test_end].to_numpy(dtype=float)
            # Classification model is converted to a conservative expected-return
            # proxy only for comparison bookkeeping; it is not a trading signal.
            scale = float(np.std(returns.iloc[:train_end])) if train_end else 0.0
            model_return = (p_model - 0.5) * 2.0 * scale
            naive_return = np.zeros(len(y_ret))
            model_rmse = float(np.sqrt(mean_squared_error(y_ret, model_return)))
            naive_rmse = float(np.sqrt(mean_squared_error(y_ret, naive_return)))

        rows.append({
            "fold": fold + 1,
            "status": "VALID",
            "reason": None,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_positive_rate": float(y_test.mean()),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": bal_acc,
            "brier": model_brier,
            "naive_brier": naive_brier,
            "brier_beats_naive": bool(model_brier < naive_brier),
            "model_rmse": model_rmse,
            "naive_rmse": naive_rmse,
            "rmse_beats_naive": bool(model_rmse < naive_rmse) if model_rmse is not None else None,
        })

    valid = [r for r in rows if r["status"] == "VALID"]
    return {
        "folds": rows,
        "valid_fold_count": len(valid),
        "invalid_fold_count": len(rows) - len(valid),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in valid])) if valid else None,
        "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in valid])) if valid else None,
        "mean_brier": float(np.mean([r["brier"] for r in valid])) if valid else None,
        "mean_naive_brier": float(np.mean([r["naive_brier"] for r in valid])) if valid else None,
        "beats_naive_brier_rate": float(np.mean([r["brier_beats_naive"] for r in valid])) if valid else 0.0,
        "beats_naive_rmse_rate": float(np.mean([r["rmse_beats_naive"] for r in valid if r["rmse_beats_naive"] is not None])) if any(r["rmse_beats_naive"] is not None for r in valid) else 0.0,
    }


def _calibrate_probability(X: pd.DataFrame, y: pd.Series, model):
    n_cal = max(20, int(len(X) * CALIBRATION_FRACTION))
    if len(X) <= n_cal + MIN_TRAIN or y.iloc[-n_cal:].nunique() < 2:
        return model, {"status": "CALIBRATION_UNAVAILABLE", "reason": "INSUFFICIENT_TWO_CLASS_CALIBRATION_DATA"}
    X_fit, X_cal = X.iloc[:-n_cal], X.iloc[-n_cal:]
    y_fit, y_cal = y.iloc[:-n_cal], y.iloc[-n_cal:]
    model.fit(X_fit, y_fit)
    raw_prob = model.predict_proba(X_cal)[:, 1]
    calibrator = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    calibrator.fit(raw_prob.reshape(-1, 1), y_cal)
    calibrated = calibrator.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
    return (model, calibrator), {"status": "CALIBRATED", "rows": len(X_cal), "brier": float(brier_score_loss(y_cal, calibrated))}


def _regime(df: pd.DataFrame) -> str:
    returns = df["close"].pct_change().dropna()
    if len(returns) < 30:
        return "UNKNOWN"
    vol = returns.tail(20).std()
    long_vol = returns.tail(min(120, len(returns))).std()
    trend = df["close"].iloc[-1] / df["close"].tail(min(30, len(df))).mean() - 1
    if long_vol and vol > long_vol * 1.5:
        return "HIGH_VOLATILITY"
    if trend > 0.002:
        return "TREND_UP"
    if trend < -0.002:
        return "TREND_DOWN"
    if long_vol and vol < long_vol * 0.65:
        return "LOW_VOLATILITY"
    return "RANGE"


def forecast_timeframe(df: pd.DataFrame, symbol: str, interval: str, horizon_bars: int = 1) -> dict:
    if len(df) < MIN_ROWS:
        return {"status": "DATA UNAVAILABLE", "reason": "INSUFFICIENT_ROWS", "symbol": symbol, "interval": interval}
    frame, X, y = _prepare_supervised(df, horizon_bars=horizon_bars)
    if len(X) < MIN_ROWS or y.nunique() < 2:
        return {"status": "DATA UNAVAILABLE", "reason": "INSUFFICIENT_TWO_CLASS_DATA", "symbol": symbol, "interval": interval}

    validation = _walk_forward_validate(X, y, frame["future_return"])
    candidate = bool(
        validation["valid_fold_count"] == VALIDATION_FOLDS
        and validation["mean_balanced_accuracy"] is not None
        and validation["mean_balanced_accuracy"] > 0.50
        and validation["mean_brier"] is not None
        and validation["mean_naive_brier"] is not None
        and validation["mean_brier"] < validation["mean_naive_brier"]
    )

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    probability_up = float(model.predict_proba(X.tail(1))[:, 1][0])
    return {
        "status": "MODEL OUTPUT",
        "symbol": symbol,
        "interval": interval,
        "model_version": MODEL_VERSION,
        "probability_up": probability_up,
        "probability_down": 1.0 - probability_up,
        "expected_return": float(frame["future_return"].tail(min(20, len(frame))).mean()),
        "confidence": float(abs(probability_up - 0.5) * 2.0),
        "regime": _regime(df),
        "validation": {**validation, "status": "ADMITTED" if candidate else "NOT_ADMITTED", "candidate": candidate},
        "data_status": "REAL_DATA",
        "output_status": "MODEL OUTPUT",
    }


def combine_timeframes(per_tf: list[dict]) -> dict:
    usable = [x for x in per_tf if x.get("status") == "MODEL OUTPUT" and x.get("probability_up") is not None]
    if not usable:
        return {"status": "DATA UNAVAILABLE", "probability_up": None, "probability_down": None, "agreement": 0.0}
    probs = [float(x["probability_up"]) for x in usable]
    p = float(np.mean(probs))
    directions = [p_i >= 0.5 for p_i in probs]
    agreement = float(max(sum(directions), len(directions) - sum(directions)) / len(directions))
    admitted = all(x.get("validation", {}).get("status") == "ADMITTED" for x in usable)
    return {"status": "MODEL OUTPUT", "probability_up": p, "probability_down": 1.0 - p, "agreement": agreement, "validation": {"status": "ADMITTED" if admitted else "NOT_ADMITTED"}}


def forecast_90d(df: pd.DataFrame, symbol: str) -> dict:
    result = forecast_timeframe(df, symbol, "90D", horizon_bars=90)
    result["horizon"] = "90D"
    return result
