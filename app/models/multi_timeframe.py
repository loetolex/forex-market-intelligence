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
FINAL_HOLDOUT_FRACTION = 0.20
MIN_HOLDOUT_ROWS = 20
MIN_HOLDOUT_ROWS_90D = 90

# Admission policy: broad, stable out-of-sample evidence must be followed by
# a completely untouched final holdout before a model can become a research
# candidate. These are research gates, not claims of profitability.
MIN_VALID_FOLDS_FOR_ADMISSION = VALIDATION_FOLDS
MIN_MEAN_BALANCED_ACCURACY = 0.55
MIN_MEDIAN_BALANCED_ACCURACY = 0.55
MIN_FOLD_BALANCED_ACCURACY = 0.50
MIN_BEATS_NAIVE_RMSE_RATE = 0.75
ADMISSION_POLICY = "STRICT_WALK_FORWARD_HOLDOUT_V2"


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
            learning_rate=0.05, max_iter=180, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )),
    ])


def _regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=180, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )),
    ])


def _validation_test_size(horizon_bars: int) -> int:
    return 90 if horizon_bars >= 90 else VALIDATION_TEST


def _holdout_rows(n: int, horizon_bars: int) -> int:
    minimum = MIN_HOLDOUT_ROWS_90D if horizon_bars >= 90 else MIN_HOLDOUT_ROWS
    return max(int(np.ceil(n * FINAL_HOLDOUT_FRACTION)), minimum)


def _metrics_for_split(
    X_train: pd.DataFrame,
    y_return_train: pd.Series,
    y_direction_train: pd.Series,
    X_test: pd.DataFrame,
    y_return_test: pd.Series,
    y_direction_test: pd.Series,
    *,
    label: str,
    fold: int | None = None,
    require_two_class_test: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    ytr_d = y_direction_train.to_numpy()
    yt_d = y_direction_test.to_numpy()
    train_classes = np.unique(ytr_d)
    test_classes = np.unique(yt_d)

    if len(train_classes) < 2:
        return None, {
            "fold": fold,
            "status": "INVALID_VALIDATION_FOLD",
            "reason": "SINGLE_CLASS_TRAIN_SET",
            "train_class_count": int(len(train_classes)),
            "test_class_count": int(len(test_classes)),
        }

    if require_two_class_test and len(test_classes) < 2:
        return None, {
            "fold": fold,
            "status": "INVALID_VALIDATION_FOLD",
            "reason": "SINGLE_CLASS_TEST_SET",
            "train_class_count": int(len(train_classes)),
            "test_class_count": int(len(test_classes)),
        }

    clf, reg = _classifier(), _regressor()
    clf.fit(X_train, y_direction_train)
    reg.fit(X_train, y_return_train)
    p = np.clip(clf.predict_proba(X_test)[:, 1], 0.0, 1.0)
    r = reg.predict(X_test)
    yt_r = y_return_test.to_numpy()
    train_prevalence = float(np.mean(ytr_d))
    baseline_probability = np.full(len(yt_d), train_prevalence, dtype=float)

    record: dict[str, Any] = {
        "status": label,
        "train_class_count": int(len(train_classes)),
        "test_class_count": int(len(test_classes)),
        "train_positive_rate": train_prevalence,
        "test_positive_rate": float(np.mean(yt_d)) if len(yt_d) else None,
        "accuracy": float(accuracy_score(yt_d, p >= 0.5)),
        "brier": float(brier_score_loss(yt_d, p)),
        "brier_baseline": float(brier_score_loss(yt_d, baseline_probability)),
        "rmse_model": float(np.sqrt(mean_squared_error(yt_r, r))),
        "rmse_naive": float(np.sqrt(mean_squared_error(yt_r, np.zeros_like(yt_r)))),
        "sample_count": int(len(yt_d)),
    }
    if fold is not None:
        record["fold"] = fold
    if len(test_classes) >= 2:
        record["balanced_accuracy"] = float(balanced_accuracy_score(yt_d, p >= 0.5))
    else:
        record["balanced_accuracy"] = None
    return record, None


def _walk_forward_validate(
    X: pd.DataFrame,
    y_return: pd.Series,
    y_direction: pd.Series,
    horizon_bars: int,
) -> dict[str, Any]:
    n = len(X)
    test_size = _validation_test_size(horizon_bars)
    holdout_rows = _holdout_rows(n, horizon_bars)
    if holdout_rows >= n:
        return {
            "status": "HOLD",
            "reason": "INSUFFICIENT_FINAL_HOLDOUT_HISTORY",
            "admission_policy": ADMISSION_POLICY,
            "validation_test_size": test_size,
            "development_rows": 0,
            "holdout_start_index": 0,
            "holdout_rows": n,
            "folds": [],
            "fold_count": 0,
            "valid_fold_count": 0,
            "invalid_fold_count": 0,
            "holdout": {
                "available": False,
                "two_class": False,
                "status": "HOLD",
            },
        }

    development_rows = n - holdout_rows
    required_development_rows = MIN_TRAIN + (VALIDATION_FOLDS * test_size)
    if development_rows < required_development_rows:
        return {
            "status": "HOLD",
            "reason": "INSUFFICIENT_WALK_FORWARD_DEVELOPMENT_HISTORY",
            "admission_policy": ADMISSION_POLICY,
            "validation_test_size": test_size,
            "required_development_rows": required_development_rows,
            "development_rows": development_rows,
            "holdout_start_index": development_rows,
            "holdout_rows": holdout_rows,
            "folds": [],
            "fold_count": 0,
            "valid_fold_count": 0,
            "invalid_fold_count": 0,
            "holdout": {
                "available": False,
                "two_class": False,
                "status": "HOLD",
            },
        }

    first_train_end = development_rows - (VALIDATION_FOLDS * test_size)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(VALIDATION_FOLDS):
        train_end = first_train_end + fold_index * test_size
        test_start = train_end
        test_end = test_start + test_size
        splits.append((np.arange(train_end), np.arange(test_start, test_end)))

    records: list[dict[str, Any]] = []
    invalid_folds: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        record, invalid = _metrics_for_split(
            X.iloc[train_idx],
            y_return.iloc[train_idx],
            y_direction.iloc[train_idx],
            X.iloc[test_idx],
            y_return.iloc[test_idx],
            y_direction.iloc[test_idx],
            label="VALID",
            fold=fold,
            require_two_class_test=True,
        )
        if invalid is not None:
            invalid_folds.append(invalid)
        elif record is not None:
            records.append(record)

    frame = pd.DataFrame(records)
    holdout_start = development_rows
    holdout_end = n
    holdout_X = X.iloc[holdout_start:holdout_end]
    holdout_y_return = y_return.iloc[holdout_start:holdout_end]
    holdout_y_direction = y_direction.iloc[holdout_start:holdout_end]

    holdout_record: dict[str, Any]
    holdout_result, holdout_invalid = _metrics_for_split(
        X.iloc[:development_rows],
        y_return.iloc[:development_rows],
        y_direction.iloc[:development_rows],
        holdout_X,
        holdout_y_return,
        holdout_y_direction,
        label="FINAL_HOLDOUT",
        require_two_class_test=False,
    )

    holdout_two_class = len(np.unique(holdout_y_direction.to_numpy())) >= 2
    holdout_train_two_class = len(np.unique(y_direction.iloc[:development_rows].to_numpy())) >= 2
    holdout_available = len(holdout_X) >= (MIN_HOLDOUT_ROWS_90D if horizon_bars >= 90 else MIN_HOLDOUT_ROWS)
    if holdout_result is None:
        holdout_record = {
            "status": "FINAL_HOLDOUT",
            "available": False,
            "two_class": holdout_two_class,
            "train_two_class": holdout_train_two_class,
            "sample_count": int(len(holdout_X)),
            "reason": holdout_invalid["reason"] if holdout_invalid else "HOLDOUT_EVALUATION_FAILED",
        }
    else:
        holdout_record = {
            **holdout_result,
            "available": bool(holdout_available),
            "two_class": bool(holdout_two_class),
            "train_two_class": bool(holdout_train_two_class),
        }

    if frame.empty:
        return {
            "status": "HOLD",
            "reason": "NO_VALID_WALK_FORWARD_FOLDS",
            "admission_policy": ADMISSION_POLICY,
            "validation_test_size": test_size,
            "development_rows": development_rows,
            "holdout_start_index": holdout_start,
            "holdout_rows": holdout_rows,
            "folds": invalid_folds,
            "fold_count": VALIDATION_FOLDS,
            "valid_fold_count": 0,
            "invalid_fold_count": len(invalid_folds),
            "holdout": holdout_record,
        }

    mean_balanced = float(frame["balanced_accuracy"].mean())
    median_balanced = float(frame["balanced_accuracy"].median())
    min_balanced = float(frame["balanced_accuracy"].min())
    mean_brier = float(frame["brier"].mean())
    mean_brier_baseline = float(frame["brier_baseline"].mean())
    beats_naive_rate = float((frame["rmse_model"] < frame["rmse_naive"]).mean())
    median_rmse_model = float(frame["rmse_model"].median())
    median_rmse_naive = float(frame["rmse_naive"].median())

    admission_checks = {
        "all_configured_folds_present": len(frame) == MIN_VALID_FOLDS_FOR_ADMISSION,
        "no_invalid_folds": len(invalid_folds) == 0,
        "mean_balanced_accuracy": mean_balanced >= MIN_MEAN_BALANCED_ACCURACY,
        "median_balanced_accuracy": median_balanced >= MIN_MEDIAN_BALANCED_ACCURACY,
        "minimum_fold_balanced_accuracy": min_balanced >= MIN_FOLD_BALANCED_ACCURACY,
        "beats_naive_rmse_rate": beats_naive_rate >= MIN_BEATS_NAIVE_RMSE_RATE,
        "mean_brier_beats_prevalence_baseline": mean_brier < mean_brier_baseline,
        "median_rmse_beats_naive": median_rmse_model < median_rmse_naive,
        "holdout_available": bool(holdout_available),
        "holdout_two_class": bool(holdout_two_class),
        "holdout_train_two_class": bool(holdout_train_two_class),
        "holdout_balanced_accuracy": bool(
            holdout_two_class
            and holdout_record.get("balanced_accuracy") is not None
            and float(holdout_record["balanced_accuracy"]) >= MIN_MEAN_BALANCED_ACCURACY
        ),
        "holdout_brier_beats_prevalence_baseline": bool(
            holdout_record.get("brier") is not None
            and holdout_record.get("brier_baseline") is not None
            and float(holdout_record["brier"]) < float(holdout_record["brier_baseline"])
        ),
        "holdout_rmse_beats_naive": bool(
            holdout_record.get("rmse_model") is not None
            and holdout_record.get("rmse_naive") is not None
            and float(holdout_record["rmse_model"]) < float(holdout_record["rmse_naive"])
        ),
    }
    failed_checks = [name for name, passed in admission_checks.items() if not passed]
    candidate = len(failed_checks) == 0

    return {
        "status": "RESEARCH_CANDIDATE" if candidate else "NOT_ADMITTED",
        "admission_policy": ADMISSION_POLICY,
        "validation_test_size": test_size,
        "development_rows": development_rows,
        "holdout_start_index": holdout_start,
        "holdout_rows": holdout_rows,
        "required_development_rows": required_development_rows,
        "admission_checks": admission_checks,
        "failed_admission_checks": failed_checks,
        "folds": records + invalid_folds,
        "fold_count": VALIDATION_FOLDS,
        "valid_fold_count": int(len(frame)),
        "invalid_fold_count": int(len(invalid_folds)),
        "beats_naive_rmse_rate": beats_naive_rate,
        "mean_balanced_accuracy": mean_balanced,
        "median_balanced_accuracy": median_balanced,
        "min_balanced_accuracy": min_balanced,
        "mean_brier": mean_brier,
        "mean_brier_baseline": mean_brier_baseline,
        "median_rmse_model": median_rmse_model,
        "median_rmse_naive": median_rmse_naive,
        "mean_rmse_model": float(frame["rmse_model"].mean()),
        "mean_rmse_naive": float(frame["rmse_naive"].mean()),
        "holdout": holdout_record,
    }


def _calibration_split_size(n_development_rows: int) -> int:
    return max(MIN_TRAIN, int(n_development_rows * (1.0 - CALIBRATION_FRACTION)))


def _calibrate_probability(raw_probability: float, X: pd.DataFrame, y_direction: pd.Series) -> tuple[float, dict[str, Any]]:
    n = len(X)
    cal_start = _calibration_split_size(n)
    if cal_start >= n - 10:
        return raw_probability, {
            "status": "NOT_CALIBRATED",
            "reason": "INSUFFICIENT_CALIBRATION_HISTORY",
            "data_role": "DEVELOPMENT_ONLY",
        }
    train_idx, cal_idx = np.arange(cal_start), np.arange(cal_start, n)
    clf = _classifier()
    clf.fit(X.iloc[train_idx], y_direction.iloc[train_idx])
    p_cal = np.clip(clf.predict_proba(X.iloc[cal_idx])[:, 1], 1e-6, 1 - 1e-6)
    y_cal = y_direction.iloc[cal_idx].to_numpy()
    if len(np.unique(y_cal)) < 2:
        return raw_probability, {
            "status": "NOT_CALIBRATED",
            "reason": "CALIBRATION_SET_SINGLE_CLASS",
            "data_role": "DEVELOPMENT_ONLY",
        }
    logits = np.log(p_cal / (1.0 - p_cal)).reshape(-1, 1)
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
        "data_role": "DEVELOPMENT_ONLY",
    }


def _regime(df: pd.DataFrame) -> dict[str, Any]:
    x = add_features(df.copy())
    atr = x["atr_pct"].dropna()
    if atr.empty:
        return {"status": "DATA UNAVAILABLE", "regime": "UNKNOWN"}
    last = x.iloc[-1]
    atr_pct = float(last["atr_pct"]) if pd.notna(last["atr_pct"]) else np.nan
    low, high = float(atr.quantile(0.20)), float(atr.quantile(0.80))
    if np.isfinite(atr_pct) and atr_pct >= high:
        state = "HIGH_VOLATILITY"
    elif np.isfinite(atr_pct) and atr_pct <= low:
        state = "LOW_VOLATILITY"
    elif pd.notna(last["sma_20"]) and pd.notna(last["sma_50"]):
        state = "TREND_UP" if last["sma_20"] > last["sma_50"] else "TREND_DOWN" if last["sma_20"] < last["sma_50"] else "RANGE"
    else:
        state = "UNKNOWN"
    return {"status": "CALCULATED", "regime": state, "atr_pct": atr_pct}


def forecast_timeframe(df: pd.DataFrame, instrument: str, timeframe: str, horizon_bars: int = 1) -> dict[str, Any]:
    X, y_return, y_direction = _feature_frame(df, horizon_bars=horizon_bars)
    base = {
        "instrument": instrument, "timeframe": timeframe, "horizon_bars": horizon_bars,
        "model_version": MODEL_VERSION, "status": "DATA UNAVAILABLE",
        "probability_up": None, "probability_down": None, "expected_return": None,
        "expected_abs_move": None, "uncertainty": None, "confidence": None,
        "regime": _regime(df),
    }
    if len(X) < MIN_ROWS or y_direction.nunique() < 2:
        base["reason"] = "INSUFFICIENT_MODEL_HISTORY"
        return base

    validation = _walk_forward_validate(X, y_return, y_direction, horizon_bars)
    if validation.get("status") == "HOLD":
        base["validation_status"] = "HOLD"
        base["validation"] = validation
        base["reason"] = validation.get("reason", "VALIDATION_HOLD")
        return base

    development_rows = int(validation["development_rows"])
    if development_rows < MIN_TRAIN:
        base["validation_status"] = "HOLD"
        base["validation"] = validation
        base["reason"] = "INSUFFICIENT_DEVELOPMENT_HISTORY"
        return base

    # Keep the final holdout untouched: it is never used for model fitting or
    # calibration. The research output is produced from the development model.
    clf, reg = _classifier(), _regressor()
    clf.fit(X.iloc[:development_rows], y_direction.iloc[:development_rows])
    reg.fit(X.iloc[:development_rows], y_return.iloc[:development_rows])
    latest = X.tail(1)
    raw_p = float(np.clip(clf.predict_proba(latest)[:, 1][0], 0.0, 1.0))
    calibrated_p, calibration = _calibrate_probability(
        raw_p,
        X.iloc[:development_rows],
        y_direction.iloc[:development_rows],
    )
    expected_return = float(reg.predict(latest)[0])
    residuals = y_return.iloc[:development_rows] - reg.predict(X.iloc[:development_rows])
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
        "training_rows": int(development_rows),
        "training_data_role": "DEVELOPMENT_ONLY",
        "final_holdout_untouched": True,
        "prediction_timestamp": str(pd.to_datetime(df["timestamp"].max(), utc=True)),
    }


def combine_timeframes(forecasts: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [x for x in forecasts if x.get("status") == "MODEL OUTPUT" and x.get("probability_up") is not None]
    if not usable:
        return {"status": "DATA UNAVAILABLE", "probability_up": None, "probability_down": None, "agreement": 0.0, "timeframes": 0, "validation_status": "HOLD"}
    p = np.array([float(x["probability_up"]) for x in usable])
    combined = float(np.mean(p))
    directions = np.array([v >= 0.5 for v in p], dtype=int)
    agreement = float(max(directions.mean(), 1.0 - directions.mean()))
    all_admitted = all(x.get("validation_status") == "RESEARCH_CANDIDATE" for x in usable)
    return {
        "status": "MODEL OUTPUT", "probability_up": combined, "probability_down": 1.0 - combined,
        "agreement": agreement, "timeframes": len(usable),
        "validation_status": "RESEARCH_CANDIDATE" if all_admitted else "NOT_ADMITTED",
        "models": [x["timeframe"] for x in usable],
    }


def forecast_90d(df: pd.DataFrame, instrument: str) -> dict[str, Any]:
    result = forecast_timeframe(df, instrument, "90D", horizon_bars=90)
    result["horizon"] = "90D"
    return result
