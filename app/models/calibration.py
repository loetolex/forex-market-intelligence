from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _logit(probability: np.ndarray | float) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _fit_platt(logits: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=1.0, solver="lbfgs", random_state=42)),
    ])
    model.fit(logits.reshape(-1, 1), y)
    return model


def fit_time_series_platt_calibrator(
    X: pd.DataFrame,
    y_direction: pd.Series,
    model_factory: Callable[[], Any],
    *,
    min_train: int = 120,
    refit_every: int = 20,
    min_calibration_rows: int = 40,
) -> tuple[Any | None, dict[str, Any]]:
    """Fit Platt scaling only from strictly out-of-sample development predictions.

    Every calibration observation is produced by a model trained strictly before
    that observation. The final holdout is never touched here.
    """
    n = len(X)
    if n <= min_train + min_calibration_rows:
        return None, {
            "status": "NOT_CALIBRATED",
            "reason": "INSUFFICIENT_CALIBRATION_HISTORY",
            "calibration_rows": 0,
            "data_role": "DEVELOPMENT_ONLY",
        }

    probabilities: list[float] = []
    targets: list[int] = []
    model = None
    last_fit = -10**9

    for i in range(min_train, n):
        if model is None or (i - last_fit) >= refit_every:
            train_y = y_direction.iloc[:i]
            if train_y.nunique() < 2:
                model = None
                continue
            model = model_factory()
            model.fit(X.iloc[:i], train_y)
            last_fit = i

        if model is None:
            continue

        probability = float(model.predict_proba(X.iloc[[i]])[:, 1][0])
        probabilities.append(float(np.clip(probability, 0.0, 1.0)))
        targets.append(int(y_direction.iloc[i]))

    if len(probabilities) < min_calibration_rows:
        return None, {
            "status": "NOT_CALIBRATED",
            "reason": "INSUFFICIENT_OOS_CALIBRATION_ROWS",
            "calibration_rows": len(probabilities),
            "data_role": "DEVELOPMENT_ONLY",
        }

    y_cal = np.asarray(targets, dtype=int)
    if len(np.unique(y_cal)) < 2:
        return None, {
            "status": "NOT_CALIBRATED",
            "reason": "CALIBRATION_SET_SINGLE_CLASS",
            "calibration_rows": len(probabilities),
            "data_role": "DEVELOPMENT_ONLY",
        }

    raw = np.asarray(probabilities, dtype=float)
    raw_brier = float(brier_score_loss(y_cal, raw))
    calibrator = _fit_platt(_logit(raw), y_cal)
    calibrated = np.clip(
        calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1],
        1e-6,
        1.0 - 1e-6,
    )
    calibrated_brier = float(brier_score_loss(y_cal, calibrated))

    status = "CALIBRATED" if calibrated_brier <= raw_brier else "RAW_MODEL_PREFERRED"
    selected_method = "platt_scaling" if status == "CALIBRATED" else "raw_probability"

    return calibrator if status == "CALIBRATED" else None, {
        "status": status,
        "method": selected_method,
        "calibration_rows": len(probabilities),
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "refit_every": refit_every,
        "min_train": min_train,
        "data_role": "DEVELOPMENT_ONLY_OOS_PREDICTIONS",
        "final_holdout_untouched": True,
    }


def apply_probability_calibrator(
    raw_probability: float,
    calibrator: Any | None,
) -> float:
    raw = float(np.clip(raw_probability, 1e-6, 1.0 - 1e-6))
    if calibrator is None:
        return raw
    calibrated = calibrator.predict_proba(
        _logit(np.asarray([raw])).reshape(-1, 1)
    )[:, 1][0]
    return float(np.clip(calibrated, 0.0, 1.0))
