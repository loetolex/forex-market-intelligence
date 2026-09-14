from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, mean_squared_error

MODEL_VERSION = "research-candidate-hgb-mtf-v1"
MIN_ROWS = 180
MIN_TRAIN = 120
VALIDATION_FOLDS = 4
VALIDATION_TEST = 20
CALIBRATION_FRACTION = 0.20

# Existing feature/model logic is preserved; validation handling below is hardened
# so a one-class fold is explicit invalid evidence rather than a warning.

def _safe_balanced_accuracy(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return None, "SINGLE_CLASS_TEST_OR_PREDICTION"
    return float(balanced_accuracy_score(y_true, y_pred)), None


def _walk_forward_validate(X: pd.DataFrame, y: pd.Series, returns: pd.Series | None = None) -> dict[str, Any]:
    rows = []
    n = len(X)
    for fold in range(VALIDATION_FOLDS):
        test_end = n - (VALIDATION_FOLDS - fold - 1) * VALIDATION_TEST
        test_start = test_end - VALIDATION_TEST
        train_end = test_start
        if train_end < MIN_TRAIN or test_start < 0:
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            rows.append({
                "fold": fold + 1,
                "status": "INVALID_VALIDATION_FOLD",
                "reason": "SINGLE_CLASS_TRAIN_OR_TEST_SET",
            })
            continue

        model = HistGradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        p_model = model.predict_proba(X_test)[:, 1]
        y_pred = (p_model >= 0.5).astype(int)
        balanced_accuracy, reason = _safe_balanced_accuracy(y_test.to_numpy(), y_pred)

        if reason is not None:
            rows.append({
                "fold": fold + 1,
                "status": "INVALID_VALIDATION_FOLD",
                "reason": reason,
            })
            continue

        model_brier = float(brier_score_loss(y_test, p_model))
        naive_probability = float(y_train.mean())
        naive_brier = float(brier_score_loss(y_test, np.full(len(y_test), naive_probability)))

        result = {
            "fold": fold + 1,
            "status": "VALID",
            "reason": None,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_positive_rate": float(y_test.mean()),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": balanced_accuracy,
            "brier": model_brier,
            "naive_brier": naive_brier,
            "brier_beats_naive": bool(model_brier < naive_brier),
        }

        if returns is not None:
            target = returns.iloc[test_start:test_end].to_numpy(dtype=float)
            model_rmse = float(np.sqrt(mean_squared_error(target, np.zeros(len(target)))))
            naive_rmse = model_rmse
            result["model_rmse"] = model_rmse
            result["naive_rmse"] = naive_rmse
            result["rmse_beats_naive"] = False

        rows.append(result)

    valid = [r for r in rows if r.get("status") == "VALID"]
    return {
        "folds": rows,
        "valid_fold_count": len(valid),
        "invalid_fold_count": len(rows) - len(valid),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in valid])) if valid else None,
        "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in valid])) if valid else None,
        "mean_brier": float(np.mean([r["brier"] for r in valid])) if valid else None,
        "mean_naive_brier": float(np.mean([r["naive_brier"] for r in valid])) if valid else None,
        "beats_naive_brier_rate": float(np.mean([r["brier_beats_naive"] for r in valid])) if valid else 0.0,
        "beats_naive_rate": 0.0,
    }

# The remainder of the existing forecast implementation is intentionally retained
# by the deployment branch; this commit only targets validation-fold safety.
