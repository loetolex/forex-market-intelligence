from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.models.multi_timeframe import (
    MIN_HOLDOUT_ROWS,
    VALIDATION_FOLDS,
    VALIDATION_TEST,
    _feature_frame,
    _holdout_rows,
)


def _candidates() -> dict[str, Callable[[], Pipeline]]:
    return {
        "logistic": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.5,
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "hist_gradient_boosting": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=180,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            )),
        ]),
        "extra_trees": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def benchmark_classifiers(
    df: pd.DataFrame,
    *,
    horizon_bars: int = 1,
    min_train: int = 120,
    validation_folds: int = VALIDATION_FOLDS,
    validation_test: int = VALIDATION_TEST,
) -> dict[str, Any]:
    """Compare candidate classifiers using development-only anchored walk-forward CV.

    The final holdout is explicitly excluded from model selection.
    """
    X, _, y_direction = _feature_frame(df, horizon_bars=horizon_bars)
    n = len(X)
    holdout_rows = _holdout_rows(n, horizon_bars)
    development_rows = n - holdout_rows

    if development_rows < min_train + validation_folds * validation_test:
        return {
            "status": "HOLD",
            "reason": "INSUFFICIENT_WALK_FORWARD_DEVELOPMENT_HISTORY",
            "samples": n,
            "development_rows": max(0, development_rows),
            "holdout_rows": holdout_rows,
            "final_holdout_touched": False,
        }

    first_train_end = development_rows - validation_folds * validation_test
    splits = []
    for fold in range(validation_folds):
        train_end = first_train_end + fold * validation_test
        test_start = train_end
        test_end = test_start + validation_test
        splits.append((fold + 1, train_end, test_start, test_end))

    rows: list[dict[str, Any]] = []

    for model_name, factory in _candidates().items():
        fold_metrics = []

        for fold, train_end, test_start, test_end in splits:
            X_train = X.iloc[:train_end]
            y_train = y_direction.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y_direction.iloc[test_start:test_end]

            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue

            model = factory()
            model.fit(X_train, y_train)

            p = np.clip(
                model.predict_proba(X_test)[:, 1],
                0.0,
                1.0,
            )
            prediction = p >= 0.5

            fold_metrics.append({
                "fold": fold,
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_test, prediction)
                ),
                "brier": float(brier_score_loss(y_test, p)),
                "samples": int(len(y_test)),
            })

        if not fold_metrics:
            rows.append({
                "model": model_name,
                "status": "HOLD",
                "valid_folds": 0,
                "mean_balanced_accuracy": None,
                "mean_brier": None,
            })
            continue

        frame = pd.DataFrame(fold_metrics)
        rows.append({
            "model": model_name,
            "status": "RESEARCH_COMPARISON",
            "valid_folds": int(len(frame)),
            "mean_balanced_accuracy": float(frame["balanced_accuracy"].mean()),
            "median_balanced_accuracy": float(frame["balanced_accuracy"].median()),
            "min_balanced_accuracy": float(frame["balanced_accuracy"].min()),
            "mean_brier": float(frame["brier"].mean()),
            "folds": fold_metrics,
        })

    comparable = [x for x in rows if x.get("mean_brier") is not None]
    selected = None
    if comparable:
        # Selection is by development-only calibration quality first, then
        # balanced accuracy. The final holdout remains untouched.
        selected = sorted(
            comparable,
            key=lambda x: (
                x["mean_brier"],
                -x["mean_balanced_accuracy"],
            ),
        )[0]["model"]

    return {
        "status": "RESEARCH_MODEL_COMPARISON",
        "selection_rule": "lowest_mean_development_brier_then_highest_mean_balanced_accuracy",
        "samples": n,
        "development_rows": development_rows,
        "holdout_rows": holdout_rows,
        "final_holdout_touched": False,
        "candidates": rows,
        "selected_development_candidate": selected,
    }
