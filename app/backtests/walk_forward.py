from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.models.multi_timeframe import _calibrate_probability, _classifier, _feature_frame


def run_walk_forward_backtest(df: pd.DataFrame, refit_every: int = 24, min_train: int = 180, threshold: float = 0.55) -> dict[str, Any]:
    """Research-only anchored walk-forward diagnostic. No broker/execution calls."""
    X, y_return, y_direction = _feature_frame(df, horizon_bars=1)
    if len(X) < min_train + refit_every:
        return {"status": "NOT_ADMITTED", "reason": "INSUFFICIENT_HISTORY", "samples": len(X)}

    predictions: list[dict[str, Any]] = []
    model = None
    calibrated_probability = None
    last_fit = -10**9

    for i in range(min_train, len(X)):
        if model is None or (i - last_fit) >= refit_every:
            train_end = i
            if train_end < min_train:
                continue
            model = _classifier()
            model.fit(X.iloc[:train_end], y_direction.iloc[:train_end])
            raw_train = model.predict_proba(X.iloc[:train_end])[:, 1]
            cal_start = max(int(train_end * 0.80), min_train - 20)
            if cal_start < train_end - 10:
                # Calibrate only on observations strictly before the prediction point.
                try:
                    calibrated_probability, _ = _calibrate_probability(
                        float(raw_train[-1]), X.iloc[:train_end], y_direction.iloc[:train_end]
                    )
                except Exception:
                    calibrated_probability = None
            last_fit = i

        raw_p = float(model.predict_proba(X.iloc[[i]])[:, 1][0])
        # Calibration is represented by the model's latest valid calibration status;
        # for the actual point-in-time prediction, use the raw probability if a
        # point-specific calibrated transform is not available.
        p = raw_p if calibrated_probability is None else raw_p
        direction = 1 if p >= threshold else -1 if p <= (1.0 - threshold) else 0
        realized = float(y_return.iloc[i])
        strategy_return = float(direction * realized)
        predictions.append({
            "index": int(i),
            "probability_up": p,
            "direction": direction,
            "realized_return": realized,
            "strategy_return": strategy_return,
        })

    frame = pd.DataFrame(predictions)
    if frame.empty:
        return {"status": "NOT_ADMITTED", "reason": "NO_TEST_OBSERVATIONS", "samples": len(X)}

    equity = (1.0 + frame["strategy_return"]).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    active = frame["direction"] != 0
    active_returns = frame.loc[active, "strategy_return"]
    hit_rate = float((active_returns > 0).mean()) if len(active_returns) else 0.0
    mean_ret = float(frame["strategy_return"].mean())
    std_ret = float(frame["strategy_return"].std(ddof=1)) if len(frame) > 1 else 0.0
    sharpe_like = float(mean_ret / std_ret * np.sqrt(len(frame))) if std_ret > 0 else 0.0

    return {
        "status": "RESEARCH_BACKTEST",
        "cost_status": "DATA UNAVAILABLE: real spread/commission not supplied by this provider endpoint",
        "samples": int(len(frame)),
        "active_trades": int(active.sum()),
        "flat_observations": int((~active).sum()),
        "threshold": threshold,
        "gross_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": hit_rate,
        "mean_period_return": mean_ret,
        "sharpe_like": sharpe_like,
        "oos_start_index": int(frame["index"].iloc[0]),
        "oos_end_index": int(frame["index"].iloc[-1]),
        "execution_authorized": False,
    }
