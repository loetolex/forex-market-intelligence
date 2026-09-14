from __future__ import annotations

import numpy as np
import pandas as pd


def direction_probability_from_momentum(df: pd.DataFrame) -> dict:
    """Deployment smoke model only; not a production trading model."""
    if df.empty:
        return {"status": "DATA UNAVAILABLE", "probability_up": None, "probability_down": None}
    row = df.iloc[-1]
    r = row.get("return_5")
    if pd.isna(r):
        return {"status": "DATA UNAVAILABLE", "probability_up": None, "probability_down": None}
    probability_up = float(np.clip(0.50 + 20.0 * float(r), 0.05, 0.95))
    return {
        "status": "MODEL OUTPUT",
        "probability_up": probability_up,
        "probability_down": 1.0 - probability_up,
        "expected_return": float(r),
        "model_version": "deployment-smoke-test-v1",
    }
