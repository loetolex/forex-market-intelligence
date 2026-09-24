import numpy as np
import pandas as pd

from app.models.calibration import fit_time_series_platt_calibrator
from app.models.multi_timeframe import _classifier


def _dataset(rows=260):
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, rows))
    high = close + np.abs(rng.normal(0, 0.5, rows))
    low = close - np.abs(rng.normal(0, 0.5, rows))
    open_ = close + rng.normal(0, 0.2, rows)
    timestamp = pd.date_range(
        "2024-01-01",
        periods=rows,
        freq="h",
        tz="UTC",
    )
    return pd.DataFrame({
        "timestamp": timestamp,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "provider": "TEST",
        "instrument": "TEST",
        "timeframe": "1h",
        "data_status": "SIMULATED DATA",
    })


def test_time_series_calibrator_uses_oos_predictions():
    df = _dataset()
    from app.models.multi_timeframe import _feature_frame
    X, _, y = _feature_frame(df, horizon_bars=1)

    calibrator, meta = fit_time_series_platt_calibrator(
        X,
        y,
        _classifier,
        min_train=120,
        refit_every=20,
        min_calibration_rows=40,
    )

    assert meta["data_role"].startswith("DEVELOPMENT_ONLY")
    assert meta["final_holdout_untouched"] is True
    assert meta["calibration_rows"] >= 40
    assert meta["status"] in {
        "CALIBRATED",
        "RAW_MODEL_PREFERRED",
        "NOT_CALIBRATED",
    }
    if meta["status"] == "CALIBRATED":
        assert calibrator is not None


def test_final_holdout_is_excluded_from_calibration():
    df = _dataset(320)
    from app.models.multi_timeframe import _feature_frame, _holdout_rows
    X, _, y = _feature_frame(df, horizon_bars=1)

    holdout_rows = _holdout_rows(len(X), 1)
    development_rows = len(X) - holdout_rows

    _, meta = fit_time_series_platt_calibrator(
        X.iloc[:development_rows],
        y.iloc[:development_rows],
        _classifier,
        min_train=120,
        refit_every=20,
        min_calibration_rows=40,
    )

    assert meta["final_holdout_untouched"] is True
    assert meta["calibration_rows"] <= development_rows - 120
