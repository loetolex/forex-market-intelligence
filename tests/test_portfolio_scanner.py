from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pandas as pd
import pytest

from app.services import portfolio_scanner as scanner
from app.services.pipeline import aggregate_from_frame, keep_closed_candles, validate_freshness


@pytest.fixture(autouse=True)
def isolated_scanner_state(monkeypatch):
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(scanner, "_EXECUTOR", executor)
    with scanner._STATE_LOCK:
        scanner._STATE.update({
            "status": "IDLE", "pairs": [], "started_at_utc": None,
            "completed_at_utc": None, "pairs_total": len(scanner.DEFAULT_PORTFOLIO_PAIRS),
            "pairs_completed": 0, "results": [], "last_error": None,
        })
    yield
    executor.shutdown(wait=True)


def test_portfolio_request_reserves_worker_and_polls_without_duplicates(monkeypatch):
    release = Event()
    calls: list[str] = []

    def fake_scan(symbol: str):
        calls.append(symbol)
        release.wait(timeout=2)
        return {"instrument": symbol, "status": "MODEL OUTPUT", "execution_authorized": False}

    monkeypatch.setattr(scanner, "_scan_pair", fake_scan)
    first = scanner.request_portfolio_scan(["EUR/USD"])
    second = scanner.request_portfolio_scan(["EUR/USD"])
    status = scanner.get_portfolio_status()

    assert first["status"] == "RUNNING"
    assert second["status"] == "RUNNING"
    assert status["pairs_completed"] == 0
    release.set()
    scanner._EXECUTOR.shutdown(wait=True)
    assert calls == ["EUR/USD"]
    assert scanner.get_portfolio_results()["results"][0]["instrument"] == "EUR/USD"


def test_failed_pair_isolated_and_results_are_compact(monkeypatch):
    def fake_scan(symbol: str):
        if symbol == "GBP/USD":
            raise RuntimeError("provider unavailable")
        return {"instrument": symbol, "status": "MODEL OUTPUT", "execution_authorized": False}

    monkeypatch.setattr(scanner, "_scan_pair", fake_scan)
    scanner.request_portfolio_scan(["EUR/USD", "GBP/USD"])
    scanner._EXECUTOR.shutdown(wait=True)
    results = scanner.get_portfolio_results()

    assert results["status"] == "COMPLETE"
    assert results["pairs_completed"] == 2
    assert results["results"][1]["status"] == "DATA UNAVAILABLE"
    assert results["execution_authorized"] is False


def test_all_twelve_pairs_complete_and_refresh_restarts_once(monkeypatch):
    calls: list[str] = []

    def fake_scan(symbol: str):
        calls.append(symbol)
        return {"instrument": symbol, "status": "MODEL OUTPUT", "execution_authorized": False}

    monkeypatch.setattr(scanner, "_scan_pair", fake_scan)
    scanner.request_portfolio_scan(scanner.DEFAULT_PORTFOLIO_PAIRS)
    scanner._EXECUTOR.shutdown(wait=True)
    assert len(scanner.get_portfolio_results()["results"]) == 12

    # A completed scan is cached until an explicit refresh request.
    scanner.request_portfolio_scan(scanner.DEFAULT_PORTFOLIO_PAIRS)
    assert len(calls) == 12

    replacement = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(scanner, "_EXECUTOR", replacement)
    scanner.request_portfolio_scan(scanner.DEFAULT_PORTFOLIO_PAIRS, refresh=True)
    replacement.shutdown(wait=True)
    assert len(calls) == 24


def test_closed_candle_freshness_and_complete_aggregation():
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    timestamps = [now - pd.Timedelta(minutes=30), now - pd.Timedelta(minutes=15), now]
    frame = pd.DataFrame({
        "timestamp": timestamps, "open": [1.0, 1.0, 1.0], "high": [1.1, 1.1, 1.1],
        "low": [0.9, 0.9, 0.9], "close": [1.0, 1.0, 1.0], "provider": "test",
        "instrument": "EUR/USD", "timeframe": "15m", "data_status": "REAL_DATA",
    })
    closed = keep_closed_candles(frame, "15m")
    assert len(closed) == 2
    aggregated = aggregate_from_frame(closed, "30min", "30m", expected_bars=2)
    assert len(aggregated) == 1
    validate_freshness(aggregated, 90, "30m")


def test_stale_data_is_rejected():
    stale = pd.DataFrame({"timestamp": [datetime.now(timezone.utc) - timedelta(hours=2)]})
    with pytest.raises(RuntimeError, match="STALE_DATA"):
        validate_freshness(stale, 30, "15m")
