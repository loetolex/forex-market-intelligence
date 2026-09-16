from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pandas as pd
import pytest

from app.services import portfolio_scanner as scanner
from app.data import tiingo_fx
from app.services.deep_analysis_router import run_deep_analysis_for_candidates, select_deep_analysis_candidates
from app.services.portfolio_ranking import rank_scanner_results
from app.services.pipeline import aggregate_from_frame, keep_closed_candles, validate_freshness


@pytest.fixture(autouse=True)
def isolated_scanner_state(monkeypatch):
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(scanner, "_EXECUTOR", executor)
    with scanner._STATE_LOCK:
        scanner._STATE.update({
            "status": "IDLE", "pairs": [], "started_at_utc": None,
            "completed_at_utc": None, "pairs_total": len(scanner.DEFAULT_PORTFOLIO_PAIRS),
            "pairs_completed": 0, "results": [], "ranking": [], "ranking_summary": {},
            "top_candidates": [], "deep_analysis_results": [], "routing_status": "IDLE", "last_error": None,
        })
    yield
    executor.shutdown(wait=True)


def test_portfolio_request_reserves_worker_and_polls_without_duplicates(monkeypatch):
    release = Event()
    started = Event()
    calls: list[str] = []

    def fake_scan(symbol: str):
        calls.append(symbol)
        started.set()
        release.wait(timeout=2)
        return {"instrument": symbol, "status": "MODEL OUTPUT", "execution_authorized": False}

    monkeypatch.setattr(scanner, "_scan_pair", fake_scan)
    first = scanner.request_portfolio_scan(["EUR/USD"])
    assert started.wait(timeout=1)
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
    # Anchor the synthetic 15m bars to a 30m boundary. The first two
    # timestamps form one complete 30m bucket; the latest closed source bar
    # is intentionally in the following bucket and is not enough to aggregate.
    now = pd.Timestamp.now(tz="UTC").floor("30min")
    timestamps = [now - pd.Timedelta(minutes=60), now - pd.Timedelta(minutes=45), now - pd.Timedelta(minutes=30)]
    frame = pd.DataFrame({
        "timestamp": timestamps, "open": [1.0, 1.0, 1.0], "high": [1.1, 1.1, 1.1],
        "low": [0.9, 0.9, 0.9], "close": [1.0, 1.0, 1.0], "provider": "test",
        "instrument": "EUR/USD", "timeframe": "15m", "data_status": "REAL_DATA",
    })
    closed = keep_closed_candles(frame, "15m")
    assert len(closed) == 3
    aggregated = aggregate_from_frame(closed, "30min", "30m", expected_bars=2)
    assert len(aggregated) == 1
    validate_freshness(aggregated, 90, "30m")


def test_stale_data_is_rejected():
    stale = pd.DataFrame({"timestamp": [datetime.now(timezone.utc) - timedelta(hours=2)]})
    with pytest.raises(RuntimeError, match="STALE_DATA"):
        validate_freshness(stale, 30, "15m")


def _rankable_result(symbol: str, directions: dict[str, str], probabilities: dict[str, float], *, stale: bool = False, rl_action: str = "NO_TRADE"):
    timestamp = datetime.now(timezone.utc) - (timedelta(days=2) if stale else timedelta(minutes=1))
    hierarchy_score = sum((probabilities[tf] - 0.5) for tf in probabilities) / len(probabilities)
    edge = abs(hierarchy_score) * 100.0
    bias = directions["1day"]
    all_aligned = all(direction == bias for direction in directions.values())
    baseline = bias if all_aligned and edge >= 10 else "NO_TRADE"
    return {
        "instrument": symbol,
        "status": "MODEL OUTPUT",
        "provenance": "REAL_DATA",
        "stages": {
            tf: {
                "direction": directions[tf],
                "probability_up": probabilities[tf],
                "regime": "TREND_DOWN" if bias == "SHORT" else "TREND_UP",
                "data_status": "REAL_DATA",
                "last_timestamp": timestamp.isoformat(),
                "validation_status": "NOT_EVALUATED",
                "forecast_status": "MODEL OUTPUT",
            }
            for tf in ("1day", "4h", "1h", "30m", "15m")
        },
        "hierarchical_forecast": {
            "primary_bias": bias,
            "higher_timeframe_alignment": directions["4h"] == bias,
            "setup_alignment": directions["4h"] == bias and directions["1h"] == bias,
            "refinement_alignment": directions["4h"] == bias and directions["1h"] == bias and directions["30m"] == bias,
            "entry_trigger": all_aligned,
            "hierarchy_score": hierarchy_score,
            "hierarchy_edge_percentage_points": edge,
            "probability_up": 0.5 + hierarchy_score,
            "probability_down": 0.5 - hierarchy_score,
            "baseline_action": baseline,
        },
        "reinforcement_learning": {"policy_action": rl_action, "advisory_only": True, "execution_authorized": False},
        "risk": {"approved": False, "reason": "PAPER_ONLY_EXECUTION_LOCK"},
        "decision": "NO TRADE",
        "execution_authorized": False,
    }


def test_full_top_down_alignment_outranks_probability_and_rl_agreement():
    short = {tf: "SHORT" for tf in ("1day", "4h", "1h", "30m", "15m")}
    aligned = _rankable_result("EUR/USD", short, {tf: 0.30 for tf in short}, rl_action="LONG")
    partial = _rankable_result(
        "USD/JPY",
        {"1day": "LONG", "4h": "LONG", "1h": "LONG", "30m": "SHORT", "15m": "SHORT"},
        {"1day": 0.95, "4h": 0.95, "1h": 0.95, "30m": 0.05, "15m": 0.05},
        rl_action="NO_TRADE",
    )

    ranked = rank_scanner_results([partial, aligned])
    assert ranked[0]["instrument"] == "EUR/USD"
    assert ranked[0]["entry_readiness"] == "READY_FOR_DEEP_ANALYSIS"
    assert ranked[0]["rl_agreement"] is False
    assert ranked[1]["rl_agreement"] is True


def test_stale_pair_stays_ranked_but_is_not_routable():
    directions = {tf: "LONG" for tf in ("1day", "4h", "1h", "30m", "15m")}
    stale = _rankable_result("EUR/USD", directions, {tf: 0.70 for tf in directions}, stale=True)
    ranked = rank_scanner_results([stale])
    assert ranked[0]["entry_readiness"] == "DATA_UNAVAILABLE"
    assert ranked[0]["opportunity_score"] == 0.0
    assert select_deep_analysis_candidates(ranked) == []


def test_router_analyzes_only_selected_top_three():
    ranked = [
        {"rank": n, "instrument": f"PAIR/{n}", "bias": "LONG", "opportunity_score": 100 - n,
         "entry_readiness": "READY_FOR_DEEP_ANALYSIS", "routing_status": "READY_FOR_DEEP_ANALYSIS",
         "routing_reasons": ["FULL_TOP_DOWN_ALIGNMENT"], "baseline_action": "LONG",
         "rl_policy_action": "NO_TRADE", "rl_agreement": False}
        for n in range(1, 5)
    ]
    selected = select_deep_analysis_candidates(ranked)
    called: list[str] = []

    def fake_market_cycle(symbol: str):
        called.append(symbol)
        return {
            "provenance": "REAL_DATA", "hierarchical_forecast": {}, "reinforcement_learning": {},
            "risk": {"approved": False, "reason": "PAPER_ONLY_EXECUTION_LOCK"}, "decision": "NO TRADE",
            "execution": {"mode": "PAPER", "live_enabled": False, "order_placement_enabled": False, "status": "LOCKED"},
        }

    analyses = run_deep_analysis_for_candidates(selected, market_cycle=fake_market_cycle)
    assert [item["instrument"] for item in selected] == ["PAIR/1", "PAIR/2", "PAIR/3"]
    assert called == ["PAIR/1", "PAIR/2", "PAIR/3"]
    assert all(item["execution_authorized"] is False for item in analyses)


def test_tiingo_429_is_retried_with_provider_pacing(monkeypatch):
    class Response:
        def __init__(self, status_code: int, payload=None):
            self.status_code = status_code
            self.payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self.payload

    responses = iter([Response(429), Response(200, [{"date": "2026-01-01T00:00:00Z"}])])
    waits: list[bool] = []
    monkeypatch.setattr(tiingo_fx, "_wait_for_rate_slot", lambda: waits.append(True))
    monkeypatch.setattr(tiingo_fx.httpx, "get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(tiingo_fx.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(tiingo_fx.settings, "tiingo_max_429_retries", 1)

    payload = tiingo_fx._get("token", "EURUSD", {"resampleFreq": "1hour"})
    assert payload == [{"date": "2026-01-01T00:00:00Z"}]
    assert len(waits) == 2
