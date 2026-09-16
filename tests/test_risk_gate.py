from __future__ import annotations

from app.risk.gate import evaluate_signal


def test_missing_probability_is_rejected():
    decision = evaluate_signal(None)
    assert decision.approved is False
    assert decision.reason == "DATA UNAVAILABLE"


def test_invalid_probability_is_rejected():
    decision = evaluate_signal(1.1)
    assert decision.approved is False
    assert decision.reason == "INVALID_PROBABILITY"


def test_probability_below_edge_threshold_is_rejected():
    decision = evaluate_signal(0.57)
    assert decision.approved is False
    assert decision.reason == "INSUFFICIENT_EDGE"


def test_probability_above_edge_threshold_is_risk_approved():
    decision = evaluate_signal(0.70)
    assert decision.approved is True
    assert decision.reason == "TIMEFRAME_RISK_APPROVED"


def test_probability_below_fifty_can_be_risk_approved():
    decision = evaluate_signal(0.30)
    assert decision.approved is True
    assert decision.reason == "TIMEFRAME_RISK_APPROVED"
