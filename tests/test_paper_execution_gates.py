from __future__ import annotations

import pytest

from app import paper_trading


def _opportunity(side: str = "LONG") -> dict:
    return {
        "prediction": side,
        "probability_up": 0.70 if side == "LONG" else 0.30,
        "exit_framework": {"method": "ATR_14_REFERENCE", "status": "CALCULATED"},
        "reference_price": 1.1000,
        "stop_loss": 1.0990 if side == "LONG" else 1.1010,
        "take_profit": 1.1015 if side == "LONG" else 1.0985,
        "holding_horizon": "15m-2h",
    }


def test_long_exit_framework_is_valid():
    result = paper_trading._validate_exit_framework(_opportunity("LONG"), "15m")
    assert result["approved"] is True
    assert result["risk_reward"] == pytest.approx(1.5)


def test_short_exit_framework_is_valid():
    result = paper_trading._validate_exit_framework(_opportunity("SHORT"), "15m")
    assert result["approved"] is True
    assert result["risk_reward"] == pytest.approx(1.5)


def test_invalid_long_exit_geometry_is_rejected():
    opportunity = _opportunity("LONG")
    opportunity["take_profit"] = 1.0995
    result = paper_trading._validate_exit_framework(opportunity, "15m")
    assert result["approved"] is False
    assert "LONG exit geometry is invalid" in result["reason"]


def test_execution_gate_requires_risk_approval(monkeypatch):
    # Isolate the risk gate from the separate deployment lock. The execution
    # gate must reject an unapproved risk decision even when paper submission
    # itself is enabled.
    monkeypatch.setattr(paper_trading, "PAPER_ORDER_PLACEMENT_ENABLED", True)
    monkeypatch.setattr(paper_trading.settings, "trading_mode", "PAPER")
    monkeypatch.setattr(paper_trading.settings, "live_trading_enabled", False)
    exit_framework = paper_trading._validate_exit_framework(_opportunity("LONG"), "15m")
    result = paper_trading._execution_gate(
        {"approved": False, "reason": "INSUFFICIENT_EDGE"},
        {"approved": True},
        exit_framework,
    )
    assert result["approved"] is False
    assert result["reason"] == "RISK_GATE: INSUFFICIENT_EDGE"


def test_execution_gate_accepts_all_paper_gates(monkeypatch):
    monkeypatch.setattr(paper_trading, "PAPER_ORDER_PLACEMENT_ENABLED", True)
    monkeypatch.setattr(paper_trading.settings, "trading_mode", "PAPER")
    monkeypatch.setattr(paper_trading.settings, "live_trading_enabled", False)
    exit_framework = paper_trading._validate_exit_framework(_opportunity("LONG"), "15m")
    result = paper_trading._execution_gate(
        {"approved": True, "reason": "TIMEFRAME_RISK_APPROVED"},
        {"approved": True, "reason": "NO_EXISTING_SYMBOL_EXPOSURE"},
        exit_framework,
    )
    assert result["approved"] is True
    assert result["reason"] == "ALL_PAPER_EXECUTION_GATES_PASSED"
