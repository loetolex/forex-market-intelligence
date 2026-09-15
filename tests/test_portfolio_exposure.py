from __future__ import annotations

from dataclasses import dataclass

from app.risk.portfolio_exposure import evaluate_symbol_exposure


@dataclass
class Contract:
    localSymbol: str


@dataclass
class Position:
    contract: Contract
    position: float


def test_missing_broker_positions_are_not_treated_as_flat():
    result = evaluate_symbol_exposure("EUR/USD", "LONG", None)
    assert result["approved"] is False
    assert result["status"] == "DATA UNAVAILABLE"


def test_flat_symbol_allows_new_direction():
    result = evaluate_symbol_exposure("EUR/USD", "LONG", [Position(Contract("GBP.USD"), 10000)])
    assert result["approved"] is True
    assert result["reason"] == "NO_EXISTING_SYMBOL_EXPOSURE"


def test_same_direction_exposure_is_blocked():
    result = evaluate_symbol_exposure("EUR/USD", "LONG", [Position(Contract("EUR.USD"), 10000)])
    assert result["approved"] is False
    assert result["reason"] == "EXISTING_SAME_DIRECTION_EXPOSURE"


def test_opposing_symbol_exposure_is_blocked_by_default():
    result = evaluate_symbol_exposure("EUR/USD", "LONG", [Position(Contract("EUR.USD"), -10000)])
    assert result["approved"] is False
    assert result["reason"] == "OPPOSING_SYMBOL_EXPOSURE_BLOCKED"


def test_opposing_exposure_requires_explicit_policy():
    result = evaluate_symbol_exposure("EUR/USD", "LONG", [Position(Contract("EUR.USD"), -10000)], allow_opposing=True)
    assert result["approved"] is True
    assert result["reason"] == "OPPOSING_EXPOSURE_POLICY_ALLOWED"
