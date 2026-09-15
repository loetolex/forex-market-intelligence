from __future__ import annotations

import pytest

from app.execution.ibkr_adapter import IBKRAdapter


def test_normalize_fx_symbol():
    assert IBKRAdapter.normalize_symbol("EURUSD") == "EUR/USD"
    assert IBKRAdapter.normalize_symbol("gbp_usd") == "GBP/USD"


def test_reject_invalid_fx_symbol():
    with pytest.raises(ValueError):
        IBKRAdapter.normalize_symbol("EUR")


def test_broker_minimum_size_is_enforced():
    contract = {"constraints": {"min_size": 20000.0, "size_increment": 1000.0}}
    result = IBKRAdapter.validate_order_quantity(contract, 10000)
    assert result["approved"] is False
    assert result["reason"] == "BELOW_BROKER_MINIMUM_SIZE"


def test_broker_size_increment_is_enforced():
    contract = {"constraints": {"min_size": 1000.0, "size_increment": 1000.0}}
    result = IBKRAdapter.validate_order_quantity(contract, 2500)
    assert result["approved"] is False
    assert result["reason"] == "INVALID_BROKER_SIZE_INCREMENT"


def test_valid_broker_quantity():
    contract = {"constraints": {"min_size": 20000.0, "size_increment": 1000.0}}
    result = IBKRAdapter.validate_order_quantity(contract, 21000)
    assert result["approved"] is True


def test_order_submission_is_hard_disabled():
    adapter = IBKRAdapter()
    with pytest.raises(RuntimeError, match="PAPER ORDER BLOCKED"):
        adapter.place_order("EUR/USD", "BUY", 20000)
