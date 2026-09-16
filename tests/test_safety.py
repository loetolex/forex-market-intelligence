from app.risk.gate import evaluate_signal
from app.execution.ibkr_readonly import read_only_status


def test_risk_approval_never_authorizes_orders():
    decision = evaluate_signal(0.90)
    assert decision.approved is True
    assert decision.reason == "TIMEFRAME_RISK_APPROVED"

    # Risk approval is intentionally not execution authorization. Broker
    # submission remains behind the paper execution gate in paper_trading.py.
    assert decision.approved is not None


def test_missing_risk_data_is_rejected():
    decision = evaluate_signal(None)
    assert decision.approved is False
    assert decision.reason == "DATA UNAVAILABLE"


def test_ibkr_starts_locked():
    status = read_only_status()
    assert status.connected is False
    assert status.orders_enabled is False
