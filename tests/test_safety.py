from app.risk.gate import evaluate_signal
from app.execution.ibkr_readonly import read_only_status


def test_risk_never_approves_orders():
    decision = evaluate_signal(0.90)
    assert decision.approved is False
    assert "PAPER_ONLY" in decision.reason


def test_ibkr_starts_locked():
    status = read_only_status()
    assert status.connected is False
    assert status.orders_enabled is False
