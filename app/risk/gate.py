from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str


def evaluate_signal(probability_up: float | None, max_probability_edge: float = 0.10) -> RiskDecision:
    """Validate signal probability risk independently from execution permissions."""
    if probability_up is None:
        return RiskDecision(False, "DATA UNAVAILABLE")
    if not 0.0 <= probability_up <= 1.0:
        return RiskDecision(False, "INVALID_PROBABILITY")
    edge = abs(probability_up - 0.50)
    if edge < max_probability_edge:
        return RiskDecision(False, "INSUFFICIENT_EDGE")
    return RiskDecision(True, "TIMEFRAME_RISK_APPROVED")
