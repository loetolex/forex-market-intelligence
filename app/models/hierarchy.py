from __future__ import annotations

from typing import Any

TIMEFRAME_ORDER = ["1day", "4h", "1h", "30m", "15m"]
TIMEFRAME_WEIGHTS = {"1day": 0.30, "4h": 0.25, "1h": 0.22, "30m": 0.15, "15m": 0.08}


def _direction(probability_up: float | None, neutral_band: float = 0.02) -> str:
    if probability_up is None: return "UNKNOWN"
    edge = float(probability_up) - 0.50
    if edge > neutral_band: return "LONG"
    if edge < -neutral_band: return "SHORT"
    return "NEUTRAL"


def _signed_edge(probability_up: float | None) -> float:
    if probability_up is None: return 0.0
    return max(-1.0, min(1.0, 2.0 * (float(probability_up) - 0.50)))


def _find(forecasts: list[dict[str, Any]], timeframe: str) -> dict[str, Any] | None:
    return next((x for x in forecasts if x.get("timeframe") == timeframe), None)


def build_hierarchical_decision(forecasts: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate MTF context while preserving independent timeframe signals.

    The hierarchy remains useful for context, bias and aggregate diagnostics.
    It no longer serves as a universal veto for an individual timeframe opportunity.
    """
    by_tf = {tf: _find(forecasts, tf) for tf in TIMEFRAME_ORDER}
    usable = {tf: item for tf, item in by_tf.items() if item and item.get("status") == "MODEL OUTPUT"}
    stages: dict[str, dict[str, Any]] = {}; signed_score = 0.0; available_weight = 0.0
    for tf in TIMEFRAME_ORDER:
        item = by_tf.get(tf); p = item.get("probability_up") if item else None; direction = _direction(p); edge = _signed_edge(p); weight = TIMEFRAME_WEIGHTS[tf]
        stages[tf] = {"direction": direction, "probability_up": p, "signed_edge": edge, "weight": weight, "regime": (item.get("regime") or {}).get("regime") if item else "UNKNOWN", "validation_status": item.get("validation_status") if item else "DATA UNAVAILABLE", "status": "AVAILABLE" if item and item.get("status") == "MODEL OUTPUT" else "UNAVAILABLE"}
        if tf in usable: signed_score += weight * edge; available_weight += weight
    if available_weight > 0: signed_score /= available_weight
    one_day, four_h, one_h, thirty, fifteen = (stages[tf]["direction"] for tf in TIMEFRAME_ORDER)
    primary_bias = one_day if one_day in {"LONG", "SHORT"} else "NEUTRAL"
    higher_timeframe_alignment = primary_bias in {"LONG", "SHORT"} and four_h == primary_bias
    setup_alignment = higher_timeframe_alignment and one_h == primary_bias
    refinement_alignment = setup_alignment and thirty in {primary_bias, "NEUTRAL"}
    entry_trigger = refinement_alignment and fifteen == primary_bias
    directions = [one_day, four_h, one_h, thirty, fifteen]
    aligned_count = sum(1 for d in directions if d == primary_bias); non_neutral_count = sum(1 for d in directions if d in {"LONG", "SHORT"})
    hierarchy_edge = abs(signed_score) * 50.0
    legacy_candidate = "NO TRADE"
    if entry_trigger and hierarchy_edge >= 10.0: legacy_candidate = "LONG CANDIDATE" if signed_score > 0 else "SHORT CANDIDATE"
    probability_up = max(0.0, min(1.0, 0.50 + signed_score / 2.0))
    baseline_action = {"LONG CANDIDATE": "LONG", "SHORT CANDIDATE": "SHORT", "NO TRADE": "NO_TRADE"}[legacy_candidate]
    return {"status": "MODEL OUTPUT" if usable else "DATA UNAVAILABLE", "method": "TOP_DOWN_HIERARCHICAL_MTF_V2_CONTEXT", "timeframe_order": TIMEFRAME_ORDER, "weights": TIMEFRAME_WEIGHTS, "stages": stages, "primary_bias": primary_bias, "higher_timeframe_alignment": higher_timeframe_alignment, "setup_alignment": setup_alignment, "refinement_alignment": refinement_alignment, "entry_trigger": entry_trigger, "aligned_count": aligned_count, "non_neutral_count": non_neutral_count, "hierarchy_score": signed_score, "hierarchy_edge_percentage_points": hierarchy_edge, "probability_up": probability_up, "probability_down": 1.0 - probability_up, "candidate_signal": legacy_candidate, "baseline_action": baseline_action, "role": "CONTEXT_AND_AGGREGATE", "conflict_policy": "CONTEXT_ONLY_NO_UNIVERSAL_VETO", "decision_rule": "1D/4H/1H/30M/15M remain aggregate context; each timeframe opportunity is evaluated independently; cross-timeframe conflicts do not veto the individual timeframe signal."}
