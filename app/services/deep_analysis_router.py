"""Route only structurally ready ranked pairs into the existing deep pipeline."""
from __future__ import annotations

from typing import Any, Callable

from app.services.pipeline import run_market_cycle


ROUTER_VERSION = "TOP_N_DEEP_ANALYSIS_ROUTER_V1"
DEFAULT_TOP_N = 3


def select_deep_analysis_candidates(ranked: list[dict[str, Any]], top_n: int = DEFAULT_TOP_N) -> list[dict[str, Any]]:
    """Select at most ``top_n`` full-alignment candidates in ranking order."""
    selected: list[dict[str, Any]] = []
    for item in ranked:
        if item.get("routing_status") != "READY_FOR_DEEP_ANALYSIS":
            continue
        selected.append({
            "rank": item.get("rank"),
            "instrument": item.get("instrument"),
            "bias": item.get("bias"),
            "opportunity_score": item.get("opportunity_score"),
            "entry_readiness": item.get("entry_readiness"),
            "routing_status": "READY_FOR_DEEP_ANALYSIS",
            "routing_reasons": list(item.get("routing_reasons") or []),
            "baseline_action": item.get("baseline_action"),
            "rl_policy_action": item.get("rl_policy_action"),
            "rl_agreement": item.get("rl_agreement"),
            "execution_authorized": False,
        })
        if len(selected) >= top_n:
            break
    return selected


def _compact_deep_result(candidate: dict[str, Any], deep: dict[str, Any]) -> dict[str, Any]:
    hierarchy = deep.get("hierarchical_forecast") or {}
    rl = deep.get("reinforcement_learning") or {}
    execution = deep.get("execution") or {}
    return {
        **candidate,
        "routing_status": "DEEP_ANALYSIS_COMPLETE",
        "deep_analysis": {
            "status": "DEEP_ANALYSIS_COMPLETE",
            "provenance": deep.get("provenance", "DATA_UNAVAILABLE"),
            "hierarchical_forecast": {
                "primary_bias": hierarchy.get("primary_bias"),
                "higher_timeframe_alignment": hierarchy.get("higher_timeframe_alignment"),
                "setup_alignment": hierarchy.get("setup_alignment"),
                "refinement_alignment": hierarchy.get("refinement_alignment"),
                "entry_trigger": hierarchy.get("entry_trigger"),
                "hierarchy_edge_percentage_points": hierarchy.get("hierarchy_edge_percentage_points"),
                "baseline_action": hierarchy.get("baseline_action"),
            },
            "reinforcement_learning": {
                "status": rl.get("status"),
                "policy_action": rl.get("policy_action"),
                "baseline_action": rl.get("baseline_action"),
                "advisory_only": True,
                "execution_authorized": False,
            },
            "risk": deep.get("risk"),
            "decision": deep.get("decision", "NO TRADE"),
            "execution": {
                "mode": execution.get("mode"),
                "live_enabled": bool(execution.get("live_enabled", False)),
                "order_placement_enabled": bool(execution.get("order_placement_enabled", False)),
                "status": execution.get("status", "LOCKED"),
            },
        },
        "execution_authorized": False,
    }


def run_deep_analysis_for_candidates(
    candidates: list[dict[str, Any]],
    market_cycle: Callable[[str], dict[str, Any]] = run_market_cycle,
) -> list[dict[str, Any]]:
    """Run direct, sequential deep analysis for the selected subset only.

    This deliberately does not issue HTTP requests and never expands the
    selection beyond the router's candidate list.
    """
    analyses: list[dict[str, Any]] = []
    for candidate in candidates:
        instrument = str(candidate["instrument"])
        try:
            analyses.append(_compact_deep_result(candidate, market_cycle(instrument)))
        except Exception as exc:
            analyses.append({
                **candidate,
                "routing_status": "DATA_UNAVAILABLE",
                "deep_analysis": {"status": "DATA_UNAVAILABLE", "error": str(exc)},
                "execution_authorized": False,
            })
    return analyses
