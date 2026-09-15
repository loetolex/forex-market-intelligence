"""Deterministic, baseline-led ranking for lightweight portfolio scans.

The scanner model is deliberately not research-admitted.  This module ranks its
compact outputs only to decide where the existing deep research pipeline should
spend its limited provider budget.  It never emits an execution authorization
and RL agreement is reported, not scored.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config import settings


RANKING_VERSION = "PORTFOLIO_HIERARCHY_RANK_V1"
TIMEFRAMES = ("1day", "4h", "1h", "30m", "15m")
ROUTING_MIN_EDGE_PERCENTAGE_POINTS = 10.0


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _freshness_summary(stages: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    by_timeframe: dict[str, str] = {}
    ages: dict[str, float] = {}
    for timeframe in TIMEFRAMES:
        stage = stages.get(timeframe) or {}
        timestamp = stage.get("last_timestamp")
        if stage.get("data_status") != "REAL_DATA" or not timestamp:
            by_timeframe[timeframe] = "DATA_UNAVAILABLE"
            continue
        try:
            candle_time = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if candle_time.tzinfo is None:
                candle_time = candle_time.replace(tzinfo=timezone.utc)
            age_minutes = max(0.0, (now - candle_time.astimezone(timezone.utc)).total_seconds() / 60.0)
            ages[timeframe] = round(age_minutes, 1)
            limit = float(settings.freshness_limits_minutes.get(timeframe, settings.max_stale_minutes))
            by_timeframe[timeframe] = "FRESH" if age_minutes <= limit else "STALE"
        except (TypeError, ValueError):
            by_timeframe[timeframe] = "DATA_UNAVAILABLE"

    values = set(by_timeframe.values())
    status = "FRESH" if values == {"FRESH"} else "STALE" if "STALE" in values else "DATA_UNAVAILABLE"
    return {
        "status": status,
        "age_minutes": ages,
        "timeframes": by_timeframe,
    }


def _validation_summary(stages: dict[str, Any]) -> dict[str, Any]:
    statuses = {
        timeframe: str((stages.get(timeframe) or {}).get("validation_status", "DATA_UNAVAILABLE"))
        for timeframe in TIMEFRAMES
    }
    complete = all((stages.get(timeframe) or {}).get("data_status") == "REAL_DATA" for timeframe in TIMEFRAMES)
    return {
        "status": "SCANNER_ONLY_NOT_RESEARCH_VALIDATED" if complete else "DATA_UNAVAILABLE",
        "timeframes": statuses,
        "research_admitted": False,
    }


def _regime_summary(stages: dict[str, Any], bias: str) -> tuple[dict[str, Any], float]:
    regimes = {timeframe: str((stages.get(timeframe) or {}).get("regime", "UNKNOWN")) for timeframe in TIMEFRAMES}
    quality_points: list[float] = []
    matching_trend = "TREND_UP" if bias == "LONG" else "TREND_DOWN"
    for regime in regimes.values():
        if regime == matching_trend:
            quality_points.append(1.0)
        elif regime in {"LOW_VOLATILITY", "RANGE"}:
            quality_points.append(0.65)
        elif regime == "HIGH_VOLATILITY":
            quality_points.append(0.30)
        else:
            quality_points.append(0.40)
    quality = sum(quality_points) / len(quality_points) if quality_points else 0.0
    label = "FAVOURABLE" if quality >= 0.75 else "MIXED" if quality >= 0.50 else "CAUTION"
    return {"quality": label, "timeframes": regimes}, round(quality * 3.0, 2)


def _entry_readiness(
    *,
    valid_data: bool,
    freshness: str,
    bias: str,
    higher_alignment: bool,
    setup_alignment: bool,
    refinement_alignment: bool,
    entry_trigger: bool,
    hierarchy_edge: float,
) -> str:
    if not valid_data or freshness != "FRESH":
        return "DATA_UNAVAILABLE"
    if bias not in {"LONG", "SHORT"}:
        return "NO_OPPORTUNITY"
    if (
        higher_alignment
        and setup_alignment
        and refinement_alignment
        and entry_trigger
        and hierarchy_edge >= ROUTING_MIN_EDGE_PERCENTAGE_POINTS
    ):
        return "READY_FOR_DEEP_ANALYSIS"
    if higher_alignment and setup_alignment and hierarchy_edge >= ROUTING_MIN_EDGE_PERCENTAGE_POINTS:
        return "CANDIDATE"
    if higher_alignment and hierarchy_edge >= 5.0:
        return "WATCH"
    return "NO_OPPORTUNITY"


def rank_scanner_result(result: dict[str, Any]) -> dict[str, Any]:
    """Create one conservative, deterministic ranking record from scan output."""
    hierarchical = result.get("hierarchical_forecast") or {}
    stages = result.get("stages") or {}
    bias = str(hierarchical.get("primary_bias", "NEUTRAL"))
    valid_data = result.get("status") == "MODEL OUTPUT" and result.get("provenance") == "REAL_DATA"
    valid_data = valid_data and all(
        (stages.get(timeframe) or {}).get("data_status") == "REAL_DATA"
        and (stages.get(timeframe) or {}).get("forecast_status", "MODEL OUTPUT") == "MODEL OUTPUT"
        for timeframe in TIMEFRAMES
    )
    freshness = _freshness_summary(stages)
    validation = _validation_summary(stages)
    hierarchy_edge = _number(hierarchical.get("hierarchy_edge_percentage_points"))
    probability_up = hierarchical.get("probability_up")
    probability_down = hierarchical.get("probability_down")
    higher_alignment = bool(hierarchical.get("higher_timeframe_alignment"))
    setup_alignment = bool(hierarchical.get("setup_alignment"))
    refinement_alignment = bool(hierarchical.get("refinement_alignment"))
    entry_trigger = bool(hierarchical.get("entry_trigger"))
    alignment_count = sum(
        1 for timeframe in TIMEFRAMES
        if (stages.get(timeframe) or {}).get("direction") == bias and bias in {"LONG", "SHORT"}
    )
    alignment_percentage = round((alignment_count / len(TIMEFRAMES)) * 100.0, 1)
    regime, regime_points = _regime_summary(stages, bias)
    readiness = _entry_readiness(
        valid_data=valid_data,
        freshness=freshness["status"],
        bias=bias,
        higher_alignment=higher_alignment,
        setup_alignment=setup_alignment,
        refinement_alignment=refinement_alignment,
        entry_trigger=entry_trigger,
        hierarchy_edge=hierarchy_edge,
    )
    rl_action = (result.get("reinforcement_learning") or {}).get("policy_action")
    baseline_action = hierarchical.get("baseline_action", "NO_TRADE")
    rl_agreement = bool(rl_action and rl_action == baseline_action)

    score = 0.0
    if valid_data and freshness["status"] == "FRESH" and bias in {"LONG", "SHORT"}:
        daily_strength = abs(_number((stages.get("1day") or {}).get("probability_up")) - 0.5)
        four_hour_strength = abs(_number((stages.get("4h") or {}).get("probability_up")) - 0.5)
        one_hour_strength = abs(_number((stages.get("1h") or {}).get("probability_up")) - 0.5)
        combined_strength = abs(_number(probability_up, 0.5) - 0.5)
        score = (
            alignment_count * 5.0                         # 0..25: full top-down alignment
            + (8.0 if higher_alignment else 0.0)          # 4H confirms 1D
            + (8.0 if setup_alignment else 0.0)           # 1H confirms structure
            + (7.0 if refinement_alignment else 0.0)      # 30M refines structure
            + (7.0 if entry_trigger else 0.0)             # 15M trigger
            + min(10.0, daily_strength * 100.0)           # 1D bias strength
            + min(7.0, four_hour_strength * 70.0)         # 4H confirmation strength
            + min(5.0, one_hour_strength * 50.0)          # 1H setup strength
            + min(10.0, (hierarchy_edge / 30.0) * 10.0)   # hierarchy edge
            + min(5.0, combined_strength * 10.0)          # combined probability strength
            + regime_points                                # 0..3: regime quality
            + 3.0                                          # fresh all five timeframes
            + 2.0                                          # complete scanner structural validation
        )

    risk = result.get("risk") or {}
    routing_reasons = []
    if readiness == "READY_FOR_DEEP_ANALYSIS":
        routing_reasons = ["REAL_DATA", "FRESH_5TF", "FULL_TOP_DOWN_ALIGNMENT", "MEANINGFUL_HIERARCHY_EDGE"]
    elif readiness == "CANDIDATE":
        routing_reasons = ["HIGHER_TIMEFRAME_AND_1H_ALIGNED", "ENTRY_REFINEMENT_PENDING"]
    elif readiness == "DATA_UNAVAILABLE":
        routing_reasons = ["DATA_OR_FRESHNESS_REQUIREMENT_FAILED"]
    else:
        routing_reasons = ["STRUCTURAL_REQUIREMENTS_NOT_MET"]

    return {
        "instrument": result.get("instrument"),
        "opportunity_score": round(score, 2),
        "bias": bias,
        "alignment_count": alignment_count,
        "alignment_percentage": alignment_percentage,
        "hierarchy_score": _number(hierarchical.get("hierarchy_score")),
        "hierarchy_edge_percentage_points": round(hierarchy_edge, 2),
        "probability_up": probability_up,
        "probability_down": probability_down,
        "entry_readiness": readiness,
        "routing_status": readiness,
        "routing_reasons": routing_reasons,
        "primary_bias": bias,
        "higher_timeframe_alignment": higher_alignment,
        "setup_alignment": setup_alignment,
        "refinement_alignment": refinement_alignment,
        "entry_trigger": entry_trigger,
        "baseline_action": baseline_action,
        "rl_policy_action": rl_action,
        "rl_agreement": rl_agreement,
        "regime_summary": regime,
        "data_freshness_summary": freshness,
        "validation_summary": validation,
        "data_status": "REAL_DATA" if valid_data and freshness["status"] == "FRESH" else "DATA_UNAVAILABLE",
        "risk_status": {"approved": bool(risk.get("approved", False)), "reason": risk.get("reason", "DATA_UNAVAILABLE")},
        "decision": result.get("decision", "NO TRADE"),
        "execution_authorized": False,
    }


def rank_scanner_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank every scanner result, retaining unavailable pairs at the bottom."""
    ranked = [rank_scanner_result(result) for result in results]
    ranked.sort(key=lambda item: (-_number(item.get("opportunity_score")), str(item.get("instrument", ""))))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def ranking_summary(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    available = [item for item in ranked if item["entry_readiness"] != "DATA_UNAVAILABLE"]
    ready = [item for item in ranked if item["entry_readiness"] == "READY_FOR_DEEP_ANALYSIS"]
    candidates = [item for item in ranked if item["entry_readiness"] in {"READY_FOR_DEEP_ANALYSIS", "CANDIDATE"}]
    unavailable = [item for item in ranked if item["entry_readiness"] == "DATA_UNAVAILABLE"]
    return {
        "pairs_scanned": len(ranked),
        "pairs_available": len(available),
        "candidates": len(candidates),
        "ready_for_deep_analysis": len(ready),
        "no_trade": sum(1 for item in ranked if item.get("decision") == "NO TRADE"),
        "data_unavailable": len(unavailable),
        "baseline_rl_comparison": {
            "agreements": sum(1 for item in available if item.get("rl_agreement") is True),
            "disagreements": sum(1 for item in available if item.get("rl_agreement") is False),
            "ranking_policy": "BASELINE_HIERARCHY_ONLY",
        },
    }
