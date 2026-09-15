"""Deterministic baseline-led ranking for the lightweight portfolio scanner."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config import settings

RANKING_VERSION = "PORTFOLIO_TIMEFRAME_OPPORTUNITY_RANK_V2"
TIMEFRAMES = ("1day", "4h", "1h", "30m", "15m")
ROUTING_MIN_EDGE_PERCENTAGE_POINTS = 10.0


def _number(value: Any, default: float = 0.0) -> float:
    try: return float(value)
    except (TypeError, ValueError): return default


def _freshness_summary(stages: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc); by_timeframe: dict[str, str] = {}; ages: dict[str, float] = {}
    for timeframe in TIMEFRAMES:
        stage = stages.get(timeframe) or {}; timestamp = stage.get("last_timestamp")
        if stage.get("data_status") != "REAL_DATA" or not timestamp: by_timeframe[timeframe] = "DATA_UNAVAILABLE"; continue
        try:
            candle_time = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")); candle_time = candle_time.replace(tzinfo=timezone.utc) if candle_time.tzinfo is None else candle_time
            age = max(0.0, (now - candle_time.astimezone(timezone.utc)).total_seconds() / 60.0); ages[timeframe] = round(age, 1)
            limit = float(settings.freshness_limits_minutes.get(timeframe, settings.max_stale_minutes)); by_timeframe[timeframe] = "FRESH" if age <= limit else "STALE"
        except (TypeError, ValueError): by_timeframe[timeframe] = "DATA_UNAVAILABLE"
    values = set(by_timeframe.values()); status = "FRESH" if values == {"FRESH"} else "STALE" if "STALE" in values else "DATA_UNAVAILABLE"
    return {"status": status, "age_minutes": ages, "timeframes": by_timeframe}


def _legacy_rank(result: dict[str, Any]) -> dict[str, Any]:
    hierarchical = result.get("hierarchical_forecast") or {}; stages = result.get("stages") or {}; bias = str(hierarchical.get("primary_bias", "NEUTRAL"))
    valid_data = result.get("status") == "MODEL OUTPUT" and result.get("provenance") == "REAL_DATA" and all((stages.get(tf) or {}).get("data_status") == "REAL_DATA" and (stages.get(tf) or {}).get("forecast_status", "MODEL OUTPUT") == "MODEL OUTPUT" for tf in TIMEFRAMES)
    freshness = _freshness_summary(stages); edge = _number(hierarchical.get("hierarchy_edge_percentage_points")); alignment = sum(1 for tf in TIMEFRAMES if (stages.get(tf) or {}).get("direction") == bias and bias in {"LONG", "SHORT"})
    readiness = "DATA_UNAVAILABLE" if not valid_data or freshness["status"] != "FRESH" else "READY_FOR_DEEP_ANALYSIS" if bool(hierarchical.get("entry_trigger")) and edge >= ROUTING_MIN_EDGE_PERCENTAGE_POINTS else "CANDIDATE" if bool(hierarchical.get("setup_alignment")) and edge >= ROUTING_MIN_EDGE_PERCENTAGE_POINTS else "WATCH" if bool(hierarchical.get("higher_timeframe_alignment")) and edge >= 5.0 else "NO_OPPORTUNITY"
    strength = abs(_number(hierarchical.get("probability_up"), 0.5) - 0.5)
    score = (alignment * 5.0 + (8.0 if hierarchical.get("higher_timeframe_alignment") else 0.0) + (8.0 if hierarchical.get("setup_alignment") else 0.0) + (7.0 if hierarchical.get("refinement_alignment") else 0.0) + (7.0 if hierarchical.get("entry_trigger") else 0.0) + min(10.0, strength * 100.0)) if valid_data and freshness["status"] == "FRESH" and bias in {"LONG", "SHORT"} else 0.0
    rl_action = (result.get("reinforcement_learning") or {}).get("policy_action"); baseline = hierarchical.get("baseline_action", "NO_TRADE")
    return {"instrument": result.get("instrument"), "opportunity_score": round(score, 2), "bias": bias, "alignment_count": alignment, "alignment_percentage": round(alignment / len(TIMEFRAMES) * 100.0, 1), "hierarchy_score": _number(hierarchical.get("hierarchy_score")), "hierarchy_edge_percentage_points": round(edge, 2), "probability_up": hierarchical.get("probability_up"), "probability_down": hierarchical.get("probability_down"), "entry_readiness": readiness, "routing_status": readiness, "routing_reasons": ["LEGACY_HIERARCHY_COMPATIBILITY"], "primary_bias": bias, "higher_timeframe_alignment": bool(hierarchical.get("higher_timeframe_alignment")), "setup_alignment": bool(hierarchical.get("setup_alignment")), "refinement_alignment": bool(hierarchical.get("refinement_alignment")), "entry_trigger": bool(hierarchical.get("entry_trigger")), "baseline_action": baseline, "rl_policy_action": rl_action, "rl_agreement": bool(rl_action and rl_action == baseline), "data_freshness_summary": freshness, "data_status": "REAL_DATA" if valid_data and freshness["status"] == "FRESH" else "DATA_UNAVAILABLE", "risk_status": result.get("risk") or {"approved": False, "reason": "DATA_UNAVAILABLE"}, "decision": result.get("decision", "NO TRADE"), "selected_timeframe": None, "strategy_id": None, "signal_id": None, "execution_authorized": False}


def _opportunity_rank(result: dict[str, Any]) -> dict[str, Any]:
    opportunities = result.get("timeframe_opportunities") or {}; stages = result.get("stages") or {}; freshness = _freshness_summary(stages)
    valid_data = result.get("status") == "MODEL OUTPUT" and result.get("provenance") == "REAL_DATA" and freshness["status"] == "FRESH"
    eligible = []
    for timeframe, op in opportunities.items():
        if op.get("status") != "MODEL OUTPUT" or op.get("data_status") != "REAL_DATA" or not op.get("entry_signal"): continue
        probability = _number(op.get("probability_up"), 0.5); edge_pp = abs(probability - 0.5) * 100.0
        context = str((op.get("context") or {}).get("classification") or "MIXED")
        context_factor = {"TREND_ALIGNED": 1.0, "PRIMARY_BIAS": 0.95, "MIXED": 0.75, "COUNTER_TREND": 0.60}.get(context, 0.50)
        score = min(100.0, edge_pp * 2.0 + _number(op.get("confidence")) * 40.0) * context_factor
        eligible.append((score, timeframe, op))
    if not valid_data or not eligible:
        return {"instrument": result.get("instrument"), "opportunity_score": 0.0, "bias": "NEUTRAL", "alignment_count": 0, "alignment_percentage": 0.0, "hierarchy_score": _number((result.get("hierarchical_forecast") or {}).get("hierarchy_score")), "hierarchy_edge_percentage_points": _number((result.get("hierarchical_forecast") or {}).get("hierarchy_edge_percentage_points")), "probability_up": None, "probability_down": None, "entry_readiness": "DATA_UNAVAILABLE" if not valid_data else "NO_OPPORTUNITY", "routing_status": "DATA_UNAVAILABLE" if not valid_data else "NO_OPPORTUNITY", "routing_reasons": ["DATA_OR_ENTRY_REQUIREMENT_FAILED"], "primary_bias": (result.get("hierarchical_forecast") or {}).get("primary_bias", "NEUTRAL"), "higher_timeframe_alignment": False, "setup_alignment": False, "refinement_alignment": False, "entry_trigger": False, "baseline_action": (result.get("hierarchical_forecast") or {}).get("baseline_action", "NO_TRADE"), "rl_policy_action": (result.get("reinforcement_learning") or {}).get("policy_action"), "rl_agreement": False, "data_freshness_summary": freshness, "data_status": "REAL_DATA" if valid_data else "DATA_UNAVAILABLE", "risk_status": result.get("risk") or {"approved": False, "reason": "PAPER_ONLY_EXECUTION_LOCK"}, "decision": result.get("decision", "NO TRADE"), "selected_timeframe": None, "strategy_id": None, "signal_id": None, "execution_authorized": False}
    score, timeframe, op = max(eligible, key=lambda item: (item[0], item[1])); direction = str(op.get("prediction") or "NEUTRAL"); aligned = sum(1 for item in opportunities.values() if item.get("prediction") == direction)
    return {"instrument": result.get("instrument"), "opportunity_score": round(score, 2), "bias": direction, "alignment_count": aligned, "alignment_percentage": round(aligned / len(TIMEFRAMES) * 100.0, 1), "hierarchy_score": _number((result.get("hierarchical_forecast") or {}).get("hierarchy_score")), "hierarchy_edge_percentage_points": _number((result.get("hierarchical_forecast") or {}).get("hierarchy_edge_percentage_points")), "probability_up": op.get("probability_up"), "probability_down": op.get("probability_down"), "entry_readiness": "READY_FOR_DEEP_ANALYSIS", "routing_status": "READY_FOR_DEEP_ANALYSIS", "routing_reasons": ["INDEPENDENT_TIMEFRAME_SIGNAL", "REAL_DATA", "FRESH_TIMEFRAME_DATA"], "primary_bias": (result.get("hierarchical_forecast") or {}).get("primary_bias", "NEUTRAL"), "higher_timeframe_alignment": False, "setup_alignment": False, "refinement_alignment": False, "entry_trigger": bool(op.get("entry_signal")), "baseline_action": (result.get("hierarchical_forecast") or {}).get("baseline_action", "NO_TRADE"), "rl_policy_action": (result.get("reinforcement_learning") or {}).get("policy_action"), "rl_agreement": False, "data_freshness_summary": freshness, "data_status": "REAL_DATA", "risk_status": {"approved": False, "reason": "PAPER_ONLY_EXECUTION_LOCK"}, "decision": result.get("decision", "MULTI_TIMEFRAME_OPPORTUNITIES"), "selected_timeframe": timeframe, "strategy_id": op.get("strategy_id"), "signal_id": op.get("signal_id"), "context": op.get("context"), "execution_authorized": False}


def rank_scanner_result(result: dict[str, Any]) -> dict[str, Any]:
    return _opportunity_rank(result) if result.get("timeframe_opportunities") is not None else _legacy_rank(result)


def rank_scanner_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [rank_scanner_result(result) for result in results]; ranked.sort(key=lambda item: (-_number(item.get("opportunity_score")), str(item.get("instrument", ""))))
    for rank, item in enumerate(ranked, start=1): item["rank"] = rank
    return ranked


def ranking_summary(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    available = [item for item in ranked if item["entry_readiness"] != "DATA_UNAVAILABLE"]; ready = [item for item in ranked if item["entry_readiness"] == "READY_FOR_DEEP_ANALYSIS"]; candidates = [item for item in ranked if item["entry_readiness"] in {"READY_FOR_DEEP_ANALYSIS", "CANDIDATE"}]
    return {"pairs_scanned": len(ranked), "pairs_available": len(available), "candidates": len(candidates), "ready_for_deep_analysis": len(ready), "no_trade": sum(1 for item in ranked if item.get("decision") == "NO TRADE"), "data_unavailable": sum(1 for item in ranked if item["entry_readiness"] == "DATA_UNAVAILABLE"), "baseline_rl_comparison": {"agreements": sum(1 for item in available if item.get("rl_agreement") is True), "disagreements": sum(1 for item in available if item.get("rl_agreement") is False), "ranking_policy": "TIMEFRAME_OPPORTUNITY_BASELINE"}}
