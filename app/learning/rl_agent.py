from __future__ import annotations

import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTIONS = ("LONG", "SHORT", "NO_TRADE")
RL_VERSION = "shadow-q-learning-mtf-v2"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def build_state(symbol: str, hierarchical: dict[str, Any]) -> str:
    stages = hierarchical.get("stages", {})
    tokens: list[str] = [f"symbol:{symbol}"]
    for tf in ("1day", "4h", "1h", "30m", "15m"):
        stage = stages.get(tf, {})
        direction = stage.get("direction", "UNKNOWN")
        p = stage.get("probability_up")
        if p is None:
            strength = "U"
        else:
            edge = abs(float(p) - 0.5)
            strength = "H" if edge >= 0.10 else "M" if edge >= 0.05 else "L"
        regime = str(stage.get("regime", "UNKNOWN"))[:24]
        tokens.append(f"{tf}:{direction}:{strength}:{regime}")
    return "|".join(tokens)


class ShadowQLearner:
    """Shadow Q learner for every configured pair.

    It learns only from hypothetical LONG/SHORT/NO_TRADE decisions and
    realized future 15m returns. It never authorizes or places broker orders.
    State keys are symbol-isolated so pairs never train one another.
    """

    def __init__(
        self,
        path: str = "/app/data/learning/rl_state.json",
        learning_rate: float = 0.10,
        discount_factor: float = 0.90,
        epsilon: float = 0.05,
        transaction_cost_bps: float = 1.5,
        reward_horizon_minutes: int = 60,
        no_trade_band_bps: float = 1.5,
    ) -> None:
        self.path = Path(path)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.reward_horizon_minutes = int(reward_horizon_minutes)
        self.no_trade_band_bps = float(no_trade_band_bps)
        self.state = self._load()

    def _default(self) -> dict[str, Any]:
        return {
            "version": RL_VERSION,
            "q_values": {},
            "pending": [],
            "state_metrics": {},
            "stats": {
                "snapshots_seen": 0,
                "transitions_learned": 0,
                "total_reward_bps": 0.0,
                "total_gross_reward_bps": 0.0,
                "total_transaction_cost_bps": 0.0,
                "positive_transitions": 0,
                "action_accuracy_count": 0,
                "action_accuracy_total": 0,
                "no_trade_total": 0,
                "no_trade_correct": 0,
                "baseline_total_reward_bps": 0.0,
                "baseline_positive_transitions": 0,
                "baseline_accuracy_count": 0,
                "baseline_accuracy_total": 0,
                "equity_bps": 0.0,
                "peak_equity_bps": 0.0,
                "max_drawdown_bps": 0.0,
                "last_learning_timestamp": None,
            },
        }

    def _load(self) -> dict[str, Any]:
        try:
            if self.path.exists():
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("version") == RL_VERSION:
                    return payload
        except Exception:
            pass
        return self._default()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="rl_state_", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2, sort_keys=True)
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def _q(self, state_key: str) -> dict[str, float]:
        q = self.state.setdefault("q_values", {}).setdefault(
            state_key,
            {action: 0.0 for action in ACTIONS},
        )
        for action in ACTIONS:
            q.setdefault(action, 0.0)
        return q

    def _metric(self, state_key: str) -> dict[str, Any]:
        return self.state.setdefault("state_metrics", {}).setdefault(
            state_key,
            {
                "transitions": 0,
                "total_reward_bps": 0.0,
                "total_gross_reward_bps": 0.0,
                "total_transaction_cost_bps": 0.0,
                "correct_actions": 0,
                "accuracy_total": 0,
                "baseline_reward_bps": 0.0,
                "baseline_correct_actions": 0,
                "baseline_accuracy_total": 0,
            },
        )

    def choose_action(self, state_key: str) -> tuple[str, str]:
        q = self._q(state_key)
        seen = int(self.state["stats"].get("snapshots_seen", 0))
        if max(abs(float(v)) for v in q.values()) == 0.0:
            return ACTIONS[seen % len(ACTIONS)], "UNSEEN_STATE_EXPLORATION"
        if random.random() < self.epsilon:
            return random.choice(ACTIONS), "EPSILON_EXPLORATION"
        action = max(ACTIONS, key=lambda a: (float(q[a]), -ACTIONS.index(a)))
        return action, "Q_GREEDY"

    @staticmethod
    def _gross_reward_bps(action: str, forward_return: float) -> float:
        if action == "LONG":
            return forward_return * 10_000.0
        if action == "SHORT":
            return -forward_return * 10_000.0
        return 0.0

    def _reward(self, action: str, forward_return: float) -> tuple[float, float, float]:
        gross = self._gross_reward_bps(action, forward_return)
        cost = 0.0 if action == "NO_TRADE" else self.transaction_cost_bps
        return gross - cost, gross, cost

    def _action_correct(self, action: str, forward_return: float) -> bool:
        move_bps = forward_return * 10_000.0
        if action == "LONG":
            return move_bps > self.no_trade_band_bps
        if action == "SHORT":
            return move_bps < -self.no_trade_band_bps
        return abs(move_bps) <= self.no_trade_band_bps

    def _update_global_stats(
        self,
        *,
        reward: float,
        gross_reward: float,
        cost: float,
        action_correct: bool,
        action: str,
        baseline_reward: float,
        baseline_correct: bool,
    ) -> None:
        stats = self.state["stats"]
        stats["transitions_learned"] += 1
        stats["total_reward_bps"] += reward
        stats["total_gross_reward_bps"] += gross_reward
        stats["total_transaction_cost_bps"] += cost
        stats["positive_transitions"] += int(reward > 0)
        stats["action_accuracy_count"] += int(action_correct)
        stats["action_accuracy_total"] += 1
        if action == "NO_TRADE":
            stats["no_trade_total"] += 1
            stats["no_trade_correct"] += int(action_correct)
        stats["baseline_total_reward_bps"] += baseline_reward
        stats["baseline_positive_transitions"] += int(baseline_reward > 0)
        stats["baseline_accuracy_count"] += int(baseline_correct)
        stats["baseline_accuracy_total"] += 1
        stats["equity_bps"] += reward
        stats["peak_equity_bps"] = max(stats["peak_equity_bps"], stats["equity_bps"])
        stats["max_drawdown_bps"] = max(
            stats["max_drawdown_bps"],
            stats["peak_equity_bps"] - stats["equity_bps"],
        )
        stats["last_learning_timestamp"] = _iso(_utc_now())

    def _learn(self, state_key: str, action: str, reward: float, next_state: str) -> None:
        q = self._q(state_key)
        next_q = self._q(next_state)
        old = float(q[action])
        target = float(reward) + self.discount_factor * max(float(v) for v in next_q.values())
        q[action] = old + self.learning_rate * (target - old)

    def _update_state_metrics(
        self,
        state_key: str,
        reward: float,
        gross_reward: float,
        cost: float,
        action_correct: bool,
        baseline_reward: float,
        baseline_correct: bool,
    ) -> None:
        metric = self._metric(state_key)
        metric["transitions"] += 1
        metric["total_reward_bps"] += reward
        metric["total_gross_reward_bps"] += gross_reward
        metric["total_transaction_cost_bps"] += cost
        metric["correct_actions"] += int(action_correct)
        metric["accuracy_total"] += 1
        metric["baseline_reward_bps"] += baseline_reward
        metric["baseline_correct_actions"] += int(baseline_correct)
        metric["baseline_accuracy_total"] += 1

    def _settle_ready(
        self,
        symbol: str,
        current_price: float,
        current_timestamp: datetime,
        next_state: str,
    ) -> list[dict[str, Any]]:
        remaining: list[dict[str, Any]] = []
        settled: list[dict[str, Any]] = []
        for pending in self.state.get("pending", []):
            if str(pending.get("symbol")) != symbol:
                remaining.append(pending)
                continue
            try:
                created = datetime.fromisoformat(str(pending["timestamp"]).replace("Z", "+00:00"))
                age_minutes = (current_timestamp - created).total_seconds() / 60.0
                if age_minutes < self.reward_horizon_minutes:
                    remaining.append(pending)
                    continue
                entry_price = float(pending["entry_price"])
                if entry_price <= 0:
                    remaining.append(pending)
                    continue

                forward_return = (float(current_price) / entry_price) - 1.0
                action = str(pending["action"])
                baseline_action = str(pending.get("baseline_action", "NO_TRADE"))
                reward, gross_reward, cost = self._reward(action, forward_return)
                baseline_reward, _, _ = self._reward(baseline_action, forward_return)
                action_correct = self._action_correct(action, forward_return)
                baseline_correct = self._action_correct(baseline_action, forward_return)

                source_state = str(pending["state"])
                self._learn(source_state, action, reward, next_state)
                self._update_global_stats(
                    reward=reward,
                    gross_reward=gross_reward,
                    cost=cost,
                    action_correct=action_correct,
                    action=action,
                    baseline_reward=baseline_reward,
                    baseline_correct=baseline_correct,
                )
                self._update_state_metrics(
                    source_state,
                    reward,
                    gross_reward,
                    cost,
                    action_correct,
                    baseline_reward,
                    baseline_correct,
                )
                settled.append({
                    "symbol": symbol,
                    "action": action,
                    "baseline_action": baseline_action,
                    "state": source_state,
                    "forward_return": forward_return,
                    "forward_return_bps": forward_return * 10_000.0,
                    "reward_bps": reward,
                    "gross_reward_bps": gross_reward,
                    "transaction_cost_bps": cost,
                    "action_correct": action_correct,
                    "baseline_reward_bps": baseline_reward,
                    "baseline_correct": baseline_correct,
                    "age_minutes": age_minutes,
                })
            except Exception:
                remaining.append(pending)
        self.state["pending"] = remaining
        return settled

    def process_snapshot(
        self,
        *,
        symbol: str,
        hierarchical: dict[str, Any],
        reference_price: float,
        reference_timestamp: Any,
    ) -> dict[str, Any]:
        timestamp = pd_timestamp(reference_timestamp)
        timestamp_iso = _iso(timestamp)
        state_key = build_state(symbol, hierarchical)
        baseline_action = str(hierarchical.get("baseline_action", "NO_TRADE"))
        settled = self._settle_ready(symbol, float(reference_price), timestamp, state_key)

        pending = self.state.setdefault("pending", [])
        existing = next(
            (
                item for item in pending
                if str(item.get("symbol")) == symbol
                and str(item.get("timestamp")) == timestamp_iso
            ),
            None,
        )
        if existing is not None:
            action = str(existing["action"])
            selection_mode = "EXISTING_SNAPSHOT"
        else:
            action, selection_mode = self.choose_action(state_key)
            pending.append({
                "symbol": symbol,
                "state": state_key,
                "action": action,
                "baseline_action": baseline_action,
                "entry_price": float(reference_price),
                "timestamp": timestamp_iso,
            })
            if len(pending) > 5000:
                del pending[:-5000]
            self.state["stats"]["snapshots_seen"] = int(self.state["stats"].get("snapshots_seen", 0)) + 1

        self._save()
        q = self._q(state_key)
        stats = self.state["stats"]
        action_accuracy = (
            float(stats["action_accuracy_count"]) / float(stats["action_accuracy_total"])
            if stats["action_accuracy_total"] else None
        )
        baseline_accuracy = (
            float(stats["baseline_accuracy_count"]) / float(stats["baseline_accuracy_total"])
            if stats["baseline_accuracy_total"] else None
        )
        no_trade_accuracy = (
            float(stats["no_trade_correct"]) / float(stats["no_trade_total"])
            if stats["no_trade_total"] else None
        )

        return {
            "status": "SHADOW_LEARNING",
            "version": RL_VERSION,
            "symbol": symbol,
            "policy_action": action,
            "baseline_action": baseline_action,
            "selection_mode": selection_mode,
            "advisory_only": True,
            "execution_authorized": False,
            "reward_horizon_minutes": self.reward_horizon_minutes,
            "reference_price": float(reference_price),
            "reference_timestamp": timestamp_iso,
            "state_key": state_key,
            "q_values": {k: float(v) for k, v in q.items()},
            "pending_experiences": len(self.state.get("pending", [])),
            "settled_transitions": settled,
            "metrics": {
                "rl_total_reward_bps": float(stats["total_reward_bps"]),
                "rl_gross_reward_bps": float(stats["total_gross_reward_bps"]),
                "transaction_cost_bps": float(stats["total_transaction_cost_bps"]),
                "action_accuracy": action_accuracy,
                "state_performance": self._metric(state_key),
                "max_drawdown_proxy_bps": float(stats["max_drawdown_bps"]),
                "no_trade_accuracy": no_trade_accuracy,
                "baseline_total_reward_bps": float(stats["baseline_total_reward_bps"]),
                "baseline_accuracy": baseline_accuracy,
                "rl_minus_baseline_reward_bps": float(stats["total_reward_bps"] - stats["baseline_total_reward_bps"]),
            },
            "stats": dict(stats),
            "persistence": "INSTANCE_LOCAL",
        }

    def status(self) -> dict[str, Any]:
        stats = self.state.get("stats", {})
        total = int(stats.get("action_accuracy_total", 0))
        baseline_total = int(stats.get("baseline_accuracy_total", 0))
        no_trade_total = int(stats.get("no_trade_total", 0))
        return {
            "status": "SHADOW_LEARNING",
            "version": RL_VERSION,
            "actions": list(ACTIONS),
            "advisory_only": True,
            "execution_authorized": False,
            "reward_horizon_minutes": self.reward_horizon_minutes,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "transaction_cost_bps": self.transaction_cost_bps,
            "no_trade_band_bps": self.no_trade_band_bps,
            "pending_experiences": len(self.state.get("pending", [])),
            "known_states": len(self.state.get("q_values", {})),
            "metrics": {
                "action_accuracy": (float(stats.get("action_accuracy_count", 0)) / total) if total else None,
                "baseline_accuracy": (float(stats.get("baseline_accuracy_count", 0)) / baseline_total) if baseline_total else None,
                "rl_total_reward_bps": float(stats.get("total_reward_bps", 0.0)),
                "baseline_total_reward_bps": float(stats.get("baseline_total_reward_bps", 0.0)),
                "rl_minus_baseline_reward_bps": float(stats.get("total_reward_bps", 0.0) - stats.get("baseline_total_reward_bps", 0.0)),
                "transaction_cost_bps": float(stats.get("total_transaction_cost_bps", 0.0)),
                "max_drawdown_proxy_bps": float(stats.get("max_drawdown_bps", 0.0)),
                "no_trade_accuracy": (float(stats.get("no_trade_correct", 0)) / no_trade_total) if no_trade_total else None,
            },
            "stats": dict(stats),
            "persistence": "INSTANCE_LOCAL",
            "state_path": str(self.path),
        }


def pd_timestamp(value: Any) -> datetime:
    try:
        import pandas as pd
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime().astimezone(timezone.utc)
    except Exception:
        return _utc_now()
