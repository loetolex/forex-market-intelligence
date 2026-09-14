from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ACTIONS = ("LONG", "SHORT", "NO_TRADE")
RL_VERSION = "shadow-q-learning-mtf-v1"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def build_state(hierarchical: dict[str, Any]) -> str:
    """Discretize the five-timeframe hierarchy into a compact RL state."""
    stages = hierarchical.get("stages", {})
    tokens: list[str] = []
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
    """Small, instance-local Q learner for simulated/shadow decisions only.

    The learner never authorizes orders. It evaluates hypothetical LONG,
    SHORT, and NO_TRADE actions against realized future 15m returns once a
    subsequent market snapshot arrives.
    """

    def __init__(
        self,
        path: str = "/app/data/learning/rl_state.json",
        learning_rate: float = 0.10,
        discount_factor: float = 0.90,
        epsilon: float = 0.05,
        transaction_cost_bps: float = 1.5,
        reward_horizon_minutes: int = 60,
    ) -> None:
        self.path = Path(path)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.reward_horizon_minutes = int(reward_horizon_minutes)
        self.state = self._load()

    def _default(self) -> dict[str, Any]:
        return {
            "version": RL_VERSION,
            "q_values": {},
            "pending": [],
            "stats": {
                "snapshots_seen": 0,
                "transitions_learned": 0,
                "total_reward_bps": 0.0,
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

    def choose_action(self, state_key: str) -> tuple[str, str]:
        """Choose a simulated action without affecting the real signal engine."""
        q = self._q(state_key)
        seen = int(self.state["stats"].get("snapshots_seen", 0))

        if max(abs(float(v)) for v in q.values()) == 0.0:
            action = ACTIONS[seen % len(ACTIONS)]
            return action, "UNSEEN_STATE_EXPLORATION"

        import random

        if random.random() < self.epsilon:
            return random.choice(ACTIONS), "EPSILON_EXPLORATION"

        action = max(ACTIONS, key=lambda a: (float(q[a]), -ACTIONS.index(a)))
        return action, "Q_GREEDY"

    @staticmethod
    def _signed_reward_bps(action: str, forward_return: float, transaction_cost_bps: float) -> float:
        if action == "LONG":
            return (forward_return * 10_000.0) - transaction_cost_bps
        if action == "SHORT":
            return (-forward_return * 10_000.0) - transaction_cost_bps
        return 0.0

    def _learn(self, state_key: str, action: str, reward: float, next_state: str) -> None:
        q = self._q(state_key)
        next_q = self._q(next_state)
        old = float(q[action])
        target = float(reward) + self.discount_factor * max(float(v) for v in next_q.values())
        q[action] = old + self.learning_rate * (target - old)
        stats = self.state["stats"]
        stats["transitions_learned"] = int(stats.get("transitions_learned", 0)) + 1
        stats["total_reward_bps"] = float(stats.get("total_reward_bps", 0.0)) + float(reward)
        stats["last_learning_timestamp"] = _iso(_utc_now())

    def _settle_ready(self, current_price: float, current_timestamp: datetime, next_state: str) -> list[dict[str, Any]]:
        remaining: list[dict[str, Any]] = []
        settled: list[dict[str, Any]] = []
        for pending in self.state.get("pending", []):
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
                reward = self._signed_reward_bps(
                    str(pending["action"]),
                    forward_return,
                    self.transaction_cost_bps,
                )
                self._learn(str(pending["state"]), str(pending["action"]), reward, next_state)
                settled.append({
                    "action": pending["action"],
                    "state": pending["state"],
                    "forward_return": forward_return,
                    "reward_bps": reward,
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
        state_key = build_state(hierarchical)
        settled = self._settle_ready(float(reference_price), timestamp, state_key)

        pending = self.state.setdefault("pending", [])
        timestamp_iso = _iso(timestamp)
        duplicate = any(
            str(item.get("symbol")) == symbol and str(item.get("timestamp")) == timestamp_iso
            for item in pending
        )
        if duplicate:
            action = str(next(item["action"] for item in pending if str(item.get("symbol")) == symbol and str(item.get("timestamp")) == timestamp_iso))
            selection_mode = "EXISTING_SNAPSHOT"
        else:
            action, selection_mode = self.choose_action(state_key)
            pending.append({
                "symbol": symbol,
                "state": state_key,
                "action": action,
                "entry_price": float(reference_price),
                "timestamp": timestamp_iso,
            })
            if len(pending) > 500:
                del pending[:-500]
            self.state["stats"]["snapshots_seen"] = int(self.state["stats"].get("snapshots_seen", 0)) + 1

        self._save()

        q = self._q(state_key)
        stats = self.state["stats"]
        return {
            "status": "SHADOW_LEARNING",
            "version": RL_VERSION,
            "policy_action": action,
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
            "stats": dict(stats),
            "persistence": "INSTANCE_LOCAL",
        }

    def status(self) -> dict[str, Any]:
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
            "pending_experiences": len(self.state.get("pending", [])),
            "known_states": len(self.state.get("q_values", {})),
            "stats": dict(self.state.get("stats", {})),
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
