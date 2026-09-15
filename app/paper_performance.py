from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


DB_PATH = Path(
    os.getenv(
        "PAPER_PERFORMANCE_DB",
        str(Path.home() / ".forex-intelligence" / "paper_performance.db"),
    )
).expanduser()
HORIZON_MINUTES = int(os.getenv("PAPER_EVALUATION_HORIZON_MINUTES", "60"))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts (
                forecast_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                snapshot_at_utc TEXT NOT NULL,
                prediction_timestamp_utc TEXT,
                horizon_minutes INTEGER NOT NULL,
                model_version TEXT,
                decision TEXT,
                baseline_action TEXT,
                primary_bias TEXT,
                hierarchy_score REAL,
                edge_pp REAL,
                probability_up REAL,
                probability_down REAL,
                entry_price REAL,
                data_route_15m TEXT,
                provenance TEXT,
                settled_at_utc TEXT,
                exit_price REAL,
                raw_return REAL,
                strategy_return REAL,
                actual_up INTEGER,
                direction_correct INTEGER,
                brier_score REAL,
                settlement_status TEXT NOT NULL DEFAULT 'PENDING',
                raw_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecasts_pending ON forecasts(settlement_status, snapshot_at_utc)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_forecasts_symbol ON forecasts(symbol, snapshot_at_utc)"
        )


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _latest_close(history: list[dict[str, Any]]) -> float | None:
    valid: list[tuple[datetime, float]] = []
    for bar in history:
        price = _safe_float(bar.get("close"))
        if price is None or price <= 0:
            continue
        try:
            ts = datetime.fromisoformat(str(bar.get("timestamp")).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            valid.append((ts.astimezone(timezone.utc), price))
        except Exception:
            valid.append((datetime.min.replace(tzinfo=timezone.utc), price))
    if not valid:
        return None
    valid.sort(key=lambda item: item[0])
    return valid[-1][1]


def record_forecast(result: dict[str, Any], quote: dict[str, Any] | None = None) -> str | None:
    symbol = str(result.get("instrument") or "").upper()
    hierarchy = result.get("hierarchical_forecast") or {}
    if not symbol or not hierarchy:
        return None

    probability_up = _safe_float(hierarchy.get("probability_up"))
    probability_down = _safe_float(hierarchy.get("probability_down"))
    if probability_up is None:
        return None

    snapshot = utc_now()
    prediction_timestamp = result.get("stages", {}).get("15m", {}).get("last_timestamp")
    entry_price = None
    if quote:
        entry_price = _safe_float(quote.get("market_price"))
        if entry_price is None:
            bid = _safe_float(quote.get("bid"))
            ask = _safe_float(quote.get("ask"))
            if bid is not None and ask is not None:
                entry_price = (bid + ask) / 2.0

    forecast_id = str(uuid4())
    with _connect() as conn:
        exists = conn.execute(
            "SELECT forecast_id FROM forecasts WHERE symbol=? AND prediction_timestamp_utc=? AND model_version=? LIMIT 1",
            (
                symbol,
                prediction_timestamp,
                str(result.get("scanner_model") or ""),
            ),
        ).fetchone()
        if exists:
            return str(exists[0])

        conn.execute(
            """
            INSERT INTO forecasts (
                forecast_id, symbol, snapshot_at_utc, prediction_timestamp_utc,
                horizon_minutes, model_version, decision, baseline_action,
                primary_bias, hierarchy_score, edge_pp, probability_up,
                probability_down, entry_price, data_route_15m, provenance,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                forecast_id,
                symbol,
                iso(snapshot),
                str(prediction_timestamp) if prediction_timestamp else None,
                HORIZON_MINUTES,
                str(result.get("scanner_model") or ""),
                str(result.get("decision") or "NO TRADE"),
                str(hierarchy.get("baseline_action") or "NO_TRADE"),
                str(hierarchy.get("primary_bias") or "NEUTRAL"),
                _safe_float(hierarchy.get("hierarchy_score")),
                _safe_float(hierarchy.get("hierarchy_edge_percentage_points")),
                probability_up,
                probability_down,
                entry_price,
                str(result.get("market_data_route_15m") or "DATA UNAVAILABLE"),
                str(result.get("provenance") or "UNKNOWN"),
                json.dumps(result, default=str, separators=(",", ":")),
            ),
        )
    return forecast_id


def settle_pending(get_history: Callable[[str, int], list[dict[str, Any]]]) -> int:
    settled = 0
    now = utc_now()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM forecasts WHERE settlement_status='PENDING' ORDER BY snapshot_at_utc ASC"
        ).fetchall()

        for row in rows:
            due_at = datetime.fromisoformat(row["snapshot_at_utc"]) + timedelta(
                minutes=int(row["horizon_minutes"])
            )
            if now < due_at:
                continue

            try:
                history = get_history(row["symbol"], 500)
            except Exception:
                continue

            target_time = due_at
            candidates: list[tuple[datetime, dict[str, Any]]] = []
            for bar in history:
                try:
                    ts = datetime.fromisoformat(str(bar["timestamp"]).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    ts = ts.astimezone(timezone.utc)
                    if ts >= target_time and _safe_float(bar.get("close")) is not None:
                        candidates.append((ts, bar))
                except Exception:
                    continue

            if not candidates:
                continue

            _, bar = sorted(candidates, key=lambda item: item[0])[0]
            exit_price = _safe_float(bar.get("close"))
            entry_price = _safe_float(row["entry_price"])
            if exit_price is None or entry_price is None or entry_price <= 0:
                conn.execute(
                    "UPDATE forecasts SET settlement_status='UNRESOLVED', settled_at_utc=? WHERE forecast_id=?",
                    (iso(now), row["forecast_id"]),
                )
                continue

            raw_return = exit_price / entry_price - 1.0
            action = str(row["decision"] or "NO TRADE").upper()
            strategy_return = (
                raw_return
                if action == "LONG"
                else -raw_return
                if action == "SHORT"
                else None
            )
            actual_up = 1 if raw_return > 0 else 0
            direction_correct = (
                1
                if action == "LONG" and raw_return > 0
                else 1
                if action == "SHORT" and raw_return < 0
                else 0
                if action in {"LONG", "SHORT"}
                else None
            )
            p = _safe_float(row["probability_up"])
            brier = (p - actual_up) ** 2 if p is not None else None

            conn.execute(
                """
                UPDATE forecasts
                SET settled_at_utc=?, exit_price=?, raw_return=?, strategy_return=?,
                    actual_up=?, direction_correct=?, brier_score=?, settlement_status='SETTLED'
                WHERE forecast_id=?
                """,
                (
                    iso(now),
                    exit_price,
                    raw_return,
                    strategy_return,
                    actual_up,
                    direction_correct,
                    brier,
                    row["forecast_id"],
                ),
            )
            settled += 1

    return settled


def summary() -> dict[str, Any]:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM forecasts WHERE settlement_status='PENDING'"
        ).fetchone()[0]
        settled = conn.execute(
            "SELECT COUNT(*) FROM forecasts WHERE settlement_status='SETTLED'"
        ).fetchone()[0]
        directional = conn.execute(
            "SELECT AVG(direction_correct) FROM forecasts WHERE settlement_status='SETTLED' AND direction_correct IS NOT NULL"
        ).fetchone()[0]
        avg_strategy = conn.execute(
            "SELECT AVG(strategy_return) FROM forecasts WHERE settlement_status='SETTLED' AND strategy_return IS NOT NULL"
        ).fetchone()[0]
        avg_brier = conn.execute(
            "SELECT AVG(brier_score) FROM forecasts WHERE settlement_status='SETTLED' AND brier_score IS NOT NULL"
        ).fetchone()[0]

        by_symbol = []
        for row in conn.execute(
            """
            SELECT symbol,
                   COUNT(*) AS total,
                   SUM(CASE WHEN settlement_status='SETTLED' THEN 1 ELSE 0 END) AS settled,
                   AVG(CASE WHEN settlement_status='SETTLED' THEN direction_correct END) AS directional_accuracy,
                   AVG(CASE WHEN settlement_status='SETTLED' THEN strategy_return END) AS avg_strategy_return,
                   AVG(CASE WHEN settlement_status='SETTLED' THEN brier_score END) AS avg_brier_score
            FROM forecasts
            GROUP BY symbol
            ORDER BY settled DESC, symbol ASC
            """
        ):
            by_symbol.append(dict(row))

        recent = [dict(row) for row in conn.execute(
            "SELECT forecast_id, symbol, snapshot_at_utc, decision, probability_up, entry_price, settlement_status, exit_price, strategy_return, direction_correct, brier_score FROM forecasts ORDER BY snapshot_at_utc DESC LIMIT 30"
        )]

    return {
        "status": "CALCULATED",
        "data_role": "PAPER_PERFORMANCE_MONITOR",
        "horizon_minutes": HORIZON_MINUTES,
        "total_forecasts": total,
        "pending_forecasts": pending,
        "settled_forecasts": settled,
        "directional_accuracy": directional,
        "average_strategy_return": avg_strategy,
        "average_brier_score": avg_brier,
        "by_symbol": by_symbol,
        "recent": recent,
        "live_trading_enabled": False,
        "order_placement_enabled": False,
        "execution_authorized": False,
        "db_path": str(DB_PATH),
    }


def worker_tick(
    fetch_results: Callable[[], list[dict[str, Any]]],
    get_quote: Callable[[str], dict[str, Any]],
    get_history: Callable[[str, int], list[dict[str, Any]]],
) -> dict[str, int]:
    init_db()
    recorded = 0
    results = fetch_results()
    for result in results:
        symbol = str(result.get("instrument") or "").strip()
        try:
            # Quotes are useful when IBKR provides a finite snapshot. They are not
            # required for paper evaluation: if quote data is unavailable, use the
            # latest completed 15m IBKR bar as the frozen entry price. This keeps the
            # evaluation on REAL_BROKER_DATA and avoids coupling the model ledger to
            # a potentially slow snapshot endpoint.
            try:
                quote = get_quote(symbol)
            except Exception:
                quote = None

            if quote and _safe_float(quote.get("market_price")) is not None:
                record = record_forecast(result, quote)
            else:
                history = get_history(symbol, 20)
                entry_price = _latest_close(history)
                if entry_price is None:
                    continue
                record = record_forecast(
                    result,
                    {
                        "market_price": entry_price,
                    },
                )
            if record:
                recorded += 1
        except Exception:
            continue
    settled = settle_pending(get_history)
    return {"recorded": recorded, "settled": settled}


def sleep_seconds() -> int:
    return max(60, int(os.getenv("PAPER_MONITOR_INTERVAL_SECONDS", "900")))
