from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from app.config import settings
from app.risk.gate import evaluate_signal
from app.risk.portfolio_exposure import evaluate_symbol_exposure

PAPER_ORDER_PLACEMENT_ENABLED = os.getenv("PAPER_ORDER_PLACEMENT_ENABLED", "false").lower() == "true"
PAPER_AUTO_TRADING_ENABLED = os.getenv("PAPER_AUTO_TRADING_ENABLED", "false").lower() == "true"
ORDER_CLIENT_ID = int(os.getenv("IBKR_ORDER_CLIENT_ID", "1902"))
TEST_QUANTITY = float(os.getenv("PAPER_TEST_QUANTITY", "20000"))
AUTO_QUANTITY = float(os.getenv("PAPER_AUTO_QUANTITY", "20000"))
AUTO_INTERVAL_SECONDS = max(60, int(os.getenv("PAPER_AUTO_INTERVAL_SECONDS", "900")))
PORTFOLIO_REFRESH_TIMEOUT_SECONDS = max(10, int(os.getenv("PAPER_PORTFOLIO_REFRESH_TIMEOUT_SECONDS", "90")))
PORTFOLIO_REFRESH_POLL_SECONDS = max(1, float(os.getenv("PAPER_PORTFOLIO_REFRESH_POLL_SECONDS", "2")))
FOREX_API_URL = os.getenv("FOREX_API_URL", "https://forex-api-production-f587.up.railway.app").rstrip("/")
DB_PATH = os.getenv("PAPER_TRADING_DB", str(os.path.expanduser("~/.forex-intelligence/paper_trading.db")))
SUPPORTED_TIMEFRAMES = {"15m", "30m", "1h", "4h", "1day"}

@dataclass
class OrderRecord:
    client_order_id: str
    broker_order_id: int | None
    symbol: str
    side: str
    quantity: float
    status: str
    filled: float
    avg_fill_price: float | None
    signal_id: str | None
    strategy_id: str
    opened_at_utc: str
    closed_at_utc: str | None = None
    close_order_id: int | None = None

_IB = None
_TRADE = None
_RECORD: OrderRecord | None = None
_LOCK = threading.RLock()
_AUTO_THREAD: threading.Thread | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_db() -> None:
    path = os.path.expanduser(DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS paper_trades (client_order_id TEXT PRIMARY KEY, broker_order_id INTEGER, symbol TEXT NOT NULL, side TEXT NOT NULL, quantity REAL NOT NULL, status TEXT NOT NULL, filled REAL NOT NULL, avg_fill_price REAL, signal_id TEXT, strategy_id TEXT NOT NULL, opened_at_utc TEXT NOT NULL, closed_at_utc TEXT, close_order_id INTEGER)""")


def _save_record(record: OrderRecord) -> None:
    with sqlite3.connect(os.path.expanduser(DB_PATH)) as conn:
        conn.execute("""INSERT OR REPLACE INTO paper_trades (client_order_id, broker_order_id, symbol, side, quantity, status, filled, avg_fill_price, signal_id, strategy_id, opened_at_utc, closed_at_utc, close_order_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", tuple(asdict(record).values()))


def _library():
    import ib_async as ibi
    return ibi


def _require_paper_order_mode() -> None:
    if str(settings.trading_mode).upper() != "PAPER": raise RuntimeError("BROKER SAFETY: paper trader requires trading_mode=PAPER.")
    if bool(settings.live_trading_enabled): raise RuntimeError("BROKER SAFETY: live_trading_enabled must remain false.")
    if not PAPER_ORDER_PLACEMENT_ENABLED: raise RuntimeError("PAPER ORDER LOCKED: set PAPER_ORDER_PLACEMENT_ENABLED=true on the Mac bridge.")


def _ib_connected():
    global _IB
    with _LOCK:
        if _IB is not None and _IB.isConnected(): return _IB
        ibi = _library(); ib = ibi.IB(); ib.connect(settings.ibkr_host, settings.ibkr_port, clientId=ORDER_CLIENT_ID, timeout=8, readonly=False); _IB = ib; return ib


def _qualify(symbol: str):
    ibi = _library(); normalized = str(symbol).upper().replace("_", "/").replace("-", "/")
    if "/" not in normalized and len(normalized) == 6: normalized = normalized[:3] + "/" + normalized[3:]
    contract = ibi.Forex(normalized.replace("/", "")); qualified = _ib_connected().qualifyContracts(contract)
    if not qualified: raise RuntimeError(f"BROKER DATA UNAVAILABLE: could not qualify {normalized}.")
    return normalized, qualified[0]


def _broker_diagnostics(trade: Any, captured_errors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for entry in list(getattr(trade, "log", []) or [])[-20:]:
        timestamp = getattr(entry, "time", None)
        entries.append({
            "time": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp) if timestamp else None,
            "status": getattr(entry, "status", None),
            "message": getattr(entry, "message", None),
            "error_code": getattr(entry, "errorCode", None),
        })
    advanced_error = str(getattr(trade, "advancedError", "") or "").strip() or None
    return {
        "trade_log": entries,
        "advanced_error": advanced_error,
        "ib_error_events": list(captured_errors or []),
    }


def _order_snapshot(trade: Any, record: OrderRecord, broker_diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    status = getattr(trade.orderStatus, "status", "UNKNOWN"); filled = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0); avg = getattr(trade.orderStatus, "avgFillPrice", None); avg_float = float(avg) if avg not in (None, 0, 0.0) else None
    record.status, record.filled, record.avg_fill_price = status, filled, avg_float
    if status in {"Filled", "PartiallyFilled"}: _save_record(record)
    return {"status": "REAL_BROKER_DATA", "broker": "INTERACTIVE_BROKERS", "order": {"client_order_id": record.client_order_id, "broker_order_id": record.broker_order_id, "symbol": record.symbol, "side": record.side, "quantity": record.quantity, "order_status": status, "filled": filled, "remaining": float(getattr(trade.orderStatus, "remaining", 0.0) or 0.0), "avg_fill_price": avg_float, "signal_id": record.signal_id, "strategy_id": record.strategy_id}, "broker_diagnostics": broker_diagnostics or _broker_diagnostics(trade), "execution_authorized": True, "paper_only": True}


def _portfolio_results() -> dict[str, Any]:
    response = httpx.get(f"{FOREX_API_URL}/portfolio/results", timeout=25.0); response.raise_for_status(); return response.json()


def _find_result(payload: dict[str, Any], symbol: str, timeframe: str) -> dict[str, Any] | None:
    for result in payload.get("results") or []:
        if str(result.get("instrument") or "").upper() != symbol.upper(): continue
        opportunity = (result.get("timeframe_opportunities") or {}).get(timeframe)
        if opportunity is not None: return {"result": result, "opportunity": opportunity}
    return None


def _fetch_timeframe_signal(symbol: str = "EUR/USD", timeframe: str = "15m") -> tuple[dict[str, Any], dict[str, Any]]:
    symbol = str(symbol).upper().replace("_", "/"); timeframe = str(timeframe).lower()
    if timeframe not in SUPPORTED_TIMEFRAMES: raise RuntimeError(f"PAPER TEST BLOCKED: unsupported timeframe {timeframe}.")
    try:
        found = _find_result(_portfolio_results(), symbol, timeframe)
    except Exception as exc:
        raise RuntimeError(f"DATA UNAVAILABLE: portfolio results could not be read: {exc}") from exc
    if found is None:
        raise RuntimeError(f"DATA UNAVAILABLE: completed portfolio snapshot does not contain {symbol} {timeframe}; refresh the portfolio scan explicitly before retrying.")
    return found["result"], found["opportunity"]


def _validate_exit_framework(opportunity: dict[str, Any], timeframe: str) -> dict[str, Any]:
    framework = opportunity.get("exit_framework") or {}
    if framework.get("status") != "CALCULATED":
        return {"approved": False, "reason": f"{timeframe} exit framework is {framework.get('status') or 'DATA UNAVAILABLE'}.", "status": "DATA UNAVAILABLE"}
    try:
        entry = float(opportunity.get("reference_price")); stop = float(opportunity.get("stop_loss")); target = float(opportunity.get("take_profit"))
    except (TypeError, ValueError):
        return {"approved": False, "reason": f"{timeframe} entry/stop/target levels are DATA UNAVAILABLE.", "status": "DATA UNAVAILABLE"}
    if not all(math.isfinite(value) and value > 0 for value in (entry, stop, target)):
        return {"approved": False, "reason": f"{timeframe} entry/stop/target levels are INVALID.", "status": "CALCULATED"}
    side = str(opportunity.get("prediction") or "").upper()
    if side == "LONG" and not (stop < entry < target):
        return {"approved": False, "reason": f"{timeframe} LONG exit geometry is invalid.", "status": "CALCULATED"}
    if side == "SHORT" and not (target < entry < stop):
        return {"approved": False, "reason": f"{timeframe} SHORT exit geometry is invalid.", "status": "CALCULATED"}
    risk_distance = abs(entry - stop); reward_distance = abs(target - entry)
    if risk_distance <= 0 or reward_distance <= 0:
        return {"approved": False, "reason": f"{timeframe} stop/target distances are INVALID.", "status": "CALCULATED"}
    return {"approved": True, "reason": "EXIT_FRAMEWORK_VALID", "status": "CALCULATED", "reference_price": entry, "stop_loss": stop, "take_profit": target, "risk_reward": reward_distance / risk_distance, "holding_horizon": opportunity.get("holding_horizon"), "method": framework.get("method")}


def _risk_check(opportunity: dict[str, Any]) -> dict[str, Any]:
    decision = evaluate_signal(opportunity.get("probability_up"))
    return {"approved": decision.approved, "reason": decision.reason, "status": "CALCULATED"}


def _exposure_check(symbol: str, side: str) -> dict[str, Any]:
    try:
        positions = _ib_connected().positions()
    except Exception as exc:
        return {"approved": False, "reason": f"DATA UNAVAILABLE: broker positions could not be verified: {exc}", "status": "DATA UNAVAILABLE"}
    return evaluate_symbol_exposure(symbol, side, positions, allow_opposing=False)


def _execution_gate(risk: dict[str, Any], exposure: dict[str, Any], exit_framework: dict[str, Any]) -> dict[str, Any]:
    if str(settings.trading_mode).upper() != "PAPER": return {"approved": False, "reason": "BROKER SAFETY: trading_mode must be PAPER."}
    if bool(settings.live_trading_enabled): return {"approved": False, "reason": "BROKER SAFETY: live_trading_enabled must remain false."}
    if not PAPER_ORDER_PLACEMENT_ENABLED: return {"approved": False, "reason": "PAPER_ORDER_PLACEMENT_ENABLED=false"}
    if not risk.get("approved"): return {"approved": False, "reason": f"RISK_GATE: {risk.get('reason')}"}
    if not exit_framework.get("approved"): return {"approved": False, "reason": f"EXIT_GATE: {exit_framework.get('reason')}"}
    if not exposure.get("approved"): return {"approved": False, "reason": f"EXPOSURE_GATE: {exposure.get('reason')}"}
    return {"approved": True, "reason": "ALL_PAPER_EXECUTION_GATES_PASSED"}


def _current_open() -> bool:
    with _LOCK: return _RECORD is not None and _RECORD.closed_at_utc is None and _RECORD.status not in {"Cancelled", "Inactive", "ApiCancelled", "Closed"}


def _validate_signal(result: dict[str, Any], opportunity: dict[str, Any], timeframe: str) -> tuple[str, str, dict[str, Any]]:
    symbol = str(result.get("instrument") or "").upper(); direction = str(opportunity.get("prediction") or "UNKNOWN").upper()
    if opportunity.get("status") != "MODEL OUTPUT" or opportunity.get("data_status") != "REAL_DATA": raise RuntimeError("PAPER TEST BLOCKED: selected timeframe model output is unavailable.")
    if direction not in {"LONG", "SHORT"}: raise RuntimeError("PAPER TEST BLOCKED: selected timeframe is not directional.")
    if not bool(opportunity.get("entry_signal")): raise RuntimeError(f"PAPER TEST BLOCKED: {timeframe} entry signal is not active: {opportunity.get('entry_reason') or 'NO TRADE'}.")
    exit_framework = _validate_exit_framework(opportunity, timeframe)
    if not exit_framework.get("approved"): raise RuntimeError(f"PAPER TEST BLOCKED: {exit_framework.get('reason')}")
    return direction, symbol, exit_framework


def controlled_test_preview(symbol: str = "EUR/USD", timeframe: str = "15m") -> dict[str, Any]:
    if not PAPER_ORDER_PLACEMENT_ENABLED: return {"status": "LOCKED", "reason": "PAPER_ORDER_PLACEMENT_ENABLED=false", "execution_authorized": False}
    result, opportunity = _fetch_timeframe_signal(symbol, timeframe); side, normalized, exit_framework = _validate_signal(result, opportunity, timeframe); _, contract = _qualify(normalized); risk = _risk_check(opportunity); exposure = _exposure_check(normalized, side); execution = _execution_gate(risk, exposure, exit_framework)
    return {"status": "PAPER_PREVIEW" if execution.get("approved") else "BLOCKED", "preview_type": "TIMEFRAME_SIGNAL_RISK_EXIT_CONTRACT_EXPOSURE_GATE_CHECK", "symbol": normalized, "timeframe": timeframe, "side": side, "strategy_id": opportunity.get("strategy_id"), "signal_id": opportunity.get("signal_id"), "quantity": TEST_QUANTITY, "broker_contract": {"con_id": int(getattr(contract, "conId", 0) or 0), "local_symbol": getattr(contract, "localSymbol", None), "exchange": getattr(contract, "exchange", None), "currency": getattr(contract, "currency", None)}, "portfolio_exposure": exposure, "risk": risk, "exit_framework": exit_framework, "execution_gate": execution, "execution_authorized": bool(execution.get("approved")), "paper_only": True, "signal": {"prediction": opportunity.get("prediction"), "probability_up": opportunity.get("probability_up"), "probability_down": opportunity.get("probability_down"), "entry_signal": opportunity.get("entry_signal"), "entry_status": opportunity.get("entry_status"), "entry_reason": opportunity.get("entry_reason"), "context": opportunity.get("context"), "reference_price": opportunity.get("reference_price"), "stop_loss": opportunity.get("stop_loss"), "take_profit": opportunity.get("take_profit")}}


def place_controlled_test(symbol: str = "EUR/USD", timeframe: str = "15m") -> dict[str, Any]:
    global _TRADE, _RECORD
    _require_paper_order_mode(); _init_db()
    with _LOCK:
        if _current_open(): raise RuntimeError("PAPER TEST BLOCKED: a controlled test order is already active.")
    result, opportunity = _fetch_timeframe_signal(symbol, timeframe); side, normalized, exit_framework = _validate_signal(result, opportunity, timeframe); _, contract = _qualify(normalized); risk = _risk_check(opportunity); exposure = _exposure_check(normalized, side); execution = _execution_gate(risk, exposure, exit_framework)
    if not execution.get("approved"): raise RuntimeError(f"PAPER TEST BLOCKED: {execution.get('reason')}")
    ibi = _library(); client_order_id = f"paper-test-{uuid4().hex}"; broker_action = "BUY" if side == "LONG" else "SELL"; order = ibi.MarketOrder(broker_action, TEST_QUANTITY); order.orderRef = client_order_id
    ib = _ib_connected(); captured_errors: list[dict[str, Any]] = []
    def _capture_error(req_id: Any, error_code: Any, error_string: Any, contract_obj: Any) -> None:
        captured_errors.append({"req_id": req_id, "error_code": error_code, "message": str(error_string), "contract": getattr(contract_obj, "localSymbol", None) if contract_obj is not None else None})
    try:
        ib.errorEvent += _capture_error
        trade = ib.placeOrder(contract, order)
        record = OrderRecord(client_order_id, int(getattr(order, "orderId", 0) or 0), normalized, side, TEST_QUANTITY, str(getattr(trade.orderStatus, "status", "Submitted")), 0.0, None, str(opportunity.get("signal_id") or "") or None, str(opportunity.get("strategy_id") or "forex-mtf-paper-test-v1"), _now())
        _TRADE, _RECORD = trade, record; _save_record(record); ib.sleep(2)
    finally:
        try: ib.errorEvent -= _capture_error
        except Exception: pass
    snapshot = _order_snapshot(trade, record, _broker_diagnostics(trade, captured_errors)); snapshot["risk"] = risk; snapshot["exit_framework"] = exit_framework; snapshot["execution_gate"] = execution; return snapshot


def order_status() -> dict[str, Any]:
    with _LOCK:
        if _TRADE is None or _RECORD is None: return {"status": "NO_ACTIVE_TEST", "execution_authorized": False, "paper_only": True}
        return _order_snapshot(_TRADE, _RECORD)


def close_controlled_test() -> dict[str, Any]:
    global _TRADE, _RECORD
    _require_paper_order_mode()
    with _LOCK:
        if _TRADE is None or _RECORD is None: raise RuntimeError("PAPER CLOSE: no controlled test order is active.")
        trade, record = _TRADE, _RECORD; status = str(getattr(trade.orderStatus, "status", "UNKNOWN")); filled = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0); ib = _ib_connected()
        if status not in {"Filled", "PartiallyFilled"} or filled <= 0:
            try: ib.cancelOrder(trade.order)
            finally: record.status, record.closed_at_utc = "Cancelled", _now(); _save_record(record)
            return {"status": "CANCELLED_UNFILLED", "order": asdict(record), "paper_only": True, "execution_authorized": False}
        ibi = _library(); _, contract = _qualify(record.symbol); close_side = "SELL" if record.side == "LONG" else "BUY"; close_order = ibi.MarketOrder(close_side, filled); close_order.orderRef = f"CLOSE-{record.client_order_id}"; close_trade = ib.placeOrder(contract, close_order); ib.sleep(2)
        record.close_order_id = int(getattr(close_order, "orderId", 0) or 0); close_status = str(getattr(close_trade.orderStatus, "status", "Submitted")); record.closed_at_utc = _now() if close_status in {"Filled", "PartiallyFilled"} else None; record.status = "Closed" if record.closed_at_utc else close_status; _save_record(record)
        return {"status": "PAPER_POSITION_CLOSED" if record.closed_at_utc else "PAPER_CLOSE_SUBMITTED", "open_order": asdict(record), "close_order": {"broker_order_id": record.close_order_id, "status": close_status, "filled": float(getattr(close_trade.orderStatus, "filled", 0.0) or 0.0), "avg_fill_price": float(getattr(close_trade.orderStatus, "avgFillPrice", 0.0) or 0.0)}, "paper_only": True, "execution_authorized": False}


def _select_auto_signal() -> tuple[dict[str, Any], dict[str, Any]] | None:
    payload = _portfolio_results(); candidates = []
    for result in payload.get("results") or []:
        for timeframe, opportunity in (result.get("timeframe_opportunities") or {}).items():
            if opportunity.get("status") == "MODEL OUTPUT" and opportunity.get("data_status") == "REAL_DATA" and opportunity.get("entry_signal"):
                risk = _risk_check(opportunity); exit_framework = _validate_exit_framework(opportunity, timeframe)
                if risk.get("approved") and exit_framework.get("approved"): candidates.append((result, opportunity))
    if not candidates: return None
    candidates.sort(key=lambda item: float(item[1].get("confidence") or 0.0), reverse=True); return candidates[0]


def _auto_tick() -> None:
    if not PAPER_AUTO_TRADING_ENABLED or not PAPER_ORDER_PLACEMENT_ENABLED or _current_open(): return
    selected = _select_auto_signal()
    if selected is None: return
    result, opportunity = selected; side = str(opportunity.get("prediction") or "").upper(); timeframe = str(opportunity.get("timeframe") or "15m")
    if side not in {"LONG", "SHORT"}: return
    exit_framework = _validate_exit_framework(opportunity, timeframe); risk = _risk_check(opportunity)
    if not exit_framework.get("approved") or not risk.get("approved"): return
    normalized, contract = _qualify(str(result.get("instrument") or "")); exposure = _exposure_check(normalized, side); execution = _execution_gate(risk, exposure, exit_framework)
    if not execution.get("approved"): return
    ibi = _library(); broker_action = "BUY" if side == "LONG" else "SELL"; order = ibi.MarketOrder(broker_action, AUTO_QUANTITY); client_order_id = f"paper-auto-{uuid4().hex}"; order.orderRef = client_order_id; trade = _ib_connected().placeOrder(contract, order)
    global _TRADE, _RECORD
    with _LOCK:
        _TRADE, _RECORD = trade, OrderRecord(client_order_id, int(getattr(order, "orderId", 0) or 0), normalized, side, AUTO_QUANTITY, str(getattr(trade.orderStatus, "status", "Submitted")), 0.0, None, str(opportunity.get("signal_id") or "") or None, str(opportunity.get("strategy_id") or "forex-mtf-paper-auto-v1"), _now()); _save_record(_RECORD)


def _auto_loop() -> None:
    _init_db()
    while True:
        try: _auto_tick()
        except Exception: pass
        time.sleep(AUTO_INTERVAL_SECONDS)


def start_auto_trader() -> None:
    global _AUTO_THREAD
    if _AUTO_THREAD is not None and _AUTO_THREAD.is_alive(): return
    if not PAPER_AUTO_TRADING_ENABLED: return
    _AUTO_THREAD = threading.Thread(target=_auto_loop, name="paper-auto-trader", daemon=True); _AUTO_THREAD.start()


def configuration_status() -> dict[str, Any]:
    return {"paper_order_placement_enabled": PAPER_ORDER_PLACEMENT_ENABLED, "paper_auto_trading_enabled": PAPER_AUTO_TRADING_ENABLED, "order_client_id": ORDER_CLIENT_ID, "test_quantity": TEST_QUANTITY, "auto_quantity": AUTO_QUANTITY, "auto_interval_seconds": AUTO_INTERVAL_SECONDS, "portfolio_refresh_timeout_seconds": PORTFOLIO_REFRESH_TIMEOUT_SECONDS, "portfolio_refresh_poll_seconds": PORTFOLIO_REFRESH_POLL_SECONDS, "live_trading_enabled": bool(settings.live_trading_enabled), "trading_mode": settings.trading_mode, "paper_only": True, "supported_timeframes": sorted(SUPPORTED_TIMEFRAMES)}
