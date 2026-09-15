from __future__ import annotations

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


PAPER_ORDER_PLACEMENT_ENABLED = os.getenv("PAPER_ORDER_PLACEMENT_ENABLED", "false").lower() == "true"
PAPER_AUTO_TRADING_ENABLED = os.getenv("PAPER_AUTO_TRADING_ENABLED", "false").lower() == "true"
ORDER_CLIENT_ID = int(os.getenv("IBKR_ORDER_CLIENT_ID", "1902"))
TEST_QUANTITY = float(os.getenv("PAPER_TEST_QUANTITY", "20000"))
AUTO_QUANTITY = float(os.getenv("PAPER_AUTO_QUANTITY", "20000"))
AUTO_INTERVAL_SECONDS = max(60, int(os.getenv("PAPER_AUTO_INTERVAL_SECONDS", "900")))
PORTFOLIO_REFRESH_TIMEOUT_SECONDS = max(10, int(os.getenv("PAPER_PORTFOLIO_REFRESH_TIMEOUT_SECONDS", "90")))
PORTFOLIO_REFRESH_POLL_SECONDS = max(1, float(os.getenv("PAPER_PORTFOLIO_REFRESH_POLL_SECONDS", "2")))
FOREX_API_URL = os.getenv(
    "FOREX_API_URL", "https://forex-api-production-f587.up.railway.app"
).rstrip("/")

DB_PATH = os.getenv(
    "PAPER_TRADING_DB",
    str(os.path.expanduser("~/.forex-intelligence/paper_trading.db")),
)


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                client_order_id TEXT PRIMARY KEY,
                broker_order_id INTEGER,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                status TEXT NOT NULL,
                filled REAL NOT NULL,
                avg_fill_price REAL,
                signal_id TEXT,
                strategy_id TEXT NOT NULL,
                opened_at_utc TEXT NOT NULL,
                closed_at_utc TEXT,
                close_order_id INTEGER
            )
            """
        )


def _save_record(record: OrderRecord) -> None:
    with sqlite3.connect(os.path.expanduser(DB_PATH)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_trades (
                client_order_id, broker_order_id, symbol, side, quantity, status,
                filled, avg_fill_price, signal_id, strategy_id, opened_at_utc,
                closed_at_utc, close_order_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.client_order_id,
                record.broker_order_id,
                record.symbol,
                record.side,
                record.quantity,
                record.status,
                record.filled,
                record.avg_fill_price,
                record.signal_id,
                record.strategy_id,
                record.opened_at_utc,
                record.closed_at_utc,
                record.close_order_id,
            ),
        )


def _library():
    import ib_async as ibi
    return ibi


def _require_paper_order_mode() -> None:
    if str(settings.trading_mode).upper() != "PAPER":
        raise RuntimeError("BROKER SAFETY: paper trader requires trading_mode=PAPER.")
    if bool(settings.live_trading_enabled):
        raise RuntimeError("BROKER SAFETY: live_trading_enabled must remain false.")
    if not PAPER_ORDER_PLACEMENT_ENABLED:
        raise RuntimeError("PAPER ORDER LOCKED: set PAPER_ORDER_PLACEMENT_ENABLED=true on the Mac bridge.")


def _ib_connected():
    global _IB
    with _LOCK:
        if _IB is not None and _IB.isConnected():
            return _IB
        ibi = _library()
        ib = ibi.IB()
        ib.connect(
            settings.ibkr_host,
            settings.ibkr_port,
            clientId=ORDER_CLIENT_ID,
            timeout=8,
            readonly=False,
        )
        _IB = ib
        return ib


def _disconnect_order_session() -> None:
    global _IB
    with _LOCK:
        if _IB is not None:
            try:
                _IB.disconnect()
            finally:
                _IB = None


def _qualify(symbol: str):
    ibi = _library()
    normalized = str(symbol).upper().replace("_", "/").replace("-", "/")
    if "/" not in normalized and len(normalized) == 6:
        normalized = normalized[:3] + "/" + normalized[3:]
    contract = ibi.Forex(normalized.replace("/", ""))
    qualified = _ib_connected().qualifyContracts(contract)
    if not qualified:
        raise RuntimeError(f"BROKER DATA UNAVAILABLE: could not qualify {normalized}.")
    return normalized, qualified[0]


def _order_snapshot(trade: Any, record: OrderRecord) -> dict[str, Any]:
    status = getattr(trade.orderStatus, "status", "UNKNOWN")
    filled = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
    avg = getattr(trade.orderStatus, "avgFillPrice", None)
    avg_float = float(avg) if avg not in (None, 0, 0.0) else None
    record.status = status
    record.filled = filled
    record.avg_fill_price = avg_float
    if status in {"Filled", "PartiallyFilled"}:
        _save_record(record)
    return {
        "status": "REAL_BROKER_DATA",
        "broker": "INTERACTIVE_BROKERS",
        "order": {
            "client_order_id": record.client_order_id,
            "broker_order_id": record.broker_order_id,
            "symbol": record.symbol,
            "side": record.side,
            "quantity": record.quantity,
            "order_status": status,
            "filled": filled,
            "remaining": float(getattr(trade.orderStatus, "remaining", 0.0) or 0.0),
            "avg_fill_price": avg_float,
            "signal_id": record.signal_id,
            "strategy_id": record.strategy_id,
        },
        "execution_authorized": False,
        "paper_only": True,
    }


def _validate_signal(signal: dict[str, Any]) -> tuple[str, str]:
    hierarchy = signal.get("hierarchical_forecast") or {}
    symbol = str(signal.get("instrument") or "").upper()
    decision = str(signal.get("decision") or "NO TRADE").upper()
    candidate = str(hierarchy.get("candidate_signal") or decision).upper()
    edge = float(hierarchy.get("hierarchy_edge_percentage_points") or 0.0)
    entry_trigger = bool(hierarchy.get("entry_trigger"))
    baseline = str(hierarchy.get("baseline_action") or "NO_TRADE").upper().replace("_", "")
    if symbol != "EUR/USD":
        raise RuntimeError("PAPER TEST BLOCKED: controlled test is restricted to EUR/USD.")
    if candidate not in {"LONG", "SHORT"} or decision not in {"LONG", "SHORT"}:
        raise RuntimeError("PAPER TEST BLOCKED: current EUR/USD signal is NO TRADE.")
    if candidate != decision:
        raise RuntimeError("PAPER TEST BLOCKED: candidate signal and decision disagree.")
    if not entry_trigger:
        raise RuntimeError("PAPER TEST BLOCKED: 15m entry trigger is not aligned.")
    if edge < 10.0:
        raise RuntimeError("PAPER TEST BLOCKED: hierarchy edge is below 10 percentage points.")
    if baseline not in {"LONG", "SHORT"}:
        raise RuntimeError("PAPER TEST BLOCKED: baseline action is not directional.")
    if baseline != candidate:
        raise RuntimeError("PAPER TEST BLOCKED: baseline and candidate signal disagree.")
    return candidate, symbol


def _portfolio_results() -> dict[str, Any]:
    response = httpx.get(f"{FOREX_API_URL}/portfolio/results", timeout=25.0)
    response.raise_for_status()
    return response.json()


def _find_eurusd_result(payload: dict[str, Any]) -> dict[str, Any] | None:
    results = payload.get("results") or []
    for result in results:
        if str(result.get("instrument") or "").upper() == "EUR/USD":
            return result
    return None


def _fetch_eurusd_signal() -> dict[str, Any]:
    """Return the current EUR/USD portfolio result, refreshing the portfolio if needed.

    The portfolio scanner owns signal generation. This function never constructs or
    substitutes a signal. If EUR/USD is absent from the current in-memory snapshot,
    it explicitly requests the scanner's existing refresh endpoint, then waits for
    that same scanner instance to finish before reading the result.
    """
    try:
        result = _find_eurusd_result(_portfolio_results())
        if result is not None:
            return result
    except Exception as exc:
        initial_error = str(exc)
    else:
        initial_error = "EUR/USD is not present in the current portfolio snapshot."

    try:
        refresh_response = httpx.get(
            f"{FOREX_API_URL}/portfolio?refresh=true",
            timeout=25.0,
        )
        refresh_response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"DATA UNAVAILABLE: unable to start portfolio refresh: {exc}") from exc

    deadline = time.monotonic() + PORTFOLIO_REFRESH_TIMEOUT_SECONDS
    last_status = "UNKNOWN"
    while time.monotonic() < deadline:
        try:
            status_response = httpx.get(f"{FOREX_API_URL}/portfolio/status", timeout=10.0)
            status_response.raise_for_status()
            status_payload = status_response.json()
            last_status = str(status_payload.get("status") or "UNKNOWN").upper()
        except Exception as exc:
            raise RuntimeError(f"DATA UNAVAILABLE: portfolio status could not be read: {exc}") from exc

        if last_status == "COMPLETE":
            try:
                result = _find_eurusd_result(_portfolio_results())
            except Exception as exc:
                raise RuntimeError(f"DATA UNAVAILABLE: completed portfolio results could not be read: {exc}") from exc
            if result is not None:
                return result
            raise RuntimeError("DATA UNAVAILABLE: portfolio scan completed without a EUR/USD result.")

        if last_status in {"FAILED", "ERROR"}:
            error_text = status_payload.get("last_error") or "portfolio scanner reported failure"
            raise RuntimeError(f"DATA UNAVAILABLE: portfolio scan failed: {error_text}")

        time.sleep(PORTFOLIO_REFRESH_POLL_SECONDS)

    raise RuntimeError(
        "DATA UNAVAILABLE: portfolio refresh timed out "
        f"after {PORTFOLIO_REFRESH_TIMEOUT_SECONDS}s (last_status={last_status}, initial={initial_error})."
    )


def _current_open() -> bool:
    with _LOCK:
        if _RECORD is not None and _RECORD.status not in {"Filled", "Cancelled", "Inactive", "ApiCancelled", "Closed"}:
            return True
        if _RECORD is not None and _RECORD.status == "Filled" and _RECORD.closed_at_utc is None:
            return True
    return False


def controlled_test_preview(signal: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate the controlled EUR/USD paper order without a blocking what-if request.

    IBKR ``whatIfOrder`` can hang in some paper/demo configurations while waiting
    for an account/margin response. The controlled preview therefore performs
    only deterministic signal gates plus broker contract qualification. Actual
    broker submission remains a separate explicit step.
    """
    if not PAPER_ORDER_PLACEMENT_ENABLED:
        return {"status": "LOCKED", "reason": "PAPER_ORDER_PLACEMENT_ENABLED=false", "execution_authorized": False}
    signal = signal or _fetch_eurusd_signal()
    side, symbol = _validate_signal(signal)
    normalized, contract = _qualify(symbol)
    return {
        "status": "PAPER_PREVIEW",
        "preview_type": "CONTRACT_AND_GATE_CHECK",
        "symbol": normalized,
        "side": side,
        "quantity": TEST_QUANTITY,
        "broker_contract": {
            "con_id": int(getattr(contract, "conId", 0) or 0),
            "local_symbol": getattr(contract, "localSymbol", None),
            "exchange": getattr(contract, "exchange", None),
            "currency": getattr(contract, "currency", None),
        },
        "margin_change": None,
        "commission": None,
        "warning_text": "WHAT_IF_DISABLED_FOR_CONTROLLED_TEST_TO_PREVENT_PAPER_DEMO_HANGS",
        "execution_authorized": False,
        "paper_only": True,
        "signal": {
            "decision": signal.get("decision"),
            "hierarchy_edge_percentage_points": (signal.get("hierarchical_forecast") or {}).get("hierarchy_edge_percentage_points"),
            "entry_trigger": (signal.get("hierarchical_forecast") or {}).get("entry_trigger"),
        },
    }


def place_controlled_test(signal: dict[str, Any] | None = None) -> dict[str, Any]:
    global _TRADE, _RECORD
    _require_paper_order_mode()
    _init_db()
    with _LOCK:
        if _RECORD is not None and _RECORD.closed_at_utc is None and _RECORD.status not in {"Cancelled", "Inactive", "ApiCancelled"}:
            raise RuntimeError("PAPER TEST BLOCKED: a controlled test order is already active.")
    signal = signal or _fetch_eurusd_signal()
    side, symbol = _validate_signal(signal)
    normalized, contract = _qualify(symbol)
    ibi = _library()
    client_order_id = f"paper-test-{uuid4().hex}"
    order = ibi.MarketOrder(side, TEST_QUANTITY)
    order.orderRef = client_order_id
    trade = _ib_connected().placeOrder(contract, order)
    record = OrderRecord(
        client_order_id=client_order_id,
        broker_order_id=int(getattr(order, "orderId", 0) or 0),
        symbol=normalized,
        side=side,
        quantity=TEST_QUANTITY,
        status=str(getattr(trade.orderStatus, "status", "Submitted")),
        filled=0.0,
        avg_fill_price=None,
        signal_id=str(signal.get("signal_id") or signal.get("forecast_id") or "") or None,
        strategy_id="forex-mtf-paper-test-v1",
        opened_at_utc=_now(),
    )
    _TRADE = trade
    _RECORD = record
    _save_record(record)
    _ib_connected().sleep(2)
    return _order_snapshot(trade, record)


def order_status() -> dict[str, Any]:
    with _LOCK:
        if _TRADE is None or _RECORD is None:
            return {"status": "NO_ACTIVE_TEST", "execution_authorized": False, "paper_only": True}
        return _order_snapshot(_TRADE, _RECORD)


def close_controlled_test() -> dict[str, Any]:
    global _TRADE, _RECORD
    _require_paper_order_mode()
    with _LOCK:
        if _TRADE is None or _RECORD is None:
            raise RuntimeError("PAPER CLOSE: no controlled test order is active.")
        trade = _TRADE
        record = _RECORD
        status = str(getattr(trade.orderStatus, "status", "UNKNOWN"))
        filled = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
        ib = _ib_connected()
        if status not in {"Filled", "PartiallyFilled"} or filled <= 0:
            try:
                ib.cancelOrder(trade.order)
            finally:
                record.status = "Cancelled"
                record.closed_at_utc = _now()
                _save_record(record)
            return {"status": "CANCELLED_UNFILLED", "order": asdict(record), "paper_only": True, "execution_authorized": False}

        ibi = _library()
        _, contract = _qualify(record.symbol)
        close_side = "SELL" if record.side == "LONG" else "BUY"
        close_order = ibi.MarketOrder(close_side, filled)
        close_order.orderRef = f"CLOSE-{record.client_order_id}"
        close_trade = ib.placeOrder(contract, close_order)
        ib.sleep(2)
        record.close_order_id = int(getattr(close_order, "orderId", 0) or 0)
        close_status = str(getattr(close_trade.orderStatus, "status", "Submitted"))
        record.closed_at_utc = _now() if close_status in {"Filled", "PartiallyFilled"} else None
        record.status = "Closed" if record.closed_at_utc else close_status
        _save_record(record)
        return {
            "status": "PAPER_POSITION_CLOSED" if record.closed_at_utc else "PAPER_CLOSE_SUBMITTED",
            "open_order": asdict(record),
            "close_order": {
                "broker_order_id": record.close_order_id,
                "status": close_status,
                "filled": float(getattr(close_trade.orderStatus, "filled", 0.0) or 0.0),
                "avg_fill_price": float(getattr(close_trade.orderStatus, "avgFillPrice", 0.0) or 0.0),
            },
            "paper_only": True,
            "execution_authorized": False,
        }


def _select_auto_signal() -> dict[str, Any] | None:
    response = httpx.get(f"{FOREX_API_URL}/portfolio/results", timeout=25.0)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results") or []
    candidates: list[dict[str, Any]] = []
    for result in results:
        hierarchy = result.get("hierarchical_forecast") or {}
        decision = str(result.get("decision") or "NO TRADE").upper()
        edge = float(hierarchy.get("hierarchy_edge_percentage_points") or 0.0)
        if decision in {"LONG", "SHORT"} and bool(hierarchy.get("entry_trigger")) and edge >= 10.0:
            candidates.append(result)
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: float((row.get("hierarchical_forecast") or {}).get("hierarchy_score") or 0.0),
        reverse=True,
    )
    return candidates[0]


def _auto_tick() -> None:
    if not PAPER_AUTO_TRADING_ENABLED or not PAPER_ORDER_PLACEMENT_ENABLED:
        return
    if _current_open():
        return
    signal = _select_auto_signal()
    if signal is None:
        return
    decision = str(signal.get("decision") or "NO TRADE").upper()
    hierarchy = signal.get("hierarchical_forecast") or {}
    edge = float(hierarchy.get("hierarchy_edge_percentage_points") or 0.0)
    if decision not in {"LONG", "SHORT"} or not hierarchy.get("entry_trigger") or edge < 10.0:
        return
    symbol = str(signal.get("instrument") or "").upper()
    normalized, contract = _qualify(symbol)
    ibi = _library()
    order = ibi.MarketOrder(decision, AUTO_QUANTITY)
    client_order_id = f"paper-auto-{uuid4().hex}"
    order.orderRef = client_order_id
    trade = _ib_connected().placeOrder(contract, order)
    global _TRADE, _RECORD
    with _LOCK:
        _TRADE = trade
        _RECORD = OrderRecord(
            client_order_id=client_order_id,
            broker_order_id=int(getattr(order, "orderId", 0) or 0),
            symbol=normalized,
            side=decision,
            quantity=AUTO_QUANTITY,
            status=str(getattr(trade.orderStatus, "status", "Submitted")),
            filled=0.0,
            avg_fill_price=None,
            signal_id=str(signal.get("signal_id") or signal.get("forecast_id") or "") or None,
            strategy_id="forex-mtf-paper-auto-v1",
            opened_at_utc=_now(),
        )
        _save_record(_RECORD)


def _auto_loop() -> None:
    _init_db()
    while True:
        try:
            _auto_tick()
        except Exception:
            pass
        time.sleep(AUTO_INTERVAL_SECONDS)


def start_auto_trader() -> None:
    global _AUTO_THREAD
    if _AUTO_THREAD is not None and _AUTO_THREAD.is_alive():
        return
    if not PAPER_AUTO_TRADING_ENABLED:
        return
    _AUTO_THREAD = threading.Thread(target=_auto_loop, name="paper-auto-trader", daemon=True)
    _AUTO_THREAD.start()


def configuration_status() -> dict[str, Any]:
    return {
        "paper_order_placement_enabled": PAPER_ORDER_PLACEMENT_ENABLED,
        "paper_auto_trading_enabled": PAPER_AUTO_TRADING_ENABLED,
        "order_client_id": ORDER_CLIENT_ID,
        "test_quantity": TEST_QUANTITY,
        "auto_quantity": AUTO_QUANTITY,
        "auto_interval_seconds": AUTO_INTERVAL_SECONDS,
        "portfolio_refresh_timeout_seconds": PORTFOLIO_REFRESH_TIMEOUT_SECONDS,
        "portfolio_refresh_poll_seconds": PORTFOLIO_REFRESH_POLL_SECONDS,
        "live_trading_enabled": bool(settings.live_trading_enabled),
        "trading_mode": settings.trading_mode,
        "paper_only": True,
    }
