from __future__ import annotations

import asyncio
import hmac
import os
import threading
import time
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query

from app.execution.ibkr_adapter import get_ibkr_adapter
from app.paper_performance import init_db, summary as paper_summary, worker_tick, sleep_seconds
from app.paper_trading import (
    close_controlled_test,
    configuration_status,
    controlled_test_preview,
    order_status as paper_order_status,
    place_controlled_test,
    start_auto_trader,
)

app = FastAPI(title="Forex Intelligence IBKR Local Bridge", version="0.4.0")

BRIDGE_TOKEN = os.getenv("IBKR_BRIDGE_TOKEN", "")
PAPER_MONITOR_ENABLED = os.getenv("PAPER_MONITOR_ENABLED", "true").lower() == "true"
FOREX_API_URL = os.getenv(
    "FOREX_API_URL",
    "https://forex-api-production-f587.up.railway.app",
).rstrip("/")


def _authorize(authorization: Annotated[str | None, Header()] = None) -> None:
    if not BRIDGE_TOKEN:
        raise HTTPException(status_code=503, detail="IBKR_BRIDGE_TOKEN is not configured.")
    expected = f"Bearer {BRIDGE_TOKEN}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized broker bridge request.")


Auth = Depends(_authorize)


def _fetch_railway_results() -> list[dict[str, Any]]:
    response = httpx.get(f"{FOREX_API_URL}/portfolio/results", timeout=20.0)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("results", [])
    return rows if isinstance(rows, list) else []


def _refresh_railway_portfolio() -> None:
    # Diagnostic/paper monitoring only. This starts the existing scanner; it does
    # not and cannot authorize an order.
    response = httpx.get(
        f"{FOREX_API_URL}/portfolio",
        params={"refresh": "true"},
        timeout=20.0,
    )
    response.raise_for_status()


def _wait_for_scan_results(timeout_seconds: float = 300.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = httpx.get(f"{FOREX_API_URL}/portfolio/status", timeout=20.0)
        response.raise_for_status()
        status = response.json()
        state = str(status.get("status") or "").upper()
        completed = int(status.get("pairs_completed") or 0)
        total = int(status.get("pairs_total") or 0)
        if state == "COMPLETE" or (total > 0 and completed >= total):
            return _fetch_railway_results()
        if state in {"FAILED", "ERROR"}:
            return []
        time.sleep(5)
    return []


def _collect_one_cycle() -> dict[str, int]:
    _refresh_railway_portfolio()
    rows = _wait_for_scan_results()
    if not rows:
        return {"recorded": 0, "settled": 0}

    adapter = get_ibkr_adapter()
    return worker_tick(
        fetch_results=lambda rows=rows: rows,
        get_quote=lambda symbol: adapter.get_quote(symbol),
        get_history=lambda symbol, size: adapter.get_historical_15m(symbol, outputsize=size),
    )


def _monitor_loop() -> None:
    init_db()
    while True:
        try:
            _collect_one_cycle()
        except Exception:
            # Monitoring failure must never interrupt the broker bridge or create
            # an order. The next scheduled tick retries.
            pass
        time.sleep(sleep_seconds())


@app.on_event("startup")
def startup() -> None:
    init_db()
    if PAPER_MONITOR_ENABLED:
        thread = threading.Thread(
            target=_monitor_loop,
            name="paper-performance-monitor",
            daemon=True,
        )
        thread.start()
    start_auto_trader()


@app.get("/health")
def health(_: None = Auth):
    adapter = get_ibkr_adapter()
    perf = paper_summary()
    return {
        "status": "ok",
        "broker": "INTERACTIVE_BROKERS",
        "connection": adapter.connection_status().__dict__,
        "paper_monitor_enabled": PAPER_MONITOR_ENABLED,
        "paper_trading": configuration_status(),
        "paper_performance": {
            "status": "CALCULATED",
            "settled_forecasts": perf["settled_forecasts"],
            "pending_forecasts": perf["pending_forecasts"],
        },
    }


@app.post("/connect")
def connect(_: None = Auth):
    return get_ibkr_adapter().connect().__dict__


@app.post("/disconnect")
def disconnect(_: None = Auth):
    get_ibkr_adapter().disconnect()
    return {"status": "DISCONNECTED"}


@app.get("/account")
def account(_: None = Auth):
    return get_ibkr_adapter().get_account()


@app.get("/positions")
def positions(_: None = Auth):
    return get_ibkr_adapter().get_positions()


@app.get("/contract/{symbol:path}")
def contract(symbol: str, _: None = Auth):
    return get_ibkr_adapter().get_contract(symbol)


@app.get("/quote/{symbol:path}")
def quote(symbol: str, _: None = Auth):
    return get_ibkr_adapter().get_quote(symbol)


@app.get("/historical/{symbol:path}")
def historical(
    symbol: str,
    outputsize: int = Query(default=420, ge=1, le=500),
    _: None = Auth,
):
    return {
        "status": "REAL_BROKER_DATA",
        "broker": "INTERACTIVE_BROKERS",
        "symbol": get_ibkr_adapter().normalize_symbol(symbol),
        "timeframe": "15m",
        "rows": get_ibkr_adapter().get_historical_15m(symbol, outputsize=outputsize),
    }


@app.get("/historical-batch")
async def historical_batch(
    symbols: Annotated[list[str], Query()],
    outputsize: int = Query(default=420, ge=1, le=500),
    _: None = Auth,
):
    """Fetch multiple unique FX 15m histories concurrently through one bridge call."""
    adapter = get_ibkr_adapter()
    unique = list(dict.fromkeys(adapter.normalize_symbol(symbol) for symbol in symbols))
    if not unique or len(unique) > 12:
        raise HTTPException(status_code=400, detail="Provide 1 to 12 unique FX symbols.")

    try:
        rows = await asyncio.wait_for(
            adapter.get_historical_15m_batch_async(unique, outputsize=outputsize),
            timeout=90.0,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="IBKR DATA TIMEOUT: 15m batch exceeded 90 seconds.") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"IBKR DATA UNAVAILABLE: {exc}") from exc

    return {
        "status": "REAL_BROKER_DATA",
        "broker": "INTERACTIVE_BROKERS",
        "timeframe": "15m",
        "rows": rows,
    }


@app.get("/paper/performance")
def paper_performance(_: None = Auth):
    return paper_summary()


@app.post("/paper/collect")
def paper_collect(_: None = Auth):
    try:
        counts = _collect_one_cycle()
        return {
            "status": "CALCULATED",
            "collection": counts,
            "performance": paper_summary(),
            "execution_authorized": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"PAPER MONITOR ERROR: {exc}") from exc


@app.get("/paper/order-config")
def paper_order_config(_: None = Auth):
    return configuration_status()


@app.post("/paper/test-order/preview")
def paper_test_order_preview(_: None = Auth):
    try:
        return controlled_test_preview()
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/paper/test-order")
def paper_test_order(_: None = Auth):
    try:
        return place_controlled_test()
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/paper/test-order/status")
def paper_test_order_status(_: None = Auth):
    return paper_order_status()


@app.post("/paper/test-order/close")
def paper_test_order_close(_: None = Auth):
    try:
        return close_controlled_test()
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/orders")
def orders(_: None = Auth):
    return {"status": "PAPER_ONLY", "order_config": configuration_status(), "execution_authorized": False}
