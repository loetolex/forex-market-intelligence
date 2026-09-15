from __future__ import annotations

import hmac
import os
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from app.execution.ibkr_adapter import get_ibkr_adapter

app = FastAPI(title="Forex Intelligence IBKR Local Bridge", version="0.1.0")

BRIDGE_TOKEN = os.getenv("IBKR_BRIDGE_TOKEN", "")


def _authorize(authorization: Annotated[str | None, Header()] = None) -> None:
    if not BRIDGE_TOKEN:
        raise HTTPException(status_code=503, detail="IBKR_BRIDGE_TOKEN is not configured.")
    expected = f"Bearer {BRIDGE_TOKEN}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized broker bridge request.")


Auth = Depends(_authorize)


@app.get("/health")
def health(_: None = Auth):
    adapter = get_ibkr_adapter()
    return {"status": "ok", "broker": "INTERACTIVE_BROKERS", "connection": adapter.connection_status().__dict__}


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
def historical_batch(
    symbols: Annotated[list[str], Query()],
    outputsize: int = Query(default=420, ge=1, le=500),
    _: None = Auth,
):
    adapter = get_ibkr_adapter()
    unique = list(dict.fromkeys(symbols))
    if not unique or len(unique) > 12:
        raise HTTPException(status_code=400, detail="Provide 1 to 12 unique FX symbols.")
    rows: dict[str, list[dict]] = {}
    for symbol in unique:
        normalized = adapter.normalize_symbol(symbol)
        try:
            rows[normalized] = adapter.get_historical_15m(normalized, outputsize=outputsize)
        except Exception as exc:
            rows[normalized] = []
    return {
        "status": "REAL_BROKER_DATA",
        "broker": "INTERACTIVE_BROKERS",
        "timeframe": "15m",
        "rows": rows,
    }


@app.get("/orders")
def orders(_: None = Auth):
    # Read-only phase: intentionally do not expose order placement.
    return {"status": "LOCKED", "orders": [], "execution_authorized": False}
