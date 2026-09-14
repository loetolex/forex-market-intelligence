from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.execution.ibkr_readonly import read_only_status
from app.services.pipeline import run_market_cycle

app = FastAPI(
    title="Forex Market Intelligence",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "forex-market-intelligence",
        "mode": settings.trading_mode,
        "live_trading_enabled": settings.live_trading_enabled,
        "order_placement_enabled": settings.order_placement_enabled,
        "broker": read_only_status().__dict__,
    }


@app.get("/market/{symbol}")
def market(symbol: str, interval: str | None = None):
    if interval and interval != settings.primary_interval:
        raise HTTPException(
            status_code=400,
            detail=f"Only configured interval {settings.primary_interval!r} is enabled in this first deployment.",
        )

    symbol = symbol.upper().replace("_", "/")
    if "/" not in symbol and len(symbol) == 6:
        symbol = f"{symbol[:3]}/{symbol[3:]}"

    try:
        return run_market_cycle(symbol)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
