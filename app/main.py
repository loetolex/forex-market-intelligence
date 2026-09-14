from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from app.backtests.walk_forward import run_walk_forward_backtest
from app.config import settings
from app.data.twelve_data import fetch_time_series, inspect_time_series
from app.execution.ibkr_readonly import read_only_status
from app.services.pipeline import normalize_symbol, run_market_cycle

app = FastAPI(
    title="Forex Market Intelligence",
    version="0.2.0",
)

DEFAULT_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD",
]


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


@app.get("/provider/inspect/{symbol}")
def provider_inspect(
    symbol: str,
    interval: str = "1h",
    outputsize: int = 10,
):
    """Diagnostic-only endpoint for one Twelve Data request.

    This bypasses the application cache and does not run forecasting,
    signal generation, risk evaluation, or execution.
    """
    if interval not in settings.forecast_intervals:
        raise HTTPException(
            status_code=400,
            detail=f"Supported intervals: {settings.forecast_intervals}",
        )
    if outputsize < 1 or outputsize > 100:
        raise HTTPException(status_code=400, detail="outputsize must be between 1 and 100.")

    try:
        return inspect_time_series(
            settings.twelve_data_api_key,
            symbol=normalize_symbol(symbol),
            interval=interval,
            outputsize=outputsize,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/market/{symbol}")
def market(symbol: str, interval: str | None = None):
    if interval and interval not in settings.forecast_intervals:
        raise HTTPException(status_code=400, detail=f"Supported intervals: {settings.forecast_intervals}")
    try:
        result = run_market_cycle(normalize_symbol(symbol))
        if interval:
            result["requested_interval"] = interval
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/portfolio")
def portfolio(
    symbols: list[str] = Query(default=DEFAULT_PAIRS),
):
    if not symbols:
        raise HTTPException(status_code=400, detail="At least one symbol is required.")
    if len(symbols) > 7:
        raise HTTPException(status_code=400, detail="Maximum 7 pairs per portfolio request in this stage.")

    results = []
    for raw_symbol in symbols:
        symbol = normalize_symbol(raw_symbol)
        try:
            results.append(run_market_cycle(symbol))
        except Exception as exc:
            results.append({
                "instrument": symbol,
                "status": "DATA UNAVAILABLE",
                "error": str(exc),
                "execution_authorized": False,
            })

    usable = [r for r in results if r.get("status") not in {"DATA UNAVAILABLE", "NOT_ADMITTED"}]
    return {
        "status": "MODEL OUTPUT" if usable else "DATA UNAVAILABLE",
        "portfolio_size": len(symbols),
        "results": results,
        "execution_authorized": False,
        "execution_mode": settings.trading_mode,
    }


@app.get("/backtest/{symbol}")
def backtest(symbol: str):
    symbol = normalize_symbol(symbol)
    try:
        raw = fetch_time_series(
            settings.twelve_data_api_key,
            symbol=symbol,
            interval="1h",
            outputsize=settings.forecast_outputsize,
        )
        if raw["data_status"].ne("REAL_DATA").any():
            raise RuntimeError("DATA UNAVAILABLE: provider provenance is not REAL_DATA.")
        result = run_walk_forward_backtest(raw)
        return {
            "instrument": symbol,
            "timeframe": "1h",
            "provenance": "REAL_DATA",
            "result": result,
            "execution_authorized": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
