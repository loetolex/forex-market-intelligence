from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from app.backtests.walk_forward import run_walk_forward_backtest
from app.config import settings
from app.data.tiingo_fx import fetch_time_series as fetch_tiingo_time_series, inspect_time_series as inspect_tiingo_time_series
from app.data.twelve_data import fetch_time_series as fetch_twelve_time_series
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
        "market_data": {
            "primary": settings.market_data_provider,
            "secondary": settings.secondary_market_data_provider,
            "tiingo_configured": bool(settings.tiingo_api_token),
            "twelve_data_configured": bool(settings.twelve_data_api_key),
        },
        "broker": read_only_status().__dict__,
    }


@app.get("/provider/inspect/{symbol}")
def provider_inspect(
    symbol: str,
    interval: str = "1h",
    outputsize: int = 10,
):
    """Diagnostic-only endpoint for one Tiingo request."""
    if interval not in settings.forecast_intervals:
        raise HTTPException(
            status_code=400,
            detail=f"Supported intervals: {settings.forecast_intervals}",
        )
    if outputsize < 1 or outputsize > 100:
        raise HTTPException(status_code=400, detail="outputsize must be between 1 and 100.")

    try:
        return inspect_tiingo_time_series(
            settings.tiingo_api_token,
            symbol=normalize_symbol(symbol),
            history_days=max(7, outputsize // 24 + 1),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/provider/crosscheck/{symbol}")
def provider_crosscheck(
    symbol: str,
    interval: str = "1h",
    outputsize: int = 10,
):
    """Compare primary Tiingo data with secondary Twelve Data data.

    This endpoint is diagnostic only. A disagreement never authorizes trading.
    """
    if interval not in {"1h", "4h", "1day"}:
        raise HTTPException(status_code=400, detail="Supported intervals: 1h, 4h, 1day")
    if outputsize < 2 or outputsize > 100:
        raise HTTPException(status_code=400, detail="outputsize must be between 2 and 100.")
    if not settings.tiingo_api_token:
        raise HTTPException(status_code=503, detail="DATA UNAVAILABLE: TIINGO_API_TOKEN is not configured.")
    if not settings.twelve_data_api_key:
        raise HTTPException(status_code=503, detail="DATA UNAVAILABLE: TWELVE_DATA_API_KEY is not configured.")

    normalized = normalize_symbol(symbol)

    try:
        tiingo = fetch_tiingo_time_series(
            settings.tiingo_api_token,
            symbol=normalized,
            interval=interval,
            history_days=30 if interval == "1h" else 120,
            cache_ttl_seconds=0,
        )
        twelve = fetch_twelve_time_series(
            settings.twelve_data_api_key,
            symbol=normalized,
            interval=interval,
            outputsize=outputsize,
        )

        tiingo_last = tiingo.iloc[-1]
        twelve_last = twelve.iloc[-1]
        tiingo_last_ts = pd.to_datetime(tiingo_last["timestamp"], utc=True)
        twelve_last_ts = pd.to_datetime(twelve_last["timestamp"], utc=True)

        return {
            "status": "CALCULATED",
            "symbol": normalized,
            "interval": interval,
            "request_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "primary": {
                "provider": "Tiingo FX",
                "rows": int(len(tiingo)),
                "first_timestamp": str(tiingo["timestamp"].min()),
                "last_timestamp": str(tiingo["timestamp"].max()),
                "last_close": float(tiingo_last["close"]),
                "data_status": str(tiingo_last["data_status"]),
            },
            "secondary": {
                "provider": "Twelve Data",
                "rows": int(len(twelve)),
                "first_timestamp": str(twelve["timestamp"].min()),
                "last_timestamp": str(twelve["timestamp"].max()),
                "last_close": float(twelve_last["close"]),
                "data_status": str(twelve_last["data_status"]),
            },
            "crosscheck": {
                "timestamp_delta_minutes": abs((tiingo_last_ts - twelve_last_ts).total_seconds()) / 60.0,
                "close_delta": float(tiingo_last["close"] - twelve_last["close"]),
                "close_delta_abs": abs(float(tiingo_last["close"] - twelve_last["close"])),
                "status": "REVIEW_REQUIRED" if tiingo_last_ts != twelve_last_ts else "ALIGNED",
            },
            "execution_authorized": False,
            "execution_status": "LOCKED",
        }
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
        raw = fetch_tiingo_time_series(
            settings.tiingo_api_token,
            symbol=symbol,
            interval="1h",
            history_days=settings.tiingo_history_days,
            cache_ttl_seconds=settings.tiingo_cache_ttl_seconds,
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
