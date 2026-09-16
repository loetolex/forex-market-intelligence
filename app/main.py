from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from app.backtests.walk_forward import run_walk_forward_backtest
from app.config import settings
from app.data.tiingo_fx import fetch_time_series as fetch_tiingo_time_series, inspect_time_series as inspect_tiingo_time_series
from app.data.twelve_data import fetch_time_series as fetch_twelve_time_series
from app.execution.ibkr_bridge_client import IBKRBridgeUnavailable, get_status as get_ibkr_bridge_status
from app.execution.ibkr_readonly import read_only_status
from app.paper_trading import (
    close_controlled_test,
    configuration_status as paper_configuration_status,
    controlled_test_preview,
    order_status as paper_order_status,
    place_controlled_test,
    start_auto_trader,
)
from app.services.pipeline import get_shadow_learning_status, normalize_symbol, run_market_cycle
from app.services.portfolio_scanner import (
    DEFAULT_PORTFOLIO_PAIRS,
    get_portfolio_candidates,
    get_portfolio_ranking,
    get_portfolio_results,
    get_portfolio_status,
    request_portfolio_scan,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Automatic trading can start only when the explicit PAPER controls are
    # enabled. The paper trader itself enforces trading_mode=PAPER and
    # live_trading_enabled=false before any broker submission.
    start_auto_trader()
    yield


app = FastAPI(
    title="Forex Market Intelligence",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    paper = paper_configuration_status()
    return {
        "status": "ok",
        "service": "forex-market-intelligence",
        "mode": settings.trading_mode,
        "live_trading_enabled": settings.live_trading_enabled,
        "order_placement_enabled": settings.order_placement_enabled,
        "paper_order_placement_enabled": paper["paper_order_placement_enabled"],
        "paper_auto_trading_enabled": paper["paper_auto_trading_enabled"],
        "market_data": {
            "primary": settings.market_data_provider,
            "secondary": settings.secondary_market_data_provider,
            "tiingo_configured": bool(settings.tiingo_api_token),
            "twelve_data_configured": bool(settings.twelve_data_api_key),
            "tiingo_intraday_history_days": settings.tiingo_intraday_history_days,
            "tiingo_daily_history_days": settings.tiingo_daily_history_days,
            "fx_daily_boundary": f"{settings.fx_daily_boundary_hour_local:02d}:00 {settings.fx_daily_boundary_timezone}",
            "forecast_intervals": settings.forecast_intervals,
            "freshness_limits_minutes": settings.freshness_limits_minutes,
        },
        "broker": {
            **read_only_status().__dict__,
            "bridge_configured": bool(settings.ibkr_bridge_url),
        },
        "execution_safety": {
            "forecast_can_place_orders": False,
            "rl_can_authorize_execution": False,
            "live_execution": "LOCKED",
            "paper_submission": "ENABLED" if paper["paper_order_placement_enabled"] else "LOCKED",
            "automatic_paper_trading": "ENABLED" if paper["paper_auto_trading_enabled"] else "LOCKED",
            "protective_exits": paper["protective_exits"],
        },
    }


@app.get("/paper/config")
def paper_config():
    return paper_configuration_status()


@app.post("/paper/controlled-test/{symbol}")
def paper_controlled_test(symbol: str, timeframe: str = "15m"):
    try:
        return place_controlled_test(normalize_symbol(symbol), timeframe)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/paper/order-status")
def paper_status():
    return paper_order_status()


@app.post("/paper/close-controlled-test")
def paper_close_controlled_test():
    try:
        return close_controlled_test()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/paper/preview/{symbol:path}")
def paper_preview(symbol: str, timeframe: str = "15m"):
    """Preview a paper execution contract without authorizing broker execution.

    The path converter intentionally accepts encoded FX separators such as
    ``EUR%2FUSD`` after URL decoding. Fixed paper routes are registered first
    so this catch-all symbol route cannot shadow ``/paper/order-status`` or
    ``/paper/close-controlled-test``.
    """
    try:
        result = controlled_test_preview(normalize_symbol(symbol), timeframe)
        # A preview may report that all pre-trade gates pass, but previewing
        # never authorizes execution. Actual authorization requires the
        # controlled execution path to submit and reconcile broker state.
        result["execution_authorized"] = False
        result["execution_status"] = "PREVIEW_ONLY"
        result["preview_only"] = True
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/learning/status")
def learning_status():
    """Read-only diagnostics for shadow RL learning and baseline comparison."""
    if not settings.rl_enabled or not settings.rl_shadow_only:
        return {"status": "DISABLED", "advisory_only": True, "execution_authorized": False}
    return get_shadow_learning_status()


@app.get("/provider/inspect/{symbol}")
def provider_inspect(
    symbol: str,
    interval: str = "1h",
    outputsize: int = 10,
):
    """Diagnostic-only endpoint for one Tiingo request."""
    if interval not in settings.forecast_intervals:
        raise HTTPException(status_code=400, detail=f"Supported intervals: {settings.forecast_intervals}")
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
    if interval not in settings.forecast_intervals:
        raise HTTPException(status_code=400, detail=f"Supported intervals: {settings.forecast_intervals}")
    if outputsize < 2 or outputsize > 100:
        raise HTTPException(status_code=400, detail="outputsize must be between 2 and 100.")
    if not settings.tiingo_api_token:
        raise HTTPException(status_code=503, detail="DATA UNAVAILABLE: TIINGO_API_TOKEN is not configured.")
    if not settings.twelve_data_api_key:
        raise HTTPException(status_code=503, detail="DATA UNAVAILABLE: TWELVE_DATA_API_KEY is not configured.")

    normalized = normalize_symbol(symbol)
    try:
        history_days = 30 if interval in {"15m", "30m", "1h"} else 120
        tiingo = fetch_tiingo_time_series(
            settings.tiingo_api_token,
            symbol=normalized,
            interval=interval,
            history_days=history_days,
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
    symbols: list[str] = Query(default=DEFAULT_PORTFOLIO_PAIRS),
    refresh: bool = False,
):
    """Return a compact five-timeframe portfolio scanner snapshot.

    The scan is asynchronous so the gateway returns immediately instead of
    waiting for all pairs to complete. Refreshing starts one background scan;
    subsequent requests return progress/results. The detailed /market/{symbol}
    endpoint remains the deep research path for an individual pair.
    """
    if not symbols:
        raise HTTPException(status_code=400, detail="At least one symbol is required.")
    if len(symbols) > len(DEFAULT_PORTFOLIO_PAIRS):
        raise HTTPException(status_code=400, detail=f"Maximum {len(DEFAULT_PORTFOLIO_PAIRS)} pairs per portfolio request in this stage.")
    try:
        return request_portfolio_scan(symbols, refresh=refresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/portfolio/status")
def portfolio_status():
    return get_portfolio_status()


@app.get("/portfolio/results")
def portfolio_results():
    return get_portfolio_results()


@app.get("/portfolio/ranking")
def portfolio_ranking():
    return get_portfolio_ranking()


@app.get("/portfolio/candidates")
def portfolio_candidates():
    return get_portfolio_candidates()


@app.get("/ibkr/bridge-status")
def ibkr_bridge_status():
    """Safe Railway-side connectivity diagnostic for the IBKR bridge."""
    if not settings.ibkr_bridge_url:
        return {
            "status": "DATA UNAVAILABLE",
            "bridge_reachable": False,
            "broker_connected": False,
            "reason": "IBKR_BRIDGE_URL is not configured.",
            "execution_locked": True,
        }
    try:
        payload = get_ibkr_bridge_status()
        connection = payload.get("connection") or {}
        return {
            "status": "CALCULATED",
            "bridge_reachable": True,
            "broker_connected": bool(connection.get("connected")),
            "broker": payload.get("broker", "INTERACTIVE_BROKERS"),
            "account_present": bool(connection.get("account")),
            "mode": connection.get("trading_mode", settings.trading_mode),
            "read_only": bool(connection.get("readonly", True)),
            "live_trading_enabled": bool(connection.get("live_trading_enabled", False)),
            "order_placement_enabled": bool(connection.get("order_placement_enabled", False)),
            "execution_locked": True,
        }
    except IBKRBridgeUnavailable as exc:
        return {
            "status": "BROKER UNAVAILABLE",
            "bridge_reachable": False,
            "broker_connected": False,
            "reason": str(exc),
            "execution_locked": True,
        }


@app.get("/backtest/{symbol}")
def backtest(symbol: str):
    symbol = normalize_symbol(symbol)
    try:
        raw = fetch_tiingo_time_series(
            settings.tiingo_api_token,
            symbol=symbol,
            interval="1h",
            history_days=settings.tiingo_intraday_history_days,
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
