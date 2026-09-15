from __future__ import annotations

from typing import Any

import httpx

from app.config import settings


class IBKRBridgeUnavailable(RuntimeError):
    pass


def _base_url() -> str:
    value = str(settings.ibkr_bridge_url or "").strip().rstrip("/")
    if not value:
        raise IBKRBridgeUnavailable("DATA UNAVAILABLE: IBKR_BRIDGE_URL is not configured.")
    if "127.0.0.1" in value or "localhost" in value:
        raise IBKRBridgeUnavailable(
            "BROKER SAFETY: Railway cannot use a local 127.0.0.1/localhost IBKR bridge URL."
        )
    return value


def _headers() -> dict[str, str]:
    token = str(settings.ibkr_bridge_token or "").strip()
    if not token:
        raise IBKRBridgeUnavailable("BROKER UNAVAILABLE: IBKR_BRIDGE_TOKEN is not configured.")
    return {"Authorization": f"Bearer {token}"}


def get_historical_15m(symbol: str, outputsize: int = 420) -> list[dict[str, Any]]:
    url = f"{_base_url()}/historical/{symbol}"
    try:
        response = httpx.get(
            url,
            params={"outputsize": int(outputsize)},
            headers=_headers(),
            timeout=float(settings.ibkr_bridge_timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise IBKRBridgeUnavailable("DATA UNAVAILABLE: IBKR bridge returned no 15m bars.")
        return rows
    except (httpx.HTTPError, ValueError) as exc:
        raise IBKRBridgeUnavailable(f"DATA UNAVAILABLE: IBKR bridge request failed: {exc}") from exc


def get_historical_15m_batch(symbols: list[str], outputsize: int = 420) -> dict[str, list[dict[str, Any]]]:
    """Fetch 15m history for the portfolio through one authenticated bridge call.

    Partial success is valid: pairs for which IBKR returned no bars are omitted
    rather than causing the entire portfolio batch to fail. The scanner can then
    evaluate available pairs and mark only the missing symbols as DATA UNAVAILABLE.
    """
    unique = list(dict.fromkeys(str(s).upper().replace("_", "/") for s in symbols))
    if not unique or len(unique) > 12:
        raise IBKRBridgeUnavailable("BROKER ERROR: bridge batch supports 1 to 12 symbols.")
    try:
        response = httpx.get(
            f"{_base_url()}/historical-batch",
            params=[("symbols", symbol) for symbol in unique] + [("outputsize", int(outputsize))],
            headers=_headers(),
            timeout=max(float(settings.ibkr_bridge_timeout_seconds), 120.0),
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("rows")
        if not isinstance(rows, dict):
            raise IBKRBridgeUnavailable("DATA UNAVAILABLE: IBKR bridge returned invalid batch history.")

        result: dict[str, list[dict[str, Any]]] = {}
        for symbol in unique:
            symbol_rows = rows.get(symbol) or rows.get(symbol.replace("/", ""))
            if isinstance(symbol_rows, list) and symbol_rows:
                result[symbol] = symbol_rows

        if not result:
            raise IBKRBridgeUnavailable("DATA UNAVAILABLE: IBKR bridge returned no usable 15m bars for the requested portfolio.")
        return result
    except (httpx.HTTPError, ValueError) as exc:
        raise IBKRBridgeUnavailable(f"DATA UNAVAILABLE: IBKR bridge batch request failed: {exc}") from exc


def get_status() -> dict[str, Any]:
    try:
        response = httpx.get(
            f"{_base_url()}/health",
            headers=_headers(),
            timeout=float(settings.ibkr_bridge_timeout_seconds),
        )
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise IBKRBridgeUnavailable(f"BROKER UNAVAILABLE: bridge health check failed: {exc}") from exc
