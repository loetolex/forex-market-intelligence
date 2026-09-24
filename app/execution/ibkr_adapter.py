from __future__ import annotations

from dataclasses import asdict, dataclass
import asyncio
import math
from threading import RLock
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class BrokerConnection:
    connected: bool
    host: str
    port: int
    client_id: int
    account: str | None
    readonly: bool
    trading_mode: str
    live_trading_enabled: bool
    order_placement_enabled: bool
    status: str


def _finite_float(value: Any) -> float | None:
    """Convert broker numeric values to JSON-safe finite floats."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class IBKRAdapter:
    """Paper-safe IBKR adapter.

    The dependency is imported lazily so the application can still boot in
    Railway without a local TWS/IB Gateway process. No broker connection is
    attempted during module import or FastAPI startup.
    """

    def __init__(self) -> None:
        self._ib: Any | None = None
        self._account: str | None = None
        self._lock = RLock()

    @staticmethod
    def _require_paper_mode() -> None:
        if str(settings.trading_mode).upper() != "PAPER":
            raise RuntimeError("BROKER SAFETY: IBKR adapter requires trading_mode=PAPER.")
        if bool(settings.live_trading_enabled):
            raise RuntimeError("BROKER SAFETY: live_trading_enabled must remain false.")

    @staticmethod
    def _library():
        try:
            import ib_async as ibi
        except ImportError as exc:
            raise RuntimeError("BROKER UNAVAILABLE: install ib_async to use the IBKR bridge.") from exc
        return ibi

    def connect(self) -> BrokerConnection:
        with self._lock:
            self._require_paper_mode()
            ibi = self._library()
            if self._ib is not None and self._ib.isConnected():
                return self.connection_status()

            ib = ibi.IB()
            # First phase is read-only. Order-enabled mode is deliberately not
            # reachable through this adapter until the application is explicitly
            # configured for paper order placement.
            ib.connect(
                settings.ibkr_host,
                settings.ibkr_port,
                clientId=settings.ibkr_client_id,
                timeout=5,
                readonly=True,
            )
            self._ib = ib
            accounts = list(ib.managedAccounts() or [])
            self._account = accounts[0] if accounts else None
            return self.connection_status()

    def disconnect(self) -> None:
        with self._lock:
            if self._ib is not None:
                try:
                    self._ib.disconnect()
                finally:
                    self._ib = None
                    self._account = None

    def _connected_ib(self):
        with self._lock:
            if self._ib is None or not self._ib.isConnected():
                self.connect()
            if self._ib is None or not self._ib.isConnected():
                raise RuntimeError("BROKER UNAVAILABLE: IBKR TWS/Gateway is not connected.")
            return self._ib

    def connection_status(self) -> BrokerConnection:
        connected = bool(self._ib is not None and self._ib.isConnected())
        return BrokerConnection(
            connected=connected,
            host=settings.ibkr_host,
            port=settings.ibkr_port,
            client_id=settings.ibkr_client_id,
            account=self._account,
            readonly=True,
            trading_mode=settings.trading_mode,
            live_trading_enabled=bool(settings.live_trading_enabled),
            order_placement_enabled=bool(settings.order_placement_enabled),
            status="CONNECTED_READ_ONLY" if connected else "BROKER CONNECTION = HOLD",
        )

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        value = str(symbol).upper().replace("_", "/").replace("-", "/")
        if "/" not in value and len(value) == 6:
            value = f"{value[:3]}/{value[3:]}"
        base, sep, quote = value.partition("/")
        if not sep or len(base) != 3 or len(quote) != 3:
            raise ValueError(f"Unsupported FX symbol: {symbol}")
        return f"{base}/{quote}"

    def _qualify_forex(self, symbol: str):
        ibi = self._library()
        ib = self._connected_ib()
        normalized = self.normalize_symbol(symbol)
        contract = ibi.Forex(normalized.replace("/", ""))
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"BROKER DATA UNAVAILABLE: IBKR could not qualify {normalized}.")
        return qualified[0]

    def search_contracts(self, pattern: str, limit: int = 50) -> dict[str, Any]:
        """Read-only IBKR contract discovery for non-FX and FX instruments.

        ``reqMatchingSymbols`` is intentionally separated from the existing
        FX-only qualification path. Discovery returns contract metadata only;
        it does not fetch prices, place orders, or authorize execution.
        """
        value = str(pattern or "").strip()
        if not value:
            raise ValueError("BROKER DISCOVERY: pattern is required.")
        if len(value) > 40:
            raise ValueError("BROKER DISCOVERY: pattern is too long.")
        capped_limit = max(1, min(int(limit), 100))

        ib = self._connected_ib()
        matches = ib.reqMatchingSymbols(value)
        rows: list[dict[str, Any]] = []
        for description in list(matches or [])[:capped_limit]:
            contract = getattr(description, "contract", None)
            if contract is None:
                continue
            rows.append({
                "con_id": int(getattr(contract, "conId", 0) or 0),
                "symbol": getattr(contract, "symbol", None),
                "sec_type": getattr(contract, "secType", None),
                "exchange": getattr(contract, "exchange", None),
                "primary_exchange": getattr(contract, "primaryExchange", None),
                "currency": getattr(contract, "currency", None),
                "local_symbol": getattr(contract, "localSymbol", None),
                "trading_class": getattr(contract, "tradingClass", None),
                "last_trade_date_or_contract_month": getattr(contract, "lastTradeDateOrContractMonth", None),
                "strike": _finite_float(getattr(contract, "strike", None)),
                "right": getattr(contract, "right", None),
                "multiplier": getattr(contract, "multiplier", None),
                "description": getattr(contract, "description", None),
                "derivative_sec_types": list(getattr(description, "derivativeSecTypes", None) or []),
                "data_status": "REAL_BROKER_DATA",
            })

        return {
            "status": "REAL_BROKER_DATA",
            "broker": "INTERACTIVE_BROKERS",
            "pattern": value,
            "matches_returned": len(rows),
            "results": rows,
            "execution_authorized": False,
        }

    def get_contract_details_by_conid(self, con_id: int) -> dict[str, Any]:
        """Return authoritative contract details for an IBKR conId."""
        try:
            normalized_con_id = int(con_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("BROKER DISCOVERY: con_id must be an integer.") from exc
        if normalized_con_id <= 0:
            raise ValueError("BROKER DISCOVERY: con_id must be positive.")

        ibi = self._library()
        ib = self._connected_ib()
        contract = ibi.Contract(conId=normalized_con_id)
        details = ib.reqContractDetails(contract)
        if not details:
            raise RuntimeError(f"BROKER DATA UNAVAILABLE: no contract details for conId {normalized_con_id}.")

        detail = details[0]
        resolved = detail.contract
        return {
            "status": "REAL_BROKER_DATA",
            "broker": "INTERACTIVE_BROKERS",
            "contract": {
                "con_id": int(getattr(resolved, "conId", normalized_con_id) or normalized_con_id),
                "symbol": getattr(resolved, "symbol", None),
                "sec_type": getattr(resolved, "secType", None),
                "exchange": getattr(resolved, "exchange", None),
                "primary_exchange": getattr(resolved, "primaryExchange", None),
                "currency": getattr(resolved, "currency", None),
                "local_symbol": getattr(resolved, "localSymbol", None),
                "trading_class": getattr(resolved, "tradingClass", None),
                "last_trade_date_or_contract_month": getattr(resolved, "lastTradeDateOrContractMonth", None),
                "strike": _finite_float(getattr(resolved, "strike", None)),
                "right": getattr(resolved, "right", None),
                "multiplier": getattr(resolved, "multiplier", None),
                "description": getattr(resolved, "description", None),
            },
            "details": {
                "market_name": getattr(detail, "marketName", None),
                "long_name": getattr(detail, "longName", None),
                "industry": getattr(detail, "industry", None),
                "category": getattr(detail, "category", None),
                "subcategory": getattr(detail, "subcategory", None),
                "time_zone_id": getattr(detail, "timeZoneId", None),
                "trading_hours": getattr(detail, "tradingHours", None),
                "liquid_hours": getattr(detail, "liquidHours", None),
                "valid_exchanges": getattr(detail, "validExchanges", None),
                "contract_month": getattr(detail, "contractMonth", None),
                "real_expiration_date": getattr(detail, "realExpirationDate", None),
                "last_trade_time": getattr(detail, "lastTradeTime", None),
                "under_con_id": int(getattr(detail, "underConId", 0) or 0),
                "under_symbol": getattr(detail, "underSymbol", None),
                "under_sec_type": getattr(detail, "underSecType", None),
                "market_rule_ids": getattr(detail, "marketRuleIds", None),
                "min_tick": _finite_float(getattr(detail, "minTick", None)),
                "min_size": _finite_float(getattr(detail, "minSize", None)),
                "size_increment": _finite_float(getattr(detail, "sizeIncrement", None)),
                "suggested_size_increment": _finite_float(getattr(detail, "suggestedSizeIncrement", None)),
                "order_types": getattr(detail, "orderTypes", None),
                "stock_type": getattr(detail, "stockType", None),
                "sec_id_list": [
                    {"tag": getattr(item, "tag", None), "value": getattr(item, "value", None)}
                    for item in list(getattr(detail, "secIdList", None) or [])
                ],
            },
            "execution_authorized": False,
        }
    def get_account(self) -> dict[str, Any]:
        ib = self._connected_ib()
        values = ib.accountValues()
        return {
            "status": "REAL_BROKER_DATA",
            "broker": "INTERACTIVE_BROKERS",
            "account": self._account,
            "values": [
                {
                    "account": x.account,
                    "tag": x.tag,
                    "value": x.value,
                    "currency": x.currency,
                    "model_code": x.modelCode,
                }
                for x in values
            ],
        }

    def get_positions(self) -> dict[str, Any]:
        ib = self._connected_ib()
        positions = ib.positions()
        return {
            "status": "REAL_BROKER_DATA",
            "broker": "INTERACTIVE_BROKERS",
            "account": self._account,
            "positions": [
                {
                    "account": p.account,
                    "symbol": p.contract.symbol,
                    "currency": p.contract.currency,
                    "sec_type": p.contract.secType,
                    "position": _finite_float(p.position) or 0.0,
                    "average_cost": _finite_float(p.avgCost) or 0.0,
                    "con_id": int(p.contract.conId),
                }
                for p in positions
            ],
        }

    def get_contract(self, symbol: str) -> dict[str, Any]:
        ib = self._connected_ib()
        contract = self._qualify_forex(symbol)
        details = ib.reqContractDetails(contract)
        if not details:
            raise RuntimeError(f"BROKER DATA UNAVAILABLE: no contract details for {symbol}.")
        detail = details[0]
        return {
            "status": "REAL_BROKER_DATA",
            "broker": "INTERACTIVE_BROKERS",
            "contract": {
                "symbol": self.normalize_symbol(symbol),
                "con_id": int(contract.conId),
                "local_symbol": contract.localSymbol,
                "trading_class": contract.tradingClass,
                "exchange": contract.exchange,
                "currency": contract.currency,
                "sec_type": contract.secType,
            },
            "constraints": {
                "min_size": _finite_float(detail.minSize) or 0.0,
                "size_increment": _finite_float(detail.sizeIncrement) or 0.0,
                "suggested_size_increment": _finite_float(detail.suggestedSizeIncrement) or 0.0,
                "min_tick": _finite_float(detail.minTick) or 0.0,
                "order_types": detail.orderTypes,
            },
        }

    def get_quote(self, symbol: str) -> dict[str, Any]:
        """Fetch a bounded-time IBKR quote without allowing snapshot hangs.

        ``reqTickers`` can wait on a snapshot-completion event that may never
        arrive in some paper/demo market-data configurations. We instead use a
        short-lived streaming subscription, wait a few seconds for a finite
        value, then explicitly cancel it. A delayed-data retry is attempted if
        live data is unavailable.
        """
        ib = self._connected_ib()
        contract = self._qualify_forex(symbol)
        normalized = self.normalize_symbol(symbol)
        last_error: str | None = None

        for market_data_type, data_label in ((1, "LIVE"), (3, "DELAYED")):
            ticker = None
            try:
                ib.reqMarketDataType(market_data_type)
                ticker = ib.reqMktData(
                    contract,
                    genericTickList="",
                    snapshot=False,
                    regulatorySnapshot=False,
                )
                # ib.sleep() yields to ib_async's event loop so the streaming
                # ticker can be populated without blocking the connection.
                for _ in range(10):
                    ib.sleep(0.5)
                    bid = _finite_float(ticker.bid)
                    ask = _finite_float(ticker.ask)
                    last = _finite_float(ticker.last)
                    close = _finite_float(ticker.close)
                    market_price = _finite_float(ticker.marketPrice())
                    if any(v is not None for v in (bid, ask, last, market_price)):
                        return {
                            "status": "REAL_BROKER_DATA",
                            "broker": "INTERACTIVE_BROKERS",
                            "symbol": normalized,
                            "timestamp": ticker.time.isoformat() if ticker.time else None,
                            "bid": bid,
                            "ask": ask,
                            "last": last,
                            "close": close,
                            "market_price": market_price,
                            "market_data_type": market_data_type,
                            "data_note": f"REAL_BROKER_DATA; market_data_type={data_label}.",
                        }
                last_error = f"No finite quote values returned for {normalized} using {data_label} data."
            except Exception as exc:
                last_error = str(exc)
            finally:
                if ticker is not None:
                    try:
                        ib.cancelMktData(contract)
                    except Exception:
                        pass

        raise RuntimeError(f"BROKER DATA UNAVAILABLE: {last_error or 'IBKR returned no quote data.'}")

    @staticmethod
    def _serialize_bars(symbol: str, bars: Any, outputsize: int) -> list[dict[str, Any]]:
        rows = list(bars or [])[-max(1, int(outputsize)):]
        return [
            {
                "timestamp": bar.date.isoformat() if hasattr(bar.date, "isoformat") else str(bar.date),
                "open": _finite_float(bar.open),
                "high": _finite_float(bar.high),
                "low": _finite_float(bar.low),
                "close": _finite_float(bar.close),
                "volume": _finite_float(bar.volume),
                "provider": "IBKR",
                "instrument": IBKRAdapter.normalize_symbol(symbol),
                "timeframe": "15m",
                "data_status": "REAL_DATA",
            }
            for bar in rows
        ]

    def get_historical_15m(self, symbol: str, outputsize: int = 420) -> list[dict[str, Any]]:
        ib = self._connected_ib()
        contract = self._qualify_forex(symbol)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="7 D",
            barSizeSetting="15 mins",
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=2,
            keepUpToDate=False,
        )
        return self._serialize_bars(symbol, bars, outputsize)

    async def get_historical_15m_async(self, symbol: str, outputsize: int = 420) -> list[dict[str, Any]]:
        """Async historical request so the portfolio batch can fan out without
        serially blocking the public bridge request."""
        ib = self._connected_ib()
        contract = self._qualify_forex(symbol)
        request_async = getattr(ib, "reqHistoricalDataAsync", None)
        if request_async is None:
            # Compatibility fallback for an older ib_async build.
            return await asyncio.to_thread(self.get_historical_15m, symbol, outputsize)
        bars = await request_async(
            contract,
            endDateTime="",
            durationStr="7 D",
            barSizeSetting="15 mins",
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=2,
            keepUpToDate=False,
        )
        return self._serialize_bars(symbol, bars, outputsize)

    async def get_historical_15m_batch_async(
        self,
        symbols: list[str],
        outputsize: int = 420,
    ) -> dict[str, list[dict[str, Any]]]:
        """Concurrent IBKR 15m history for up to 12 unique FX symbols."""
        unique = list(dict.fromkeys(self.normalize_symbol(symbol) for symbol in symbols))
        if not unique or len(unique) > 12:
            raise ValueError("Provide 1 to 12 unique FX symbols.")

        results = await asyncio.gather(
            *(self.get_historical_15m_async(symbol, outputsize) for symbol in unique),
            return_exceptions=True,
        )

        rows: dict[str, list[dict[str, Any]]] = {}
        errors: list[str] = []
        for symbol, result in zip(unique, results):
            if isinstance(result, Exception):
                rows[symbol] = []
                errors.append(f"{symbol}: {type(result).__name__}: {result}")
            else:
                rows[symbol] = result

        if not any(rows.values()) and errors:
            raise RuntimeError("BROKER DATA UNAVAILABLE: no IBKR 15m batches returned. " + " | ".join(errors))
        return rows

    @staticmethod
    def validate_order_quantity(contract_info: dict[str, Any], quantity: float) -> dict[str, Any]:
        qty = float(quantity)
        min_size = float(contract_info.get("constraints", {}).get("min_size") or 0.0)
        increment = float(contract_info.get("constraints", {}).get("size_increment") or 0.0)
        if qty <= 0:
            return {"approved": False, "reason": "INVALID_ORDER_QUANTITY"}
        if min_size > 0 and qty < min_size:
            return {"approved": False, "reason": "BELOW_BROKER_MINIMUM_SIZE", "minimum_size": min_size}
        if increment > 0:
            steps = qty / increment
            if abs(steps - round(steps)) > 1e-9:
                return {
                    "approved": False,
                    "reason": "INVALID_BROKER_SIZE_INCREMENT",
                    "size_increment": increment,
                }
        return {"approved": True, "quantity": qty, "minimum_size": min_size, "size_increment": increment}

    def preview_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MKT",
        risk_approved: bool = False,
        signal_id: str | None = None,
        strategy_id: str = "forex-mtf-v1",
    ) -> dict[str, Any]:
        self._require_paper_mode()
        if not risk_approved:
            raise RuntimeError("EXECUTION BLOCKED: risk engine approval is required.")
        info = self.get_contract(symbol)
        size_check = self.validate_order_quantity(info, quantity)
        if not size_check["approved"]:
            return {
                "status": "NO TRADE",
                "reason": size_check["reason"],
                "broker": "INTERACTIVE_BROKERS",
                "execution_authorized": False,
            }
        ibi = self._library()
        ib = self._connected_ib()
        contract = self._qualify_forex(symbol)
        order = ibi.MarketOrder(str(side).upper(), float(quantity))
        order.orderRef = signal_id or strategy_id
        state = ib.whatIfOrder(contract, order)
        return {
            "status": "PAPER_PREVIEW",
            "broker": "INTERACTIVE_BROKERS",
            "symbol": self.normalize_symbol(symbol),
            "side": str(side).upper(),
            "quantity": float(quantity),
            "order_type": order_type,
            "signal_id": signal_id,
            "strategy_id": strategy_id,
            "margin_change": _finite_float(getattr(state, "initMarginChange", None)),
            "commission": _finite_float(getattr(state, "commission", None)),
            "warning_text": getattr(state, "warningText", ""),
            "execution_authorized": False,
        }

    def place_order(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # Deliberately hard-disabled in this implementation phase.
        raise RuntimeError(
            "PAPER ORDER BLOCKED: broker submission is not enabled. "
            "Complete connectivity, contract, quote, risk and preview tests first."
        )


_adapter = IBKRAdapter()


def get_ibkr_adapter() -> IBKRAdapter:
    return _adapter


def adapter_status() -> dict[str, Any]:
    return asdict(_adapter.connection_status())
