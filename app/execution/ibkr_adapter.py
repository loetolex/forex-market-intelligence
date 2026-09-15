from __future__ import annotations

from dataclasses import asdict, dataclass
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
        ib = self._connected_ib()
        contract = self._qualify_forex(symbol)
        ticker = ib.reqTickers(contract)[0]
        market_price = _finite_float(ticker.marketPrice())
        return {
            "status": "REAL_BROKER_DATA",
            "broker": "INTERACTIVE_BROKERS",
            "symbol": self.normalize_symbol(symbol),
            "timestamp": ticker.time.isoformat() if ticker.time else None,
            "bid": _finite_float(ticker.bid),
            "ask": _finite_float(ticker.ask),
            "last": _finite_float(ticker.last),
            "close": _finite_float(ticker.close),
            "market_price": market_price,
            "market_data_type": int(ticker.marketDataType) if ticker.marketDataType is not None else None,
            "data_note": "REAL_BROKER_DATA; null means IBKR did not provide a finite value for that field.",
        }

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
                "instrument": self.normalize_symbol(symbol),
                "timeframe": "15m",
                "data_status": "REAL_DATA",
            }
            for bar in rows
        ]

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
