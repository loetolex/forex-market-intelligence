from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SEC_TYPE_TO_ASSET_CLASS = {
    "STK": "STOCK",
    "IND": "INDEX",
    "OPT": "OPTION",
    "FUT": "FUTURE",
    "FOP": "FUTURE_OPTION",
    "CASH": "FX",
    "WAR": "WARRANT",
    "BOND": "BOND",
    "CMDTY": "COMMODITY",
    "FUND": "FUND",
    "CFD": "CFD",
    "BAG": "COMBO",
}

SUPPORTED_ASSET_CLASSES = {
    "FX",
    "STOCK",
    "ETF",
    "INDEX",
    "OPTION",
    "FUTURE",
    "FUTURE_OPTION",
    "BOND",
    "COMMODITY",
    "FUND",
    "WARRANT",
    "CFD",
    "COMBO",
    "CRYPTO",
    "UNKNOWN",
}


@dataclass(frozen=True)
class InstrumentRecord:
    asset_id: str
    asset_class: str
    symbol: str
    con_id: int | None
    sec_type: str | None
    exchange: str | None
    primary_exchange: str | None
    currency: str | None
    trading_class: str | None
    local_symbol: str | None
    expiry: str | None
    strike: float | None
    right: str | None
    multiplier: str | None
    description: str | None
    source: str
    contract_status: str
    market_data_status: str
    data_status: str
    research_enabled: bool
    paper_enabled: bool
    live_enabled: bool


def asset_class_from_sec_type(sec_type: str | None) -> str:
    value = str(sec_type or "").upper().strip()
    return SEC_TYPE_TO_ASSET_CLASS.get(value, "UNKNOWN")


def make_asset_id(
    asset_class: str,
    symbol: str,
    con_id: int | None = None,
) -> str:
    asset_class = str(asset_class or "UNKNOWN").upper().strip()
    symbol = str(symbol or "").upper().strip().replace("/", "")
    if not symbol:
        raise ValueError("INSTRUMENT REGISTRY: symbol is required.")

    if con_id is not None:
        return f"{asset_class}.CONID.{int(con_id)}"
    return f"{asset_class}.{symbol}"


def from_ibkr_contract(
    payload: dict[str, Any],
    *,
    source: str = "IBKR_CONTRACT_DISCOVERY",
) -> InstrumentRecord:
    sec_type = str(payload.get("sec_type") or "").upper() or None
    asset_class = asset_class_from_sec_type(sec_type)

    # IBKR crypto representation may differ by venue/security type. Until a
    # verified crypto contract is returned, do not infer CRYPTO from a symbol.
    symbol = str(payload.get("symbol") or "").upper().strip()
    if not symbol:
        raise ValueError("INSTRUMENT REGISTRY: IBKR contract is missing symbol.")

    con_id = payload.get("con_id")
    con_id = int(con_id) if con_id not in (None, "", 0) else None

    return InstrumentRecord(
        asset_id=make_asset_id(asset_class, symbol, con_id),
        asset_class=asset_class,
        symbol=symbol,
        con_id=con_id,
        sec_type=sec_type,
        exchange=payload.get("exchange"),
        primary_exchange=payload.get("primary_exchange"),
        currency=payload.get("currency"),
        trading_class=payload.get("trading_class"),
        local_symbol=payload.get("local_symbol"),
        expiry=payload.get("last_trade_date_or_contract_month"),
        strike=payload.get("strike"),
        right=payload.get("right"),
        multiplier=payload.get("multiplier"),
        description=payload.get("description"),
        source=source,
        contract_status="DISCOVERED" if con_id else "NOT_DISCOVERED",
        market_data_status="NOT_CHECKED",
        data_status=str(payload.get("data_status") or "DATA UNAVAILABLE"),
        research_enabled=False,
        paper_enabled=False,
        live_enabled=False,
    )


def record_to_dict(record: InstrumentRecord) -> dict[str, Any]:
    return asdict(record)


def validate_instrument_record(record: InstrumentRecord) -> dict[str, Any]:
    errors: list[str] = []

    if record.asset_class not in SUPPORTED_ASSET_CLASSES:
        errors.append(f"UNKNOWN_ASSET_CLASS:{record.asset_class}")

    if not record.symbol:
        errors.append("SYMBOL_MISSING")

    if record.con_id is None:
        errors.append("CONID_MISSING")

    if record.live_enabled:
        errors.append("LIVE_EXECUTION_MUST_REMAIN_DISABLED")

    if record.paper_enabled:
        errors.append("PAPER_EXECUTION_MUST_REMAIN_DISABLED")

    if record.research_enabled and record.market_data_status != "READY":
        errors.append("RESEARCH_REQUIRES_MARKET_DATA_READY")

    return {
        "status": "VALID" if not errors else "INVALID",
        "errors": errors,
        "asset_id": record.asset_id,
        "con_id": record.con_id,
        "asset_class": record.asset_class,
        "symbol": record.symbol,
    }


def merge_discovery_rows(
    existing: list[InstrumentRecord],
    discovered: list[dict[str, Any]],
) -> list[InstrumentRecord]:
    records: dict[str, InstrumentRecord] = {
        record.asset_id: record for record in existing
    }

    for row in discovered:
        record = from_ibkr_contract(row)
        records[record.asset_id] = record

    return list(records.values())
