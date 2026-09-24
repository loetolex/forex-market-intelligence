from app.market_universe.registry import (
    InstrumentRecord,
    asset_class_from_sec_type,
    from_ibkr_contract,
    make_asset_id,
    validate_instrument_record,
)


def test_asset_class_mapping():
    assert asset_class_from_sec_type("STK") == "STOCK"
    assert asset_class_from_sec_type("IND") == "INDEX"
    assert asset_class_from_sec_type("FUT") == "FUTURE"
    assert asset_class_from_sec_type("OPT") == "OPTION"
    assert asset_class_from_sec_type("CASH") == "FX"


def test_asset_id_is_conid_stable_for_discovered_contracts():
    assert make_asset_id("STOCK", "AAPL", 12345) == "STOCK.CONID.12345"


def test_discovered_contract_is_research_locked_until_data_validation():
    record = from_ibkr_contract(
        {
            "con_id": 12345,
            "symbol": "AAPL",
            "sec_type": "STK",
            "exchange": "SMART",
            "primary_exchange": "NASDAQ",
            "currency": "USD",
            "trading_class": "NMS",
            "local_symbol": "AAPL",
            "data_status": "REAL_BROKER_DATA",
        }
    )

    assert record.asset_class == "STOCK"
    assert record.contract_status == "DISCOVERED"
    assert record.market_data_status == "NOT_CHECKED"
    assert record.research_enabled is False
    assert record.paper_enabled is False
    assert record.live_enabled is False


def test_validation_rejects_execution_enablement():
    record = InstrumentRecord(
        asset_id="STOCK.CONID.12345",
        asset_class="STOCK",
        symbol="AAPL",
        con_id=12345,
        sec_type="STK",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        trading_class="NMS",
        local_symbol="AAPL",
        expiry=None,
        strike=None,
        right=None,
        multiplier=None,
        description=None,
        source="TEST",
        contract_status="DISCOVERED",
        market_data_status="NOT_CHECKED",
        data_status="REAL_BROKER_DATA",
        research_enabled=False,
        paper_enabled=False,
        live_enabled=False,
    )

    result = validate_instrument_record(record)
    assert result["status"] == "VALID"
    assert result["errors"] == []
