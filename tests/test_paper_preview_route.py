from fastapi.testclient import TestClient

from app import main


def test_encoded_fx_preview_route_accepts_slash_and_never_authorizes_execution(monkeypatch):
    seen = {}

    def fake_preview(symbol, timeframe):
        seen["symbol"] = symbol
        seen["timeframe"] = timeframe
        return {
            "status": "PAPER_PREVIEW",
            "execution_gate": {"approved": True},
            "execution_authorized": True,
        }

    monkeypatch.setattr(main, "controlled_test_preview", fake_preview)

    with TestClient(main.app) as client:
        response = client.get("/paper/preview/EUR%2FUSD?timeframe=15m")

    assert response.status_code == 200
    assert seen == {"symbol": "EUR/USD", "timeframe": "15m"}
    payload = response.json()
    assert payload["execution_gate"]["approved"] is True
    assert payload["execution_authorized"] is False
    assert payload["execution_status"] == "PREVIEW_ONLY"
    assert payload["preview_only"] is True


def test_fixed_paper_routes_are_not_shadowed_by_symbol_path_converter(monkeypatch):
    monkeypatch.setattr(main, "paper_order_status", lambda: {"status": "NO_ACTIVE_TEST", "execution_authorized": False})

    with TestClient(main.app) as client:
        response = client.get("/paper/order-status")

    assert response.status_code == 200
    assert response.json()["status"] == "NO_ACTIVE_TEST"
