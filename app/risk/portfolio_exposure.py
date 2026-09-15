from __future__ import annotations

from typing import Any


def _position_symbol(position: Any) -> str:
    contract = getattr(position, "contract", None)
    local_symbol = str(getattr(contract, "localSymbol", "") or "")
    if local_symbol:
        return local_symbol.replace(".", "/").upper()
    symbol = str(getattr(contract, "symbol", "") or "").upper()
    currency = str(getattr(contract, "currency", "") or "").upper()
    return f"{symbol}/{currency}" if symbol and currency else symbol


def evaluate_symbol_exposure(
    symbol: str,
    side: str,
    positions: list[Any] | None,
    *,
    allow_opposing: bool = False,
) -> dict[str, Any]:
    """Deterministically gate a new opportunity against verified broker positions.

    ``positions=None`` means broker position data was not available and therefore
    cannot be treated as an empty account. No account balance is inferred here.
    """
    normalized = str(symbol).upper().replace(".", "/").replace("_", "/").replace("-", "/")
    side = str(side).upper()
    if side not in {"LONG", "SHORT"}:
        return {"approved": False, "reason": "INVALID_SIDE", "status": "CALCULATED"}
    if positions is None:
        return {"approved": False, "reason": "DATA UNAVAILABLE: broker positions could not be verified.", "status": "DATA UNAVAILABLE"}

    matches: list[dict[str, Any]] = []
    for position in positions:
        position_symbol = _position_symbol(position)
        if position_symbol != normalized:
            continue
        quantity = float(getattr(position, "position", 0.0) or 0.0)
        if quantity == 0:
            continue
        position_side = "LONG" if quantity > 0 else "SHORT"
        matches.append({"side": position_side, "quantity": abs(quantity)})

    if not matches:
        return {"approved": True, "reason": "NO_EXISTING_SYMBOL_EXPOSURE", "status": "CALCULATED", "existing_exposure": []}
    if any(item["side"] == side for item in matches):
        return {"approved": False, "reason": "EXISTING_SAME_DIRECTION_EXPOSURE", "status": "CALCULATED", "existing_exposure": matches}
    if not allow_opposing:
        return {"approved": False, "reason": "OPPOSING_SYMBOL_EXPOSURE_BLOCKED", "status": "CALCULATED", "existing_exposure": matches}
    return {"approved": True, "reason": "OPPOSING_EXPOSURE_POLICY_ALLOWED", "status": "CALCULATED", "existing_exposure": matches}
