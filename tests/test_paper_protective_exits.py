from __future__ import annotations

from types import SimpleNamespace

from app.paper_trading import _build_protective_orders


class _FakeClient:
    def __init__(self):
        self._next = 100

    def getReqId(self):
        value = self._next
        self._next += 1
        return value


class _FakeIB:
    def __init__(self):
        self.client = _FakeClient()


def _ibi():
    class Order:
        def __init__(self, action, quantity):
            self.action = action
            self.totalQuantity = quantity
            self.orderId = 0
            self.parentId = 0
            self.transmit = True
            self.orderRef = ""
            self.ocaGroup = ""
            self.ocaType = 0

    class MarketOrder(Order):
        pass

    class LimitOrder(Order):
        def __init__(self, action, quantity, price):
            super().__init__(action, quantity)
            self.lmtPrice = price

    class StopOrder(Order):
        def __init__(self, action, quantity, price):
            super().__init__(action, quantity)
            self.auxPrice = price

    return SimpleNamespace(MarketOrder=MarketOrder, LimitOrder=LimitOrder, StopOrder=StopOrder)


def test_long_bracket_has_correct_side_and_geometry():
    parent, take_profit, stop_loss = _build_protective_orders(
        _ibi(),
        _FakeIB(),
        "BUY",
        20_000,
        {"stop_loss": 1.0990, "take_profit": 1.1015},
        "paper-auto-test",
    )

    assert parent.action == "BUY"
    assert parent.transmit is False
    assert take_profit.action == "SELL"
    assert take_profit.lmtPrice == 1.1015
    assert stop_loss.action == "SELL"
    assert stop_loss.auxPrice == 1.0990
    assert take_profit.parentId == parent.orderId
    assert stop_loss.parentId == parent.orderId
    assert take_profit.transmit is False
    assert stop_loss.transmit is True
    assert take_profit.ocaGroup == stop_loss.ocaGroup
    assert take_profit.ocaType == 1
    assert stop_loss.ocaType == 1


def test_short_bracket_reverses_protective_sides():
    parent, take_profit, stop_loss = _build_protective_orders(
        _ibi(),
        _FakeIB(),
        "SELL",
        20_000,
        {"stop_loss": 1.1010, "take_profit": 1.0985},
        "paper-auto-test",
    )

    assert parent.action == "SELL"
    assert take_profit.action == "BUY"
    assert take_profit.lmtPrice == 1.0985
    assert stop_loss.action == "BUY"
    assert stop_loss.auxPrice == 1.1010
    assert take_profit.parentId == parent.orderId
    assert stop_loss.parentId == parent.orderId
