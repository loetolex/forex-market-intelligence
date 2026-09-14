from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IBKRStatus:
    connected: bool
    account: str | None
    market_data: bool
    orders_enabled: bool
    status: str


def read_only_status() -> IBKRStatus:
    return IBKRStatus(
        connected=False,
        account=None,
        market_data=False,
        orders_enabled=False,
        status="BROKER CONNECTION = HOLD",
    )


def assert_paper_only() -> None:
    raise RuntimeError(
        "PAPER/DEMO SAFETY: order placement is intentionally disabled "
        "in this first deployment."
    )
