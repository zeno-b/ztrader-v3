"""Order manager abstractions used by execution agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderRequest:
    """Order request payload."""

    symbol: str
    side: str
    quantity: float
    order_type: str


@dataclass(frozen=True)
class OrderResponse:
    """Order placement result payload."""

    accepted: bool
    order_id: str | None
    reason: str
