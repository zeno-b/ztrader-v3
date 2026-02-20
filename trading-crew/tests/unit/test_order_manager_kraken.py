"""Unit tests for Kraken order manager behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from tools.order_manager import KrakenOrderManager, OrderRequest


@dataclass
class _FakeKrakenClient:
    """Deterministic fake async Kraken client for tests."""

    should_timeout: bool = False
    captured: dict[str, Any] | None = None

    async def create_order(
        self,
        *,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: float | None,
    ) -> dict[str, str]:
        self.captured = {
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
        }
        if self.should_timeout:
            raise TimeoutError("temporary upstream timeout")
        return {"id": "kraken-123"}

    async def close(self) -> None:
        return None


def test_kraken_manager_places_market_order() -> None:
    client = _FakeKrakenClient()
    manager = KrakenOrderManager(
        api_key="key",
        api_secret="secret",
        sandbox=True,
        client=client,
    )
    response = asyncio.run(
        manager.place_order(
            OrderRequest(
                symbol="BTC/USD",
                side="buy",
                quantity=0.01,
                order_type="market",
                exchange="kraken",
            )
        )
    )
    assert response.accepted is True
    assert response.order_id == "kraken-123"
    assert client.captured is not None
    assert client.captured["symbol"] == "BTC/USD"


def test_kraken_manager_rejects_limit_without_price() -> None:
    client = _FakeKrakenClient()
    manager = KrakenOrderManager(
        api_key="key",
        api_secret="secret",
        sandbox=True,
        client=client,
    )
    response = asyncio.run(
        manager.place_order(
            OrderRequest(
                symbol="BTC/USD",
                side="sell",
                quantity=0.01,
                order_type="limit",
                exchange="kraken",
            )
        )
    )
    assert response.accepted is False
    assert response.retryable is False
    assert "requires explicit price" in response.reason


def test_kraken_manager_flags_retryable_transient_errors() -> None:
    client = _FakeKrakenClient(should_timeout=True)
    manager = KrakenOrderManager(
        api_key="key",
        api_secret="secret",
        sandbox=True,
        client=client,
    )
    response = asyncio.run(
        manager.place_order(
            OrderRequest(
                symbol="ETH/USD",
                side="buy",
                quantity=0.02,
                order_type="market",
                exchange="kraken",
            )
        )
    )
    assert response.accepted is False
    assert response.retryable is True
    assert "Transient Kraken API error" in response.reason
