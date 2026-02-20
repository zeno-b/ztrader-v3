"""Order manager abstractions used by execution agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from loguru import logger

ExchangeName = Literal["alpaca", "kraken"]
OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


@dataclass(frozen=True)
class OrderRequest:
    """Order request payload."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    exchange: ExchangeName
    price: float | None = None


@dataclass(frozen=True)
class OrderResponse:
    """Order placement result payload."""

    accepted: bool
    order_id: str | None
    reason: str
    retryable: bool = False


class OrderManager(Protocol):
    """Protocol for broker and exchange order placement clients."""

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place an order and return typed response."""


@dataclass(frozen=True)
class PaperOrderManager:
    """Paper-only order manager used by default."""

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Simulate an accepted paper order."""

        return OrderResponse(
            accepted=True,
            order_id=f"paper-{request.exchange}-{request.symbol}",
            reason="Paper order simulated.",
            retryable=False,
        )


class KrakenOrderManager:
    """
    Kraken exchange order manager using ccxt async client.

    The manager supports dependency injection of a preconfigured client to keep
    tests deterministic.
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        sandbox: bool,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._sandbox = sandbox
        self._client: Any | None = client

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place a Kraken order using ccxt create_order."""

        if request.exchange != "kraken":
            return OrderResponse(
                accepted=False,
                order_id=None,
                reason="Kraken manager received non-Kraken exchange request.",
                retryable=False,
            )
        if request.quantity <= 0:
            return OrderResponse(
                accepted=False,
                order_id=None,
                reason="Order quantity must be positive.",
                retryable=False,
            )
        if request.order_type == "limit" and request.price is None:
            return OrderResponse(
                accepted=False,
                order_id=None,
                reason="Limit order requires explicit price.",
                retryable=False,
            )

        client = self._get_or_create_client()
        try:
            created = await client.create_order(
                symbol=request.symbol,
                type=request.order_type,
                side=request.side,
                amount=request.quantity,
                price=request.price,
            )
            order_id = str(created.get("id")) if created.get("id") is not None else None
            return OrderResponse(
                accepted=order_id is not None,
                order_id=order_id,
                reason="Order accepted by Kraken." if order_id else "Kraken did not return order id.",
                retryable=False,
            )
        except (TimeoutError, ConnectionError) as exc:
            logger.error("kraken_transient_error", error=str(exc))
            return OrderResponse(
                accepted=False,
                order_id=None,
                reason=f"Transient Kraken API error: {exc}",
                retryable=True,
            )
        except Exception as exc:
            logger.error("kraken_order_error", error=str(exc))
            return OrderResponse(
                accepted=False,
                order_id=None,
                reason=f"Kraken order rejected: {exc}",
                retryable=False,
            )

    async def close(self) -> None:
        """Close ccxt client if it exposes async close()."""

        if self._client is None:
            return
        close_fn = getattr(self._client, "close", None)
        if close_fn is not None:
            await close_fn()

    def _get_or_create_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import ccxt.async_support as ccxt_async
        except ImportError as exc:
            raise RuntimeError("ccxt is required for Kraken trading support.") from exc

        self._client = ccxt_async.kraken(
            {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
            }
        )
        if self._sandbox:
            self._client.set_sandbox_mode(True)
        return self._client


def build_live_order_manager(
    *,
    exchange: ExchangeName,
    kraken_api_key: str,
    kraken_api_secret: str,
    kraken_sandbox: bool,
) -> OrderManager:
    """Build live-trading order manager for supported exchanges."""

    if exchange == "kraken":
        return KrakenOrderManager(
            api_key=kraken_api_key,
            api_secret=kraken_api_secret,
            sandbox=kraken_sandbox,
        )
    raise ValueError(f"Unsupported live exchange for this build: {exchange}")
