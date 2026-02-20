"""Market data abstraction with source fallback semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from models.schemas import OHLCVCandle


@dataclass(frozen=True)
class MarketSnapshot:
    """Market snapshot fetched from a data source."""

    asset: str
    source: str
    fetched_at: datetime
    candles: list[OHLCVCandle]


class MarketDataClient:
    """Source facade for Alpaca, Yahoo Finance, and CCXT providers."""

    async def get_ohlcv(self, asset: str, timeframe: str, limit: int = 200) -> MarketSnapshot:
        """
        Fetch OHLCV candles.

        This scaffold intentionally omits external API integration details.
        """

        raise NotImplementedError("Integrate Alpaca/Yahoo/CCXT provider clients.")
