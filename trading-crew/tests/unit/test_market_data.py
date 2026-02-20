"""Unit tests for multi-source market data acquisition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from models.schemas import OHLCVCandle
from tools.market_data import DataSourceError, MarketDataClient, MarketSnapshot


@dataclass(frozen=True)
class _FakeProvider:
    source_name: str
    supported_asset_classes: frozenset[str]
    snapshot: MarketSnapshot | None = None
    error: Exception | None = None

    async def fetch_ohlcv(
        self,
        *,
        asset: str,
        timeframe: str,
        limit: int,
    ) -> MarketSnapshot:
        if self.error is not None:
            raise self.error
        if self.snapshot is None:
            raise RuntimeError("missing snapshot")
        return self.snapshot


def _snapshot(source: str, close: float, minutes_ago: int = 1) -> MarketSnapshot:
    timestamp = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    return MarketSnapshot(
        asset="SPY",
        source=source,
        fetched_at=datetime.now(UTC),
        candles=[
            OHLCVCandle(
                timestamp=timestamp,
                open=close - 0.5,
                high=close + 0.5,
                low=close - 1.0,
                close=close,
                volume=1_000.0,
            )
        ],
    )


def test_market_data_client_falls_back_when_first_source_fails() -> None:
    provider_1 = _FakeProvider(
        source_name="source_1",
        supported_asset_classes=frozenset({"equity"}),
        error=DataSourceError("downstream timeout"),
    )
    provider_2 = _FakeProvider(
        source_name="source_2",
        supported_asset_classes=frozenset({"equity"}),
        snapshot=_snapshot("source_2", 101.0),
    )
    client = MarketDataClient(providers=[provider_1, provider_2], max_provider_failures=5)
    bundle = asyncio.run(
        client.get_trade_inputs(
            asset="SPY",
            timeframe="1h",
            limit=50,
            asset_class="equity",
            min_sources=1,
        )
    )
    assert bundle.primary.source == "source_2"


def test_market_data_client_computes_consensus_price_and_spread() -> None:
    provider_1 = _FakeProvider(
        source_name="source_1",
        supported_asset_classes=frozenset({"equity"}),
        snapshot=_snapshot("source_1", 100.0),
    )
    provider_2 = _FakeProvider(
        source_name="source_2",
        supported_asset_classes=frozenset({"equity"}),
        snapshot=_snapshot("source_2", 101.0),
    )
    client = MarketDataClient(providers=[provider_1, provider_2], max_provider_failures=5)
    bundle = asyncio.run(
        client.get_trade_inputs(
            asset="SPY",
            timeframe="1h",
            limit=50,
            asset_class="equity",
            min_sources=2,
        )
    )
    assert bundle.consensus_close == 100.5
    assert len(bundle.secondary) == 1
    assert bundle.price_spread_bps > 0.0


def test_market_data_client_skips_stale_snapshots() -> None:
    stale = MarketSnapshot(
        asset="SPY",
        source="stale_source",
        fetched_at=datetime.now(UTC),
        candles=[
            OHLCVCandle(
                timestamp=datetime.now(UTC) - timedelta(days=3),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1_000.0,
            )
        ],
    )
    provider_1 = _FakeProvider(
        source_name="stale_source",
        supported_asset_classes=frozenset({"equity"}),
        snapshot=stale,
    )
    provider_2 = _FakeProvider(
        source_name="fresh_source",
        supported_asset_classes=frozenset({"equity"}),
        snapshot=_snapshot("fresh_source", 102.0),
    )
    client = MarketDataClient(providers=[provider_1, provider_2], max_provider_failures=5)
    bundle = asyncio.run(
        client.get_trade_inputs(
            asset="SPY",
            timeframe="1h",
            limit=50,
            asset_class="equity",
            min_sources=1,
        )
    )
    assert bundle.primary.source == "fresh_source"
