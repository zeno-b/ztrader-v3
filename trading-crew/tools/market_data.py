"""Market data layer with multi-source fallback and source consensus."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import fmean
from typing import Any, Collection, Protocol, Sequence

import aiohttp
from loguru import logger

from models.schemas import AssetClass, OHLCVCandle, Timeframe


@dataclass(frozen=True)
class MarketSnapshot:
    """Market snapshot fetched from one data source."""

    asset: str
    source: str
    fetched_at: datetime
    candles: list[OHLCVCandle]


@dataclass(frozen=True)
class TradeInputBundle:
    """Multi-source market input used for trade decisions."""

    asset: str
    timeframe: Timeframe
    primary: MarketSnapshot
    secondary: list[MarketSnapshot]
    consensus_close: float
    price_spread_bps: float


@dataclass(frozen=True)
class DataFreshnessPolicy:
    """Freshness thresholds for intraday and swing-style timeframes."""

    intraday_max_age_minutes: int = 15
    swing_max_age_days: int = 1

    def max_age_for_timeframe(self, timeframe: Timeframe) -> timedelta:
        if timeframe in {"1m", "5m", "15m", "1h", "4h"}:
            return timedelta(minutes=self.intraday_max_age_minutes)
        return timedelta(days=self.swing_max_age_days)

    def is_stale(self, snapshot: MarketSnapshot, timeframe: Timeframe) -> bool:
        if not snapshot.candles:
            return True
        latest = max(candle.timestamp for candle in snapshot.candles)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=UTC)
        age = datetime.now(UTC) - latest
        return age > self.max_age_for_timeframe(timeframe)


class DataSourceError(RuntimeError):
    """Typed source failure with retryability hint."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class MarketDataProvider(Protocol):
    """Provider contract for external market data sources."""

    @property
    def source_name(self) -> str:
        """Human-readable unique source name."""
        ...

    @property
    def supported_asset_classes(self) -> Collection[AssetClass]:
        """Supported asset classes for this provider."""
        ...

    async def fetch_ohlcv(
        self,
        *,
        asset: str,
        timeframe: Timeframe,
        limit: int,
    ) -> MarketSnapshot:
        """Fetch OHLCV candles for one asset."""


@dataclass
class _ProviderState:
    failures: int = 0
    circuit_open_until: datetime | None = None


@dataclass(frozen=True)
class AlpacaMarketDataProvider:
    """Alpaca market data provider for equities and ETFs."""

    api_key: str
    api_secret: str
    base_url: str = "https://data.alpaca.markets"
    source_name: str = "alpaca"
    supported_asset_classes: frozenset[AssetClass] = frozenset({"equity", "etf"})

    async def fetch_ohlcv(
        self,
        *,
        asset: str,
        timeframe: Timeframe,
        limit: int,
    ) -> MarketSnapshot:
        if not self.api_key or not self.api_secret:
            raise DataSourceError("Alpaca credentials are missing.", retryable=False)

        timeframe_map: dict[Timeframe, str] = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "1h": "1Hour",
            "4h": "4Hour",
            "1d": "1Day",
        }
        mapped = timeframe_map.get(timeframe)
        if mapped is None:
            raise DataSourceError(f"Unsupported Alpaca timeframe: {timeframe}", retryable=False)

        url = f"{self.base_url.rstrip('/')}/v2/stocks/{asset}/bars"
        params = {"timeframe": mapped, "limit": str(limit), "adjustment": "raw"}
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status >= 400:
                    raise DataSourceError(f"Alpaca returned status {response.status}", retryable=True)
                payload = await response.json()

        bars = payload.get("bars", [])
        candles = [_alpaca_bar_to_candle(item) for item in bars]
        if not candles:
            raise DataSourceError("Alpaca returned empty bars.", retryable=True)
        return MarketSnapshot(
            asset=asset,
            source=self.source_name,
            fetched_at=datetime.now(UTC),
            candles=candles,
        )


@dataclass(frozen=True)
class YahooFinanceMarketDataProvider:
    """Yahoo chart endpoint market data provider."""

    source_name: str = "yahoo_chart"
    supported_asset_classes: frozenset[AssetClass] = frozenset(
        {"equity", "etf", "crypto", "fx"}
    )

    async def fetch_ohlcv(
        self,
        *,
        asset: str,
        timeframe: Timeframe,
        limit: int,
    ) -> MarketSnapshot:
        interval_map: dict[Timeframe, str] = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "60m",
            "4h": "60m",
            "1d": "1d",
        }
        interval = interval_map.get(timeframe)
        if interval is None:
            raise DataSourceError(f"Unsupported Yahoo timeframe: {timeframe}", retryable=False)

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{asset}"
        params = {
            "interval": interval,
            "range": _range_for_request(timeframe=timeframe, limit=limit),
            "includePrePost": "false",
        }
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status >= 400:
                    raise DataSourceError(f"Yahoo returned status {response.status}", retryable=True)
                payload = await response.json()

        result = payload.get("chart", {}).get("result")
        if not result:
            raise DataSourceError("Yahoo chart payload missing result.", retryable=True)
        first_result = result[0]
        timestamps = first_result.get("timestamp", [])
        quote = first_result.get("indicators", {}).get("quote", [{}])[0]
        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])
        candles: list[OHLCVCandle] = []
        for values in zip(timestamps, opens, highs, lows, closes, volumes, strict=False):
            ts_value, open_value, high_value, low_value, close_value, volume_value = values
            if None in {open_value, high_value, low_value, close_value, volume_value}:
                continue
            candles.append(
                OHLCVCandle(
                    timestamp=datetime.fromtimestamp(int(ts_value), tz=UTC),
                    open=float(open_value),
                    high=float(high_value),
                    low=float(low_value),
                    close=float(close_value),
                    volume=float(volume_value),
                )
            )
        if not candles:
            raise DataSourceError("Yahoo returned no complete candles.", retryable=True)
        return MarketSnapshot(
            asset=asset,
            source=self.source_name,
            fetched_at=datetime.now(UTC),
            candles=candles[-limit:],
        )


@dataclass(frozen=True)
class CCXTMarketDataProvider:
    """CCXT market data provider for crypto exchanges."""

    exchange_id: str
    sandbox: bool = False
    supported_asset_classes: frozenset[AssetClass] = frozenset({"crypto"})

    @property
    def source_name(self) -> str:
        return f"ccxt_{self.exchange_id}"

    async def fetch_ohlcv(
        self,
        *,
        asset: str,
        timeframe: Timeframe,
        limit: int,
    ) -> MarketSnapshot:
        try:
            import ccxt.async_support as ccxt_async
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise DataSourceError("ccxt dependency is unavailable.", retryable=False) from exc

        exchange_class = getattr(ccxt_async, self.exchange_id, None)
        if exchange_class is None:
            raise DataSourceError(
                f"Unsupported ccxt exchange: {self.exchange_id}",
                retryable=False,
            )
        exchange = exchange_class({"enableRateLimit": True})
        if self.sandbox and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)
        try:
            rows = await exchange.fetch_ohlcv(symbol=asset, timeframe=timeframe, limit=limit)
        except Exception as exc:
            raise DataSourceError(f"ccxt {self.exchange_id} fetch failed: {exc}", retryable=True) from exc
        finally:
            await exchange.close()

        candles = [_ccxt_row_to_candle(row) for row in rows]
        if not candles:
            raise DataSourceError(f"ccxt {self.exchange_id} returned no candles.", retryable=True)
        return MarketSnapshot(
            asset=asset,
            source=self.source_name,
            fetched_at=datetime.now(UTC),
            candles=candles,
        )


class MarketDataClient:
    """Fetch market data from multiple providers with fallback and consensus."""

    def __init__(
        self,
        *,
        providers: Sequence[MarketDataProvider] | None = None,
        freshness_policy: DataFreshnessPolicy | None = None,
        max_provider_failures: int = 3,
        circuit_cooldown_seconds: int = 120,
        provider_timeout_seconds: int = 20,
        alpaca_api_key: str = "",
        alpaca_api_secret: str = "",
        alpaca_base_url: str = "https://data.alpaca.markets",
    ) -> None:
        if providers is None:
            self._providers: list[MarketDataProvider] = [
                AlpacaMarketDataProvider(
                    api_key=alpaca_api_key,
                    api_secret=alpaca_api_secret,
                    base_url=alpaca_base_url,
                ),
                YahooFinanceMarketDataProvider(),
                CCXTMarketDataProvider(exchange_id="binance"),
                CCXTMarketDataProvider(exchange_id="kraken"),
            ]
        else:
            self._providers = list(providers)

        self._freshness_policy = freshness_policy or DataFreshnessPolicy()
        self._max_provider_failures = max_provider_failures
        self._circuit_cooldown_seconds = circuit_cooldown_seconds
        self._provider_timeout_seconds = provider_timeout_seconds
        self._provider_state: dict[str, _ProviderState] = {
            provider.source_name: _ProviderState() for provider in self._providers
        }

    async def get_ohlcv(
        self,
        asset: str,
        timeframe: Timeframe,
        limit: int = 200,
        asset_class: AssetClass = "equity",
    ) -> MarketSnapshot:
        """Return the primary snapshot selected from provider consensus."""

        bundle = await self.get_trade_inputs(
            asset=asset,
            timeframe=timeframe,
            limit=limit,
            asset_class=asset_class,
            min_sources=1,
        )
        return bundle.primary

    async def get_trade_inputs(
        self,
        *,
        asset: str,
        timeframe: Timeframe,
        limit: int = 200,
        asset_class: AssetClass,
        min_sources: int = 1,
    ) -> TradeInputBundle:
        """Fetch trading inputs from multiple sources and compute consensus."""

        providers = [provider for provider in self._providers if asset_class in provider.supported_asset_classes]
        if not providers:
            raise RuntimeError(f"No providers configured for asset_class={asset_class}.")

        successful: list[MarketSnapshot] = []
        failures: list[str] = []
        for provider in providers:
            if self._is_circuit_open(provider.source_name):
                failures.append(f"{provider.source_name}: circuit_open")
                continue
            try:
                snapshot = await asyncio.wait_for(
                    provider.fetch_ohlcv(asset=asset, timeframe=timeframe, limit=limit),
                    timeout=self._provider_timeout_seconds,
                )
                if self._freshness_policy.is_stale(snapshot, timeframe):
                    raise DataSourceError("snapshot is stale", retryable=True)
                successful.append(snapshot)
                self._register_success(provider.source_name)
            except Exception as exc:
                reason = str(exc)
                failures.append(f"{provider.source_name}: {reason}")
                logger.warning("market_source_failed", source=provider.source_name, reason=reason)
                self._register_failure(provider.source_name)

        if len(successful) < min_sources:
            joined = "; ".join(failures) if failures else "unknown failure"
            raise RuntimeError(f"Unable to gather market data from enough sources: {joined}")

        closes = [snapshot.candles[-1].close for snapshot in successful]
        consensus_close = fmean(closes)
        if len(closes) <= 1 or consensus_close == 0.0:
            spread_bps = 0.0
        else:
            spread_bps = ((max(closes) - min(closes)) / consensus_close) * 10_000

        primary = successful[0]
        secondary = successful[1:]
        return TradeInputBundle(
            asset=asset,
            timeframe=timeframe,
            primary=primary,
            secondary=secondary,
            consensus_close=consensus_close,
            price_spread_bps=spread_bps,
        )

    def _is_circuit_open(self, source_name: str) -> bool:
        state = self._provider_state[source_name]
        if state.circuit_open_until is None:
            return False
        return datetime.now(UTC) < state.circuit_open_until

    def _register_success(self, source_name: str) -> None:
        self._provider_state[source_name] = _ProviderState()

    def _register_failure(self, source_name: str) -> None:
        previous = self._provider_state[source_name]
        failures = previous.failures + 1
        open_until: datetime | None = previous.circuit_open_until
        if failures >= self._max_provider_failures:
            open_until = datetime.now(UTC) + timedelta(seconds=self._circuit_cooldown_seconds)
            failures = 0
        self._provider_state[source_name] = _ProviderState(
            failures=failures,
            circuit_open_until=open_until,
        )


def _alpaca_bar_to_candle(payload: dict[str, Any]) -> OHLCVCandle:
    timestamp_raw = payload.get("t")
    if not isinstance(timestamp_raw, str):
        raise DataSourceError("Alpaca bar timestamp missing.", retryable=True)
    timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
    return OHLCVCandle(
        timestamp=timestamp,
        open=float(payload["o"]),
        high=float(payload["h"]),
        low=float(payload["l"]),
        close=float(payload["c"]),
        volume=float(payload["v"]),
    )


def _ccxt_row_to_candle(row: list[int | float]) -> OHLCVCandle:
    if len(row) < 6:
        raise DataSourceError("Malformed ccxt OHLCV row.", retryable=True)
    return OHLCVCandle(
        timestamp=datetime.fromtimestamp(int(row[0]) / 1000, tz=UTC),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def _range_for_request(*, timeframe: Timeframe, limit: int) -> str:
    minutes_per_candle: dict[Timeframe, int] = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }
    total_minutes = minutes_per_candle[timeframe] * limit
    if total_minutes <= 7 * 24 * 60:
        return "7d"
    if total_minutes <= 30 * 24 * 60:
        return "1mo"
    if total_minutes <= 90 * 24 * 60:
        return "3mo"
    if total_minutes <= 365 * 24 * 60:
        return "1y"
    return "5y"
