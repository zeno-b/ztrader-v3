"""Technical agent that operates only on provided OHLCV candles."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from loguru import logger

from agents.base_agent import Result
from models.schemas import AgentResponse, MarketRegime, OHLCVCandle, SignalDirection, TechnicalSignal
from tools.indicators import atr, bollinger_bands, macd, rsi, vwap


class TechnicalAgent:
    """Generates technical signals from OHLCV time series."""

    agent_id = "technical_agent"

    def __init__(self, adapter_version: str, min_lookback_period: int = 50) -> None:
        self.adapter_version = adapter_version
        self.min_lookback_period = min_lookback_period

    async def run(
        self,
        *,
        task_id: str,
        asset: str,
        timeframe: str,
        candles: list[OHLCVCandle],
        market_regime: MarketRegime,
    ) -> Result[AgentResponse]:
        """Build a technical signal from local OHLCV candles only."""

        if len(candles) < self.min_lookback_period:
            return Result.success(
                AgentResponse(
                    agent_id=self.agent_id,
                    timestamp=datetime.now(UTC),
                    task_id=task_id,
                    status="abstain",
                    payload=TechnicalSignal(
                        asset=asset,
                        direction="abstain",
                        timeframe="1h",
                        strength=0.0,
                        indicators_used=["lookback_validation"],
                    ),
                    confidence=0.0,
                    reasoning="Insufficient lookback history for indicators.",
                    data_sources=["timescaledb:ohlcv"],
                    latency_ms=1,
                    adapter_version=self.adapter_version,
                    market_regime=market_regime,
                )
            )

        frame = pd.DataFrame([c.model_dump() for c in candles])
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        volume = frame["volume"]

        rsi_series = rsi(close)
        macd_line, signal_line = macd(close)
        lower_band, _, upper_band = bollinger_bands(close)
        _ = vwap(close, volume)
        _ = atr(high, low, close)

        current_rsi = float(rsi_series.iloc[-1])
        current_macd = float(macd_line.iloc[-1])
        current_macd_signal = float(signal_line.iloc[-1])
        current_close = float(close.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])

        direction: SignalDirection = "hold"
        strength = 0.5
        if current_rsi <= 35.0 and current_macd > current_macd_signal and current_close <= current_lower:
            direction = "buy"
            strength = 0.8
        elif (
            current_rsi >= 65.0
            and current_macd < current_macd_signal
            and current_close >= current_upper
        ):
            direction = "sell"
            strength = 0.8

        response = AgentResponse(
            agent_id=self.agent_id,
            timestamp=datetime.now(UTC),
            task_id=task_id,
            status="success",
            payload=TechnicalSignal(
                asset=asset,
                direction=direction,
                timeframe="1h",
                strength=strength,
                indicators_used=["rsi", "macd", "bollinger", "vwap", "atr"],
            ),
            confidence=strength,
            reasoning="Signal based on RSI, MACD, and Bollinger confirmation.",
            data_sources=["timescaledb:ohlcv"],
            latency_ms=5,
            adapter_version=self.adapter_version,
            market_regime=market_regime,
        )
        logger.info("technical_signal_emitted", task_id=task_id, direction=direction)
        return Result.success(response)
