"""Typed domain models for trading and training pipelines."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypeAlias
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

MarketRegime: TypeAlias = Literal[
    "trending_bull",
    "trending_bear",
    "mean_reverting",
    "high_volatility",
]

AgentStatus: TypeAlias = Literal["success", "abstain", "error"]
SignalDirection: TypeAlias = Literal["buy", "sell", "hold", "abstain"]
Timeframe: TypeAlias = Literal["1m", "5m", "15m", "1h", "4h", "1d"]
AssetClass: TypeAlias = Literal["equity", "crypto", "etf", "fx", "other"]


class BaseSignal(BaseModel):
    """Common signal payload used by all agents."""

    asset: str = Field(min_length=1)
    direction: SignalDirection
    timeframe: Timeframe = "1h"


class OHLCVCandle(BaseModel):
    """Single OHLCV candle used by technical computations."""

    timestamp: datetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)


class SentimentSignal(BaseSignal):
    """Signal produced by research agent."""

    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)


class TechnicalSignal(BaseSignal):
    """Signal produced by technical agent."""

    strength: float = Field(ge=0.0, le=1.0)
    indicators_used: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Risk gate output with veto semantics."""

    approved: bool
    reason: str
    adjusted_size: float = Field(ge=0.0, le=1.0)


class RiskContext(BaseModel):
    """Inputs required by the risk veto agent."""

    portfolio_value: float = Field(gt=0.0)
    proposed_position_value: float = Field(ge=0.0)
    current_daily_drawdown_pct: float = Field(ge=0.0)
    sector_exposure_pct: float = Field(ge=0.0, le=1.0)
    minutes_to_major_event: int
    instrument_history_days: int = Field(ge=0)


class TradeDecision(BaseModel):
    """Final coordinator decision emitted before execution."""

    task_id: str = Field(min_length=1)
    asset: str = Field(min_length=1)
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    approved: bool
    veto_reason: str | None = None
    position_size: float = Field(ge=0.0, le=1.0)
    weighted_votes: dict[str, float] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Strict response contract for all agent outputs."""

    agent_id: str = Field(min_length=1)
    timestamp: datetime
    task_id: str = Field(min_length=1)
    status: AgentStatus
    payload: BaseSignal
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
    data_sources: list[str] = Field(default_factory=list)
    latency_ms: int = Field(ge=0)
    adapter_version: str = Field(min_length=1)
    market_regime: MarketRegime


class DecisionLogRecord(BaseModel):
    """Immutable decision log entry used for retraining datasets."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    agent_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    asset: str = Field(min_length=1)
    asset_class: AssetClass = "other"
    timeframe: Timeframe = "1h"
    signal_type: str = Field(min_length=1)
    signal_value: BaseSignal
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    data_sources: list[str] = Field(default_factory=list)
    market_regime: MarketRegime
    outcome_pnl: float | None = None
    outcome_latency_days: int | None = None
    contributed_to_trade: bool = False
    trade_was_profitable: bool | None = None

    @field_validator("trade_was_profitable")
    @classmethod
    def validate_outcome_pair(
        cls, value: bool | None, info: object
    ) -> bool | None:
        """Require profitability label whenever outcome PnL is present."""

        if not hasattr(info, "data"):
            return value
        data = getattr(info, "data")
        outcome_pnl = data.get("outcome_pnl")
        if outcome_pnl is not None and value is None:
            raise ValueError("trade_was_profitable must be set when outcome_pnl exists")
        return value


class TrainingPairMetadata(BaseModel):
    """Metadata for each JSONL training pair."""

    regime: MarketRegime
    agent_id: str
    outcome_pnl: float
    confidence: float = Field(ge=0.0, le=1.0)
    is_replay: bool = False
    dataset_version: str
    unmatched_negative: bool = False


class TrainingPair(BaseModel):
    """Single train/eval pair encoded in JSONL."""

    prompt: str
    completion: str
    metadata: TrainingPairMetadata


class EvaluationMetrics(BaseModel):
    """Adapter evaluation metrics on holdout data."""

    signal_accuracy: float = Field(ge=0.0, le=1.0)
    abstain_rate: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0)
    regime_accuracy: dict[MarketRegime, float]
    consistency_variance: float = Field(ge=0.0)


class PromotionDecision(BaseModel):
    """Promotion result and rationale."""

    approved: bool
    reasons: list[str] = Field(default_factory=list)
