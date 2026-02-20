"""Base agent primitives with explicit Result pattern."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, Protocol, TypeVar

from loguru import logger

from models.schemas import AgentResponse, BaseSignal, MarketRegime

T = TypeVar("T")


@dataclass(frozen=True)
class Result(Generic[T]):
    """Success-or-error container used by agent boundaries."""

    ok: bool
    value: T | None = None
    error: str | None = None

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """Construct a successful result."""

        return cls(ok=True, value=value, error=None)

    @classmethod
    def failure(cls, error: str) -> "Result[T]":
        """Construct a failed result."""

        return cls(ok=False, value=None, error=error)


class Agent(Protocol):
    """Protocol implemented by all trading and training agents."""

    agent_id: str
    adapter_version: str

    async def run(self, task_id: str) -> Result[AgentResponse]:
        """Run an agent task and return a typed response."""


def build_error_response(
    *,
    agent_id: str,
    task_id: str,
    asset: str,
    reasoning: str,
    market_regime: MarketRegime,
    adapter_version: str,
) -> AgentResponse:
    """Create a standardized error response for boundary failures."""

    logger.error(
        "agent_error",
        agent_id=agent_id,
        task_id=task_id,
        reasoning=reasoning,
    )
    return AgentResponse(
        agent_id=agent_id,
        timestamp=datetime.now(UTC),
        task_id=task_id,
        status="error",
        payload=BaseSignal(asset=asset, direction="abstain", timeframe="1h"),
        confidence=0.0,
        reasoning=reasoning,
        data_sources=[],
        latency_ms=0,
        adapter_version=adapter_version,
        market_regime=market_regime,
    )
