"""Research agent that emits sentiment signals from cited sources."""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger

from agents.base_agent import Result
from models.schemas import AgentResponse, SentimentSignal


class ResearchAgent:
    """Research agent with conservative abstain behavior."""

    agent_id = "research_agent"

    def __init__(self, adapter_version: str) -> None:
        self.adapter_version = adapter_version

    async def run(self, task_id: str) -> Result[AgentResponse]:
        """
        Emit ABSTAIN by default when no validated research bundle is provided.

        The production implementation should inject verified news, filings,
        and macro context before calling this method.
        """

        response = AgentResponse(
            agent_id=self.agent_id,
            timestamp=datetime.now(UTC),
            task_id=task_id,
            status="abstain",
            payload=SentimentSignal(
                asset="UNKNOWN",
                direction="abstain",
                timeframe="1h",
                score=0.0,
                confidence=0.0,
                sources=[],
            ),
            confidence=0.0,
            reasoning="No validated source bundle provided; abstaining.",
            data_sources=[],
            latency_ms=1,
            adapter_version=self.adapter_version,
            market_regime="mean_reverting",
        )
        logger.info("research_agent_abstain", task_id=task_id)
        return Result.success(response)
