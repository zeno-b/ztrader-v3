"""Coordinator agent for signal aggregation and trade decisioning."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from loguru import logger

from models.schemas import (
    AgentResponse,
    MarketRegime,
    RiskAssessment,
    SignalDirection,
    TradeDecision,
)


@dataclass(frozen=True)
class CoordinatorConfig:
    """Configuration for weighted signal aggregation."""

    signal_timeout_seconds: int = 30
    min_confidence: float = 0.60
    default_position_size: float = 0.01
    min_agent_weight: float = 0.05


class Coordinator:
    """Aggregate typed agent outputs and apply risk veto semantics."""

    def __init__(self, weights: dict[str, float], config: CoordinatorConfig | None = None):
        self._config = config or CoordinatorConfig()
        self._weights = self._normalize_weights(weights)

    @property
    def weights(self) -> dict[str, float]:
        """Return active normalized agent weights."""

        return dict(self._weights)

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update agent weights while preserving minimum floors."""

        self._weights = self._normalize_weights(new_weights)
        logger.info("coordinator_weights_updated", weights=self._weights)

    def aggregate(
        self,
        *,
        task_id: str,
        asset: str,
        responses: list[AgentResponse],
        risk_assessment: RiskAssessment,
        market_regime: MarketRegime,
    ) -> TradeDecision:
        """Aggregate validated responses into a typed trade decision."""

        votes: dict[SignalDirection, float] = defaultdict(float)
        valid_responses = [
            response
            for response in responses
            if response.status == "success" and response.confidence >= self._config.min_confidence
        ]

        for response in valid_responses:
            weight = self._weights.get(response.agent_id, self._config.min_agent_weight)
            response_direction = response.payload.direction
            votes[response_direction] += response.confidence * weight

        direction: SignalDirection = "abstain"
        confidence = 0.0
        if votes:
            direction, confidence = max(votes.items(), key=lambda item: item[1])

        approved = risk_assessment.approved and direction in {"buy", "sell"}
        veto_reason = None if approved else risk_assessment.reason
        position_size = (
            min(self._config.default_position_size, risk_assessment.adjusted_size)
            if approved
            else 0.0
        )

        logger.bind(
            timestamp=datetime.now(UTC).isoformat(),
            task_id=task_id,
            asset=asset,
            market_regime=market_regime,
        ).info(
            "trade_decision",
            direction=direction,
            approved=approved,
            confidence=confidence,
            veto_reason=veto_reason,
        )

        return TradeDecision(
            task_id=task_id,
            asset=asset,
            direction=direction,
            confidence=min(confidence, 1.0),
            approved=approved,
            veto_reason=veto_reason,
            position_size=position_size,
            weighted_votes={str(direction): value for direction, value in votes.items()},
        )

    def _normalize_weights(self, raw_weights: dict[str, float]) -> dict[str, float]:
        """Normalize agent weights to sum to 1.0 with minimum floor."""

        if not raw_weights:
            return {}

        floored = {agent: max(weight, self._config.min_agent_weight) for agent, weight in raw_weights.items()}
        total = sum(floored.values())
        if total <= 0:
            return {agent: 1.0 / len(floored) for agent in floored}
        return {agent: value / total for agent, value in floored.items()}


def main() -> None:
    """Entrypoint for containerized coordinator service."""

    logger.info("coordinator_standby")


if __name__ == "__main__":
    main()
