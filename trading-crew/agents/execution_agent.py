"""Execution agent with paper/live guardrails and retry logic."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from models.schemas import TradeDecision


@dataclass(frozen=True)
class ExecutionResult:
    """Result for an order placement attempt."""

    success: bool
    order_id: str | None
    reason: str


class ExecutionAgent:
    """Places orders only after successful risk approval."""

    agent_id = "execution_agent"

    def __init__(self, *, live_trading: bool, max_retries: int = 3) -> None:
        self.live_trading = live_trading
        self.max_retries = max_retries

    async def execute(self, decision: TradeDecision) -> ExecutionResult:
        """Execute approved decision with exponential backoff retries."""

        if not decision.approved:
            return ExecutionResult(success=False, order_id=None, reason="Risk not approved.")
        if decision.direction not in {"buy", "sell"}:
            return ExecutionResult(success=False, order_id=None, reason="No executable direction.")
        if self.live_trading is False:
            logger.info("paper_order_submitted", task_id=decision.task_id, asset=decision.asset)
            return ExecutionResult(
                success=True,
                order_id=f"paper-{decision.task_id}",
                reason="Paper order simulated.",
            )

        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                # Placeholder for broker API call.
                logger.info("live_order_attempt", task_id=decision.task_id, attempt=attempt)
                return ExecutionResult(
                    success=True,
                    order_id=f"live-{decision.task_id}-{attempt}",
                    reason="Live order placed.",
                )
            except RuntimeError as exc:
                logger.error("order_attempt_failed", attempt=attempt, error=str(exc))
                await asyncio.sleep(delay)
                delay *= 2
        return ExecutionResult(success=False, order_id=None, reason="Exhausted retries.")
