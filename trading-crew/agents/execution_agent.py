"""Execution agent with paper/live guardrails and retry logic."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from models.schemas import TradeDecision
from tools.order_manager import (
    ExchangeName,
    OrderManager,
    OrderRequest,
    OrderSide,
    PaperOrderManager,
)


@dataclass(frozen=True)
class ExecutionResult:
    """Result for an order placement attempt."""

    success: bool
    order_id: str | None
    reason: str


class ExecutionAgent:
    """Places orders only after successful risk approval."""

    agent_id = "execution_agent"

    def __init__(
        self,
        *,
        live_trading: bool,
        max_retries: int = 3,
        initial_retry_delay_seconds: float = 1.0,
        order_manager: OrderManager | None = None,
        exchange: ExchangeName = "alpaca",
    ) -> None:
        self.live_trading = live_trading
        self.max_retries = max_retries
        self.initial_retry_delay_seconds = initial_retry_delay_seconds
        self.order_manager = order_manager or PaperOrderManager()
        self.exchange = exchange

    async def execute(self, decision: TradeDecision) -> ExecutionResult:
        """Execute approved decision with exponential backoff retries."""

        if not decision.approved:
            return ExecutionResult(success=False, order_id=None, reason="Risk not approved.")
        order_side: OrderSide
        if decision.direction == "buy":
            order_side = "buy"
        elif decision.direction == "sell":
            order_side = "sell"
        else:
            return ExecutionResult(success=False, order_id=None, reason="No executable direction.")
        if self.live_trading is False:
            logger.info("paper_order_submitted", task_id=decision.task_id, asset=decision.asset)
            return ExecutionResult(
                success=True,
                order_id=f"paper-{decision.task_id}",
                reason="Paper order simulated.",
            )

        delay = self.initial_retry_delay_seconds
        order_request = OrderRequest(
            symbol=decision.asset,
            side=order_side,
            quantity=max(decision.position_size, 0.0),
            order_type="market",
            exchange=self.exchange,
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "live_order_attempt",
                    task_id=decision.task_id,
                    attempt=attempt,
                    exchange=self.exchange,
                )
                response = await self.order_manager.place_order(order_request)
                if response.accepted:
                    return ExecutionResult(
                        success=True,
                        order_id=response.order_id,
                        reason=response.reason,
                    )
                if not response.retryable:
                    return ExecutionResult(
                        success=False,
                        order_id=response.order_id,
                        reason=response.reason,
                    )
                logger.warning(
                    "retryable_order_rejection",
                    attempt=attempt,
                    reason=response.reason,
                )
            except RuntimeError as exc:
                logger.error("order_attempt_failed", attempt=attempt, error=str(exc))
            if attempt < self.max_retries:
                await asyncio.sleep(delay)
                delay *= 2
        return ExecutionResult(success=False, order_id=None, reason="Exhausted retries.")
