"""Unit tests for execution agent order routing and retry behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from agents.execution_agent import ExecutionAgent
from models.schemas import TradeDecision
from tools.order_manager import OrderRequest, OrderResponse


@dataclass
class _StubOrderManager:
    responses: list[OrderResponse]
    calls: int = 0
    captured_requests: list[OrderRequest] = field(default_factory=list)

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        self.calls += 1
        self.captured_requests.append(request)
        index = min(self.calls - 1, len(self.responses) - 1)
        return self.responses[index]


def _trade_decision() -> TradeDecision:
    return TradeDecision(
        task_id="task-1",
        asset="BTC/USD",
        direction="buy",
        confidence=0.9,
        approved=True,
        veto_reason=None,
        position_size=0.01,
        weighted_votes={"technical_agent": 0.8},
    )


def test_execution_agent_uses_order_manager_for_live_trading() -> None:
    manager = _StubOrderManager(
        responses=[OrderResponse(accepted=True, order_id="kraken-1", reason="ok")]
    )
    agent = ExecutionAgent(
        live_trading=True,
        max_retries=2,
        initial_retry_delay_seconds=0.0,
        order_manager=manager,
        exchange="kraken",
    )
    result = asyncio.run(agent.execute(_trade_decision()))
    assert result.success is True
    assert result.order_id == "kraken-1"
    assert manager.calls == 1
    assert manager.captured_requests[0].exchange == "kraken"


def test_execution_agent_retries_on_retryable_rejection() -> None:
    manager = _StubOrderManager(
        responses=[
            OrderResponse(accepted=False, order_id=None, reason="temporary", retryable=True),
            OrderResponse(accepted=True, order_id="kraken-2", reason="ok"),
        ]
    )
    agent = ExecutionAgent(
        live_trading=True,
        max_retries=3,
        initial_retry_delay_seconds=0.0,
        order_manager=manager,
        exchange="kraken",
    )
    result = asyncio.run(agent.execute(_trade_decision()))
    assert result.success is True
    assert result.order_id == "kraken-2"
    assert manager.calls == 2


def test_execution_agent_stops_on_non_retryable_rejection() -> None:
    manager = _StubOrderManager(
        responses=[
            OrderResponse(accepted=False, order_id=None, reason="invalid order", retryable=False),
            OrderResponse(accepted=True, order_id="kraken-3", reason="unexpected"),
        ]
    )
    agent = ExecutionAgent(
        live_trading=True,
        max_retries=3,
        initial_retry_delay_seconds=0.0,
        order_manager=manager,
        exchange="kraken",
    )
    result = asyncio.run(agent.execute(_trade_decision()))
    assert result.success is False
    assert "invalid order" in result.reason
    assert manager.calls == 1
