"""Unit tests for strict schema validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from models.schemas import AgentResponse, BaseSignal


def test_agent_response_requires_confidence_bounds() -> None:
    with pytest.raises(ValueError):
        AgentResponse(
            agent_id="technical_agent",
            timestamp=datetime.now(UTC),
            task_id="task-1",
            status="success",
            payload=BaseSignal(asset="SPY", direction="buy", timeframe="1h"),
            confidence=1.1,
            reasoning="invalid",
            data_sources=["unit"],
            latency_ms=1,
            adapter_version="v1",
            market_regime="trending_bull",
        )
