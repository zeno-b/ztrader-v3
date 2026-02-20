"""Unit tests for hard-rule risk veto logic."""

from __future__ import annotations

from agents.risk_agent import RiskAgent
from models.schemas import RiskContext


def _base_context() -> RiskContext:
    return RiskContext(
        portfolio_value=100_000.0,
        proposed_position_value=1_000.0,
        current_daily_drawdown_pct=0.01,
        sector_exposure_pct=0.05,
        minutes_to_major_event=120,
        instrument_history_days=120,
    )


def test_risk_rejects_drawdown_breach() -> None:
    agent = RiskAgent()
    context = _base_context().model_copy(update={"current_daily_drawdown_pct": 0.06})
    assessment = agent.assess(context)
    assert assessment.approved is False
    assert "drawdown" in assessment.reason.lower()


def test_risk_rejects_event_window() -> None:
    agent = RiskAgent()
    context = _base_context().model_copy(update={"minutes_to_major_event": 3})
    assessment = agent.assess(context)
    assert assessment.approved is False
    assert "event" in assessment.reason.lower()


def test_risk_adjusts_position_size_to_limit() -> None:
    agent = RiskAgent()
    context = _base_context().model_copy(update={"proposed_position_value": 5_000.0})
    assessment = agent.assess(context)
    assert assessment.approved is True
    assert assessment.adjusted_size == 0.02


def test_risk_approves_when_all_limits_pass() -> None:
    agent = RiskAgent()
    assessment = agent.assess(_base_context())
    assert assessment.approved is True
    assert assessment.adjusted_size == 0.01
