"""Risk agent with hard veto constraints that cannot be overridden."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from models.schemas import RiskAssessment, RiskContext


@dataclass(frozen=True)
class RiskLimits:
    """Non-adjustable automated risk limits."""

    max_position_pct: float = 0.02
    max_daily_drawdown_pct: float = 0.05
    max_correlated_exposure_pct: float = 0.10
    no_trade_event_window_minutes: int = 5
    min_history_days: int = 30


class RiskAgent:
    """Evaluates risk constraints and emits a veto-capable assessment."""

    agent_id = "risk_agent"

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self._limits = limits or RiskLimits()

    def assess(self, context: RiskContext) -> RiskAssessment:
        """Apply hard risk checks and return approval decision."""

        max_position_value = context.portfolio_value * self._limits.max_position_pct
        adjusted_size = min(
            context.proposed_position_value / context.portfolio_value
            if context.portfolio_value > 0
            else 0.0,
            self._limits.max_position_pct,
        )

        if context.current_daily_drawdown_pct >= self._limits.max_daily_drawdown_pct:
            return self._reject(
                "Daily drawdown breach: trading halted.",
                adjusted_size=0.0,
            )

        if abs(context.minutes_to_major_event) <= self._limits.no_trade_event_window_minutes:
            return self._reject(
                "Within major economic event no-trade window.",
                adjusted_size=0.0,
            )

        if context.instrument_history_days < self._limits.min_history_days:
            return self._reject(
                "Instrument has fewer than 30 days of history.",
                adjusted_size=0.0,
            )

        if context.sector_exposure_pct > self._limits.max_correlated_exposure_pct:
            return self._reject(
                "Sector correlated exposure exceeds 10%.",
                adjusted_size=0.0,
            )

        if context.proposed_position_value > max_position_value:
            logger.warning(
                "risk_position_adjusted",
                proposed=context.proposed_position_value,
                max_allowed=max_position_value,
            )
            return RiskAssessment(
                approved=True,
                reason="Position size adjusted to risk limit.",
                adjusted_size=self._limits.max_position_pct,
            )

        return RiskAssessment(
            approved=True,
            reason="Approved",
            adjusted_size=adjusted_size,
        )

    def _reject(self, reason: str, adjusted_size: float) -> RiskAssessment:
        """Emit a rejected assessment with structured logging."""

        logger.error("risk_veto", reason=reason)
        return RiskAssessment(
            approved=False,
            reason=reason,
            adjusted_size=adjusted_size,
        )
