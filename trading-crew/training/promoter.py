"""Promotion gate logic, including shadow deployment checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from models.schemas import PromotionDecision


@dataclass(frozen=True)
class ShadowDeploymentResult:
    """Outcome of a 48-hour shadow deployment period."""

    started_at: datetime
    ended_at: datetime
    agreement_rate: float
    total_samples: int


@dataclass(frozen=True)
class PromotionResult:
    """Resolved promotion decision after shadow phase."""

    promoted: bool
    reason: str
    retain_previous_for_days: int
    reset_failure_streak: bool


class Promoter:
    """Handles shadow deployment and final promotion gating."""

    SHADOW_DURATION_HOURS = 48
    MIN_SHADOW_AGREEMENT = 0.85

    def begin_shadow_window(self) -> tuple[datetime, datetime]:
        """Return deterministic shadow deployment start and end times."""

        start = datetime.now(UTC)
        return start, start + timedelta(hours=self.SHADOW_DURATION_HOURS)

    def resolve(
        self,
        *,
        evaluation_decision: PromotionDecision,
        shadow_result: ShadowDeploymentResult,
    ) -> PromotionResult:
        """Resolve final promotion decision from evaluation and shadow metrics."""

        if not evaluation_decision.approved:
            return PromotionResult(
                promoted=False,
                reason="Evaluation gate rejected candidate.",
                retain_previous_for_days=90,
                reset_failure_streak=False,
            )
        if shadow_result.agreement_rate < self.MIN_SHADOW_AGREEMENT:
            return PromotionResult(
                promoted=False,
                reason="Shadow agreement below 85%; human review required.",
                retain_previous_for_days=90,
                reset_failure_streak=False,
            )
        return PromotionResult(
            promoted=True,
            reason="Candidate promoted to champion after successful shadow deployment.",
            retain_previous_for_days=90,
            reset_failure_streak=True,
        )
