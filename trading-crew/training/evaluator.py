"""Evaluation utilities for champion vs candidate adapter comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random

from models.schemas import EvaluationMetrics, MarketRegime, PromotionDecision

REGIMES: tuple[MarketRegime, ...] = (
    "trending_bull",
    "trending_bear",
    "mean_reverting",
    "high_volatility",
)


@dataclass(frozen=True)
class HoldoutPrediction:
    """Single holdout prediction record for metric computation."""

    regime: MarketRegime
    predicted_profitable: bool
    actual_profitable: bool
    confidence: float
    abstained: bool


class Evaluator:
    """Computes metrics and applies hard promotion criteria."""

    def __init__(self, seed: int = 11) -> None:
        self._rng = Random(seed)

    def compute_metrics(self, predictions: list[HoldoutPrediction]) -> EvaluationMetrics:
        """Compute required evaluation metrics from holdout predictions."""

        if not predictions:
            raise ValueError("Predictions cannot be empty.")

        signal_rows = [row for row in predictions if not row.abstained]
        total = len(predictions)
        signal_total = max(1, len(signal_rows))
        matches = sum(
            1 for row in signal_rows if row.predicted_profitable == row.actual_profitable
        )
        signal_accuracy = matches / signal_total
        abstain_rate = (total - len(signal_rows)) / total

        brier_sum = 0.0
        for row in predictions:
            if row.abstained:
                probability = 0.5
            elif row.predicted_profitable:
                probability = row.confidence
            else:
                probability = 1.0 - row.confidence
            target = 1.0 if row.actual_profitable else 0.0
            brier_sum += (probability - target) ** 2
        brier_score = brier_sum / total

        regime_accuracy: dict[MarketRegime, float] = {}
        for regime in REGIMES:
            regime_rows = [row for row in signal_rows if row.regime == regime]
            if not regime_rows:
                regime_accuracy[regime] = 0.0
                continue
            regime_matches = sum(
                1
                for row in regime_rows
                if row.predicted_profitable == row.actual_profitable
            )
            regime_accuracy[regime] = regime_matches / len(regime_rows)

        consistency_variance = self._compute_consistency_variance(predictions)
        return EvaluationMetrics(
            signal_accuracy=signal_accuracy,
            abstain_rate=abstain_rate,
            brier_score=brier_score,
            regime_accuracy=regime_accuracy,
            consistency_variance=consistency_variance,
        )

    def evaluate_promotion(
        self,
        *,
        champion: EvaluationMetrics,
        candidate: EvaluationMetrics,
        champion_dataset_version: str,
        candidate_dataset_version: str,
    ) -> PromotionDecision:
        """Apply non-negotiable promotion criteria."""

        failures: list[str] = []
        if candidate.signal_accuracy - champion.signal_accuracy < 0.02:
            failures.append("Signal accuracy improvement is below 2%.")
        if candidate.brier_score > champion.brier_score:
            failures.append("Brier score degraded versus champion.")
        if not (0.15 <= candidate.abstain_rate <= 0.40):
            failures.append("Candidate abstain rate is outside healthy 15%-40% range.")
        for regime in REGIMES:
            champion_score = champion.regime_accuracy.get(regime, 0.0)
            candidate_score = candidate.regime_accuracy.get(regime, 0.0)
            if champion_score - candidate_score > 0.05:
                failures.append(f"Regime degradation exceeds 5% for {regime}.")
        if candidate.consistency_variance >= 0.05:
            failures.append("Candidate consistency variance is not stable (<0.05 required).")
        if not self._is_newer_dataset(candidate_dataset_version, champion_dataset_version):
            failures.append("Candidate dataset_version must be newer than champion.")

        return PromotionDecision(approved=not failures, reasons=failures)

    def _compute_consistency_variance(self, predictions: list[HoldoutPrediction]) -> float:
        accuracies: list[float] = []
        sample_size = max(1, int(len(predictions) * 0.7))
        for _ in range(5):
            sample = self._rng.sample(predictions, k=sample_size)
            signals = [row for row in sample if not row.abstained]
            if not signals:
                accuracies.append(0.0)
                continue
            matches = sum(
                1 for row in signals if row.predicted_profitable == row.actual_profitable
            )
            accuracies.append(matches / len(signals))
        mean = sum(accuracies) / len(accuracies)
        return sum((value - mean) ** 2 for value in accuracies) / len(accuracies)

    def _is_newer_dataset(self, candidate: str, champion: str) -> bool:
        return self._extract_numeric_version(candidate) > self._extract_numeric_version(
            champion
        )

    def _extract_numeric_version(self, value: str) -> int:
        match = re.search(r"(\d+)", value)
        if not match:
            return -1
        return int(match.group(1))
