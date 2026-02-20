"""Unit tests for evaluation and promotion gates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from models.schemas import EvaluationMetrics, MarketRegime, PromotionDecision
from training.evaluator import Evaluator, HoldoutPrediction
from training.promoter import Promoter, ShadowDeploymentResult


def _predictions(accurate: bool, abstain_every: int = 5) -> list[HoldoutPrediction]:
    rows: list[HoldoutPrediction] = []
    regimes: tuple[MarketRegime, ...] = (
        "trending_bull",
        "trending_bear",
        "mean_reverting",
        "high_volatility",
    )
    for index in range(80):
        regime = regimes[index % len(regimes)]
        actual = index % 2 == 0
        abstained = index % abstain_every == 0
        predicted = actual if accurate else not actual
        rows.append(
            HoldoutPrediction(
                regime=regime,
                predicted_profitable=predicted,
                actual_profitable=actual,
                confidence=0.8,
                abstained=abstained,
            )
        )
    return rows


def test_evaluator_promotion_gate_accepts_valid_candidate() -> None:
    evaluator = Evaluator(seed=1)
    champion = EvaluationMetrics(
        signal_accuracy=0.70,
        abstain_rate=0.22,
        brier_score=0.20,
        regime_accuracy={
            "trending_bull": 0.70,
            "trending_bear": 0.72,
            "mean_reverting": 0.69,
            "high_volatility": 0.71,
        },
        consistency_variance=0.03,
    )
    candidate = EvaluationMetrics(
        signal_accuracy=0.73,
        abstain_rate=0.20,
        brier_score=0.19,
        regime_accuracy={
            "trending_bull": champion.regime_accuracy["trending_bull"],
            "trending_bear": champion.regime_accuracy["trending_bear"],
            "mean_reverting": champion.regime_accuracy["mean_reverting"],
            "high_volatility": champion.regime_accuracy["high_volatility"],
        },
        consistency_variance=0.01,
    )
    decision = evaluator.evaluate_promotion(
        champion=champion,
        candidate=candidate,
        champion_dataset_version="v10",
        candidate_dataset_version="v11",
    )
    assert decision.approved is True
    assert decision.reasons == []


def test_evaluator_promotion_gate_rejects_regime_degradation() -> None:
    evaluator = Evaluator(seed=1)
    champion = EvaluationMetrics(
        signal_accuracy=0.70,
        abstain_rate=0.22,
        brier_score=0.20,
        regime_accuracy={
            "trending_bull": 0.70,
            "trending_bear": 0.72,
            "mean_reverting": 0.69,
            "high_volatility": 0.71,
        },
        consistency_variance=0.03,
    )
    candidate = EvaluationMetrics(
        signal_accuracy=0.73,
        abstain_rate=0.20,
        brier_score=0.19,
        regime_accuracy={
            "trending_bull": max(0.0, champion.regime_accuracy["trending_bull"] - 0.10),
            "trending_bear": champion.regime_accuracy["trending_bear"],
            "mean_reverting": champion.regime_accuracy["mean_reverting"],
            "high_volatility": champion.regime_accuracy["high_volatility"],
        },
        consistency_variance=0.01,
    )
    decision = evaluator.evaluate_promotion(
        champion=champion,
        candidate=candidate,
        champion_dataset_version="v10",
        candidate_dataset_version="v9",
    )
    assert decision.approved is False
    assert any("Regime degradation" in reason for reason in decision.reasons)
    assert any("dataset_version" in reason for reason in decision.reasons)


def test_promoter_requires_shadow_agreement() -> None:
    promoter = Promoter()
    start = datetime.now(UTC)
    low_agreement = ShadowDeploymentResult(
        started_at=start,
        ended_at=start + timedelta(hours=48),
        agreement_rate=0.80,
        total_samples=500,
    )
    result = promoter.resolve(
        evaluation_decision=PromotionDecision(approved=True, reasons=[]),
        shadow_result=low_agreement,
    )
    assert result.promoted is False
    assert "human review" in result.reason.lower()
