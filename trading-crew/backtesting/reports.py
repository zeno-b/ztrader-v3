"""Backtesting report helpers."""

from __future__ import annotations

from backtesting.engine import BacktestMetrics


def is_likely_overfit(metrics: BacktestMetrics) -> bool:
    """Flag suspiciously high in-sample Sharpe ratios."""

    return metrics.sharpe_ratio > 3.0
