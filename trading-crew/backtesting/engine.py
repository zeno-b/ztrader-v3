"""Backtesting engine facade for Backtrader and vectorbt workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestMetrics:
    """Backtest metric bundle."""

    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float


class BacktestEngine:
    """Executes backtests and returns standardized metrics."""

    def run(self) -> BacktestMetrics:
        """Run backtest with deterministic settings."""

        return BacktestMetrics(sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0)
