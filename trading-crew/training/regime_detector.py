"""Hidden Markov Model based market regime classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from models.schemas import MarketRegime

GaussianHMMType: Any | None = None
_HMM_IMPORT_ERROR: ImportError | None = None
try:
    from hmmlearn.hmm import GaussianHMM as ImportedGaussianHMM
except ImportError as exc:  # pragma: no cover - dependency availability
    _HMM_IMPORT_ERROR = exc
else:
    GaussianHMMType = ImportedGaussianHMM

REGIME_ORDER: tuple[MarketRegime, ...] = (
    "trending_bull",
    "trending_bear",
    "mean_reverting",
    "high_volatility",
)


@dataclass
class RegimeModelState:
    """Model state and previous inferred regime."""

    model: Any | None
    previous_regime: MarketRegime | None = None


class RegimeDetector:
    """Detects market regimes using 4-state Gaussian HMM."""

    def __init__(self, rolling_window: int = 252, random_state: int = 19) -> None:
        self.rolling_window = rolling_window
        self.random_state = random_state
        self.state = RegimeModelState(model=None)

    def fit(self, market_frame: pd.DataFrame) -> None:
        """Train the HMM model on rolling market features."""

        if GaussianHMMType is None:  # pragma: no cover
            raise RuntimeError(f"hmmlearn unavailable: {_HMM_IMPORT_ERROR}")

        features = self._build_features(market_frame).tail(self.rolling_window)
        if len(features) < self.rolling_window:
            raise ValueError("Insufficient history for regime detector fit.")
        model = GaussianHMMType(
            n_components=4,
            covariance_type="diag",
            n_iter=200,
            random_state=self.random_state,
        )
        model.fit(features.to_numpy())
        self.state.model = model

    def current_regime(self, market_frame: pd.DataFrame) -> MarketRegime:
        """Infer current regime from latest rolling feature row."""

        if self.state.model is None:
            raise RuntimeError("Regime detector model is not fitted.")
        features = self._build_features(market_frame).tail(1).to_numpy()
        hidden_state = int(self.state.model.predict(features)[0])
        regime = REGIME_ORDER[hidden_state % len(REGIME_ORDER)]
        self.state.previous_regime = regime
        return regime

    def _build_features(self, market_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Build HMM feature matrix.

        Expected columns: close, volume, vix_proxy
        """

        close = market_frame["close"]
        volume = market_frame["volume"]
        vix_proxy = market_frame["vix_proxy"]

        log_returns = (close / close.shift(1)).map(np.log).fillna(0.0)
        realized_volatility = log_returns.rolling(window=20, min_periods=20).std().fillna(0.0)
        volume_z = ((volume - volume.rolling(window=20).mean()) / volume.rolling(window=20).std()).fillna(0.0)

        return pd.DataFrame(
            {
                "log_returns": log_returns,
                "realized_volatility": realized_volatility,
                "volume_z": volume_z,
                "vix_proxy": vix_proxy.ffill().fillna(0.0),
            }
        )
