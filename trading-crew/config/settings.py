"""Runtime settings for trading and training services."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = Field(default="INFO")
    live_trading: bool = Field(default=False, alias="LIVE_TRADING")

    max_position_pct: float = Field(default=0.02, alias="MAX_POSITION_PCT")
    max_daily_drawdown_pct: float = Field(
        default=0.05, alias="MAX_DAILY_DRAWDOWN_PCT"
    )
    max_correlated_exposure_pct: float = Field(
        default=0.10, alias="MAX_CORRELATED_EXPOSURE_PCT"
    )

    postgres_host: str = Field(default="timescaledb", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="trading", alias="POSTGRES_USER")
    postgres_password: str = Field(default="trading", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="trading", alias="POSTGRES_DB")

    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment: str = Field(
        default="trading-crew", alias="MLFLOW_EXPERIMENT"
    )

    training_min_outcome_records: int = Field(
        default=500, alias="TRAINING_MIN_OUTCOME_RECORDS"
    )
    training_failure_streak_pause: int = Field(
        default=3, alias="TRAINING_FAILURE_STREAK_PAUSE"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached settings instance."""

    return Settings()
