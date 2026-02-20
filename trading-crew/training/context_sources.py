"""Training context sources used to enrich dataset prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Protocol

from models.schemas import ContextValue, DecisionLogRecord

ContextMap = dict[str, ContextValue]


@dataclass(frozen=True)
class TrainingContext:
    """Context emitted by one training enrichment source."""

    source_name: str
    fields: ContextMap


class TrainingContextSource(Protocol):
    """Protocol for dataset prompt/context enrichers."""

    @property
    def source_name(self) -> str:
        """Stable source identifier used in prompts and metadata."""
        ...

    def build_context(self, record: DecisionLogRecord) -> TrainingContext:
        """Build context fields for one decision record."""


@dataclass(frozen=True)
class DataSourceDiversityContextSource:
    """Summarize source diversity used by the originating decision."""

    source_name: str = "source_diversity"

    def build_context(self, record: DecisionLogRecord) -> TrainingContext:
        unique_sources = sorted(set(record.data_sources))
        return TrainingContext(
            source_name=self.source_name,
            fields={
                "source_count": len(unique_sources),
                "has_multi_source": len(unique_sources) > 1,
                "primary_source": unique_sources[0] if unique_sources else "none",
            },
        )


@dataclass(frozen=True)
class OutcomeQualityContextSource:
    """Encode outcome quality labels for calibration-aware training."""

    source_name: str = "outcome_quality"

    def build_context(self, record: DecisionLogRecord) -> TrainingContext:
        outcome_pnl = float(record.outcome_pnl or 0.0)
        latency_days = int(record.outcome_latency_days or 0)
        return TrainingContext(
            source_name=self.source_name,
            fields={
                "pnl_sign": "positive" if outcome_pnl > 0.0 else "non_positive",
                "latency_bucket": _latency_bucket(latency_days),
                "trade_was_profitable": bool(record.trade_was_profitable),
            },
        )


@dataclass(frozen=True)
class TemporalRegimeContextSource:
    """Add temporal markers and regime for richer sequence conditioning."""

    source_name: str = "temporal_regime"

    def build_context(self, record: DecisionLogRecord) -> TrainingContext:
        timestamp = record.timestamp
        return TrainingContext(
            source_name=self.source_name,
            fields={
                "weekday": timestamp.weekday(),
                "hour_utc": timestamp.hour,
                "regime": record.market_regime,
            },
        )


@dataclass(frozen=True)
class MacroSnapshotContextSource:
    """
    Inject exogenous macro features from daily snapshots.

    Snapshot keys should use `YYYY-MM-DD` date format.
    """

    snapshots: Mapping[str, Mapping[str, ContextValue]]
    source_name: str = "macro_snapshot"

    def build_context(self, record: DecisionLogRecord) -> TrainingContext:
        day_key = record.timestamp.date().isoformat()
        row = self.snapshots.get(day_key, {})
        normalized: ContextMap = {"available": bool(row)}
        for key, value in row.items():
            if isinstance(value, (str, float, int, bool)):
                normalized[key] = value
        return TrainingContext(source_name=self.source_name, fields=normalized)


def render_context_lines(contexts: list[TrainingContext]) -> list[str]:
    """Render contexts as stable text lines for training prompts."""

    lines: list[str] = []
    for context in contexts:
        serialized = json.dumps(context.fields, sort_keys=True, separators=(",", ":"))
        lines.append(f"- {context.source_name}: {serialized}")
    return lines


def flatten_contexts(contexts: list[TrainingContext]) -> ContextMap:
    """Flatten context maps into metadata-safe dotted keys."""

    flattened: ContextMap = {}
    for context in contexts:
        for key, value in context.fields.items():
            flattened[f"{context.source_name}.{key}"] = value
    return flattened


def _latency_bucket(latency_days: int) -> str:
    if latency_days <= 1:
        return "fast"
    if latency_days <= 5:
        return "medium"
    return "slow"
