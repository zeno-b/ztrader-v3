"""Unit tests for deterministic dataset builder behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from models.schemas import BaseSignal, DecisionLogRecord, MarketRegime
from training.context_sources import MacroSnapshotContextSource
from training.dataset_builder import DatasetBuilder, DatasetBuilderConfig

REGIMES: tuple[MarketRegime, ...] = (
    "trending_bull",
    "trending_bear",
    "mean_reverting",
    "high_volatility",
)


def _build_records(count: int) -> list[DecisionLogRecord]:
    now = datetime.now(UTC)
    records: list[DecisionLogRecord] = []
    for index in range(count):
        regime = REGIMES[index % len(REGIMES)]
        profitable = index % 2 == 0
        records.append(
            DecisionLogRecord(
                id=uuid4(),
                timestamp=now - timedelta(minutes=count - index),
                agent_id="technical_agent",
                task_id=f"task-{index}",
                asset="SPY",
                asset_class="etf",
                timeframe="1h",
                signal_type="technical",
                signal_value=BaseSignal(asset="SPY", direction="buy", timeframe="1h"),
                confidence=0.8,
                reasoning="synthetic",
                data_sources=["synthetic"],
                market_regime=regime,
                outcome_pnl=0.01 if profitable else -0.01,
                outcome_latency_days=1,
                contributed_to_trade=True,
                trade_was_profitable=profitable,
            )
        )
    return records


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_dataset_builder_creates_and_reuses_holdout_lock(tmp_path: Path) -> None:
    output_dir = tmp_path / "datasets"
    builder = DatasetBuilder(
        output_dir=output_dir,
        config=DatasetBuilderConfig(min_outcome_records=20, seed=2),
    )

    records_v1 = _build_records(80)
    first = builder.build(records_v1, "v1")
    assert first.test_path.exists()

    lock_path = output_dir / "holdout_lock.json"
    assert lock_path.exists()
    first_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    first_locked_ids = set(first_lock["test_ids"])
    assert first_locked_ids

    records_v2 = records_v1 + _build_records(20)
    _ = builder.build(records_v2, "v2")
    second_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert set(second_lock["test_ids"]) == first_locked_ids


def test_dataset_builder_enforces_replay_and_regime_floor(tmp_path: Path) -> None:
    output_dir = tmp_path / "datasets"
    builder = DatasetBuilder(
        output_dir=output_dir,
        config=DatasetBuilderConfig(min_outcome_records=20, seed=3),
    )

    built = builder.build(_build_records(120), "v3")
    rows = _read_jsonl(built.train_path)
    assert rows

    replay_rows = [row for row in rows if row["metadata"]["is_replay"] is True]
    replay_ratio = len(replay_rows) / len(rows)
    assert replay_ratio >= 0.30

    regime_counts: dict[str, int] = {regime: 0 for regime in REGIMES}
    for row in rows:
        regime = row["metadata"]["regime"]
        regime_counts[str(regime)] += 1
    for regime in REGIMES:
        assert regime_counts[regime] / len(rows) >= 0.20


def test_dataset_builder_adds_context_source_features(tmp_path: Path) -> None:
    output_dir = tmp_path / "datasets"
    records = _build_records(60)
    snapshot_key = records[0].timestamp.date().isoformat()
    builder = DatasetBuilder(
        output_dir=output_dir,
        config=DatasetBuilderConfig(min_outcome_records=20, seed=4),
        context_sources=[MacroSnapshotContextSource(snapshots={snapshot_key: {"vix": 21.5}})],
    )

    built = builder.build(records, "v4")
    rows = _read_jsonl(built.train_path)
    assert rows
    first = rows[0]
    assert "Additional input sources:" in first["prompt"]
    metadata = first["metadata"]
    assert "macro_snapshot" in metadata["input_sources"]
    context_features = metadata["context_features"]
    assert "macro_snapshot.available" in context_features
