"""Builds temporally-safe, regime-balanced training datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from uuid import UUID

from loguru import logger

from models.schemas import DecisionLogRecord, MarketRegime, TrainingPair, TrainingPairMetadata

REGIMES: tuple[MarketRegime, ...] = (
    "trending_bull",
    "trending_bear",
    "mean_reverting",
    "high_volatility",
)


@dataclass(frozen=True)
class DatasetBuilderConfig:
    """Configuration for deterministic dataset generation."""

    min_outcome_records: int = 500
    replay_ratio: float = 0.30
    min_regime_ratio: float = 0.20
    seed: int = 7
    holdout_lock_filename: str = "holdout_lock.json"


@dataclass(frozen=True)
class SelectedRecord:
    """Record wrapper with replay metadata."""

    record: DecisionLogRecord
    is_replay: bool


@dataclass(frozen=True)
class BuiltDataset:
    """Result object with output paths and split sizes."""

    dataset_version: str
    train_path: Path
    validation_path: Path
    test_path: Path
    split_counts: dict[str, int]
    regime_distribution: dict[str, float]


class DatasetBuilder:
    """Converts decision logs into JSONL train/validation/test artifacts."""

    def __init__(self, output_dir: Path, config: DatasetBuilderConfig | None = None) -> None:
        self._output_dir = output_dir
        self._config = config or DatasetBuilderConfig()
        self._rng = Random(self._config.seed)

    def build(self, records: list[DecisionLogRecord], dataset_version: str) -> BuiltDataset:
        """
        Build and persist dataset splits with holdout lock semantics.

        Training records with null outcomes are excluded by design.
        """

        eligible = sorted(
            [
                record
                for record in records
                if record.outcome_pnl is not None and record.trade_was_profitable is not None
            ],
            key=lambda item: item.timestamp,
        )
        if len(eligible) < self._config.min_outcome_records:
            raise ValueError(
                "Insufficient outcome-labeled records for training: "
                f"{len(eligible)} < {self._config.min_outcome_records}"
            )

        dataset_root = self._output_dir / dataset_version
        dataset_root.mkdir(parents=True, exist_ok=True)
        holdout_lock_path = self._output_dir / self._config.holdout_lock_filename

        train_records: list[DecisionLogRecord]
        validation_records: list[DecisionLogRecord]
        test_records: list[DecisionLogRecord]
        if holdout_lock_path.exists():
            train_records, validation_records, test_records = self._split_with_locked_holdout(
                eligible,
                holdout_lock_path,
            )
        else:
            train_records, validation_records, test_records = self._initial_time_split(eligible)
            self._persist_holdout_lock(holdout_lock_path, test_records)

        balanced_train = self._balance_regimes(train_records, eligible)
        replay_enriched_train = self._inject_replay_buffer(
            base_records=balanced_train,
            historical_pool=eligible,
        )

        train_pairs = self._build_pairs(
            selected_records=replay_enriched_train,
            historical_pool=eligible,
            dataset_version=dataset_version,
        )
        validation_pairs = self._build_pairs(
            selected_records=[SelectedRecord(record=item, is_replay=False) for item in validation_records],
            historical_pool=validation_records,
            dataset_version=dataset_version,
        )
        test_pairs = self._build_pairs(
            selected_records=[SelectedRecord(record=item, is_replay=False) for item in test_records],
            historical_pool=test_records,
            dataset_version=dataset_version,
        )

        train_path = dataset_root / "train.jsonl"
        validation_path = dataset_root / "validation.jsonl"
        test_path = dataset_root / "test.jsonl"
        self._write_jsonl(train_path, train_pairs)
        self._write_jsonl(validation_path, validation_pairs)
        self._write_jsonl(test_path, test_pairs)

        regime_distribution = self._regime_distribution([entry.record for entry in replay_enriched_train])
        logger.info(
            "dataset_built",
            dataset_version=dataset_version,
            train=len(train_pairs),
            validation=len(validation_pairs),
            test=len(test_pairs),
            regime_distribution=regime_distribution,
        )
        return BuiltDataset(
            dataset_version=dataset_version,
            train_path=train_path,
            validation_path=validation_path,
            test_path=test_path,
            split_counts={
                "train": len(train_pairs),
                "validation": len(validation_pairs),
                "test": len(test_pairs),
            },
            regime_distribution=regime_distribution,
        )

    def _initial_time_split(
        self, records: list[DecisionLogRecord]
    ) -> tuple[list[DecisionLogRecord], list[DecisionLogRecord], list[DecisionLogRecord]]:
        count = len(records)
        train_end = int(count * 0.70)
        validation_end = int(count * 0.85)
        return (
            records[:train_end],
            records[train_end:validation_end],
            records[validation_end:],
        )

    def _split_with_locked_holdout(
        self,
        records: list[DecisionLogRecord],
        holdout_lock_path: Path,
    ) -> tuple[list[DecisionLogRecord], list[DecisionLogRecord], list[DecisionLogRecord]]:
        with holdout_lock_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        test_ids = {UUID(identifier) for identifier in payload["test_ids"]}

        test_records = [record for record in records if record.id in test_ids]
        remainder = [record for record in records if record.id not in test_ids]
        split_ratio = 0.70 / 0.85
        train_end = int(len(remainder) * split_ratio)
        return (
            remainder[:train_end],
            remainder[train_end:],
            test_records,
        )

    def _persist_holdout_lock(self, lock_path: Path, test_records: list[DecisionLogRecord]) -> None:
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "test_ids": [str(record.id) for record in test_records],
        }
        with lock_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def _balance_regimes(
        self,
        base_records: list[DecisionLogRecord],
        historical_pool: list[DecisionLogRecord],
    ) -> list[SelectedRecord]:
        selected: list[SelectedRecord] = [
            SelectedRecord(record=record, is_replay=False) for record in base_records
        ]
        pool_by_regime: dict[MarketRegime, list[DecisionLogRecord]] = {regime: [] for regime in REGIMES}
        for record in historical_pool:
            pool_by_regime[record.market_regime].append(record)

        for regime in REGIMES:
            if not pool_by_regime[regime]:
                raise ValueError(f"No historical records available for regime: {regime}")

        max_iterations = len(base_records) * 8 if base_records else 0
        iterations = 0
        while not self._regimes_meet_floor([item.record for item in selected]) and iterations < max_iterations:
            iterations += 1
            counts = self._regime_counts([item.record for item in selected])
            total = len(selected)
            for regime in REGIMES:
                min_count = math.ceil(total * self._config.min_regime_ratio)
                if counts[regime] >= min_count:
                    continue
                sample = self._sample_with_recency_weight(pool_by_regime[regime], 1)[0]
                selected.append(SelectedRecord(record=sample, is_replay=False))
                counts[regime] += 1
                total += 1

        if not self._regimes_meet_floor([item.record for item in selected]):
            raise ValueError("Unable to satisfy minimum regime distribution constraints.")
        return selected

    def _regimes_meet_floor(self, records: list[DecisionLogRecord]) -> bool:
        if not records:
            return False
        counts = self._regime_counts(records)
        total = len(records)
        for regime in REGIMES:
            if counts[regime] < math.ceil(total * self._config.min_regime_ratio):
                return False
        return True

    def _regime_counts(self, records: list[DecisionLogRecord]) -> dict[MarketRegime, int]:
        counts: dict[MarketRegime, int] = {regime: 0 for regime in REGIMES}
        for record in records:
            counts[record.market_regime] += 1
        return counts

    def _inject_replay_buffer(
        self,
        *,
        base_records: list[SelectedRecord],
        historical_pool: list[DecisionLogRecord],
    ) -> list[SelectedRecord]:
        replay_records = list(base_records)
        base_count = len(base_records)
        min_replay = math.ceil(
            (self._config.replay_ratio * base_count) / (1.0 - self._config.replay_ratio)
        )
        sampled = self._sample_with_recency_weight(historical_pool, min_replay)
        replay_records.extend(SelectedRecord(record=record, is_replay=True) for record in sampled)
        return replay_records

    def _sample_with_recency_weight(
        self, records: list[DecisionLogRecord], count: int
    ) -> list[DecisionLogRecord]:
        if count <= 0:
            return []
        ordered = sorted(records, key=lambda item: item.timestamp)
        weights = [1.0 + (index / max(1, len(ordered) - 1)) for index in range(len(ordered))]
        return [self._rng.choices(ordered, weights=weights, k=1)[0] for _ in range(count)]

    def _build_pairs(
        self,
        *,
        selected_records: list[SelectedRecord],
        historical_pool: list[DecisionLogRecord],
        dataset_version: str,
    ) -> list[TrainingPair]:
        negatives_by_context: dict[tuple[MarketRegime, str, str], list[DecisionLogRecord]] = {}
        for record in historical_pool:
            if record.trade_was_profitable is True:
                continue
            key = (record.market_regime, record.asset_class, record.timeframe)
            negatives_by_context.setdefault(key, []).append(record)

        pairs: list[TrainingPair] = []
        for selected in selected_records:
            record = selected.record
            unmatched = False
            if record.trade_was_profitable is True:
                context_key = (record.market_regime, record.asset_class, record.timeframe)
                candidate_list = negatives_by_context.get(context_key, [])
                if candidate_list:
                    negative_match = candidate_list.pop(0)
                    pairs.append(
                        self._record_to_pair(
                            negative_match,
                            dataset_version=dataset_version,
                            is_replay=selected.is_replay,
                            unmatched_negative=False,
                        )
                    )
                else:
                    unmatched = True
            pairs.append(
                self._record_to_pair(
                    record,
                    dataset_version=dataset_version,
                    is_replay=selected.is_replay,
                    unmatched_negative=unmatched,
                )
            )

        pairs.sort(key=lambda item: self._extract_timestamp(item.prompt))
        return pairs

    def _record_to_pair(
        self,
        record: DecisionLogRecord,
        *,
        dataset_version: str,
        is_replay: bool,
        unmatched_negative: bool,
    ) -> TrainingPair:
        prompt = (
            "Agent context:\n"
            f"- timestamp: {record.timestamp.isoformat()}\n"
            f"- task_id: {record.task_id}\n"
            f"- agent_id: {record.agent_id}\n"
            f"- asset: {record.asset}\n"
            f"- asset_class: {record.asset_class}\n"
            f"- timeframe: {record.timeframe}\n"
            f"- market_regime: {record.market_regime}\n"
            f"- confidence: {record.confidence:.4f}\n"
            f"- signal: {record.signal_value.model_dump_json()}\n"
            f"- reasoning: {record.reasoning}\n"
            "Return a valid AgentResponse JSON."
        )
        completion = json.dumps(
            {
                "agent_id": record.agent_id,
                "timestamp": record.timestamp.isoformat(),
                "task_id": record.task_id,
                "status": "success",
                "payload": record.signal_value.model_dump(mode="json"),
                "confidence": record.confidence,
                "reasoning": record.reasoning,
                "data_sources": record.data_sources,
                "latency_ms": 1,
                "adapter_version": "label_from_record",
                "market_regime": record.market_regime,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return TrainingPair(
            prompt=prompt,
            completion=completion,
            metadata=TrainingPairMetadata(
                regime=record.market_regime,
                agent_id=record.agent_id,
                outcome_pnl=float(record.outcome_pnl or 0.0),
                confidence=record.confidence,
                is_replay=is_replay,
                dataset_version=dataset_version,
                unmatched_negative=unmatched_negative,
            ),
        )

    def _extract_timestamp(self, prompt: str) -> datetime:
        for line in prompt.splitlines():
            if line.startswith("- timestamp: "):
                value = line.removeprefix("- timestamp: ").strip()
                return datetime.fromisoformat(value)
        return datetime.now(UTC)

    def _regime_distribution(self, records: list[DecisionLogRecord]) -> dict[str, float]:
        counts = self._regime_counts(records)
        total = max(1, len(records))
        return {regime: counts[regime] / total for regime in REGIMES}

    def _write_jsonl(self, target: Path, rows: list[TrainingPair]) -> None:
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row.model_dump_json())
                handle.write("\n")
