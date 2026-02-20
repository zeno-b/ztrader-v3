"""Adapter registry contracts for Minio and MLflow metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class AdapterRecord:
    """Persisted adapter metadata entry."""

    agent_id: str
    adapter_version: str
    dataset_version: str
    run_id: str
    stage: str
    created_at: str


class AdapterRegistry:
    """Local metadata registry used as a stand-in for Minio + MLflow."""

    def __init__(self, registry_path: Path) -> None:
        self._registry_path = registry_path
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._registry_path.exists():
            self._registry_path.write_text("[]\n", encoding="utf-8")

    def register(
        self,
        *,
        agent_id: str,
        adapter_version: str,
        dataset_version: str,
        run_id: str,
        stage: str,
    ) -> AdapterRecord:
        """Append a new adapter metadata record."""

        record = AdapterRecord(
            agent_id=agent_id,
            adapter_version=adapter_version,
            dataset_version=dataset_version,
            run_id=run_id,
            stage=stage,
            created_at=datetime.now(UTC).isoformat(),
        )
        payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        payload.append(record.__dict__)
        self._registry_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return record

    def latest_for_agent(self, agent_id: str, stage: str) -> AdapterRecord | None:
        """Return most recent adapter metadata by stage."""

        payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
        filtered = [
            entry
            for entry in payload
            if entry["agent_id"] == agent_id and entry["stage"] == stage
        ]
        if not filtered:
            return None
        latest = filtered[-1]
        return AdapterRecord(**latest)
