"""Trainer agent orchestrating autonomous retraining workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from loguru import logger

from training.dataset_builder import DatasetBuilder


@dataclass
class TrainingState:
    """Mutable training control state."""

    running: bool = False
    failure_streak: int = 0


class TrainerAgent:
    """Coordinates dataset build, training, evaluation, and promotion triggers."""

    agent_id = "trainer_agent"

    def __init__(self, dataset_output_dir: Path, min_outcomes: int = 500) -> None:
        self._state = TrainingState()
        self._lock = Lock()
        self._min_outcomes = min_outcomes
        self._dataset_builder = DatasetBuilder(output_dir=dataset_output_dir)

    def should_trigger(self, outcome_record_count: int) -> bool:
        """Return true when threshold-based training criteria is met."""

        return outcome_record_count >= self._min_outcomes

    def begin_run(self) -> bool:
        """Acquire run lock; return False when a run is already active."""

        with self._lock:
            if self._state.running:
                return False
            self._state.running = True
            return True

    def complete_run(self, succeeded: bool) -> None:
        """Finalize run state and update failure streak policy."""

        with self._lock:
            self._state.running = False
            if succeeded:
                self._state.failure_streak = 0
            else:
                self._state.failure_streak += 1

            logger.info(
                "training_run_completed",
                succeeded=succeeded,
                failure_streak=self._state.failure_streak,
            )

    @property
    def failure_streak(self) -> int:
        """Expose current failure streak count."""

        return self._state.failure_streak


def main() -> None:
    """Entrypoint for containerized trainer service."""

    logger.info("trainer_standby")


if __name__ == "__main__":
    main()
