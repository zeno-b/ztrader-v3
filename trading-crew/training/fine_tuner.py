"""LoRA fine-tuning pipeline configuration and orchestration stub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(frozen=True)
class FineTuneConfig:
    """Fixed fine-tuning hyperparameters (human-governed defaults)."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    seed: int = 42


class FineTuner:
    """Wrapper for Unsloth + TRL adapter-only fine-tuning."""

    def __init__(self, config: FineTuneConfig | None = None) -> None:
        self._config = config or FineTuneConfig()

    def run(
        self,
        *,
        agent_id: str,
        base_model: str,
        train_jsonl: Path,
        validation_jsonl: Path,
        output_dir: Path,
    ) -> Path:
        """
        Execute LoRA training and return adapter artifact path.

        This repository scaffold provides a strict, typed contract and logging.
        Production implementation should call Unsloth + TRL SFTTrainer here.
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = output_dir / f"{agent_id}_adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "fine_tune_started",
            agent_id=agent_id,
            base_model=base_model,
            train_jsonl=str(train_jsonl),
            validation_jsonl=str(validation_jsonl),
            config=self._config.__dict__,
        )
        logger.info("fine_tune_completed", adapter_path=str(adapter_path))
        return adapter_path
