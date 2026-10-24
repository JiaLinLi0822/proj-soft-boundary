from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Config for a single Trainer.fit() run."""

    hidden_dim: int = 2
    seed: int = 0
    device: str = "auto"

    batch_size: int = 1024
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 0.0

    recurrent_l1_lambda: float = 0.0
    diagonal_readout: bool = False

    early_stopping_patience: int | None = None

    record_history: bool = True
    show_progress: bool = True
    progress_desc: str = "Training"
    progress_leave: bool = True

__all__ = ["TrainConfig"]
