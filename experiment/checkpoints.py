"""Checkpoint I/O and loss-curve plotting."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import torch

from utils import get_device
from networks import TinyGRUPolicy


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_policy_checkpoint(ckpt_path: str, device: str):
    """Load a trusted training checkpoint and return (model, ckpt_dict, device)."""
    # Nested-CV checkpoints contain NumPy metadata as well as tensor weights.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    model = TinyGRUPolicy(
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_actions=int(ckpt["num_actions"]),
        diagonal_readout=bool(ckpt["diagonal_readout"]),
    )
    model.load_state_dict(state_dict)
    dev = get_device(device)
    return model.to(dev).eval(), ckpt, dev


def save_loss_curve_png(history: List[Dict[str, float]], path: str) -> None:
    import matplotlib.pyplot as plt

    if not history:
        return

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    epochs_h = [int(h["epoch"]) for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs_h, [h["train_loss"] for h in history], label="train (epoch mean)", color="C0")
    ax.plot(epochs_h, [h["val_loss"] for h in history], label="val", color="C1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
