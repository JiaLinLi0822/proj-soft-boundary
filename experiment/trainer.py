"""Trainer wrapper class plus stateless model ops shared with non-training callers.

The caller is responsible for constructing the model and the dataloaders
externally and then passing them in.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import TrainConfig
from networks import TinyGRUPolicy


Metrics = Dict[str, Any]


def rollout_sequence(model: TinyGRUPolicy, x: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Roll the GRU over a padded (B, T, D) batch respecting `lengths`.

    Returns dict with 'states' (B, T, H) and 'action_logits' (B, T, A).
    """
    batch_size, time_steps, _ = x.shape
    valid_mask = torch.arange(time_steps, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

    hidden = torch.zeros(batch_size, model.hidden_dim, device=x.device, dtype=x.dtype)
    hidden_steps: List[torch.Tensor] = []
    action_logits_steps: List[torch.Tensor] = []

    for t in range(time_steps):
        active_t = valid_mask[:, t].unsqueeze(1)
        hidden_candidate = model.gru(x[:, t, :], hidden)
        hidden = torch.where(active_t, hidden_candidate, hidden)
        hidden_steps.append(hidden)
        action_logits_steps.append(model.readout(hidden))

        if t + 1 < time_steps:
            keep_next = valid_mask[:, t + 1].unsqueeze(1)
            hidden = torch.where(keep_next, hidden, torch.zeros_like(hidden))

    return {
        "states": torch.stack(hidden_steps, dim=1),
        "action_logits": torch.stack(action_logits_steps, dim=1),
    }


def compute_batch_loss(action_logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    loss_pos = F.cross_entropy(
        action_logits.reshape(-1, action_logits.shape[-1]),
        y.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(mask)
    mean_loss = loss_pos.sum() / mask.sum().clamp_min(1.0)
    stats = {
        "loss_sum": float(loss_pos.sum().item()),
        "loss_weight": float(mask.sum().item()),
    }
    return mean_loss, stats


def recurrent_l1_penalty(model: TinyGRUPolicy) -> torch.Tensor:
    """Sum of |W_hh| across all GRU recurrent weight matrices."""
    total = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if "weight_hh" in name:
            total = total + param.abs().sum()
    return total


def evaluate(model: TinyGRUPolicy, loader: DataLoader, device: torch.device) -> Metrics:
    """Evaluate timestep-mean loss, trial-mean trajectory NLL, and accuracy."""
    model.eval()

    total_loss = 0.0
    total_weight = 0.0
    n_final = 0
    n_final_correct = 0
    confusion = np.zeros((2, 2), dtype=int)

    with torch.no_grad():
        for x, y, lengths, mask in loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            mask = mask.to(device)

            outputs = rollout_sequence(model, x, lengths)
            logits = outputs["action_logits"]
            _, batch_stats = compute_batch_loss(logits, y, mask)
            total_loss += batch_stats["loss_sum"]
            total_weight += batch_stats["loss_weight"]

            batch_index = torch.arange(logits.shape[0], device=device)
            t_last = lengths - 1
            pred = torch.argmax(logits[batch_index, t_last], dim=-1)
            true = y[batch_index, t_last]

            for true_i, pred_i in zip(true.tolist(), pred.tolist()):
                if true_i in (0, 1) and pred_i in (0, 1):
                    confusion[int(true_i), int(pred_i)] += 1
                n_final += 1
                n_final_correct += int(int(pred_i) == int(true_i))

    return {
        "loss": float(total_loss / max(total_weight, 1e-9)),
        "trajectory_nll": float(total_loss / max(n_final, 1)),
        "final_acc": float(n_final_correct / max(n_final, 1)),
        "n_final": int(n_final),
        "confusion_2x2": confusion.tolist(),
    }


class Trainer:
    """Train an externally constructed model and DataLoaders."""

    def __init__(self, model: TinyGRUPolicy, train_loader: DataLoader, val_loader: Optional[DataLoader], device: torch.device, config: TrainConfig, train_eval_loader: Optional[DataLoader] = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_eval_loader = (
            train_eval_loader if train_eval_loader is not None else train_loader
        )
        self.device = device
        self.cfg = config

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _epoch_train(self) -> float:
        self.model.train()
        loss_sum = 0.0
        weight_sum = 0.0

        for x, y, lengths, mask in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            lengths = lengths.to(self.device)
            mask = mask.to(self.device)

            outputs = rollout_sequence(self.model, x, lengths)
            loss, stats = compute_batch_loss(outputs["action_logits"], y, mask)

            if self.cfg.recurrent_l1_lambda > 0.0:
                loss = loss + self.cfg.recurrent_l1_lambda * recurrent_l1_penalty(self.model)

            loss_sum += stats["loss_sum"]
            weight_sum += stats["loss_weight"]

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss_sum / max(weight_sum, 1e-9)

    def fit(self) -> Dict[str, Any]:
        best: Dict[str, Any] = {
            "val_loss": float("inf"),
            "val_final_acc": -1.0,
            "epoch": -1,
            "state_dict": None,
        }
        history: List[Dict[str, float]] = []
        epochs_without_improve = 0
        epochs_trained = 0
        stopped_early = False

        epoch_iter = range(1, self.cfg.epochs + 1)
        if self.cfg.show_progress:
            epoch_iter = tqdm(epoch_iter, desc=self.cfg.progress_desc, leave=self.cfg.progress_leave)

        for epoch in epoch_iter:
            epochs_trained = int(epoch)
            epoch_train_loss = self._epoch_train()

            # Final nested-CV refits use all outer-training data for a fixed
            # number of epochs, so there is deliberately no validation pass.
            if self.val_loader is None:
                if self.cfg.show_progress:
                    epoch_iter.set_postfix(refresh=False, train_loss=f"{epoch_train_loss:.4f}")
                continue

            val_metrics = evaluate(self.model, self.val_loader, self.device)

            if self.cfg.record_history:
                train_metrics = evaluate(self.model, self.train_eval_loader, self.device)
                history.append(
                    {
                        "epoch": float(epoch),
                        "train_loss": float(epoch_train_loss),
                        "train_final_acc": float(train_metrics["final_acc"]),
                        "val_loss": float(val_metrics["loss"]),
                        "val_final_acc": float(val_metrics["final_acc"]),
                    }
                )

            if self.cfg.show_progress:
                epoch_iter.set_postfix(
                    refresh=False,
                    train_loss=f"{epoch_train_loss:.4f}",
                    val_loss=f"{val_metrics['loss']:.4f}",
                    val_acc=f"{val_metrics['final_acc']:.4f}",
                )

            if val_metrics["loss"] < best["val_loss"]:
                best = {
                    "val_loss": float(val_metrics["loss"]),
                    "val_final_acc": float(val_metrics["final_acc"]),
                    "epoch": int(epoch),
                    "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                }
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if (
                self.cfg.early_stopping_patience is not None
                and epochs_without_improve >= int(self.cfg.early_stopping_patience)
            ):
                stopped_early = True
                break

        if self.val_loader is None:
            return {
                "model": self.model,
                "device": str(self.device),
                "best_epoch": int(epochs_trained),
                "best_val_loss": None,
                "best_val_final_acc": None,
                "epochs_trained": int(epochs_trained),
                "stopped_early": False,
                "final_train": evaluate(self.model, self.train_eval_loader, self.device),
                "final_val": None,
                "history": history,
            }

        if best["state_dict"] is not None:
            self.model.load_state_dict(best["state_dict"])

        final_train = evaluate(self.model, self.train_eval_loader, self.device)
        final_val = evaluate(self.model, self.val_loader, self.device)

        return {
            "model": self.model,
            "device": str(self.device),
            "best_epoch": int(best["epoch"]),
            "best_val_loss": float(best["val_loss"]),
            "best_val_final_acc": float(best["val_final_acc"]),
            "epochs_trained": int(epochs_trained),
            "stopped_early": bool(stopped_early),
            "final_train": final_train,
            "final_val": final_val,
            "history": history,
        }


__all__ = [
    "rollout_sequence",
    "compute_batch_loss",
    "recurrent_l1_penalty",
    "evaluate",
    "Trainer",
]
