from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from checkpoints import load_policy_checkpoint, save_checkpoint
from config import TrainConfig
from dataset import collate_trialseq_batch, stratified_kfold
from networks import TinyGRUPolicy
from trainer import Trainer, evaluate
from utils import get_device, set_seed

_WORKER_SEQS = None
_WORKER_CV = None


def _worker_init(seqs, cv):
    global _WORKER_SEQS, _WORKER_CV
    import torch

    torch.set_num_threads(1)
    _WORKER_SEQS = seqs
    _WORKER_CV = cv


def _run_inner(task):
    out = _WORKER_CV._fit_model(
        [_WORKER_SEQS[i] for i in task["train_idx"]],
        [_WORKER_SEQS[i] for i in task["val_idx"]],
        hidden_dim=task["hidden_dim"],
        recurrent_l1=task["recurrent_l1"],
        seed=task["init_seed"],
        shuffle_seed=task["shuffle_seed"],
        epochs=_WORKER_CV.inner_epochs,
    )
    result = {key: value for key, value in task.items() if key not in ("train_idx", "val_idx")} | {
        "best_epoch": out["best_epoch"],
        "epochs_trained": out["epochs_trained"],
        "stopped_early": out["stopped_early"],
        "best_val_loss": out["best_val_loss"],
        "best_val_accuracy": out["best_val_final_acc"],
        "train_metrics": out["final_train"],
        "val_metrics": out["final_val"],
        "train_size": len(task["train_idx"]),
        "val_size": len(task["val_idx"]),
    }
    l1 = (
        f'{task["recurrent_l1"]:.8e}'
        .replace(".", "p").replace("+", "P").replace("-", "m")
    )
    path = (
        _WORKER_CV.base_dir / "cv_models"
        / f'outer{task["outer_fold"]:02d}'
        / f'h{task["hidden_dim"]}_l1_{l1}_seed{task["init_seed"]}'
        / f'inner{task["inner_fold"]:02d}.pt'
    )
    save_checkpoint(
        path,
        task | result | {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in out["model"].state_dict().items()
            },
            "input_dim": out["model"].input_dim,
            "num_actions": 3,
            "diagonal_readout": _WORKER_CV.diagonal_readout,
        },
    )
    return result | {"model_path": str(path)}


class NestedCV:
    """Run nested CV for one hidden size, selecting L1 and initialization seed
    by mean inner validation loss, then testing the best inner model.
    """

    def __init__(
        self,
        base_dir,
        outer_splits=5,
        inner_splits=None,
        hidden_dim=2,
        recurrent_l1_grid=(0.0,),
        init_seeds=(0, 1, 2),
        fold_seed=0,
        device="auto",
        num_workers=0,
        inner_epochs=1000,
        batch_size=1024,
        lr=1e-3,
        weight_decay=0.0,
        diagonal_readout=False,
        early_stopping_patience=100,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.hidden_dim = hidden_dim
        self.recurrent_l1_grid = recurrent_l1_grid
        self.init_seeds = init_seeds
        self.fold_seed = fold_seed
        self.device = device
        self.num_workers = num_workers
        self.inner_epochs = inner_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.diagonal_readout = diagonal_readout
        self.early_stopping_patience = early_stopping_patience

    def _fit_model(self, train_data, val_data, *, hidden_dim, recurrent_l1, seed, shuffle_seed, epochs):
        
        config = TrainConfig(
            hidden_dim=hidden_dim,
            seed=seed,
            device=self.device,
            batch_size=self.batch_size,
            epochs=epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            recurrent_l1_lambda=recurrent_l1,
            diagonal_readout=self.diagonal_readout,
            early_stopping_patience=self.early_stopping_patience,
            record_history=False,
            show_progress=False,
        )
        set_seed(seed)
        device = get_device(self.device)
        train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_trialseq_batch,
            generator=torch.Generator().manual_seed(shuffle_seed),
        )
        val_loader = DataLoader(
            val_data, batch_size=self.batch_size, collate_fn=collate_trialseq_batch,
        )
        
        model = TinyGRUPolicy(
            input_dim=train_data[0].x.shape[1],
            hidden_dim=hidden_dim,
            num_actions=3,
            diagonal_readout=self.diagonal_readout,
        )

        return Trainer(
            model,
            train_loader,
            val_loader,
            device=device,
            config=config,
            train_eval_loader=DataLoader(
                train_data, batch_size=self.batch_size,
                collate_fn=collate_trialseq_batch,
            ),
        ).fit()

    def fit(self, data, *, outer_fold=None):

        if self.inner_splits is None:
            inner_splits = self.outer_splits - 1
        else:
            inner_splits = self.inner_splits

        if self.outer_splits < 2:
            raise ValueError("outer_splits must be >= 2")
        if inner_splits < 2:
            raise ValueError("inner_splits must be >= 2")
        if outer_fold is not None and not 0 <= outer_fold < self.outer_splits:
            raise ValueError("outer_fold must be between 0 and outer_splits - 1")

        seqs = list(data)
        input_dim = seqs[0].x.shape[1]
        
        self.base_dir.mkdir(parents=True, exist_ok=True)

        grid = [
            (f"l1={l1:.8g}|seed={seed}", l1, seed)
            for l1 in self.recurrent_l1_grid
            for seed in self.init_seeds
        ]

        tasks = []
        outer_folds = dict(enumerate(stratified_kfold(
            [seq.y[-1] for seq in seqs], self.outer_splits, self.fold_seed
        )))
        if outer_fold is not None:
            outer_folds = {outer_fold: outer_folds[outer_fold]}

        for outer_fold, (train_idx, test_idx) in outer_folds.items():
            outer_seed = self.fold_seed + 1000 * outer_fold
            inner_folds = stratified_kfold(
                [seqs[i].y[-1] for i in train_idx],
                inner_splits,
                outer_seed + 7,
            )
            for config_key, recurrent_l1, init_seed in grid:
                for inner_fold, (inner_train, inner_val) in enumerate(inner_folds):
                    tasks.append(
                        {
                            "outer_fold": outer_fold,
                            "outer_seed": outer_seed,
                            "inner_fold": inner_fold,
                            "config_key": config_key,
                            "hidden_dim": self.hidden_dim,
                            "recurrent_l1": recurrent_l1,
                            "init_seed": init_seed,
                            "shuffle_seed": outer_seed + 17 * inner_fold,
                            "train_idx": train_idx[inner_train],
                            "val_idx": train_idx[inner_val],
                        }
                    )

        if self.num_workers > 1:
            with ProcessPoolExecutor(
                max_workers=min(self.num_workers, len(tasks)),
                initializer=_worker_init,
                initargs=(seqs, self),
            ) as pool:
                results = list(
                    tqdm(
                        pool.map(_run_inner, tasks),
                        total=len(tasks),
                        desc="inner fits",
                    )
                )
        else:
            global _WORKER_SEQS, _WORKER_CV
            _WORKER_SEQS, _WORKER_CV = seqs, self
            results = [_run_inner(task) for task in tqdm(tasks, desc="inner fits")]

        runs = {}
        for result in results:
            key = result["outer_fold"], result["config_key"]
            runs.setdefault(key, []).append(result)

        pd.DataFrame([
            {
                "outer_fold": run["outer_fold"],
                "inner_fold": run["inner_fold"],
                "hidden_dim": run["hidden_dim"],
                "recurrent_l1": run["recurrent_l1"],
                "init_seed": run["init_seed"],
                "shuffle_seed": run["shuffle_seed"],
                "best_epoch": run["best_epoch"],
                "val_loss": run["best_val_loss"],
                "val_accuracy": run["best_val_accuracy"],
            }
            for run in results
        ]).to_csv(self.base_dir / f"inner_results_h{self.hidden_dim}.csv", index=False)

        rows = []
        for outer_fold, (_, test_idx) in outer_folds.items():
            
            mean_val_loss = {key: float(np.mean([run["best_val_loss"] for run in runs[outer_fold, key]])) for key, _, _ in grid}
            best_key = min(mean_val_loss, key=mean_val_loss.get)
            _, recurrent_l1, init_seed = next(
                config for config in grid if config[0] == best_key
            )
            selected = min(runs[outer_fold, best_key], key=lambda run: run["best_val_loss"])
            test_data = [seqs[i] for i in test_idx]

            model, checkpoint, device = load_policy_checkpoint(selected["model_path"], self.device)
            test_metrics = evaluate(
                model,
                DataLoader(
                    test_data, batch_size=self.batch_size,
                    collate_fn=collate_trialseq_batch,
                ),
                device=device,
            )
            path = self.base_dir / (
                f"outer{outer_fold}_besth{self.hidden_dim}_l1_{recurrent_l1:.0e}_seed{init_seed}.pt"
            )

            save_checkpoint(
                str(path),
                {
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "hidden_dim": self.hidden_dim,
                    "input_dim": input_dim,
                    "num_actions": 3,
                    "diagonal_readout": self.diagonal_readout,
                    "recurrent_l1": recurrent_l1,
                    "train_init_seed": init_seed,
                    "outer_seed": self.fold_seed + 1000 * outer_fold,
                    "outer_fold": outer_fold,
                    "mean_inner_val_loss_by_config": mean_val_loss,
                    "inner_fold": selected["inner_fold"],
                    "best_epoch": selected["best_epoch"],
                    "train_idx": checkpoint["train_idx"],
                    "val_idx": checkpoint["val_idx"],
                    "train_size": selected["train_size"],
                    "test_size": len(test_data),
                    "train_metrics": selected["train_metrics"],
                    "test_metrics": test_metrics,
                },
            )
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "hidden_dim": self.hidden_dim,
                    "recurrent_l1": recurrent_l1,
                    "initialization_seed": init_seed,
                    "inner_val_loss": mean_val_loss[best_key],
                    "inner_fold": selected["inner_fold"],
                    "best_epoch": selected["best_epoch"],
                    "train_size": selected["train_size"],
                    "test_size": len(test_data),
                    "train_loss": selected["train_metrics"]["loss"],
                    "train_trajectory_nll": selected["train_metrics"]["trajectory_nll"],
                    "train_accuracy": selected["train_metrics"]["final_acc"],
                    "test_loss": test_metrics["loss"],
                    "test_trajectory_nll": test_metrics["trajectory_nll"],
                    "test_accuracy": test_metrics["final_acc"],
                    "model_path": str(path.relative_to(self.base_dir)),
                }
            )

        return pd.DataFrame(rows)


__all__ = ["NestedCV"]
