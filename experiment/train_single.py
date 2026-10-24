#!/usr/bin/env python3
"""Train one or more tiny GRUs with a train/validation split."""

import argparse
import time
from pathlib import Path

from torch.utils.data import DataLoader

from checkpoints import save_checkpoint, save_loss_curve_png
from config import TrainConfig
from dataset import KiraDataset, collate_trialseq_batch, split_dataset
from networks import TinyGRUPolicy
from trainer import Trainer
from utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat")
    parser.add_argument("--monkey-id", type=int, choices=[1, 2], default=2)
    parser.add_argument("--hidden", type=int, nargs="+", default=[2])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recurrent-l1-lambda", type=float, default=1e-3)
    parser.add_argument("--diagonal-readout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=200)
    parser.add_argument("--use-num-accum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remove-trump-trials", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--last-accum-override", type=float)
    parser.add_argument("--outdir", default="runs/single")
    parser.add_argument("--save-loss-curve", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    data_path = Path(args.mat) if args.mat else (
        project_dir
        / "neuralSPRT"
        / "data"
        / ("dataJ.mat" if args.monkey_id == 2 else "dataE.mat")
    )
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    dataset = KiraDataset(data_path, args.monkey_id,
        use_num_accum=args.use_num_accum,
        remove_trump_trials=args.remove_trump_trials,
        last_accum_override=args.last_accum_override,
    )
    if not dataset:
        raise RuntimeError("No valid sequences built from input data")

    train_data, val_data, split = split_dataset(
        dataset, val_frac=args.val_frac, seed=args.seed
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_trialseq_batch,)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_trialseq_batch,)
    train_eval_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_trialseq_batch,)
    device = get_device(args.device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    monkey = "MonkeyE" if args.monkey_id == 1 else "MonkeyJ"

    for hidden_dim in args.hidden:
        set_seed(args.seed)
        config = TrainConfig(
            hidden_dim=hidden_dim,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            recurrent_l1_lambda=args.recurrent_l1_lambda,
            diagonal_readout=args.diagonal_readout,
            early_stopping_patience=args.early_stopping_patience,
            progress_desc=f"h={hidden_dim}",
        )
        model = TinyGRUPolicy(
            input_dim=train_data[0].x.shape[1],
            hidden_dim=hidden_dim,
            num_actions=3,
            diagonal_readout=args.diagonal_readout,
        )
        result = Trainer(
            model,
            train_loader,
            val_loader,
            device=device,
            config=config,
            train_eval_loader=train_eval_loader,
        ).fit()

        run_dir = (Path(args.outdir)/ monkey / f"h{hidden_dim}_l1{args.recurrent_l1_lambda:g}_seed{args.seed}_{timestamp}")
        metrics = {key: value for key, value in result.items() if key not in {"model", "device", "history"}}
        
        save_checkpoint(
            str(run_dir / "model.pt"),
            {
                "state_dict": result["model"].state_dict(),
                "hidden_dim": hidden_dim,
                "input_dim": train_data[0].x.shape[1],
                "num_actions": 3,
                "diagonal_readout": args.diagonal_readout,
                "config": vars(args) | {"mat": str(data_path)},
                "split": split,
                **metrics,
            },
        )
        if args.save_loss_curve:
            save_loss_curve_png(result["history"], str(run_dir / "loss_curve.png"))
        print(f"Saved {run_dir / 'model.pt'}")
        print(metrics)


if __name__ == "__main__":
    main()
