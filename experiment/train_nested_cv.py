#!/usr/bin/env python3
"""Train nested-CV tiny GRUs on one monkey dataset."""

import argparse
import time
from pathlib import Path

import pandas as pd

from dataset import KiraDataset
from nested_cv import NestedCV, plot_test_trajectory_nll_by_hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey-id", type=int, choices=[1, 2], default=1)
    parser.add_argument("--mat")

    #hyperparameters
    parser.add_argument("--hidden", type=int, nargs="+", default=[1, 2, 3, 4, 5, 10, 20])
    parser.add_argument("--recurrent-l1-lambda", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2])
    
    #nested cross validation
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int)
    parser.add_argument("--inner-epochs", type=int, default=1000)

    #training setup
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--diagonal-readout", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-num-accum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remove-trump-trials", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--last-accum-override", type=float)
    parser.add_argument("--outdir", default="runs/nested_cv")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    data_path = Path(args.mat) if args.mat else (
        project_dir / "neuralSPRT" / "data" /
        ("dataJ.mat" if args.monkey_id == 2 else "dataE.mat")
    )
    dataset = KiraDataset(
        str(data_path),
        args.monkey_id,
        use_num_accum=args.use_num_accum,
        remove_trump_trials=args.remove_trump_trials,
        last_accum_override=args.last_accum_override,
    )
    monkey = "MonkeyE" if args.monkey_id == 1 else "MonkeyJ"
    outdir = Path(args.outdir) / monkey / time.strftime(
        f"nestedcv_%Y%m%d_%H%M%S_seed{args.seed}"
    )

    summaries = []
    for hidden_dim in args.hidden:
        print(f"\nNested CV for hidden_dim={hidden_dim}")
        summaries.append(NestedCV(
            base_dir=outdir,
            outer_splits=args.outer_folds,
            inner_splits=args.inner_folds,
            hidden_dim=hidden_dim,
            recurrent_l1_grid=args.recurrent_l1_lambda,
            init_seeds=args.init_seeds,
            fold_seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            inner_epochs=args.inner_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            diagonal_readout=args.diagonal_readout,
            early_stopping_patience=args.early_stopping_patience,
        ).fit(dataset))
    summary = pd.concat(summaries, ignore_index=True)

    metadata = {
        "data_mat": str(data_path.resolve()),
        "monkey_id": args.monkey_id,
        "use_num_accum": args.use_num_accum,
        "remove_trump_trials": args.remove_trump_trials,
        "last_accum_override": args.last_accum_override,
        "outer_splits": args.outer_folds,
        "fold_seed": args.seed,
        "initialization_seeds": str(args.init_seeds),
    }
    
    for name, value in metadata.items():
        summary[name] = value
    summary = summary[list(metadata) + [c for c in summary if c not in metadata]]
    summary.to_csv(outdir / "summary.csv", index=False)
    plot_test_trajectory_nll_by_hidden(
        summary, outdir / "test_trajectory_nll_vs_hidden"
    )
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nWrote nested-CV results to: {outdir}")


if __name__ == "__main__":
    main()
