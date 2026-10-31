#!/usr/bin/env python3
"""Analyze both monkeys using their best h=2 nested-CV checkpoints by default."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from checkpoints import load_policy_checkpoint
from dataset import KiraDataset
from behavior import (
    plot_behavior,
    plot_behavior2,
    plot_decision_threshold_distribution,
    trials_from_dataset,
)
from representation import (
    compute_representation_noise,
    plot_hidden_by_evidence,
    plot_hidden_by_policy,
    plot_hidden_by_timestep,
    plot_representation_noise,
)
from policy import (
    plot_psample,
    plot_psample_scatter,
    plot_sampling_variability,
)
from simulate import simulate_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey-id", type=int, choices=[1, 2], help="Analyze one monkey; default: both.")
    parser.add_argument("--ckpt", help="Override automatic selection; requires --monkey-id.")
    parser.add_argument("--mat", default=None)
    parser.add_argument("--outdir", default=None)  # Default: experiment/results
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-trials", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--rollout-seed", type=int, default=45)
    parser.add_argument("--include-trump", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.monkey_id is None and (args.ckpt or args.mat):
        parser.error("--ckpt and --mat require --monkey-id to identify the matching monkey")
    return args


def select_best_h2_checkpoint(root, monkey_id):
    """Select minimum outer-test trajectory NLL after keeping the latest fold reruns."""
    monkey = {1: "MonkeyE", 2: "MonkeyJ"}[monkey_id]
    paths = sorted(Path(root).glob(f"slurm_*/h2/outer*/{monkey}/nestedcv_*/summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No h=2 summaries found for {monkey} in {root}")
    summary = pd.concat([
        pd.read_csv(path).assign(source_summary=str(path), run_timestamp=path.parent.name)
        for path in paths
    ], ignore_index=True)
    summary = summary[(summary.monkey_id == monkey_id) & (summary.hidden_dim == 2)]
    summary = summary.sort_values("run_timestamp").drop_duplicates(
        ["fold_seed", "outer_fold"], keep="last"
    )
    if summary.empty or not np.isfinite(summary.test_trajectory_nll).all():
        raise ValueError(f"Missing or nonfinite h=2 test NLL for {monkey}")
    best = summary.loc[summary.test_trajectory_nll.idxmin()]
    checkpoint = (Path(best.source_summary).parent / best.model_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint}")
    print(f"{monkey}: selected {checkpoint}; test trajectory NLL={best.test_trajectory_nll:.6f}", flush=True)
    return checkpoint


def analyze_monkey(args, monkey_id, ckpt_path):
    script_dir = Path(__file__).resolve().parent
    monkey_name = "Monkey E" if monkey_id == 1 else "Monkey J"
    mat_path = Path(args.mat) if args.mat else (
        script_dir / "data"
        / ("dataE.mat" if monkey_id == 1 else "dataJ.mat")
    )
    outdir = (Path(args.outdir) if args.outdir else script_dir / "results") / monkey_name
    outdir.mkdir(parents=True, exist_ok=True)

    model, ckpt, device = load_policy_checkpoint(str(ckpt_path), args.device)
    hidden_dim = int(ckpt["hidden_dim"])
    if args.ckpt is None and hidden_dim != 2:
        raise ValueError(f"Automatically selected checkpoint must have h=2: {ckpt_path}")
    dataset = KiraDataset(str(mat_path), int(monkey_id), use_num_accum=True, remove_trump_trials=True)

    # Simulate the model and the monkey trials
    monkey_trials = trials_from_dataset(dataset)
    model_trials = simulate_model(
        model,
        monkey_id=monkey_id,
        n_trials=args.n_trials,
        rollout_seed=args.rollout_seed,
        include_trump=args.include_trump,
        max_steps=args.max_steps,
    )

    # Plot the behavior
    plot_behavior(
        monkey_trials,
        model_trials,
        int(monkey_id),
        out_path=str(outdir / "behavior"),
        title=None,
    )

    plot_behavior2(
        monkey_trials,
        model_trials,
        int(monkey_id),
        out_path=str(outdir / "behavior2"),
        title=None,
    )

    plot_decision_threshold_distribution(
        monkey_trials,
        int(monkey_id),
        out_path=str(outdir / "monkey_decision_threshold_distribution"),
    )

    # Plot the policy
    timesteps = list(range(1, 11))
    plot_psample(
        model_trials,
        out_path=str(outdir / "psample"),
        timesteps=timesteps,
        n_bins=50,
        min_bin_count=10,
        bin_start=-4.0,
        bin_end=4.1,
        figsize=(3.35, 2.55),
    )

    plot_psample_scatter(
        model_trials,
        out_path=str(outdir / "psample_scatter"),
        max_time_step=10,
        point_size=5.0,
        point_alpha=1.0,
        highlight_trial=-1,
        highlight_length=-1,
    )
    plot_sampling_variability(
        model_trials,
        out_path=str(outdir / "psample_variability"),
        timesteps=timesteps,
        evidence_range=3.2,
        evidence_bin_width=0.2,
        min_bin_count=10,
        title=None,
    )

    # Plot the hidden states
    if hidden_dim == 2:
        plot_hidden_by_timestep(model_trials, out_path=str(outdir / "hidden_by_timestep"), max_time_step=10)
        plot_hidden_by_evidence(model_trials, out_path=str(outdir / "hidden_by_evidence"), max_time_step=10)
        plot_hidden_by_policy(model_trials, out_path=str(outdir / "hidden_by_policy"), max_time_step=10, n_bins=100, min_bin_count=1)
    
    noise, noise_by_time = compute_representation_noise(model_trials)
    plot_representation_noise(noise, noise_by_time, out_path=str(outdir / "representation_noise"))

    summary = {
        "monkey": monkey_name,
        "monkey_id": int(monkey_id),
        "model": {
            "checkpoint": str(ckpt_path),
            "directory": str(ckpt_path.parent),
            "hidden_dim": hidden_dim,
            "recurrent_l1": ckpt.get("recurrent_l1"),
            "train_seed": ckpt.get("train_init_seed", ckpt.get("seed")),
            "outer_seed": ckpt.get("outer_seed"),
            "outer_fold": ckpt.get("outer_fold"),
            "inner_fold": ckpt.get("inner_fold"),
        },
        "device": str(device),
        "data_mat": str(mat_path),
        "output_directory": str(outdir),
        "n_monkey_trials": len(monkey_trials),
        "n_model_trials": len(model_trials),
        "rollout": {
            "seed": args.rollout_seed,
            "max_steps": args.max_steps,
            "include_trump": args.include_trump,
        },
    }
    summary_path = outdir / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "summary_json": str(summary_path)}, indent=2))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent / "runs" / "nested_cv"
    for monkey_id in ([args.monkey_id] if args.monkey_id else [1, 2]):
        checkpoint = Path(args.ckpt) if args.ckpt else select_best_h2_checkpoint(root, monkey_id)
        analyze_monkey(args, monkey_id, checkpoint)


if __name__ == "__main__":
    main()
