#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    plot_representation_noise_heatmap,
    plot_representation_noise_over_time,
)
from policy import (
    plot_psample,
    plot_psample_scatter,
    plot_sampling_variability,
)
from simulate import simulate_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey-id", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--ckpt",
        required=True,
    )
    parser.add_argument("--mat", default=None)
    parser.add_argument("--outdir", default=None)  # None uses fit_tinyrnns/results
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-trials", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--rollout-seed", type=int, default=45)
    parser.add_argument("--include-trump", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    monkey_id = args.monkey_id
    monkey_name = "Monkey E" if monkey_id == 1 else "Monkey J"
    ckpt_path = Path(args.ckpt)
    mat_path = Path(args.mat) if args.mat else (
        project_dir
        / "neuralSPRT"
        / "data"
        / ("dataE.mat" if monkey_id == 1 else "dataJ.mat")
    )
    outdir = (Path(args.outdir) if args.outdir else script_dir / "results") / monkey_name
    outdir.mkdir(parents=True, exist_ok=True)

    model, ckpt, device = load_policy_checkpoint(str(ckpt_path), args.device)
    hidden_dim = int(ckpt["hidden_dim"])
    dataset = KiraDataset(
        str(mat_path), int(monkey_id), use_num_accum=True, remove_trump_trials=True
    )

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
        evidence_range=4.0,
        evidence_bin_width=0.1,
        min_bin_count=30,
        title=None,
    )

    # Plot the hidden states
    if hidden_dim >= 2:
        plot_hidden_by_timestep(
            model_trials,
            out_path=str(outdir / "hidden_by_timestep"),
            max_time_step=10,
        )
        plot_hidden_by_evidence(
            model_trials,
            out_path=str(outdir / "hidden_by_evidence"),
            max_time_step=10,
        )
        plot_hidden_by_policy(
            model_trials,
            out_path=str(outdir / "hidden_by_policy"),
            max_time_step=10,
            n_bins=100,
            min_bin_count=1,
        )
    noise, noise_by_time = compute_representation_noise(model_trials)
    plot_representation_noise_over_time(
        noise_by_time, str(outdir / "representation_noise_over_time")
    )
    plot_representation_noise_heatmap(
        noise, str(outdir / "representation_noise_heatmap")
    )

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


if __name__ == "__main__":
    main()
