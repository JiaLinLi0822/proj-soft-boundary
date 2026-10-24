# Normative Models and Figures 1–4

## Overview

This folder contains the normative sequential-sampling models accompanying
[*Stochastic Decision Policy Is Optimal Under Costly Control*](https://www.biorxiv.org/content/10.64898/2026.10.06.757200v1.abstract).
Observers choose between two hypotheses or pay a cost to sample another cue.
The models compare deterministic decisions, stochastic action selection,
representation noise, and their combination under finite and infinite horizons.

## Folder Contents

| File | Purpose |
| --- | --- |
| `noise_model_runner.py` | Shared dynamic-programming solver, trial simulation, and diagnostic plotting. Supports representation, execution, and mixed noise. |
| `fig1.py` | Task schematic, action values, and deterministic versus soft policies. |
| `fig2.py` | Soft policies and the KL cost-of-control tradeoff; uses its own grid-aligned value solver. |
| `fig3.py` | Internal hard boundaries and apparent soft boundaries under representation noise. |
| `fig4.py` | Sampling policies for four observer types with and without a deadline. |
| `fig1.ipynb`–`fig4.ipynb` | Interactive figure versions. Some differ from the scripts; the commands below use the `.py` versions. |
| `fig1.png`–`fig4.png`, `fig1.svg`–`fig4.svg` | Generated figure files. |

## Setup

Install the dependencies using the [main README](../README.md#installation).
No behavioral data, fitted checkpoints, or GPU are required.

## Reproduce Figures 1–4

Run these commands from the **repository root** with the environment activated:

```bash
python normative/fig1.py
python normative/fig2.py
python normative/fig3.py
python normative/fig4.py
```

Each script computes its models and saves `figN.png` and `figN.svg` beside the
script in `normative/`. Existing files with those names are overwritten.

| Figure | Analyses and panels | Supplied settings |
| --- | --- | --- |
| 1 | Task and cue distributions (A), action values (B), deterministic policy and simulations (C–E), soft policy and simulations (F–H). | Sampling cost `0.04`, softmax temperature `0.04`, 50,000 trials per observer, seed `1`. |
| 2 | Policies in the action-probability simplex (A), sampling/choice probability versus belief (B), expected gain and KL-regularized gain versus control cost (C). | Sampling cost `0.04`, temperature levels in `LEVEL_T`, KL weight `LAMBDA_A = 0.12`. |
| 3 | Internal versus ground-truth policies (A–B), the same trials in internal versus ground-truth evidence coordinates (C–D), and a representation-noise sweep (E–F). | Cost `0.04`, per-step representation-noise SD `0.25`, 50,000 trials, seed `1`. |
| 4 | Sampling probability for ideal, soft-policy, noisy-belief, and mixed observers: infinite horizon (A–D) and finite horizon (E–H). | Cost `0.004`, representation-noise SD `0.30`, temperature `0.010`, time steps 1–10, deadline `11`. Curves are analytical. |

Retain the constants at the top of each script to reproduce its supplied
configuration. Changing trial counts or seeds changes Monte Carlo distributions;
changing costs, noise levels, or grid settings changes model predictions.
Figures 1, 3, and 4 use the shared solver; Figure 2 is self-contained.

## Run Additional Model Analyses

For example, simulate a mixed-noise observer with a finite horizon:

```bash
python normative/noise_model_runner.py \
  --noise mixed --horizon finite \
  --n-trials 1000 --seed 1 --n-gh 81 \
  --no-show --save-dir normative/figures
```

Choose `--noise representation`, `execution`, or `mixed`, and
`--horizon finite` or `infinite`. Additional settings include `--cost`,
`--sigma-repr`, `--tau-exec`, `--max-timestep`, `--deadline`, and `--n-l`.
Use `python normative/noise_model_runner.py --help` for all options.

The runner saves three diagnostic SVGs in `--save-dir`, with filenames beginning
with the noise and horizon names: `*_value_summary.svg`,
`*_ground_distribution.svg`, and `*_ground_policy.svg`. These are diagnostic
plots; the numbered manuscript layouts are assembled by `fig1.py`–`fig4.py`.
Use `--skip-plots` to run the model and simulation without saving plots.

## Citation

See the [project citation](../README.md#citation).
