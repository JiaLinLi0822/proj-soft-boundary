# Soft Boundary

Code for studying soft decision boundaries in sequential evidence accumulation.
The project compares ideal observers, stochastic action selection (execution
noise), and noisy internal evidence (representation noise), and fits small GRU
policies to monkey choices. 

## Setup

Use Python 3.10 or later. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` includes the numerical, plotting, PyTorch, HDF5, and statistical
packages used by both workflows. For a headless machine, set `MPLBACKEND=Agg`.
The notebooks additionally require Jupyter; the commands below use Python scripts.

`requirements-lock.txt` records exact dependency versions. To install those
versions on a compatible platform, use
`python -m pip install -r requirements-lock.txt`.

## Code structure

| Location | Purpose |
| --- | --- |
| `normative/noise_model_runner.py` | Shared dynamic programming, simulation, and diagnostic plots for representation, execution, and mixed noise with finite or infinite horizons. |
| `normative/fig1.py`–`fig4.py` | Standalone scripts assembling the numbered normative figures. |
| `normative/*.ipynb` | Interactive figure versions; some differ from the scripts. Use the `.py` files for the reproduction instructions below. |
| `experiment/dataset.py`, `environment.py`, `utils.py` | Read monkey data, encode shapes, split trials, and generate task stimuli. |
| `experiment/networks.py`, `config.py`, `trainer.py` | Tiny GRU policies, training settings, and training/evaluation. |
| `experiment/train_single.py`, `train_nested_cv.py`, `nested_cv.py` | Single fits, hidden-size sweeps, and nested cross-validation. |
| `experiment/checkpoints.py`, `simulate.py` | Save/load fitted models and generate model trials. |
| `experiment/analysis.py`, `behavior.py`, `policy.py`, `representation.py`, `plotting_style.py` | Behavior, sampling policy, and hidden-state figures. |
| `experiment/train_nested_cv.sbatch` | Cluster-specific Slurm example; edit paths, account, and environment before use. |

## Reproduce Figures 1–4

Run from the project root with the environment activated:

```bash
python normative/fig1.py
python normative/fig2.py
python normative/fig3.py
python normative/fig4.py
```

Each script writes `normative/figN.png` and `normative/figN.svg`, replacing any
existing files with those names. No external data or GPU is needed. Keep the
constants in the scripts unchanged to use the supplied figure settings.

| Figure | Content | Main settings |
| --- | --- | --- |
| 1 | Task, action values, and deterministic versus soft policies and trial distributions. | Sampling cost `0.04`, temperature `0.04`, 50,000 trials per observer, seed `1`. |
| 2 | Policy softening and the KL cost-of-control tradeoff. Uses its own grid-aligned value solver. | Cost `0.04`, example temperatures in `LEVEL_T`, KL weight `0.12`. |
| 3 | Hard internal boundaries appearing soft in ground-truth evidence coordinates under representation noise. | Cost `0.04`, per-step noise SD `0.25`, 50,000 trials, seed `1`. |
| 4 | Sampling policies for ideal, soft, noisy-belief, and mixed observers with and without a deadline. | Cost `0.004`, noise SD `0.30`, temperature `0.010`, 10 plotted steps; analytical curves. |

For other noise/horizon combinations, for example:

```bash
python normative/noise_model_runner.py \
  --noise mixed --horizon finite --n-trials 1000 \
  --n-gh 81 --no-show --save-dir normative/figures
```

This produces diagnostic SVGs in the requested directory. It does not assemble
one of the four numbered figures. The default high-resolution noisy models can
be slower than the deterministic models.

## Reproduce the experimental analyses

**Original data and fitted checkpoints are not included.** Obtain the Kira-lab
MATLAB v7.3/HDF5 files `dataE.mat` (Eli, `--monkey-id 1`) and `dataJ.mat` (Joey,
`--monkey-id 2`). The loader expects an `info/TM` matrix in MATLAB storage
orientation; columns of its transpose contain choices, timing, and shape IDs
(see `dataset.py`). Ordinary pre-v7.3 MAT files are not supported.

Always pass `--mat` explicitly. Without it, the scripts look for the legacy path
`neuralSPRT/data/dataE.mat` or `dataJ.mat` under the project root. From the root,
train a two-unit model and make analysis figures:

```bash
python experiment/train_single.py \
  --mat /path/to/dataE.mat --monkey-id 1 --hidden 2 \
  --epochs 500 --lr 1e-3 --batch-size 128 --val-frac 0.2 \
  --seed 30 --device cpu --recurrent-l1-lambda 1e-5 \
  --outdir experiment/runs/single

# Replace the checkpoint path below with the model.pt printed by training.
python experiment/analysis.py \
  --mat /path/to/dataE.mat --monkey-id 1 \
  --ckpt experiment/runs/single/MonkeyE/RUN_DIRECTORY/model.pt \
  --n-trials 10000 --rollout-seed 45 --device cpu \
  --outdir experiment/results
```

Training writes a checkpoint and loss curve for each hidden size. Analysis writes
PNG/SVG pairs and `analysis_summary.json` under `experiment/results/Monkey E/`
(or `Monkey J/`): behavior (`behavior`, `behavior2`), decision thresholds,
`psample`, `psample_scatter`, `psample_variability`, hidden-state plots, and
representation-noise plots. Two-dimensional hidden-state plots require at least
two hidden units and are skipped for a one-unit model. Use separate `--outdir`
values to retain analyses of multiple checkpoints. Load only trusted `.pt` files.

For the capacity-versus-likelihood plot, run nested CV:

```bash
python experiment/train_nested_cv.py \
  --mat /path/to/dataE.mat --monkey-id 1 \
  --hidden 1 2 3 4 5 10 20 --recurrent-l1-lambda 1e-5 \
  --init-seeds 0 1 2 --outer-folds 5 --inner-epochs 1000 \
  --batch-size 128 --lr 3e-3 --seed 0 --device cpu \
  --num-workers 0 --outdir experiment/runs/nested_cv
```

Each hidden size is evaluated separately; inner folds select L1 and initialization
seed, then the selected configuration is refitted on all outer-training trials.
Outputs include inner/outer checkpoints, CSV summaries, and
`test_trajectory_nll_vs_hidden.{png,svg}`. Increasing `--num-workers` above 1
parallelizes inner fits; use CPU for this mode. These commands are examples,
not a claim that the unavailable paper's empirical fits used these settings.
