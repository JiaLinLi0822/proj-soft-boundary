# Experimental GRU Models and Analyses

## Overview

This folder contains the experimental modeling workflow for
[*Stochastic Decision Policy Is Optimal Under Costly Control*](https://www.biorxiv.org/content/10.64898/2026.10.06.757200v1.abstract).
Small GRU policies are fitted to macaque choices in a sequential-sampling task.
At each timestep, the model receives one of eight shape categories and produces
probabilities over `choose_A`, `choose_B`, and `continue`.

The workflow fits models, compares held-out likelihood across hidden sizes,
and generates behavior, sampling-policy, and hidden-state plots associated
with Figures 5–8.

## Folder Contents

| Files | Purpose |
| --- | --- |
| `dataset.py`, `utils.py` | Load MATLAB data, encode shapes, collate variable-length trials, and split data. |
| `environment.py` | Generate task stimuli for model rollouts. |
| `networks.py`, `config.py`, `trainer.py` | GRU architecture, training configuration, optimization, and evaluation. |
| `train_single.py` | Train one model or sweep hidden sizes using a train/validation split. |
| `train_nested_cv.py`, `nested_cv.py` | Nested cross-validation, hyperparameter selection, outer refits, and likelihood-versus-capacity plots. |
| `checkpoints.py` | Save/load models and plot training loss. |
| `simulate.py` | Generate model trials and matched-evidence stimulus permutations. |
| `analysis.py` | Generate the experimental analysis suite from one fitted checkpoint. |
| `behavior.py`, `policy.py`, `representation.py` | Compute and plot behavioral, sampling-policy, and hidden-state analyses. |
| `plotting_style.py` | Shared figure style and PNG/SVG export. |
| `train_nested_cv.sbatch` | Slurm job-array example for cluster fitting. |

## Setup and Data

Install dependencies using the [main README](../README.md#installation).
Run all commands below from the **repository root**.

Original data and fitted checkpoints are not included. Supply the Kira-lab
MATLAB v7.3/HDF5 files for the subject you want to analyze:

| Subject | Data file | CLI argument |
| --- | --- | --- |
| Eli | `dataE.mat` | `--monkey-id 1` |
| Joey | `dataJ.mat` | `--monkey-id 2` |

Pass the file location explicitly with `--mat /path/to/dataE.mat` or
`--mat /path/to/dataJ.mat`. Without it, scripts look under the legacy
`neuralSPRT/data/` directory at the repository root.

The loader expects `info/TM` in HDF5. After transposition, its columns contain
choices, timing, and up to 20 shape presentations; see `dataset.py` for the
subject-specific column definitions and accumulation-time correction. Ordinary
pre-v7.3 MAT files are not supported. Default preprocessing excludes trump
trials and retains the estimated number of accumulated cues.

## Fit the GRU Models

### Single model or hidden-size sweep

For example, fit a two-unit GRU to Eli:

```bash
python experiment/train_single.py \
  --mat /path/to/dataE.mat --monkey-id 1 --hidden 2 \
  --epochs 500 --lr 1e-3 --batch-size 128 --val-frac 0.2 \
  --seed 30 --device cpu --recurrent-l1-lambda 1e-5 \
  --outdir experiment/runs/single
```

Use `--hidden 1 2 3 5 10` to fit several model sizes. Each fit writes
`model.pt` and `loss_curve.png` under:

```text
experiment/runs/single/MonkeyE/h{H}_l1{L}_seed{S}_{timestamp}/
```

Joey's fits use `MonkeyJ`. The checkpoint path is printed after training.
The example settings above demonstrate the workflow; use the intended fitting
settings and data to reproduce a particular fitted model.

### Nested cross-validation and likelihood versus capacity

```bash
python experiment/train_nested_cv.py \
  --mat /path/to/dataE.mat --monkey-id 1 \
  --hidden 1 2 3 4 5 10 20 \
  --recurrent-l1-lambda 1e-1 1e-2 1e-3 1e-4 1e-5 \
  --init-seeds 0 1 2 --outer-folds 10 --inner-epochs 1000 \
  --batch-size 128 --lr 3e-3 --early-stopping-patience 200 \
  --seed 0 --device cpu --num-workers 0 \
  --outdir experiment/runs/nested_cv
```

Each hidden size runs a separate nested CV. Inner folds select recurrent L1
strength and initialization seed by mean validation loss. The selected model
is refitted on all outer-training data before evaluation on the outer test set.
By default, the number of inner folds is `outer-folds - 1`; use `--inner-folds`
to override it. Set `--num-workers` above 1 to parallelize inner fits on CPU.

Each timestamped `MonkeyE` or `MonkeyJ` run directory contains:

| Output | Content |
| --- | --- |
| `cv_models/outer*/h*/inner*.pt` | Individual inner-fold checkpoints. |
| `inner_results_hH.csv` | Inner-fit scores and selected training epochs for each hidden size. |
| `outerN_besthH_l1_*_seed*.pt` | Selected, refitted outer-fold models. |
| `summary.csv` | Outer-fold configuration and held-out performance. |
| `test_trajectory_nll_vs_hidden.png`, `.svg` | Model capacity versus held-out trajectory negative log-likelihood. |

For each outer fold, trajectory NLL is averaged across trials after summing
action NLL over all timesteps in each trial. The plotted mean weights folds by
their test-set sizes; error bars show the standard deviation across folds.

### Cluster execution

`train_nested_cv.sbatch` is a cluster-specific example. Before submitting, edit
its account, module, Python path, email, resource settings, and data location.
The script currently relies on the default `neuralSPRT/data/` path; add an
explicit `--mat /path/to/dataE.mat` argument to its Python command if needed.
It launches `train_nested_cv.py` relative to the submission directory, so submit
from `experiment/`:

```bash
cd experiment
mkdir -p cluster
sbatch train_nested_cv.sbatch
```

The array distributes hidden sizes across jobs. Its supplied epoch count is
2,000, whereas the CLI example above uses 1,000; align settings when comparing
runs. Return to the repository root for the analysis commands below.

## Reproduce the Experimental Analyses and Figures

After fitting, supply a checkpoint and its matching monkey data:

```bash
python experiment/analysis.py \
  --mat /path/to/dataE.mat --monkey-id 1 \
  --ckpt experiment/runs/single/MonkeyE/RUN_DIRECTORY/model.pt \
  --n-trials 10000 --max-steps 200 --rollout-seed 45 --device cpu \
  --outdir experiment/results
```

Replace `RUN_DIRECTORY` with the actual directory printed during training.
You can also pass an outer-fold checkpoint produced by nested CV. For Joey,
change the data file, `--monkey-id`, and checkpoint together. Load only trusted
`.pt` files.

`analysis.py` loads the model, constructs monkey trials, generates one shared set
of model rollouts, and exports the following PNG/SVG pairs under
`experiment/results/Monkey E/` or `experiment/results/Monkey J/`:

| Output stem | Analysis |
| --- | --- |
| `behavior`, `behavior2` | Alternate layouts of shape-count distributions, psychometric curves, subjective shape weights, and decision thresholds. |
| `monkey_decision_threshold_distribution` | Monkey evidence distributions for sampling and commitment across timesteps. |
| `monkey_decision_threshold_distribution_legend` | Separate legend for the threshold-distribution plot. |
| `psample` | Mean sampling probability versus cumulative evidence, grouped by timestep. |
| `psample_scatter` | Individual sampling probabilities with a highlighted trial. |
| `psample_variability` | Sampling-probability variability across matched evidence and timestep bins. |
| `hidden_by_timestep`, `hidden_by_evidence` | Hidden-state trajectories colored by time or evidence. |
| `hidden_by_policy` | Action probabilities represented in two-dimensional hidden space. |
| `representation_noise_over_time` | Hidden-state variability over time at matched evidence. |
| `representation_noise_heatmap` | Hidden-state variability by timestep and evidence level. |

`analysis_summary.json` records checkpoint, data, subject, and rollout settings.
Two-dimensional hidden-state plots are skipped for one-unit models. For models
with more than two units, these plots display the first two hidden units.
Use different `--outdir` values for different checkpoints to retain each result.

These exports provide the experimental analysis plots rather than automatically
assembling four numbered manuscript layouts. The capacity plot is generated
by nested CV, not `analysis.py`. Use the original data, selected checkpoints,
and matching settings for numerical reproduction of the preprint. Binning,
minimum counts, axis limits, and styling are set in `analysis.py` and its plotting
functions; CLI options are listed by `python experiment/analysis.py --help`.

### Matched-evidence permutations

This additional analysis is exposed through Python functions rather than the
analysis CLI. From the repository root, start Python and run:

```python
import sys
sys.path.insert(0, "experiment")
from checkpoints import load_policy_checkpoint
from simulate import simulate_permuted_evidence
from representation import plot_permuted_evidence_trajectories

model, _, _ = load_policy_checkpoint("/path/to/model.pt", "cpu")
result = simulate_permuted_evidence(
    model, monkey_id=1, rollout_seed=45, permutation_seed=0,
    shape_ids=[4, 6, 8], match_timestep=3,
)
plot_permuted_evidence_trajectories(
    result["hidden_paths"], result["hidden_df"],
    out_path="experiment/results/permuted_evidence_hidden_trajectories",
    match_timestep=3,
)
```

The permutations preserve the set of cues, hence their cumulative evidence at
the matched timestep, while changing cue order. The function exports PNG/SVG
trajectories; replace the example sequence with the desired stimulus sequence.

## Citation

See the [project citation](../README.md#citation).
