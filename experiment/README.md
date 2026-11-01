# Model fitting and experimental analyses for Macques data

## Overview

This folder contains the experimental modeling for
*[Stochastic Decision Policy Is Optimal Under Costly Control](https://www.biorxiv.org/content/10.64898/2026.10.06.757200v1.abstract)*.
Small GRU networks are fitted to macaque choices in a sequential-sampling task.
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
| `train_nested_cv.py`, `nested_cv.py` | Nested cross-validation, hyperparameter selection, outer-test evaluation, and likelihood-versus-capacity plots. |
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
  --mat experiment/data/dataE.mat --monkey-id 1 --hidden 2 \
  --epochs 2000 --lr 1e-3 --batch-size 128 --val-frac 0.2 \
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
  --init-seeds 0 1 2 --outer-folds 10 --inner-epochs 2000 \
  --batch-size 128 --lr 3e-3 --early-stopping-patience 200 \
  --seed 0 --device cpu --num-workers 0 \
  --outdir experiment/runs/nested_cv
```

Each hidden size runs a separate nested CV. Inner folds select recurrent L1
strength and initialization seed by mean validation loss. Among the inner models
for the winning configuration, the one with the lowest validation loss is
evaluated directly on the outer test set. Its weights come from its best
validation epoch, and its training data consists only of that inner training fold.
By default, the number of inner folds is `outer-folds - 1`; use `--inner-folds`
to override it. Set `--num-workers` above 1 to parallelize inner fits on CPU.
Use `--outer-fold 0` (zero-based) to run only one outer fold while preserving  
the full run's splits and seeds. Single-fold runs write a summary but skip  
the capacity plot, which requires combining outer-fold results.

Each timestamped `MonkeyE` or `MonkeyJ` run directory contains:

| Output | Content |
| --- | --- |
| `cv_models/outer*/h*/inner*.pt` | Individual inner-fold checkpoints. |
| `inner_results_hH.csv` | Inner-fit scores and selected training epochs for each hidden size. |
| `outerN_besthH_l1_*_seed*.pt` | Selected inner models with outer-test metrics. |
| `summary.csv` | Outer-fold configuration and held-out performance. |
| `test_trajectory_nll_vs_hidden.png`, `.svg` | Model capacity versus held-out trajectory negative log-likelihood. |

Outer checkpoints and summary rows record the selected `inner_fold`,
`best_epoch`, and actual inner-model `train_size`.

For each outer fold, trajectory NLL is averaged across trials after summing
action NLL over all timesteps in each trial. The plotted mean weights folds by
their test-set sizes; error bars show the standard deviation across folds.

### Cluster execution

To train the model on batch, use `train_nested_cv.sbatch`:

```bash
cd experiment
mkdir -p cluster
sbatch train_nested_cv.sbatch
```

The array contains 70 tasks in parallel, one per `(hidden size, outer fold)`. Tasks 0–9  
run the ten outer folds for H=1, tasks 10–19 for H=2, and tasks 60–69 for H=20.  
Each task trains 135 inner models with the default grid, using up to 32 workers,  
and requests 32 CPUs and 32 GB. There is no array concurrency cap: all 70 tasks  
may run simultaneously (up to 2,240 simultaneous fits), subject to cluster  
availability and account limits.

Task outputs are isolated under
`runs/nested_cv/slurm_JOBID/hH/outerF/MonkeyE/nestedcv_TIMESTAMP_seedS/`
(or `MonkeyJ`). Each task writes one row to its own `summary.csv`, plus its  
inner checkpoints, inner-results CSV, and selected outer model. The total remains 9,450 trained models per monkey.

## Reproduce the Experimental Analyses and Figures

By default, analyze both monkeys using the h=2 checkpoint with the lowest
outer-test trajectory NLL for each monkey:

```bash
python experiment/analysis.py --device cpu
```

The script searches `experiment/runs/nested_cv/slurm_*/h2/`, keeps the latest
rerun of each fold, and uses `experiment/data/dataE.mat` and `dataJ.mat`.
Use `--monkey-id 1` or `--monkey-id 2` to analyze only one monkey. The selected
checkpoint, selection criterion, and test NLL are recorded in `analysis_summary.json`.
This minimum-test-NLL selection is for illustrative analyses, rather than an
unbiased estimate of generalization performance.

To analyze a specific checkpoint, supply it and its matching monkey data:

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
| `behavior` | Distribution for number of shapes used, psychometric curves, subjective shape weights, and decision thresholds. |
| `monkey_decision_threshold_distribution` | Monkey evidence distributions for sampling and commitment across timesteps. |
| `psample` | Mean sampling probability versus cumulative evidence, grouped by timestep. |
| `psample_scatter` | Individual sampling probabilities with a highlighted trial. |
| `psample_variability` | Sampling-probability variability across matched evidence and timestep bins. |
| `hidden_by_timestep`, `hidden_by_evidence`, `hidden_by_policy`  | Hidden-state trajectories colored by time step, evidence and policy. |
| `representation_noise` | Vertically stacked hidden-state variability time course and evidence heatmap, with a horizontal colorbar. |

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

## Citation

See the [project citation](../README.md#citation).
