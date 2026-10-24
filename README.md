# Stochastic Decision Policy Is Optimal Under Costly Control

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.10.06.757200-red)](https://www.biorxiv.org/content/10.64898/2026.10.06.757200v1.abstract)

**Authors: Jialin Li, Kenway Louie, Aadith Vittala, Marcelo G. Mattar, Paul Glimcher, Bo Shen**

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Reproduce the Analyses and Figures](#reproduce-the-analyses-and-figures)
- [Citation](#citation)

## Overview

This repository contains the modeling and analysis code accompanying the
[preprint](https://www.biorxiv.org/content/10.64898/2026.10.06.757200v1.abstract)
*Stochastic Decision Policy Is Optimal Under Costly Control*.

The project studies how the cost of controlling action selection can make
stochastic decision policies optimal in sequential evidence accumulation. It
compares soft policies with noisy internal beliefs and fits small recurrent
neural networks (GRUs) to macaque choices to characterize sampling policies
and hidden-state variability.

## Repo Contents

- [Normative models](./normative/): Dynamic programming, simulations, and scripts
  for Figures 1–4. See the [normative README](./normative/README.md) for commands,
  model settings, and output files.
- [Experimental analyses](./experiment/): Monkey-data loading, GRU fitting,
  nested cross-validation, and behavioral, policy, and representation analyses
  associated with Figures 5–8. See the [experiment README](./experiment/README.md)
  for data requirements, training, and figure-generation instructions.
- [Dependencies](./requirements.txt): Python packages used by both workflows.
  [requirements-lock.txt](./requirements-lock.txt) records exact package versions.

Original monkey data and fitted checkpoints are not included. The normative
figures can be generated without external data.

## System Requirements

### Hardware and Software Requirements

Use Python 3.10 or later with NumPy, SciPy, pandas, Matplotlib, PyTorch, h5py,
statsmodels, and tqdm. A standard CPU computer can run the normative figures
and individual GRU fits. Nested cross-validation involves many fits; a multicore
machine or computing cluster is useful for the full sweep. GPU execution is
optional.

### OS Requirements

The scripts use Python packages available on macOS, Linux, and Windows. Setup
commands below use a macOS/Linux shell. On Windows, activate the environment
with `.venv\Scripts\Activate.ps1` in PowerShell. The Slurm example requires a
Linux cluster with Slurm and local configuration.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To install the recorded exact versions on a compatible platform, use
`python -m pip install -r requirements-lock.txt` instead. Jupyter is needed only
for the optional notebooks. On a headless machine, set `MPLBACKEND=Agg`.

## Reproduce the Analyses and Figures

After installation, follow the instructions in each subfolder:

1. [Figures 1–4: normative models](./normative/README.md).
2. [GRU fitting and nested cross-validation](./experiment/README.md#fit-the-gru-models).
3. [Experimental analyses and figure outputs](./experiment/README.md#reproduce-the-experimental-analyses-and-figures).

Commands in both subfolder READMEs are run from the **repository root**.

## Citation

If you use this code, please cite:

Li, J., Louie, K., Vittala, A., Mattar, M. G., Glimcher, P., & Shen, B. (2026).
*Stochastic Decision Policy Is Optimal Under Costly Control*. bioRxiv.
[https://doi.org/10.64898/2026.10.06.757200](https://doi.org/10.64898/2026.10.06.757200).
