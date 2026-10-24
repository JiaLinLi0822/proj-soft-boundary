# Experimental GRU models

Fit small GRU policies to sequential-sampling monkey choices, then compare
behavior, sampling policies, and hidden-state variability. Actions are
`choose_A`, `choose_B`, and `continue`; inputs are one-hot vectors over eight
shape categories.

See the [project README](../README.md) for installation, module structure,
data requirements, training commands, and figure reproduction. Scripts can be
called from the project root as `python experiment/train_single.py`, etc.
Always provide your data path with `--mat`; analysis also requires `--ckpt`
and the matching `--monkey-id`.

The original `.mat` files and fitted `.pt` checkpoints are not distributed in
this folder. The Slurm script is an environment-specific example and needs
local cluster configuration.
