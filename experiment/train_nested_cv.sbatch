#!/bin/bash
#SBATCH --job-name=tinyrnn_ncv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-69
#SBATCH --output=cluster/%x_%A_%a.out
#SBATCH --error=cluster/%x_%A_%a.err
#SBATCH --account=bornstea_lab

set -euo pipefail
module load python/3.10.2
PYTHON="$HOME/venvs/soft-boundary-py310/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

HIDDEN_SIZES=(1 2 3 4 5 10 20)
OUTER_FOLDS="${OUTER_FOLDS:-10}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
if (( OUTER_FOLDS < 3 || TASK_ID < 0 || TASK_ID >= ${#HIDDEN_SIZES[@]} * OUTER_FOLDS )); then
    echo "Invalid task index or fold count; array range must match hidden sizes * outer folds." >&2
    exit 1
fi
HIDDEN="${HIDDEN_SIZES[$((TASK_ID / OUTER_FOLDS))]}"
OUTER_FOLD="$((TASK_ID % OUTER_FOLDS))"

MONKEY_ID="${MONKEY_ID:-1}"
RECURRENT_L1="${RECURRENT_L1:-1e-3 1e-4 1e-5}"
INIT_SEEDS="${INIT_SEEDS:-0 1 2}"
INNER_EPOCHS="${INNER_EPOCHS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-3}"
PATIENCE="${PATIENCE:-100}"
FOLD_SEED="${FOLD_SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK}}"
OUTDIR="${OUTDIR:-runs/nested_cv/slurm_${SLURM_ARRAY_JOB_ID}}/h${HIDDEN}/outer${OUTER_FOLD}"

echo "hidden=${HIDDEN} outer_fold=${OUTER_FOLD} workers=${NUM_WORKERS} l1=${RECURRENT_L1} init_seeds=${INIT_SEEDS}"

srun "$PYTHON" -u train_nested_cv.py \
    --monkey-id "${MONKEY_ID}" \
    --hidden "${HIDDEN}" \
    --recurrent-l1-lambda ${RECURRENT_L1} \
    --init-seeds ${INIT_SEEDS} \
    --outer-folds "${OUTER_FOLDS}" \
    --outer-fold "${OUTER_FOLD}" \
    --inner-epochs "${INNER_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --early-stopping-patience "${PATIENCE}" \
    --device cpu \
    --num-workers "${NUM_WORKERS}" \
    --seed "${FOLD_SEED}" \
    --outdir "${OUTDIR}"
