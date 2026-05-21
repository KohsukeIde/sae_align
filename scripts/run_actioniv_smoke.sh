#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-outputs/actioniv_smoke}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
rm -rf "$OUT"
mkdir -p "$OUT/data"

PYTHONPATH=src:. python scripts/make_actioniv_task_dataset.py \
  --backend synthetic \
  --out "$OUT/data/task_dataset.npz" \
  --n-states 64 \
  --actions-per-state 4 \
  --grid-size 32 \
  --seed 0

PYTHONPATH=src:. python scripts/train_actioniv_task_oracle.py \
  --data "$OUT/data/task_dataset.npz" \
  --out "$OUT/task_oracle" \
  --input-channels rgb \
  --methods uniform oracle_task change_mask observability shuffled_observability \
  --alphas 1 2 4 8 16 \
  --split-seeds 0 1 \
  --channel-dim 16 \
  --epochs 40

PYTHONPATH=src:. python scripts/train_actioniv_effect_encoder.py \
  --data "$OUT/data/task_dataset.npz" \
  --out "$OUT/effect_encoder" \
  --channels rgb range \
  --input-dim 64 \
  --latent-dims 8 \
  --k-values 1 5 10 \
  --seed 0

echo "Action-IV smoke complete: $OUT"
