#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-outputs/staged0p_smoke}
DATA=${2:-outputs/stageb_b6_primary_cell_v1_cpu/seed_0/split_17/pca_32/stage0/stage0_dataset.npz}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
if [[ ! -f "$DATA" ]]; then
  echo "Missing data file: $DATA" >&2
  echo "Pass a Stage-0 dataset path as the second argument." >&2
  exit 2
fi
PYTHONPATH=${PYTHONPATH:-src:.} python scripts/train_staged0p_predictor_grounded.py \
  --data "$DATA" \
  --out "$OUT" \
  --input-channels rgb \
  --observability-channels rgb range \
  --methods uniform change_mask observability shuffled_observability predictor_uncertainty predictor_grounded shuffled_predictor_grounded lossgrad_observability oracle_event \
  --alphas 2 4 8 16 32 \
  --seeds 0 1 \
  --channel-dim 64 \
  --epochs 80 \
  --lr 0.2 \
  --l2 1e-4
PYTHONPATH=${PYTHONPATH:-src:.} python scripts/summarize_staged0p_grid.py --root "$OUT" --out "$OUT"
echo "Wrote $OUT"
