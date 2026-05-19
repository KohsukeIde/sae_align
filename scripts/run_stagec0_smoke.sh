#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-outputs/stagec0_smoke}
DATA=${2:-$OUT/data/stage0_dataset.npz}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$OUT/data"

# Generate a small Stage-0 dataset with static observations if the caller did not
# pass an existing dataset. This keeps the smoke test self-contained.
if [ ! -f "$DATA" ]; then
  PYTHONPATH=src python scripts/make_stage0_dataset.py \
    --config configs/stage0_toy.json \
    --out "$DATA" \
    --n-states 160 \
    --k-actions 24 \
    --grid-size 32 \
    --horizon 3 \
    --seed 0 \
    --dense-sampling full-states \
    --max-delta-samples 1920 \
    --store-static-obs
fi

PYTHONPATH=src python scripts/train_stagec0_prediction.py \
  --data "$DATA" \
  --out "$OUT" \
  --input-channels rgb \
  --target-channel rgb \
  --event-channel event_response \
  --observability-channels rgb range \
  --observability-mix geom \
  --methods uniform change_mask observability shuffled_observability oracle_event \
  --seeds 0 1 \
  --channel-dim 64 \
  --ridge 1.0 \
  --weight-alpha 4.0 \
  --max-samples 1600

PYTHONPATH=src python scripts/summarize_stagec0_grid.py --root "$OUT"

echo "Stage C0 smoke completed: $OUT"
