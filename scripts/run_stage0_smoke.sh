#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

OUT=${1:-outputs/stage0_smoke}
mkdir -p "$OUT/data"

python scripts/make_stage0_dataset.py \
  --config configs/stage0_toy.json \
  --out "$OUT/data/stage0_dataset.npz" \
  --n-states 128 \
  --k-actions 16 \
  --grid-size 32 \
  --horizon 3 \
  --seed 0

python scripts/analyze_stage0_channels.py \
  --data "$OUT/data/stage0_dataset.npz" \
  --out "$OUT" \
  --k-sweep 8 16 \
  --threshold-quantiles 0.05 0.10 0.20 \
  --default-threshold 0.10 \
  --max-probe-samples 1000 \
  --projection-dim 64

echo "Stage 0 smoke test complete: $OUT"
