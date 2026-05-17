#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-outputs/stage0_smoke}
mkdir -p "$OUT/data"

python scripts/make_stage0_dataset.py \
  --config configs/stage0_toy.json \
  --out "$OUT/data/stage0_dataset.npz" \
  --n-states 16 \
  --k-actions 8 \
  --grid-size 32 \
  --horizon 3 \
  --seed 0 \
  --max-delta-samples 128

python scripts/analyze_stage0_channels.py \
  --data "$OUT/data/stage0_dataset.npz" \
  --out "$OUT" \
  --k-sweep 4 8 \
  --k-bootstrap 2 \
  --threshold-quantiles 0.05 0.10 0.20 \
  --default-threshold 0.10 \
  --max-probe-samples 128 \
  --seed 0

printf '\nStage 0 smoke complete: %s\n' "$OUT"
