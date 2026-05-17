#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-outputs/stage0_v1_large}
SEED=${2:-0}

python scripts/make_stage0_dataset.py \
  --config configs/stage0_v1_large.json \
  --out "${OUT}/data/stage0_dataset.npz" \
  --seed "${SEED}" \
  --max-delta-samples 8000

python scripts/analyze_stage0_channels.py \
  --data "${OUT}/data/stage0_dataset.npz" \
  --out "${OUT}" \
  --k-sweep 16 32 64 128 \
  --k-bootstrap 20 \
  --threshold-quantiles 0.05 0.10 0.20 \
  --default-threshold 0.10 \
  --max-probe-samples 5000 \
  --projection-dim 512 \
  --seed "${SEED}"
