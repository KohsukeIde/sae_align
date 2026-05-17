#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

OUT=${1:-outputs/stage0_v1}
mkdir -p "${OUT}/data"

python scripts/make_stage0_dataset.py \
  --config configs/stage0_v1.json \
  --out "${OUT}/data/stage0_dataset.npz" \
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
  --seed 0

printf '\nStage 0 v1 complete: %s\n' "${OUT}"
