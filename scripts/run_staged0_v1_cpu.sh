#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-outputs/staged0_v1_cpu}
export PYTHONPATH=${PYTHONPATH:-src:.}
python scripts/make_staged0_powderworld_dataset.py \
  --config configs/staged0_powderworld.json \
  --backend powderworld \
  --out "$OUT/data/staged0_dataset.npz"
python scripts/train_staged0p_predictor_grounded.py \
  --data "$OUT/data/staged0_dataset.npz" \
  --out "$OUT/audit" \
  --input-channels rgb \
  --observability-channels rgb range \
  --seeds 0 1 2 \
  --alphas 2 4 8 16 32 \
  --epochs 300 \
  --channel-dim 64
python scripts/summarize_staged0p_grid.py --root "$OUT" --out "$OUT" --expected-report-dirs 1 --phase d0
echo "Stage D0 v1 CPU completed: $OUT"
