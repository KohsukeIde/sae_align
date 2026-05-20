#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-outputs/staged0_smoke_toy}
export PYTHONPATH=${PYTHONPATH:-src:.}
python scripts/make_staged0_powderworld_dataset.py \
  --config configs/staged0_toy_smoke.json \
  --backend toy \
  --out "$OUT/data/staged0_dataset.npz"
python scripts/train_staged0p_predictor_grounded.py \
  --data "$OUT/data/staged0_dataset.npz" \
  --out "$OUT/audit" \
  --input-channels rgb \
  --observability-channels rgb range \
  --seeds 0 \
  --alphas 2 4 \
  --epochs 50 \
  --channel-dim 16
python scripts/summarize_staged0p_grid.py --root "$OUT" --out "$OUT" --expected-report-dirs 1 --phase d0
echo "Stage D0 toy smoke completed: $OUT"
