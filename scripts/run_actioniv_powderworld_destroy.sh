#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-outputs/actioniv_destroy}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
rm -rf "$OUT"
mkdir -p "$OUT/data"

PYTHONPATH=src:. python scripts/make_actioniv_task_dataset.py \
  --backend powderworld \
  --task-gen PWTaskGenDestroyAll \
  --out "$OUT/data/task_dataset.npz" \
  --n-states 128 \
  --actions-per-state 8 \
  --device cpu \
  --time-per-action 2 \
  --n-world-settle 64 \
  --erase-action-fraction 0.5 \
  --seed 0

PYTHONPATH=src:. python scripts/train_actioniv_task_oracle.py \
  --data "$OUT/data/task_dataset.npz" \
  --out "$OUT/task_oracle" \
  --input-channels rgb \
  --methods uniform oracle_task change_mask observability shuffled_observability \
  --alphas 1 2 4 8 16 32 \
  --split-seeds 0 1 2 \
  --channel-dim 64 \
  --epochs 120

BRANCH=$(PYTHONPATH=src:. python - <<PY
import json
with open("$OUT/task_oracle/reports/actioniv_task_oracle_decision.json") as f:
    print(json.load(f).get("branch", ""))
PY
)

if [[ "$BRANCH" == "oracle_pass" ]]; then
  PYTHONPATH=src:. python scripts/train_actioniv_effect_encoder.py \
    --data "$OUT/data/task_dataset.npz" \
    --out "$OUT/effect_encoder" \
    --channels rgb range \
    --input-dim 256 \
    --latent-dims 8 16 32 \
    --k-values 1 5 10 \
    --seed 0
else
  echo "Skipping Action-IV effect prototype because task oracle branch is $BRANCH"
fi

echo "Action-IV Powderworld DestroyAll audit complete: $OUT"
