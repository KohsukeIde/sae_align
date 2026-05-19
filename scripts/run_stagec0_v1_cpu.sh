#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-outputs/stagec0_v1_cpu}
DATA=${2:-outputs/stage0_v1/data/stage0_dataset.npz}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [ ! -f "$DATA" ]; then
  echo "Missing Stage-0 dataset: $DATA" >&2
  echo "Generate one with --store-static-obs, e.g. scripts/run_stage0_v1.sh plus static obs enabled." >&2
  exit 1
fi

mkdir -p "$OUT"

# Run a small but useful grid over observability score and input-feature choices.
for OBS_MIX in geom mean product min; do
  for INPUT in rgb "rgb range"; do
    SAFE_INPUT=$(echo "$INPUT" | tr ' ' '_')
    CELL="$OUT/input_${SAFE_INPUT}/obs_${OBS_MIX}"
    PYTHONPATH=src python scripts/train_stagec0_prediction.py \
      --data "$DATA" \
      --out "$CELL" \
      --input-channels $INPUT \
      --target-channel rgb \
      --event-channel event_response \
      --observability-channels rgb range \
      --observability-mix "$OBS_MIX" \
      --methods uniform change_mask observability shuffled_observability oracle_event \
      --seeds 0 1 2 \
      --channel-dim 128 \
      --ridge 1.0 \
      --weight-alpha 4.0
  done
done

PYTHONPATH=src python scripts/summarize_stagec0_grid.py --root "$OUT"

echo "Stage C0 v1 CPU grid completed: $OUT"
