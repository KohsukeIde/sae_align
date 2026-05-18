#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-outputs/stageb_b2_smoke}
DATA="${OUT}/stage0/data/stage0_dataset.npz"

mkdir -p "${OUT}/stage0/data"

python scripts/make_stage0_dataset.py \
  --config configs/stage0_toy.json \
  --out "${DATA}" \
  --n-states 48 \
  --k-actions 16 \
  --grid-size 32 \
  --horizon 3 \
  --seed 0 \
  --max-delta-samples 384 \
  --dense-sampling full-states \
  --store-static-obs

python scripts/train_transition_encoder.py \
  --data "${DATA}" \
  --out "${OUT}/action_effect" \
  --n-components 16 \
  --max-train-samples 384 \
  --seed 0 \
  --train-action-ids 0 1 2 3 4 5 6 7

python scripts/train_transition_encoder.py \
  --data "${DATA}" \
  --out "${OUT}/static" \
  --feature-kind static \
  --n-components 16 \
  --max-train-samples 384 \
  --seed 0 \
  --train-action-ids 0 1 2 3 4 5 6 7

python scripts/analyze_state_signature_knn.py \
  --data "${DATA}" \
  --model "${OUT}/action_effect/transition_encoders.npz" \
  --static-model "${OUT}/static/transition_encoders.npz" \
  --out "${OUT}/state_signature_knn" \
  --k 5 \
  --max-states 24 \
  --probe-action-ids 0 1 2 3 4 5 6 7 \
  --probe-fraction 0.5 \
  --regular-state-threshold 0.25 \
  --blind-state-threshold 0.25 \
  --physical-state-threshold 0.25 \
  --bootstrap-repeats 20 \
  --bootstrap-seed 0 \
  --seed 0

printf '\nStage B.2 state signature smoke complete: %s\n' "${OUT}"
