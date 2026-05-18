#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-outputs/stageb_b1_smoke}
DATA="${OUT}/stage0/data/stage0_dataset.npz"

mkdir -p "${OUT}/stage0/data"

python scripts/make_stage0_dataset.py \
  --config configs/stage0_toy.json \
  --out "${DATA}" \
  --n-states 32 \
  --k-actions 12 \
  --grid-size 32 \
  --horizon 3 \
  --seed 0 \
  --max-delta-samples 256 \
  --store-static-obs

python scripts/train_transition_encoder.py \
  --data "${DATA}" \
  --out "${OUT}/action_effect" \
  --n-components 16 \
  --max-train-samples 256 \
  --seed 0

python scripts/analyze_stratified_knn.py \
  --data "${DATA}" \
  --model "${OUT}/action_effect/transition_encoders.npz" \
  --out "${OUT}/action_effect_knn" \
  --k 5 \
  --max-points 256 \
  --seed 0

python scripts/train_transition_encoder.py \
  --data "${DATA}" \
  --out "${OUT}/static" \
  --feature-kind static \
  --n-components 16 \
  --max-train-samples 256 \
  --seed 0

python scripts/compare_static_action_effect_knn.py \
  --data "${DATA}" \
  --action-effect-model "${OUT}/action_effect/transition_encoders.npz" \
  --static-model "${OUT}/static/transition_encoders.npz" \
  --out "${OUT}/static_vs_action_effect" \
  --k 5 \
  --max-points 256 \
  --seed 0

printf '\nStage B.1 smoke complete: %s\n' "${OUT}"
