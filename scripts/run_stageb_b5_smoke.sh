#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-outputs/stageb_b5_smoke}
DATA="${OUT}/stage0/data/stage0_dataset.npz"
SPLIT="${OUT}/split"

mkdir -p "${OUT}/stage0/data"

python scripts/make_stage0_dataset.py \
  --config configs/stage0_toy.json \
  --out "${DATA}" \
  --n-states 64 \
  --k-actions 16 \
  --grid-size 32 \
  --horizon 3 \
  --seed 0 \
  --max-delta-samples 512 \
  --dense-sampling full-states \
  --store-static-obs

python scripts/make_balanced_action_split.py \
  --data "${DATA}" \
  --out "${SPLIT}" \
  --probe-fraction 0.5 \
  --seed 17 \
  --n-candidates 500

PROBE_IDS=$(cat "${SPLIT}/probe_action_ids.txt")

python scripts/train_transition_encoder.py \
  --data "${DATA}" \
  --out "${OUT}/action_effect" \
  --channels rgb range local noisy_rgb gray_rgb blur_rgb \
  --n-components 16 \
  --max-train-samples 512 \
  --seed 0 \
  --train-action-ids ${PROBE_IDS}

python scripts/train_transition_encoder.py \
  --data "${DATA}" \
  --out "${OUT}/static" \
  --feature-kind static \
  --channels rgb range local noisy_rgb gray_rgb blur_rgb \
  --n-components 16 \
  --max-train-samples 512 \
  --seed 0 \
  --train-action-ids ${PROBE_IDS}

python scripts/analyze_stageb5_heldout_alignment.py \
  --data "${DATA}" \
  --model "${OUT}/action_effect/transition_encoders.npz" \
  --static-model "${OUT}/static/transition_encoders.npz" \
  --out "${OUT}/heldout_alignment" \
  --channels rgb range local noisy_rgb gray_rgb blur_rgb \
  --probe-action-ids ${PROBE_IDS} \
  --representations pca_probe_only raw_delta random_projection \
  --k 5 \
  --max-states 32 \
  --bootstrap-repeats 20 \
  --bootstrap-seed 0 \
  --normalization-modes none probe_global_apply probe_action_type_apply \
  --seed 0

printf '\nStage B.5 smoke complete: %s\n' "${OUT}"
