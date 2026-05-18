#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-outputs/stageb_b4_v1_cpu}
DATA_SEEDS=${DATA_SEEDS:-"0 1"}
SPLIT_SEEDS=${SPLIT_SEEDS:-"17 29"}

for DATA_SEED in ${DATA_SEEDS}; do
  DATA_DIR="${OUT}/seed_${DATA_SEED}/stage0"
  DATA="${DATA_DIR}/stage0_dataset.npz"
  mkdir -p "${DATA_DIR}"

  python scripts/make_stage0_dataset.py \
    --config configs/stage0_toy.json \
    --out "${DATA}" \
    --n-states 256 \
    --k-actions 32 \
    --grid-size 32 \
    --horizon 3 \
    --seed "${DATA_SEED}" \
    --max-delta-samples 4096 \
    --dense-sampling full-states \
    --store-static-obs

  for SPLIT_SEED in ${SPLIT_SEEDS}; do
    RUN="${OUT}/seed_${DATA_SEED}/split_${SPLIT_SEED}"
    SPLIT="${RUN}/split"

    python scripts/make_balanced_action_split.py \
      --data "${DATA}" \
      --out "${SPLIT}" \
      --probe-fraction 0.5 \
      --seed "${SPLIT_SEED}" \
      --n-candidates 3000

    PROBE_IDS=$(cat "${SPLIT}/probe_action_ids.txt")
    BOOTSTRAP_SEED=$((2000 + DATA_SEED * 100 + SPLIT_SEED))

    python scripts/train_transition_encoder.py \
      --data "${DATA}" \
      --out "${RUN}/action_effect" \
      --channels rgb range local noisy_rgb gray_rgb blur_rgb \
      --n-components 32 \
      --max-train-samples 2048 \
      --seed "${DATA_SEED}" \
      --train-action-ids ${PROBE_IDS}

    python scripts/analyze_stageb4_reliability.py \
      --data "${DATA}" \
      --model "${RUN}/action_effect/transition_encoders.npz" \
      --out "${RUN}/reliability" \
      --channels rgb range local noisy_rgb gray_rgb blur_rgb \
      --probe-action-ids ${PROBE_IDS} \
      --k 10 \
      --max-states 128 \
      --bootstrap-repeats 200 \
      --bootstrap-seed "${BOOTSTRAP_SEED}" \
      --normalization-modes none probe_global_apply probe_action_type_apply split_global_diagnostic per_action_diagnostic \
      --seed "${DATA_SEED}"
  done
done

printf '\nStage B.4 v1 CPU grid complete: %s\n' "${OUT}"
