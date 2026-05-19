#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-outputs/stageb_b6_v1_cpu}"
DATA_SEED="${DATA_SEED:?DATA_SEED is required}"
SPLIT_SEED="${SPLIT_SEED:?SPLIT_SEED is required}"
PCA_DIM="${PCA_DIM:-32}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHANNELS="${CHANNELS:-rgb range local noisy_rgb gray_rgb blur_rgb}"
N_STATES="${N_STATES:-256}"
K_ACTIONS="${K_ACTIONS:-32}"
GRID_SIZE="${GRID_SIZE:-32}"
HORIZON="${HORIZON:-3}"
MAX_DELTA_SAMPLES="${MAX_DELTA_SAMPLES:-4096}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2048}"
MAX_STATES_ANALYZE="${MAX_STATES_ANALYZE:-128}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-$((3000 + DATA_SEED * 100 + SPLIT_SEED + PCA_DIM))}"
RANDOM_PROJECTION_DIM="${RANDOM_PROJECTION_DIM:-32}"
K_VALUES="${K_VALUES:-5 10 20}"
JITTER_EPSILONS="${JITTER_EPSILONS:-0 1e-6 1e-5 1e-4 1e-3}"
JITTER_SEEDS="${JITTER_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
NORMALIZATION_MODES="${NORMALIZATION_MODES:-none probe_global_apply probe_action_type_apply}"
REPRESENTATIONS="${REPRESENTATIONS:-pca_probe_only pca_all_action raw_delta random_projection}"
PERMUTATION_REPEATS="${PERMUTATION_REPEATS:-200}"
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-200}"
RIDGE_SPLITS="${RIDGE_SPLITS:-10}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
TARGET_PAIRS="${TARGET_PAIRS:-rgb:range}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# qsub -v cannot safely carry shell-space lists. Accept colon-separated lists
# from the submitter and convert them back to argv lists here.
CHANNELS="${CHANNELS//:/ }"
K_VALUES="${K_VALUES//:/ }"
JITTER_EPSILONS="${JITTER_EPSILONS//:/ }"
JITTER_SEEDS="${JITTER_SEEDS//:/ }"
NORMALIZATION_MODES="${NORMALIZATION_MODES//:/ }"
REPRESENTATIONS="${REPRESENTATIONS//:/ }"
TARGET_PAIRS="${TARGET_PAIRS//,/ }"
TARGET_PAIRS="${TARGET_PAIRS//|/ }"

export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}/src}"

RUN="${OUT}/seed_${DATA_SEED}/split_${SPLIT_SEED}/pca_${PCA_DIM}"
DATA_DIR="${RUN}/stage0"
DATA="${DATA_DIR}/stage0_dataset.npz"
SPLIT="${RUN}/split"

if [[ "${SKIP_EXISTING}" == "1" && -f "${RUN}/stageb6_diagnostics/reports/stageb6_summary.json" ]]; then
  printf 'Stage B.6 task already complete: data_seed=%s split_seed=%s pca_dim=%s out=%s\n' \
    "${DATA_SEED}" "${SPLIT_SEED}" "${PCA_DIM}" "${RUN}/stageb6_diagnostics"
  exit 0
fi

mkdir -p "${DATA_DIR}" "${SPLIT}"

"${PYTHON_BIN}" scripts/make_stage0_dataset.py \
  --config configs/stage0_toy.json \
  --out "${DATA}" \
  --n-states "${N_STATES}" \
  --k-actions "${K_ACTIONS}" \
  --grid-size "${GRID_SIZE}" \
  --horizon "${HORIZON}" \
  --seed "${DATA_SEED}" \
  --max-delta-samples "${MAX_DELTA_SAMPLES}" \
  --dense-sampling full-states \
  --store-static-obs

"${PYTHON_BIN}" scripts/make_balanced_action_split.py \
  --data "${DATA}" \
  --out "${SPLIT}" \
  --probe-fraction 0.5 \
  --seed "${SPLIT_SEED}" \
  --n-candidates 3000

PROBE_IDS="$(cat "${SPLIT}/probe_action_ids.txt")"

"${PYTHON_BIN}" scripts/train_transition_encoder.py \
  --data "${DATA}" \
  --out "${RUN}/action_effect_probe" \
  --channels ${CHANNELS} \
  --n-components "${PCA_DIM}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --seed "${DATA_SEED}" \
  --train-action-ids ${PROBE_IDS}

"${PYTHON_BIN}" scripts/train_transition_encoder.py \
  --data "${DATA}" \
  --out "${RUN}/action_effect_all_action" \
  --channels ${CHANNELS} \
  --n-components "${PCA_DIM}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --seed "${DATA_SEED}"

"${PYTHON_BIN}" scripts/train_transition_encoder.py \
  --data "${DATA}" \
  --out "${RUN}/static" \
  --feature-kind static \
  --channels ${CHANNELS} \
  --n-components "${PCA_DIM}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --seed "${DATA_SEED}" \
  --train-action-ids ${PROBE_IDS}

"${PYTHON_BIN}" scripts/analyze_stageb6_diagnostics.py \
  --data "${DATA}" \
  --model "${RUN}/action_effect_probe/transition_encoders.npz" \
  --static-model "${RUN}/static/transition_encoders.npz" \
  --all-action-model "${RUN}/action_effect_all_action/transition_encoders.npz" \
  --out "${RUN}/stageb6_diagnostics" \
  --channels ${CHANNELS} \
  --probe-action-ids ${PROBE_IDS} \
  --representations ${REPRESENTATIONS} \
  --random-projection-dim "${RANDOM_PROJECTION_DIM}" \
  --random-projection-seed "${BOOTSTRAP_SEED}" \
  --k-values ${K_VALUES} \
  --jitter-epsilons ${JITTER_EPSILONS} \
  --jitter-seeds ${JITTER_SEEDS} \
  --normalization-modes ${NORMALIZATION_MODES} \
  --target-pairs ${TARGET_PAIRS} \
  --max-states "${MAX_STATES_ANALYZE}" \
  --seed "${DATA_SEED}" \
  --permutation-repeats "${PERMUTATION_REPEATS}" \
  --permutation-seed "${BOOTSTRAP_SEED}" \
  --bootstrap-repeats "${BOOTSTRAP_REPEATS}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  --ridge-splits "${RIDGE_SPLITS}" \
  --ridge-alpha "${RIDGE_ALPHA}"

printf 'Stage B.6 task complete: data_seed=%s split_seed=%s pca_dim=%s out=%s\n' \
  "${DATA_SEED}" "${SPLIT_SEED}" "${PCA_DIM}" "${RUN}/stageb6_diagnostics"
