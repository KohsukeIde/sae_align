#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-outputs/stageb_b6_realpw_v1_cpu}"
DATA_SEED="${DATA_SEED:?DATA_SEED is required}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BACKEND="${BACKEND:-powderworld}"
CHANNELS="${CHANNELS:-rgb range noisy_rgb gray_rgb blur_rgb}"
N_STATES="${N_STATES:-128}"
K_ACTIONS="${K_ACTIONS:-32}"
GRID_SIZE="${GRID_SIZE:-64}"
HORIZON="${HORIZON:-4}"
DEVICE="${DEVICE:-cpu}"
USE_JIT="${USE_JIT:-0}"
MAX_DELTA_SAMPLES="${MAX_DELTA_SAMPLES:-$((N_STATES * K_ACTIONS))}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CHANNELS="${CHANNELS//:/ }"
export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}/src}"

DATA_DIR="${OUT}/seed_${DATA_SEED}/stage0"
DATA="${DATA_DIR}/stage0_dataset.npz"

if [[ "${SKIP_EXISTING}" == "1" && -f "${DATA}" ]]; then
  printf 'Real Powderworld B6 dataset already exists: data_seed=%s data=%s\n' "${DATA_SEED}" "${DATA}"
  exit 0
fi

mkdir -p "${DATA_DIR}"

GENERATOR_ARGS=(
  scripts/make_staged0_powderworld_dataset.py
  --config configs/staged0_powderworld.json
  --out "${DATA}"
  --backend "${BACKEND}"
  --n-states "${N_STATES}"
  --k-actions "${K_ACTIONS}"
  --grid-size "${GRID_SIZE}"
  --horizon "${HORIZON}"
  --seed "${DATA_SEED}"
  --device "${DEVICE}"
  --channels ${CHANNELS}
  --max-delta-samples "${MAX_DELTA_SAMPLES}"
  --dense-sampling full-states
)
if [[ "${USE_JIT}" == "1" ]]; then
  GENERATOR_ARGS+=(--use-jit)
fi

"${PYTHON_BIN}" "${GENERATOR_ARGS[@]}"
printf 'Real Powderworld B6 dataset complete: data_seed=%s data=%s\n' "${DATA_SEED}" "${DATA}"
