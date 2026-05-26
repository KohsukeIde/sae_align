#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-outputs/stageb_b6_realpw_v1_cpu}"
DATA_SEEDS="${DATA_SEEDS:-0 1 2}"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
ABCI_QUEUE="${ABCI_QUEUE:-rt_HC}"
WALLTIME="${WALLTIME:-04:00:00}"
SELECT="${SELECT:-1}"
DRY_RUN="${DRY_RUN:-0}"
STAGEB_ENV_PREFIX="${STAGEB_ENV_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

BACKEND="${BACKEND:-powderworld}"
CHANNELS="${CHANNELS:-rgb range noisy_rgb gray_rgb blur_rgb}"
N_STATES="${N_STATES:-128}"
K_ACTIONS="${K_ACTIONS:-32}"
GRID_SIZE="${GRID_SIZE:-64}"
HORIZON="${HORIZON:-4}"
DEVICE="${DEVICE:-cpu}"
USE_JIT="${USE_JIT:-0}"
MAX_DELTA_SAMPLES="${MAX_DELTA_SAMPLES:-$((N_STATES * K_ACTIONS))}"

TASK_FILE="${TASK_FILE:-${OUT}/stageb6_realpw_generate_tasks.txt}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${OUT}/qsub_logs}"
mkdir -p "$(dirname "${TASK_FILE}")" "${SUBMIT_LOG_DIR}"

printf "%s\n" ${DATA_SEEDS} > "${TASK_FILE}"
N_TASKS="$(wc -l < "${TASK_FILE}" | tr -d ' ')"
if [[ "${N_TASKS}" == "0" ]]; then
  echo "[error] no real Powderworld generation tasks generated" >&2
  exit 2
fi

colon_list() {
  printf "%s" "$1" | tr -s '[:space:]' ':' | sed 's/^://; s/:$//'
}

VARLIST="TASK_FILE=${TASK_FILE},OUT=${OUT},STAGEB_ENV_PREFIX=${STAGEB_ENV_PREFIX},PYTHON_BIN=${PYTHON_BIN},SKIP_EXISTING=${SKIP_EXISTING}"
VARLIST+=",BACKEND=${BACKEND},DEVICE=${DEVICE},USE_JIT=${USE_JIT}"
VARLIST+=",CHANNELS=$(colon_list "${CHANNELS}")"
VARLIST+=",N_STATES=${N_STATES},K_ACTIONS=${K_ACTIONS},GRID_SIZE=${GRID_SIZE},HORIZON=${HORIZON},MAX_DELTA_SAMPLES=${MAX_DELTA_SAMPLES}"

CMD=(
  qsub
  -P "${ABCI_GROUP}"
  -q "${ABCI_QUEUE}"
  -l "select=${SELECT}"
  -l "walltime=${WALLTIME}"
  -N "b6rpw_gen"
  -J "1-${N_TASKS}"
  -j oe
  -o "${SUBMIT_LOG_DIR}/stageb6_realpw_generate_array.pbs.log"
  -v "${VARLIST}"
  scripts/stageb_b6_realpw_generate_array.pbs
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${CMD[@]}"; printf '\n'
  printf '[dry-run] task_file=%s n_tasks=%s\n' "${TASK_FILE}" "${N_TASKS}"
else
  JOBID="$("${CMD[@]}")"
  JOBID="${JOBID%% *}"
  printf '[submitted] real Powderworld B6 generation array job=%s tasks=%s task_file=%s\n' \
    "${JOBID}" "${N_TASKS}" "${TASK_FILE}"
fi
