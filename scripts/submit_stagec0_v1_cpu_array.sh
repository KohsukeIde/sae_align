#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

DATA="${DATA:-outputs/stage0_v1/data/stage0_dataset.npz}"
OUT="${OUT:-outputs/stagec0_v1_cpu_array}"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
ABCI_QUEUE="${ABCI_QUEUE:-rt_HC}"
WALLTIME="${WALLTIME:-08:00:00}"
SELECT="${SELECT:-1}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGEC0_ENV_PREFIX="${STAGEC0_ENV_PREFIX:-}"

SEEDS="${SEEDS:-0 1 2 3 4}"
CHANNEL_DIM="${CHANNEL_DIM:-128}"
RIDGE="${RIDGE:-1.0}"
WEIGHT_ALPHA="${WEIGHT_ALPHA:-4.0}"
OOD_NOISE_STD="${OOD_NOISE_STD:-0.35}"
OMP_NUM_THREADS_VALUE="${OMP_NUM_THREADS_VALUE:-8}"
OPENBLAS_NUM_THREADS_VALUE="${OPENBLAS_NUM_THREADS_VALUE:-8}"
MKL_NUM_THREADS_VALUE="${MKL_NUM_THREADS_VALUE:-8}"

TASK_FILE="${TASK_FILE:-${OUT}/stagec0_v1_tasks.tsv}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${OUT}/qsub_logs}"
mkdir -p "$(dirname "${TASK_FILE}")" "${SUBMIT_LOG_DIR}"

: > "${TASK_FILE}"
for OBS_MIX in geom mean product min; do
  printf "%s\t%s\t%s\n" "${OBS_MIX}" "rgb" "rgb" >> "${TASK_FILE}"
  printf "%s\t%s\t%s\n" "${OBS_MIX}" "rgb_range" "rgb:range" >> "${TASK_FILE}"
done

N_TASKS="$(wc -l < "${TASK_FILE}" | tr -d ' ')"
if [[ "${N_TASKS}" == "0" ]]; then
  echo "[error] no Stage C0 tasks generated" >&2
  exit 2
fi

colon_list() {
  printf "%s" "$1" | tr -s '[:space:]' ':' | sed 's/^://; s/:$//'
}

VARLIST="TASK_FILE=${TASK_FILE},DATA=${DATA},OUT=${OUT},PYTHON_BIN=${PYTHON_BIN},STAGEC0_ENV_PREFIX=${STAGEC0_ENV_PREFIX}"
VARLIST+=",SEEDS=$(colon_list "${SEEDS}")"
VARLIST+=",CHANNEL_DIM=${CHANNEL_DIM},RIDGE=${RIDGE},WEIGHT_ALPHA=${WEIGHT_ALPHA},OOD_NOISE_STD=${OOD_NOISE_STD}"
VARLIST+=",OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE},OPENBLAS_NUM_THREADS_VALUE=${OPENBLAS_NUM_THREADS_VALUE},MKL_NUM_THREADS_VALUE=${MKL_NUM_THREADS_VALUE}"

CMD=(
  qsub
  -P "${ABCI_GROUP}"
  -q "${ABCI_QUEUE}"
  -l "select=${SELECT}"
  -l "walltime=${WALLTIME}"
  -N "stagec0_v1"
  -J "1-${N_TASKS}"
  -j oe
  -o "${SUBMIT_LOG_DIR}/stagec0_v1_array.pbs.log"
  -v "${VARLIST}"
  scripts/stagec0_v1_cpu_array.pbs
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${CMD[@]}"; printf '\n'
  printf '[dry-run] task_file=%s n_tasks=%s\n' "${TASK_FILE}" "${N_TASKS}"
else
  JOBID="$("${CMD[@]}")"
  JOBID="${JOBID%% *}"
  printf '[submitted] Stage C0 v1 array job=%s tasks=%s task_file=%s\n' "${JOBID}" "${N_TASKS}" "${TASK_FILE}"
fi
