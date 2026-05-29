#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

ROOT="${ROOT:-outputs/static_weak_realpw_v1_cpu}"
OUT_CONTROLS="${OUT_CONTROLS:-${ROOT}/static_controls_static_weak_v1}"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
ABCI_QUEUE="${ABCI_QUEUE:-rt_HC}"
WALLTIME="${WALLTIME:-04:00:00}"
SELECT="${SELECT:-1}"
DRY_RUN="${DRY_RUN:-0}"
STAGEB_ENV_PREFIX="${STAGEB_ENV_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${ROOT}/qsub_logs}"
mkdir -p "${SUBMIT_LOG_DIR}"

VARLIST="ROOT=${ROOT},OUT_CONTROLS=${OUT_CONTROLS},STAGEB_ENV_PREFIX=${STAGEB_ENV_PREFIX},PYTHON_BIN=${PYTHON_BIN}"

CMD=(
  qsub
  -P "${ABCI_GROUP}"
  -q "${ABCI_QUEUE}"
  -l "select=${SELECT}"
  -l "walltime=${WALLTIME}"
  -N "sw_ctrl"
  -j oe
  -o "${SUBMIT_LOG_DIR}/static_weak_static_controls.pbs.log"
  -v "${VARLIST}"
  scripts/static_weak_static_controls.pbs
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${CMD[@]}"; printf '\n'
else
  JOBID="$("${CMD[@]}")"
  JOBID="${JOBID%% *}"
  printf '[submitted] static-weak static controls job=%s root=%s out=%s\n' \
    "${JOBID}" "${ROOT}" "${OUT_CONTROLS}"
fi

