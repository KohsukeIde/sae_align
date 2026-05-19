#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-outputs/stageb_b6_v1_cpu}"
DATA_SEEDS="${DATA_SEEDS:-0 1}"
SPLIT_SEEDS="${SPLIT_SEEDS:-17 29}"
PCA_DIMS="${PCA_DIMS:-16 32 64 128}"

ABCI_GROUP="${ABCI_GROUP:-gag51404}"
ABCI_QUEUE="${ABCI_QUEUE:-rt_HC}"
WALLTIME="${WALLTIME:-24:00:00}"
SELECT="${SELECT:-1}"
DRY_RUN="${DRY_RUN:-0}"
STAGEB_ENV_PREFIX="${STAGEB_ENV_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CHANNELS="${CHANNELS:-rgb range local noisy_rgb gray_rgb blur_rgb}"
N_STATES="${N_STATES:-256}"
K_ACTIONS="${K_ACTIONS:-32}"
GRID_SIZE="${GRID_SIZE:-32}"
HORIZON="${HORIZON:-3}"
MAX_DELTA_SAMPLES="${MAX_DELTA_SAMPLES:-4096}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2048}"
MAX_STATES_ANALYZE="${MAX_STATES_ANALYZE:-128}"
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

TASK_FILE="${TASK_FILE:-${OUT}/stageb6_tasks.tsv}"
SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${OUT}/qsub_logs}"
mkdir -p "$(dirname "${TASK_FILE}")" "${SUBMIT_LOG_DIR}"

: > "${TASK_FILE}"
for DATA_SEED in ${DATA_SEEDS}; do
  for SPLIT_SEED in ${SPLIT_SEEDS}; do
    for PCA_DIM in ${PCA_DIMS}; do
      printf "%s\t%s\t%s\n" "${DATA_SEED}" "${SPLIT_SEED}" "${PCA_DIM}" >> "${TASK_FILE}"
    done
  done
done

N_TASKS="$(wc -l < "${TASK_FILE}" | tr -d ' ')"
if [[ "${N_TASKS}" == "0" ]]; then
  echo "[error] no Stage B.6 tasks generated" >&2
  exit 2
fi

colon_list() {
  printf "%s" "$1" | tr -s '[:space:]' ':' | sed 's/^://; s/:$//'
}

pipe_list() {
  printf "%s" "$1" | tr -s '[:space:]' '|' | sed 's/^|//; s/|$//'
}

VARLIST="TASK_FILE=${TASK_FILE},OUT=${OUT},STAGEB_ENV_PREFIX=${STAGEB_ENV_PREFIX},PYTHON_BIN=${PYTHON_BIN},SKIP_EXISTING=${SKIP_EXISTING}"
VARLIST+=",CHANNELS=$(colon_list "${CHANNELS}")"
VARLIST+=",N_STATES=${N_STATES},K_ACTIONS=${K_ACTIONS},GRID_SIZE=${GRID_SIZE},HORIZON=${HORIZON}"
VARLIST+=",MAX_DELTA_SAMPLES=${MAX_DELTA_SAMPLES},MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES},MAX_STATES_ANALYZE=${MAX_STATES_ANALYZE}"
VARLIST+=",RANDOM_PROJECTION_DIM=${RANDOM_PROJECTION_DIM}"
VARLIST+=",K_VALUES=$(colon_list "${K_VALUES}")"
VARLIST+=",JITTER_EPSILONS=$(colon_list "${JITTER_EPSILONS}")"
VARLIST+=",JITTER_SEEDS=$(colon_list "${JITTER_SEEDS}")"
VARLIST+=",NORMALIZATION_MODES=$(colon_list "${NORMALIZATION_MODES}")"
VARLIST+=",REPRESENTATIONS=$(colon_list "${REPRESENTATIONS}")"
VARLIST+=",PERMUTATION_REPEATS=${PERMUTATION_REPEATS},BOOTSTRAP_REPEATS=${BOOTSTRAP_REPEATS},RIDGE_SPLITS=${RIDGE_SPLITS},RIDGE_ALPHA=${RIDGE_ALPHA}"
VARLIST+=",TARGET_PAIRS=$(pipe_list "${TARGET_PAIRS}")"

CMD=(
  qsub
  -P "${ABCI_GROUP}"
  -q "${ABCI_QUEUE}"
  -l "select=${SELECT}"
  -l "walltime=${WALLTIME}"
  -N "stageb6_v1"
  -J "1-${N_TASKS}"
  -j oe
  -o "${SUBMIT_LOG_DIR}/stageb6_v1_array.pbs.log"
  -v "${VARLIST}"
  scripts/stageb_b6_v1_cpu_array.pbs
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${CMD[@]}"; printf '\n'
  printf '[dry-run] task_file=%s n_tasks=%s\n' "${TASK_FILE}" "${N_TASKS}"
else
  JOBID="$("${CMD[@]}")"
  JOBID="${JOBID%% *}"
  printf '[submitted] Stage B.6 array job=%s tasks=%s task_file=%s\n' "${JOBID}" "${N_TASKS}" "${TASK_FILE}"
fi
