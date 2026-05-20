#!/usr/bin/env bash
# ABCI/PBS template for Stage D0 real-Powderworld oracle-positive audits.
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-outputs/staged0_v1_cpu_array}
CONFIG=${CONFIG:-configs/staged0_powderworld.json}
DATA_SEEDS=${DATA_SEEDS:-"0 1 2"}
ABCI_GROUP=${ABCI_GROUP:-gag51404}
ABCI_QUEUE=${ABCI_QUEUE:-rt_HC}
WALLTIME=${WALLTIME:-04:00:00}
SELECT=${SELECT:-1}
DRY_RUN=${DRY_RUN:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE:-1}
OPENBLAS_NUM_THREADS_VALUE=${OPENBLAS_NUM_THREADS_VALUE:-1}
MKL_NUM_THREADS_VALUE=${MKL_NUM_THREADS_VALUE:-1}
NUMEXPR_NUM_THREADS_VALUE=${NUMEXPR_NUM_THREADS_VALUE:-1}

TASK_FILE="${TASK_FILE:-${OUT_ROOT}/staged0_tasks.tsv}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/qsub_logs}"
mkdir -p "$(dirname "${TASK_FILE}")" "${LOG_DIR}"
: > "${TASK_FILE}"
for SEED in ${DATA_SEEDS}; do
  printf "%s\n" "${SEED}" >> "${TASK_FILE}"
done
N_TASKS=$(wc -l < "${TASK_FILE}" | tr -d ' ')
if [[ "${N_TASKS}" == "0" ]]; then
  echo "[error] no Stage D0 tasks generated" >&2
  exit 2
fi

cat > "${OUT_ROOT}/staged0_v1_cpu_array.pbs" <<'PBS'
#!/usr/bin/env bash
#PBS -j oe
set -euo pipefail
cd "${PBS_O_WORKDIR}"
export PYTHONPATH=src:.
export OMP_NUM_THREADS="${OMP_NUM_THREADS_VALUE:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS_VALUE:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_VALUE:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS_VALUE:-1}"

TASK_ID="${PBS_ARRAY_INDEX:-1}"
SEED=$(sed -n "${TASK_ID}p" "${TASK_FILE}")
RUN_OUT="${OUT_ROOT}/seed_${SEED}"
DATA="${RUN_OUT}/data/staged0_dataset.npz"

"${PYTHON_BIN:-python}" scripts/make_staged0_powderworld_dataset.py \
  --config "${CONFIG}" \
  --backend powderworld \
  --seed "${SEED}" \
  --out "${DATA}"

"${PYTHON_BIN:-python}" scripts/train_staged0p_predictor_grounded.py \
  --data "${DATA}" \
  --out "${RUN_OUT}/audit" \
  --input-channels rgb \
  --observability-channels rgb range \
  --seeds 0 1 2 \
  --alphas 2 4 8 16 32 \
  --epochs 300 \
  --channel-dim 64
PBS

VARLIST="OUT_ROOT=${OUT_ROOT},CONFIG=${CONFIG},TASK_FILE=${TASK_FILE},PYTHON_BIN=${PYTHON_BIN}"
VARLIST+=",OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE},OPENBLAS_NUM_THREADS_VALUE=${OPENBLAS_NUM_THREADS_VALUE},MKL_NUM_THREADS_VALUE=${MKL_NUM_THREADS_VALUE},NUMEXPR_NUM_THREADS_VALUE=${NUMEXPR_NUM_THREADS_VALUE}"
CMD=(
  qsub
  -P "${ABCI_GROUP}"
  -q "${ABCI_QUEUE}"
  -l "select=${SELECT}"
  -l "walltime=${WALLTIME}"
  -N "staged0_v1"
  -J "1-${N_TASKS}"
  -j oe
  -o "${LOG_DIR}/staged0_v1_array.pbs.log"
  -v "${VARLIST}"
  "${OUT_ROOT}/staged0_v1_cpu_array.pbs"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${CMD[@]}"; printf '\n'
  printf '[dry-run] task_file=%s n_tasks=%s\n' "${TASK_FILE}" "${N_TASKS}"
else
  JOBID="$("${CMD[@]}")"
  JOBID="${JOBID%% *}"
  printf '[submitted] Stage D0 v1 array job=%s tasks=%s task_file=%s\n' "${JOBID}" "${N_TASKS}" "${TASK_FILE}"
  printf 'After completion: PYTHONPATH=src:. python scripts/summarize_staged0p_grid.py --root %s --out %s --expected-report-dirs %s --phase d0\n' "${OUT_ROOT}" "${OUT_ROOT}" "${N_TASKS}"
fi
