#!/usr/bin/env bash
# Site-local qsub template for Action-IV official-task sanity.
# Edit PBS directives for the local cluster before use.
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-outputs/actioniv_task_oracle_array}
TASK_GEN=${TASK_GEN:-PWTaskGenDestroyAll}
SEEDS=${SEEDS:-"0 1 2"}
N_STATES=${N_STATES:-256}
ACTIONS_PER_STATE=${ACTIONS_PER_STATE:-16}
TIME_PER_ACTION=${TIME_PER_ACTION:-2}
N_WORLD_SETTLE=${N_WORLD_SETTLE:-64}
ERASE_ACTION_FRACTION=${ERASE_ACTION_FRACTION:-0.5}
WALLTIME=${WALLTIME:-02:00:00}
ABCI_GROUP=${ABCI_GROUP:-gag51404}
ABCI_QUEUE=${ABCI_QUEUE:-rt_HC}
SELECT=${SELECT:-1:ncpus=8:mem=32gb}
PYTHON_BIN=${PYTHON_BIN:-python}
DRY_RUN=${DRY_RUN:-0}
LOG_DIR=${LOG_DIR:-${OUT_ROOT}/qsub_logs}
OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE:-1}
OPENBLAS_NUM_THREADS_VALUE=${OPENBLAS_NUM_THREADS_VALUE:-1}
MKL_NUM_THREADS_VALUE=${MKL_NUM_THREADS_VALUE:-1}
NUMEXPR_NUM_THREADS_VALUE=${NUMEXPR_NUM_THREADS_VALUE:-1}

mkdir -p "$OUT_ROOT" "$LOG_DIR"
SEED_LIST=($SEEDS)
N=${#SEED_LIST[@]}

cat > "$OUT_ROOT/job.sh" <<'EOS'
#!/usr/bin/env bash
#PBS -j oe
set -euo pipefail
cd "$PBS_O_WORKDIR"
IDX=$((PBS_ARRAY_INDEX-1))
SEED=$(sed -n "$((IDX+1))p" "$OUT_ROOT/seeds.txt")
RUN_DIR="$OUT_ROOT/seed_${SEED}"
mkdir -p "$RUN_DIR/data"
export OMP_NUM_THREADS="${OMP_NUM_THREADS_VALUE:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS_VALUE:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_VALUE:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS_VALUE:-1}"
PYTHONPATH=src:. "${PYTHON_BIN:-python}" scripts/make_actioniv_task_dataset.py \
  --backend powderworld \
  --task-gen "$TASK_GEN" \
  --out "$RUN_DIR/data/task_dataset.npz" \
  --n-states "$N_STATES" \
  --actions-per-state "$ACTIONS_PER_STATE" \
  --device cpu \
  --time-per-action "$TIME_PER_ACTION" \
  --n-world-settle "$N_WORLD_SETTLE" \
  --erase-action-fraction "$ERASE_ACTION_FRACTION" \
  --seed "$SEED"
PYTHONPATH=src:. "${PYTHON_BIN:-python}" scripts/train_actioniv_task_oracle.py \
  --data "$RUN_DIR/data/task_dataset.npz" \
  --out "$RUN_DIR/task_oracle" \
  --input-channels rgb \
  --methods uniform oracle_task change_mask observability shuffled_observability \
  --alphas 1 2 4 8 16 32 \
  --split-seeds 0 1 2 \
  --channel-dim 64 \
  --epochs 120
EOS

printf "%s\n" ${SEED_LIST[@]} > "$OUT_ROOT/seeds.txt"
VARLIST="OUT_ROOT=${OUT_ROOT},TASK_GEN=${TASK_GEN},N_STATES=${N_STATES},ACTIONS_PER_STATE=${ACTIONS_PER_STATE},TIME_PER_ACTION=${TIME_PER_ACTION},N_WORLD_SETTLE=${N_WORLD_SETTLE},ERASE_ACTION_FRACTION=${ERASE_ACTION_FRACTION},PYTHON_BIN=${PYTHON_BIN}"
VARLIST+=",OMP_NUM_THREADS_VALUE=${OMP_NUM_THREADS_VALUE},OPENBLAS_NUM_THREADS_VALUE=${OPENBLAS_NUM_THREADS_VALUE},MKL_NUM_THREADS_VALUE=${MKL_NUM_THREADS_VALUE},NUMEXPR_NUM_THREADS_VALUE=${NUMEXPR_NUM_THREADS_VALUE}"
CMD=(
  qsub
  -P "${ABCI_GROUP}"
  -q "${ABCI_QUEUE}"
  -l "select=${SELECT}"
  -l "walltime=${WALLTIME}"
  -N "actioniv_task"
  -J "1-${N}"
  -j oe
  -o "${LOG_DIR}/actioniv_task_oracle_array.pbs.log"
  -v "${VARLIST}"
  "$OUT_ROOT/job.sh"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[dry-run]'; printf ' %q' "${CMD[@]}"; printf '\n'
  printf '[dry-run] seeds=%s task_file=%s\n' "$N" "$OUT_ROOT/seeds.txt"
else
  "${CMD[@]}"
fi
