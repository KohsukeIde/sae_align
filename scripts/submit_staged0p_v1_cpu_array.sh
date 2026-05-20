#!/usr/bin/env bash
# ABCI/PBS template for Stage D0'. Adjust group, queue, and walltime locally.
set -euo pipefail
ROOT=${ROOT:-outputs/staged0p_v1_cpu_array}
DATASETS_FILE=${DATASETS_FILE:-configs/staged0p_datasets.txt}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
if [[ ! -f "$DATASETS_FILE" ]]; then
  cat >&2 <<EOF
Missing $DATASETS_FILE.
Create one Stage-0 dataset path per line, for example:
outputs/stageb_b6_primary_cell_v1_cpu/seed_0/split_17/pca_32/stage0/stage0_dataset.npz
outputs/stageb_b6_primary_cell_v1_cpu/seed_1/split_17/pca_32/stage0/stage0_dataset.npz
EOF
  exit 2
fi
N=$(grep -v '^[[:space:]]*#' "$DATASETS_FILE" | grep -vc '^[[:space:]]*$')
mkdir -p "$ROOT"
cat > "$ROOT/staged0p_array.pbs" <<'PBS'
#!/usr/bin/env bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=01:00:00
#PBS -j oe
set -euo pipefail
cd "$PBS_O_WORKDIR"
TASK_ID=${PBS_ARRAY_INDEX:-1}
DATASETS_FILE=${DATASETS_FILE:-configs/staged0p_datasets.txt}
DATA=$(grep -v '^[[:space:]]*#' "$DATASETS_FILE" | grep -v '^[[:space:]]*$' | sed -n "${TASK_ID}p")
OUT=${ROOT:-outputs/staged0p_v1_cpu_array}/task_${TASK_ID}
PYTHONPATH=src:. python scripts/train_staged0p_predictor_grounded.py \
  --data "$DATA" \
  --out "$OUT" \
  --input-channels rgb \
  --observability-channels rgb range \
  --methods uniform change_mask observability shuffled_observability predictor_uncertainty predictor_grounded shuffled_predictor_grounded lossgrad_observability oracle_event \
  --alphas 2 4 8 16 32 \
  --seeds 0 1 2 3 4 \
  --channel-dim 64 \
  --epochs 120 \
  --lr 0.2 \
  --l2 1e-4
PBS
qsub -v ROOT="$ROOT",DATASETS_FILE="$DATASETS_FILE",OMP_NUM_THREADS="$OMP_NUM_THREADS",OPENBLAS_NUM_THREADS="$OPENBLAS_NUM_THREADS",MKL_NUM_THREADS="$MKL_NUM_THREADS",NUMEXPR_NUM_THREADS="$NUMEXPR_NUM_THREADS" -J 1-"$N" "$ROOT/staged0p_array.pbs"
echo "Submitted $N Stage D0' tasks. After completion, run:"
echo "PYTHONPATH=src:. python scripts/summarize_staged0p_grid.py --root $ROOT --out $ROOT --expected-report-dirs $N"
