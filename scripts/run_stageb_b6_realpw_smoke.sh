#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-outputs/stageb_b6_realpw_smoke}"

DATA_SEED="${DATA_SEED:-0}" \
SPLIT_SEED="${SPLIT_SEED:-17}" \
PCA_DIM="${PCA_DIM:-4}" \
OUT="${OUT}" \
BACKEND="${BACKEND:-powderworld}" \
CHANNELS="${CHANNELS:-rgb range noisy_rgb gray_rgb blur_rgb}" \
N_STATES="${N_STATES:-8}" \
K_ACTIONS="${K_ACTIONS:-4}" \
GRID_SIZE="${GRID_SIZE:-32}" \
HORIZON="${HORIZON:-2}" \
MAX_DELTA_SAMPLES="${MAX_DELTA_SAMPLES:-32}" \
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-32}" \
MAX_STATES_ANALYZE="${MAX_STATES_ANALYZE:-8}" \
K_VALUES="${K_VALUES:-2 3}" \
JITTER_EPSILONS="${JITTER_EPSILONS:-0}" \
JITTER_SEEDS="${JITTER_SEEDS:-0}" \
PERMUTATION_REPEATS="${PERMUTATION_REPEATS:-10}" \
BOOTSTRAP_REPEATS="${BOOTSTRAP_REPEATS:-20}" \
RIDGE_SPLITS="${RIDGE_SPLITS:-2}" \
SANITY_KNN_K="${SANITY_KNN_K:-2}" \
SVCCA_COMPONENTS="${SVCCA_COMPONENTS:-4}" \
EXTENDED_METRIC_REPEATS="${EXTENDED_METRIC_REPEATS:-5}" \
SKIP_EXISTING="${SKIP_EXISTING:-0}" \
bash scripts/run_stageb_b6_realpw_task.sh

PYTHONPATH=src python scripts/summarize_stageb6_grid.py \
  --root "${OUT}" \
  --out "${OUT}" \
  --expected-report-dirs 1
