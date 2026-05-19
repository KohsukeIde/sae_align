#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${1:-outputs/stageb_b6_v1_cpu}"
DATA_SEEDS="${DATA_SEEDS:-0 1}"
SPLIT_SEEDS="${SPLIT_SEEDS:-17 29}"
PCA_DIMS="${PCA_DIMS:-16 32 64 128}"

for DATA_SEED in ${DATA_SEEDS}; do
  for SPLIT_SEED in ${SPLIT_SEEDS}; do
    for PCA_DIM in ${PCA_DIMS}; do
      DATA_SEED="${DATA_SEED}" \
      SPLIT_SEED="${SPLIT_SEED}" \
      PCA_DIM="${PCA_DIM}" \
      OUT="${OUT}" \
      bash scripts/run_stageb_b6_v1_cpu_task.sh
    done
  done
done

python scripts/summarize_stageb6_grid.py --root "${OUT}" --out "${OUT}"

printf '\nStage B.6 v1 CPU grid complete: %s\n' "${OUT}"
