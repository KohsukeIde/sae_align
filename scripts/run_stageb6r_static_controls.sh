#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

ROOT="${1:-outputs/stageb6r_realpw_v1_cpu}"
OUT="${2:-${ROOT}/static_controls_b6r_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}/src}"

"${PYTHON_BIN}" scripts/analyze_stageb6_static_controls.py \
  --root "${ROOT}" \
  --out "${OUT}" \
  --expected-report-dirs 9 \
  --pca-dims 32 \
  --channels rgb range \
  --normalization-modes probe_action_type_apply \
  --k-values 10 \
  --target-pairs rgb:range \
  --max-states 128 \
  --permutation-repeats 10

printf 'Stage B6R static controls complete: root=%s out=%s\n' "${ROOT}" "${OUT}"
