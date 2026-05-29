#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

ROOT="${1:-outputs/static_weak_realpw_v1_cpu}"
OUT="${2:-${ROOT}/static_controls_static_weak_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}/src}"

"${PYTHON_BIN}" scripts/analyze_stageb6_static_controls.py \
  --root "${ROOT}" \
  --out "${OUT}" \
  --expected-report-dirs 9 \
  --pca-dims 32 \
  --channels rgb range local edge noisy_rgb gray_rgb blur_rgb \
  --normalization-modes probe_action_type_apply \
  --k-values 10 \
  --target-pairs rgb:range rgb:edge rgb:local range:local rgb:noisy_rgb rgb:gray_rgb rgb:blur_rgb \
  --max-states 128 \
  --permutation-repeats 10

printf 'Static-weak static controls complete: root=%s out=%s\n' "${ROOT}" "${OUT}"
