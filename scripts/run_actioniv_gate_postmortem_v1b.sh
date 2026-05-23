#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-outputs/actioniv_gate_postmortem_v1b}"
ROOT="${2:-outputs/actioniv_task_oracle_v1b_cpu_array}"

PYTHONPATH=src:. python scripts/postmortem_actioniv_gate.py \
  --root "${ROOT}" \
  --out "${OUT}"

