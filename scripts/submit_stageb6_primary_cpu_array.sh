#!/usr/bin/env bash
# Template for the larger Stage-B6 primary-cell replication requested before
# final framing. This assumes existing B6 scripts are already present.
#$ -cwd
#$ -l h_rt=08:00:00
#$ -l s_vmem=8G
#$ -l mem_req=8G
#$ -t 1-9
#$ -j y
#$ -o logs/stageb6_primary.$TASK_ID.log

set -euo pipefail
mkdir -p logs

# Map 9 jobs to 3 data seeds x 3 split seeds.
DATA_SEEDS=(0 1 2)
SPLIT_SEEDS=(17 29 43)
IDX=$((SGE_TASK_ID - 1))
DATA_SEED=${DATA_SEEDS[$((IDX / 3))]}
SPLIT_SEED=${SPLIT_SEEDS[$((IDX % 3))]}
ROOT=${ROOT:-outputs/stageb_b6_primary_v2_cpu}
OUT="$ROOT/data${DATA_SEED}_split${SPLIT_SEED}"

# This is intentionally a template because local sites differ in qsub modules.
# It calls the existing Stage-B6 diagnostic script with the preregistered primary
# representation family. Adjust DATA/MODEL/STATIC_MODEL paths as needed.
DATA=${DATA:-outputs/stage0_v1/data/stage0_dataset.npz}
MODEL=${MODEL:-outputs/stageb_v1/transition_encoders.npz}
STATIC_MODEL=${STATIC_MODEL:-outputs/stageb_v1_static/transition_encoders.npz}
PROBE_ACTION_IDS=${PROBE_ACTION_IDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"}

PYTHONPATH=src python scripts/analyze_stageb6_diagnostics.py \
  --data "$DATA" \
  --model "$MODEL" \
  --static-model "$STATIC_MODEL" \
  --out "$OUT" \
  --probe-action-ids $PROBE_ACTION_IDS \
  --representations pca_probe_only \
  --normalization-modes probe_action_type_apply \
  --k-values 5 10 20 \
  --jitter-epsilons 0 1e-5 1e-4 \
  --jitter-seeds 0 1 2 3 4 \
  --target-pairs rgb:range \
  --bootstrap-repeats 500 \
  --bootstrap-seed "$SPLIT_SEED" \
  --seed "$DATA_SEED"
