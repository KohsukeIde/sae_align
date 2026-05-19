#!/usr/bin/env bash
set -euo pipefail

TASK_FILE="${TASK_FILE:?TASK_FILE is required}"
INDEX="${PBS_ARRAY_INDEX:?PBS_ARRAY_INDEX is required}"
LINE="$(sed -n "${INDEX}p" "${TASK_FILE}")"
if [[ -z "${LINE}" ]]; then
  echo "[error] no task line for PBS_ARRAY_INDEX=${INDEX} in ${TASK_FILE}" >&2
  exit 2
fi

read -r OBS_MIX INPUT_LABEL INPUT_CHANNELS_COLON <<< "${LINE}"
INPUT_CHANNELS="${INPUT_CHANNELS_COLON//:/ }"

DATA="${DATA:?DATA is required}"
OUT="${OUT:-outputs/stagec0_v1_cpu_array}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS="${SEEDS:-0:1:2:3:4}"
CHANNEL_DIM="${CHANNEL_DIM:-128}"
RIDGE="${RIDGE:-1.0}"
WEIGHT_ALPHA="${WEIGHT_ALPHA:-4.0}"
OOD_NOISE_STD="${OOD_NOISE_STD:-0.35}"

CELL="${OUT}/input_${INPUT_LABEL}/obs_${OBS_MIX}"

"${PYTHON_BIN}" scripts/train_stagec0_prediction.py \
  --data "${DATA}" \
  --out "${CELL}" \
  --input-channels ${INPUT_CHANNELS} \
  --target-channel rgb \
  --event-channel event_response \
  --observability-channels rgb range \
  --observability-mix "${OBS_MIX}" \
  --methods uniform change_mask observability shuffled_observability oracle_event \
  --seeds ${SEEDS//:/ } \
  --channel-dim "${CHANNEL_DIM}" \
  --ridge "${RIDGE}" \
  --weight-alpha "${WEIGHT_ALPHA}" \
  --ood-noise-std "${OOD_NOISE_STD}"
