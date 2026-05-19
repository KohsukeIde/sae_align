#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:?usage: scripts/monitor_stageb6_job.sh JOB_ID [OUT_DIR] [INTERVAL_SECONDS]}"
OUT="${2:-outputs/stageb_b6_v1_cpu}"
INTERVAL="${3:-300}"
LOG_DIR="${OUT}/monitor_logs"
LOG="${LOG_DIR}/stageb6_${JOB_ID}_monitor.log"
mkdir -p "${LOG_DIR}"

log_status() {
  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    qstat -t "${JOB_ID}[].pbs1" 2>&1 || qstat -t | rg "${JOB_ID}|stageb6" || true
    printf "completed summaries: "
    find "${OUT}" -path "*/stageb6_diagnostics/reports/stageb6_summary.json" | wc -l
    printf "output size: "
    du -sh "${OUT}" 2>/dev/null || true
    echo "recent qsub log:"
    find "${OUT}/qsub_logs" -maxdepth 1 -type f -print -exec tail -n 20 {} \; 2>/dev/null || true
  } >> "${LOG}" 2>&1
}

log_final() {
  {
    echo "===== job left qstat: $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    for i in $(seq 1 16); do
      printf "%s " "${i}"
      qstat -fx "${JOB_ID}[${i}].pbs1" 2>/dev/null \
        | awk -F"= " '/resources_used.walltime|resources_used.cput|Exit_status|job_state/{gsub(/^ +/,"",$1); printf "%s=%s ",$1,$2} END{print ""}'
    done
    completed="$(find "${OUT}" -path "*/stageb6_diagnostics/reports/stageb6_summary.json" | wc -l | tr -d " ")"
    echo "completed summaries: ${completed}"
    if [[ "${completed}" == "16" ]]; then
      PYTHONPATH=src python scripts/summarize_stageb6_grid.py --root "${OUT}" --out "${OUT}"
      echo "summarized Stage B.6 grid"
    else
      echo "not summarized: incomplete task summaries"
    fi
  } >> "${LOG}" 2>&1
}

while true; do
  log_status
  if ! qstat -t "${JOB_ID}[].pbs1" >/dev/null 2>&1; then
    log_final
    break
  fi
  sleep "${INTERVAL}"
done
