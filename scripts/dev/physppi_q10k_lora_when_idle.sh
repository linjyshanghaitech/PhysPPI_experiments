#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/101045/jinyu/PhysPPI/openfold-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/physppi_q10k_local_msa_training/lora_no_source1}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-1000}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DEVICES="${DEVICES:-4}"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/heartbeat"
LOG_FILE="${LOG_DIR}/heartbeat.log"
SMOKE_DONE="${REPO_ROOT}/${OUTPUT_ROOT}/smoke/status.done"
FULL_DONE="${REPO_ROOT}/${OUTPUT_ROOT}/full/status.done"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "${LOG_FILE}"
}

active_blocking_processes() {
  pgrep -af "physppi_run_q10k_local_official_shards|physppi_broad_predict_compatible_eval|openfold3.run_openfold train" \
    | grep -v "physppi_q10k_lora_when_idle" \
    | grep -v "pgrep -af" \
    || true
}

gpu_busy() {
  while IFS=, read -r idx mem util; do
    mem=${mem// /}
    if [[ -n "${mem}" && "${mem}" -ge "${GPU_MEM_LIMIT_MB}" ]]; then
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  return 1
}

ready_now() {
  [[ -z "$(active_blocking_processes)" ]] && ! gpu_busy
}

next_stage() {
  STAGE=smoke
  if [[ ! -f "${SMOKE_DONE}" ]]; then
    echo smoke
    return
  fi
  STAGE=full
  if [[ ! -f "${FULL_DONE}" ]]; then
    echo full
    return
  fi
  echo none
}

cd "${REPO_ROOT}"
stage=$(next_stage)
if [[ "${stage}" == "none" ]]; then
  log "nothing_to_launch"
  exit 0
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  if ready_now; then
    log "check_only_ready=yes next_stage=${stage}"
    exit 0
  fi
  log "check_only_ready=no next_stage=${stage}"
  exit 3
fi

while ! ready_now; do
  log "not_ready_waiting next_stage=${stage}"
  sleep "${SLEEP_SECONDS}"
done

log "ready_launching stage=${stage}"
STAGE="${stage}" DEVICES="${DEVICES}" DRY_RUN_ONLY=0 \
  scripts/dev/physppi_run_q10k_lora_no_source1_train.sh
