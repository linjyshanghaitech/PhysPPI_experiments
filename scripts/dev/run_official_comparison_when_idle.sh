#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/101045/jinyu/PhysPPI/openfold-3}
LOG_DIR="$REPO_ROOT/output/openfold3_official_docker_migration/reports"
LOG_FILE="$LOG_DIR/official_comparison_when_idle.log"
SLEEP_SECONDS=${SLEEP_SECONDS:-300}
GPU_MEM_LIMIT_MB=${GPU_MEM_LIMIT_MB:-1000}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
PIXI_ENV=${PIXI_ENV:-openfold3-cuda12}
CHECK_ONLY=${CHECK_ONLY:-0}

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$LOG_FILE"
}

training_processes() {
  pgrep -af "run_sequence_resume_after_epoch5_eval|openfold3.run_openfold train|launch_4gpu" \
    | grep -v "run_official_comparison_when_idle" \
    | grep -v "pgrep -af" || true
}

gpu_busy() {
  local rows
  rows=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  while IFS=, read -r index mem util; do
    mem=${mem// /}
    if [[ -n "$mem" && "$mem" -ge "$GPU_MEM_LIMIT_MB" ]]; then
      return 0
    fi
  done <<<"$rows"
  return 1
}

print_state() {
  log "git_head=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
  local gpu_state
  gpu_state=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')
  log "gpu_state=$gpu_state"
  local procs
  procs=$(training_processes)
  if [[ -n "$procs" ]]; then
    log "training_processes_present=yes"
    echo "$procs" | tee -a "$LOG_FILE"
  else
    log "training_processes_present=no"
  fi
}

ready_now() {
  [[ -z "$(training_processes)" ]] && ! gpu_busy
}

main() {
  cd "$REPO_ROOT"
  log "watcher_start sleep_seconds=$SLEEP_SECONDS gpu_mem_limit_mb=$GPU_MEM_LIMIT_MB cuda_visible_devices=$CUDA_VISIBLE_DEVICES pixi_env=$PIXI_ENV"
  print_state

  if [[ "$CHECK_ONLY" == "1" ]]; then
    if ready_now; then
      log "check_only_ready=yes"
      return 0
    fi
    log "check_only_ready=no"
    return 3
  fi

  while ! ready_now; do
    log "not_ready_waiting"
    sleep "$SLEEP_SECONDS"
    print_state
  done

  log "ready_starting_official_comparison"
  DRY_RUN=0 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PIXI_ENV="$PIXI_ENV" \
    scripts/dev/run_official_docker_vs_github_comparison.sh \
    >"$LOG_DIR/official_comparison_run.log" 2>&1
  log "official_comparison_finished"

  scripts/dev/collect_official_docker_vs_github_metrics.sh \
    >"$LOG_DIR/official_comparison_collect_metrics.log" 2>&1
  log "metrics_collection_finished"

  /101045/jinyu/.pixi/bin/pixi run -e openfold3-cpu \
    python scripts/dev/summarize_official_docker_vs_github_metrics.py \
    >"$LOG_DIR/official_comparison_summarize.log" 2>&1
  log "summary_finished"
}

main "$@"
