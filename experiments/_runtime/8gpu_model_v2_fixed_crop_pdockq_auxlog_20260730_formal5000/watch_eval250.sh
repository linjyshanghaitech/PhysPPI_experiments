#!/usr/bin/env bash
set -u

REPO_ROOT="/101045/jinyu/PhysPPI_jinyu/PhysPPI"
RUN_DIR="${REPO_ROOT}/experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_formal5000"
VAL_TEMPLATE="${RUN_DIR}/validation16_eval_template.yaml"
RESUME_BASE="${RUN_DIR}/8gpu_resume_base.yaml"
CHECKPOINT_DIR="${RUN_DIR}/checkpoint"
EVAL_ROOT="${RUN_DIR}/eval_every250"
LOG_FILE="${RUN_DIR}/eval250_watcher.log"
PYTHON="/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python"
TORCHRUN="/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/torchrun"

mkdir -p "${EVAL_ROOT}"

# ── environment setup (shared by training and eval) ──────────────────────
CUDA_LIBS=$(find /101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH="/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/101045/jinyu/PPI/Prior/src:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL settings to prevent SIGABRT timeouts on large-model gradient sync
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${LOG_FILE}"
}

# ── find the latest checkpoint step number ──────────────────────────────
find_latest_checkpoint_step() {
  local max_step=0
  local ckpt_file
  for ckpt_file in "${CHECKPOINT_DIR}"/step-*.ckpt; do
    [[ -f "${ckpt_file}" ]] || continue
    local fname=$(basename "${ckpt_file}")
    # Extract step number from "step-XXXXXX.ckpt" or "step-XXXXXX-vYY.ckpt"
    local step_num
    step_num=$(echo "${fname}" | sed -n 's/^step-0*\([0-9]\+\).*\.ckpt$/\1/p')
    if [[ -n "${step_num}" ]] && [[ "${step_num}" -gt "${max_step}" ]]; then
      max_step="${step_num}"
    fi
  done
  echo "${max_step}"
}

# ── generate per-step eval YAML from template ───────────────────────────
make_eval_yaml() {
  local step="$1"
  local ckpt="$2"
  local eval_dir="$3"
  local eval_yaml="$4"
  "${PYTHON}" - "${VAL_TEMPLATE}" "${eval_yaml}" "${ckpt}" "${eval_dir}" <<'PY'
from pathlib import Path
import sys
import yaml

template = Path(sys.argv[1])
target = Path(sys.argv[2])
checkpoint = Path(sys.argv[3])
eval_dir = Path(sys.argv[4])
repo = Path("/101045/jinyu/PhysPPI_jinyu/PhysPPI")

with template.open() as f:
    cfg = yaml.safe_load(f)

cfg["experiment_settings"]["restart_checkpoint_path"] = str(checkpoint)
cfg["experiment_settings"]["output_dir"] = str(eval_dir.relative_to(repo))
cfg["checkpoint_config"]["dirpath"] = str((eval_dir / "checkpoint").relative_to(repo))
settings = cfg["model_update"]["custom"]["settings"]
settings["validation_use_ema_weights"] = False
settings["validation_artifacts"]["output_dir"] = str(
    (eval_dir / "validation_artifacts").relative_to(repo)
)

target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

# ── generate resume YAML for training from a given checkpoint ───────────
make_resume_yaml() {
  local ckpt="$1"
  local out_yaml="${RUN_DIR}/8gpu_resume_active.yaml"
  "${PYTHON}" - "${RESUME_BASE}" "${out_yaml}" "${ckpt}" <<'PY'
from pathlib import Path
import sys
import yaml

base = Path(sys.argv[1])
target = Path(sys.argv[2])
ckpt = Path(sys.argv[3])

with base.open() as f:
    cfg = yaml.safe_load(f)

cfg["experiment_settings"]["restart_checkpoint_path"] = str(ckpt)

with target.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
  echo "${out_yaml}"
}

# ── kill training and all its children, wait for GPU memory to free ─────
stop_training() {
  local pid="$1"
  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  log "stopping training (PID ${pid})..."

  # SIGTERM to the torchrun parent and all child processes
  local children
  children=$(pgrep -P "${pid}" 2>/dev/null || true)
  kill "${pid}" 2>/dev/null || true
  for c in ${children}; do
    kill "${c}" 2>/dev/null || true
  done

  # Wait up to 30 s for graceful exit
  local waited=0
  while kill -0 "${pid}" 2>/dev/null && [[ ${waited} -lt 30 ]]; do
    sleep 2
    waited=$((waited + 2))
  done

  # Force kill if still alive
  if kill -0 "${pid}" 2>/dev/null; then
    log "force-killing training (PID ${pid})"
    children=$(pgrep -P "${pid}" 2>/dev/null || true)
    kill -9 "${pid}" 2>/dev/null || true
    for c in ${children}; do
      kill -9 "${c}" 2>/dev/null || true
    done
    sleep 2
  fi

  wait "${pid}" 2>/dev/null || true

  # Wait for GPU memory to be fully released
  log "waiting for GPU memory to free..."
  sleep 10
}

# ── run eval for a given step ───────────────────────────────────────────
run_eval() {
  local step="$1"
  local ckpt="${CHECKPOINT_DIR}/step-$(printf '%06d' "${step}").ckpt"
  local eval_dir="${EVAL_ROOT}/step_$(printf '%06d' "${step}")"
  local done_marker="${eval_dir}/.done"
  local running_marker="${eval_dir}/.running"
  local failed_marker="${eval_dir}/.failed"
  local eval_yaml="${eval_dir}/validation16_eval_step_$(printf '%06d' "${step}").yaml"
  local parent_log="${eval_dir}/validation16_eval_step_$(printf '%06d' "${step}").log"

  if [[ -f "${done_marker}" ]]; then
    log "eval step=${step} already done, skipping"
    return 0
  fi
  if [[ ! -f "${ckpt}" ]]; then
    log "checkpoint not found: ${ckpt}, skipping eval"
    return 1
  fi

  mkdir -p "${eval_dir}"
  date '+%Y-%m-%d %H:%M:%S %Z' > "${running_marker}"
  rm -f "${failed_marker}"
  make_eval_yaml "${step}" "${ckpt}" "${eval_dir}" "${eval_yaml}"

  log "starting val16 step=${step} ckpt=${ckpt}"
  (
    cd "${REPO_ROOT}" || exit 1
    "${TORCHRUN}" --standalone --nproc_per_node=8 -m openfold3.run_openfold train --runner-yaml "${eval_yaml}"
  ) > "${parent_log}" 2>&1
  local status=$?

  rm -f "${running_marker}"
  if [[ "${status}" -eq 0 ]]; then
    date '+%Y-%m-%d %H:%M:%S %Z' > "${done_marker}"
    log "completed val16 step=${step} status=0"
  else
    printf '%s\n' "${status}" > "${failed_marker}"
    log "FAILED val16 step=${step} status=${status}; see ${parent_log}"
  fi
  return "${status}"
}

# ── launch training from a given checkpoint step ────────────────────────
launch_training() {
  local step="$1"
  local ckpt="${CHECKPOINT_DIR}/step-$(printf '%06d' "${step}").ckpt"

  if [[ ! -f "${ckpt}" ]]; then
    log "FATAL: checkpoint not found: ${ckpt}"
    return 1
  fi

  local resume_yaml
  resume_yaml=$(make_resume_yaml "${ckpt}")
  log "launching training from step=${step} ckpt=${ckpt}"

  cd "${REPO_ROOT}" || exit 1
  "${TORCHRUN}" --standalone --nproc_per_node=8 -m openfold3.run_openfold train \
    --runner-yaml "${resume_yaml}" &
  local train_pid=$!
  echo "${train_pid}"
}

# ══════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════

START_STEP="${1:-250}"   # first checkpoint already available
MAX_STEP=5000
INTERVAL=250

log "======================================================"
log "watcher v2 started (coordinated train-eval loop)"
log "  start_step=${START_STEP}  max_step=${MAX_STEP}  interval=${INTERVAL}"
log "  checkpoint_dir=${CHECKPOINT_DIR}"
log "  eval_root=${EVAL_ROOT}"
log "======================================================"

CURRENT_STEP="${START_STEP}"

while true; do
  NEXT_MILESTONE=$((CURRENT_STEP + INTERVAL))

  if [[ "${CURRENT_STEP}" -ge "${MAX_STEP}" ]]; then
    log "all milestones done (current=${CURRENT_STEP}, max=${MAX_STEP})"
    exit 0
  fi

  # Check if the NEXT milestone eval is already done
  NEXT_DONE="${EVAL_ROOT}/step_$(printf '%06d' ${NEXT_MILESTONE})/.done"
  if [[ -f "${NEXT_DONE}" ]]; then
    log "step=${NEXT_MILESTONE} eval already done, advancing"
    CURRENT_STEP="${NEXT_MILESTONE}"
    continue
  fi

  # 1. Launch training from current checkpoint
  TRAIN_PID=$(launch_training "${CURRENT_STEP}")
  if [[ -z "${TRAIN_PID}" ]]; then
    log "FATAL: failed to launch training from step=${CURRENT_STEP}"
    exit 1
  fi

  MILESTONE_CKPT="${CHECKPOINT_DIR}/step-$(printf '%06d' ${NEXT_MILESTONE}).ckpt"

  # 2. Wait for the next milestone checkpoint
  log "waiting for next milestone: step=${NEXT_MILESTONE}"
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    if [[ -f "${MILESTONE_CKPT}" ]]; then
      # Small safety: wait 5s so the checkpoint is fully written
      sleep 5
      log "milestone checkpoint reached: ${MILESTONE_CKPT}"

      # 3. Stop training
      stop_training "${TRAIN_PID}"
      TRAIN_PID=""

      # 4. Run eval
      run_eval "${NEXT_MILESTONE}"

      # 5. Advance
      CURRENT_STEP="${NEXT_MILESTONE}"
      break
    fi
    sleep 30
  done

  # If training died unexpectedly (not killed by us)
  if [[ -n "${TRAIN_PID}" ]] && ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    log "WARNING: training exited unexpectedly (PID ${TRAIN_PID})"

    # Find the latest checkpoint — training may have progressed past CURRENT_STEP
    latest_ckpt_step=$(find_latest_checkpoint_step)
    log "latest checkpoint step = ${latest_ckpt_step}, current = ${CURRENT_STEP}"

    if [[ "${latest_ckpt_step}" -gt "${CURRENT_STEP}" ]]; then
      log "advancing from step=${CURRENT_STEP} to step=${latest_ckpt_step} (latest checkpoint)"
      CURRENT_STEP="${latest_ckpt_step}"
    fi

    # If we've reached/passed a milestone, run its eval
    if [[ -f "${MILESTONE_CKPT}" ]]; then
      log "milestone checkpoint exists, running eval"
      run_eval "${NEXT_MILESTONE}" || true
      if [[ "${NEXT_MILESTONE}" -gt "${CURRENT_STEP}" ]]; then
        CURRENT_STEP="${NEXT_MILESTONE}"
      fi
    else
      log "no milestone checkpoint yet, resuming from step=${CURRENT_STEP}"
    fi
  fi

  if [[ "${CURRENT_STEP}" -ge "${MAX_STEP}" ]]; then
    log "all milestones complete; watcher exiting"
    exit 0
  fi
done
