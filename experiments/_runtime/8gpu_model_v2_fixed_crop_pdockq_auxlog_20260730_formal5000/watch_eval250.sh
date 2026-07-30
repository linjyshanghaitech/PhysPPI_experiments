#!/usr/bin/env bash
set -u

REPO_ROOT="/101045/jinyu/PhysPPI_jinyu/PhysPPI"
RUN_DIR="${REPO_ROOT}/experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_formal5000"
VAL_TEMPLATE="${RUN_DIR}/validation16_eval_template.yaml"
CHECKPOINT_DIR="${RUN_DIR}/checkpoint"
EVAL_ROOT="${RUN_DIR}/eval_every250"
LOG_FILE="${RUN_DIR}/eval250_watcher.log"
PYTHON="/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python"
TORCHRUN="/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/torchrun"

mkdir -p "${EVAL_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${LOG_FILE}"
}

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

run_eval() {
  local step="$1"
  local ckpt="${CHECKPOINT_DIR}/step-$(printf '%06d' "${step}").ckpt"
  local eval_dir="${EVAL_ROOT}/step_$(printf '%06d' "${step}")"
  local done_marker="${eval_dir}/.done"
  local running_marker="${eval_dir}/.running"
  local failed_marker="${eval_dir}/.failed"
  local eval_yaml="${eval_dir}/validation16_eval_step_$(printf '%06d' "${step}").yaml"
  local parent_log="${eval_dir}/validation16_eval_step_$(printf '%06d' "${step}").log"

  if [[ -f "${done_marker}" || -f "${running_marker}" ]]; then
    return 0
  fi
  if [[ ! -f "${ckpt}" ]]; then
    return 0
  fi

  sleep 20
  if [[ ! -f "${ckpt}" ]]; then
    log "checkpoint disappeared before eval: ${ckpt}"
    return 0
  fi

  mkdir -p "${eval_dir}"
  date '+%Y-%m-%d %H:%M:%S %Z' > "${running_marker}"
  rm -f "${failed_marker}"
  make_eval_yaml "${step}" "${ckpt}" "${eval_dir}" "${eval_yaml}"

  log "starting val16 step=${step} ckpt=${ckpt} output=${eval_dir}"
  (
    cd "${REPO_ROOT}" || exit 1
    CUDA_LIBS=$(find /101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia -type d -name lib | paste -sd: -)
    export LD_LIBRARY_PATH="/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="/101045/jinyu/PPI/Prior/src:${REPO_ROOT}:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    "${TORCHRUN}" --standalone --nproc_per_node=8 -m openfold3.run_openfold train --runner-yaml "${eval_yaml}"
  ) > "${parent_log}" 2>&1
  status=$?

  rm -f "${running_marker}"
  if [[ "${status}" -eq 0 ]]; then
    date '+%Y-%m-%d %H:%M:%S %Z' > "${done_marker}"
    log "completed val16 step=${step} status=0"
  else
    printf '%s\n' "${status}" > "${failed_marker}"
    log "failed val16 step=${step} status=${status}; see ${parent_log}"
  fi
}

log "watcher started; checkpoint_dir=${CHECKPOINT_DIR}; eval_root=${EVAL_ROOT}"
while true; do
  for step in $(seq 250 250 5000); do
    run_eval "${step}"
  done

  if [[ -f "${CHECKPOINT_DIR}/step-005000.ckpt" && -f "${EVAL_ROOT}/step_005000/.done" ]]; then
    log "watcher finished: step-005000 eval is done"
    exit 0
  fi

  sleep 60
done
