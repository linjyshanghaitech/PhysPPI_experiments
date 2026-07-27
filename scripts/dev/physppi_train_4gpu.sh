#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/dev/physppi_train_4gpu.sh examples/training_yamls/physppi_training.yml" >&2
  exit 2
fi

yaml_path="$1"
run_dir="output/physppi_openfold3_training/four_gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
echo "$run_dir" > output/physppi_openfold3_training/latest_four_gpu_run.txt

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
      --format=csv,noheader,nounits >> "$run_dir/gpu.csv" 2>/dev/null || true
    sleep 2
  done
) &
mon_pid=$!
cleanup() {
  kill "$mon_pid" 2>/dev/null || true
}
trap cleanup EXIT

# Set GPUs before validation; default to 0,1,2,3 if not set
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES
gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
yaml_devices=$(sed -n 's/^[[:space:]]*devices:[[:space:]]*\([0-9]*\).*/\1/p' "$yaml_path")
if [[ "$gpu_count" -ne "$yaml_devices" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES exposes $gpu_count GPU(s) but yaml requests $yaml_devices device(s)" >&2
  exit 1
fi

PYTHONUNBUFFERED=1 \
  /101045/jinyu/.pixi/bin/pixi run -e openfold3-cuda12 \
  run_openfold train --runner-yaml "$yaml_path" --seed 42 \
  2>&1 | tee "$run_dir/train.log"

status=${PIPESTATUS[0]}
cleanup
exit "$status"
