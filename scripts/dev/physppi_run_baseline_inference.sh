#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/dev/physppi_run_baseline_inference.sh [options]

Runs pretrained OpenFold3 baseline inference with physppi.enabled=false.

Options:
  --query-json PATH       Query JSON to run.
                          Default: output/physppi_openfold3_baseline/query_test_small.json
  --output-dir PATH       Output directory for this run.
                          Default: output/physppi_openfold3_baseline/runs/small_${timestamp}
  --runner-yaml PATH      Runner YAML with physppi.enabled=false.
                          Default: examples/example_runner_yamls/physppi_baseline_predict.yml
  --checkpoint-name NAME  OpenFold3 checkpoint registry name.
                          Default: openfold3-p2-155k
  --cuda-devices IDS      CUDA_VISIBLE_DEVICES value.
                          Default: 0
  --execute               Launch inference.
  --print-command-only    Print command and write metadata without launching, even if --execute is present.
  -h, --help              Show this help.

Positional compatibility:
  A single positional *.json argument is treated as --query-json PATH.
USAGE
}

QUERY_JSON="output/physppi_openfold3_baseline/query_test_small.json"
RUNNER_YAML="examples/example_runner_yamls/physppi_baseline_predict.yml"
CHECKPOINT_NAME="openfold3-p2-155k"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EXECUTE=0
PRINT_COMMAND_ONLY=0
OUT_DIR=""

# Auto-detect OpenFold3 Python: prefer pixi, fall back to system python3
PYTHON="python3"
if command -v pixi &>/dev/null && pixi run -e openfold3-cuda12 python3 -c "import openfold3" 2>/dev/null; then
  PYTHON="pixi run -e openfold3-cuda12 python3"
elif command -v /101045/jinyu/.pixi/bin/pixi &>/dev/null; then
  PYTHON="/101045/jinyu/.pixi/bin/pixi run -e openfold3-cuda12 python3"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --query-json)
      QUERY_JSON="${2:?missing value for --query-json}"
      shift 2
      ;;
    --output-dir)
      OUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --runner-yaml)
      RUNNER_YAML="${2:?missing value for --runner-yaml}"
      shift 2
      ;;
    --checkpoint-name)
      CHECKPOINT_NAME="${2:?missing value for --checkpoint-name}"
      shift 2
      ;;
    --cuda-devices)
      CUDA_DEVICES="${2:?missing value for --cuda-devices}"
      shift 2
      ;;
    --execute)
      EXECUTE=1
      shift
      ;;
    --print-command-only)
      PRINT_COMMAND_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *.json)
      QUERY_JSON="$1"
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$QUERY_JSON" ]]; then
  echo "ERROR: query JSON not found: $QUERY_JSON" >&2
  exit 1
fi

if [[ ! -f "$RUNNER_YAML" ]]; then
  echo "ERROR: baseline runner YAML not found: $RUNNER_YAML" >&2
  exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
  RUN_NAME="small_$(date +%Y%m%d_%H%M%S)"
  OUT_DIR="output/physppi_openfold3_baseline/runs/${RUN_NAME}"
else
  RUN_NAME="$(basename "$OUT_DIR")"
fi

mkdir -p "$OUT_DIR"

if [[ "$PRINT_COMMAND_ONLY" == "1" ]]; then
  EXECUTE_EFFECTIVE=0
else
  EXECUTE_EFFECTIVE="$EXECUTE"
fi

cat > "$OUT_DIR/run_info.json" <<EOF
{
  "query_json": "$QUERY_JSON",
  "run_name": "$RUN_NAME",
  "output_dir": "$OUT_DIR",
  "physppi_enabled": false,
  "runner_yaml": "$RUNNER_YAML",
  "pretrained_checkpoint": "$CHECKPOINT_NAME",
  "cuda_visible_devices": "$CUDA_DEVICES",
  "num_model_seeds": 1,
  "num_diffusion_samples": 1,
  "git_commit": "$(git rev-parse --short HEAD)",
  "execute_requested": $([[ "$EXECUTE" == "1" ]] && echo true || echo false),
  "execute": $([[ "$EXECUTE_EFFECTIVE" == "1" ]] && echo true || echo false)
}
EOF

echo "$OUT_DIR" > output/physppi_openfold3_baseline/latest_small_run.txt

echo "Baseline inference run: $RUN_NAME"
echo "  query: $QUERY_JSON"
echo "  output: $OUT_DIR"
echo "  physppi: disabled"
echo "  runner_yaml: $RUNNER_YAML"
echo "  checkpoint: $CHECKPOINT_NAME"
echo "  cuda: $CUDA_DEVICES"
echo ""
echo "Command:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES $PYTHON openfold3/run_openfold.py predict \\"
echo "    --query-json $QUERY_JSON \\"
echo "    --runner-yaml $RUNNER_YAML \\"
echo "    --inference-ckpt-name $CHECKPOINT_NAME \\"
echo "    --output-dir $OUT_DIR \\"
echo "    --num-model-seeds 1 \\"
echo "    --num-diffusion-samples 1"
echo ""

if [[ "$EXECUTE_EFFECTIVE" != "1" ]]; then
  echo "Dry run complete. Add --execute to launch."
  exit 0
fi

echo "Launching inference..."
GPU_LOG="$OUT_DIR/gpu.csv"
(
  echo "timestamp,index,utilization.gpu,memory.used,memory.total"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 5
  done
) > "$GPU_LOG" &
GPU_MONITOR_PID=$!

cleanup() {
  kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

START_SECONDS="$(date +%s)"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" $PYTHON openfold3/run_openfold.py predict \
  --query-json "$QUERY_JSON" \
  --runner-yaml "$RUNNER_YAML" \
  --inference-ckpt-name "$CHECKPOINT_NAME" \
  --output-dir "$OUT_DIR" \
  --num-model-seeds 1 \
  --num-diffusion-samples 1 \
  2>&1 | tee "$OUT_DIR/inference.log"
END_SECONDS="$(date +%s)"

cleanup
trap - EXIT

python3 - "$OUT_DIR/run_info.json" "$START_SECONDS" "$END_SECONDS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
data = json.loads(path.read_text())
data["start_epoch_seconds"] = start
data["end_epoch_seconds"] = end
data["elapsed_seconds"] = end - start
path.write_text(json.dumps(data, indent=2) + "\n")
PY

echo "Done. Output in $OUT_DIR"
