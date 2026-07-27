#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/101045/jinyu/PhysPPI/openfold-3}
PIXI=${PIXI:-/101045/jinyu/.pixi/bin/pixi}
PIXI_ENV=${PIXI_ENV:-openfold3-cuda12}
CKPT=${CKPT:-/101045/jinyu/.openfold3/of3-p2-155k.pt}
DRY_RUN=${DRY_RUN:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_DIFFUSION_SAMPLES=${NUM_DIFFUSION_SAMPLES:-1}
NUM_MODEL_SEEDS=${NUM_MODEL_SEEDS:-1}

BASE="$REPO_ROOT/output/openfold3_official_docker_migration"
INPUT_DIR="$BASE/comparison_inputs"
RUN_DIR="$BASE/comparison_runs"
REPORT_DIR="$BASE/reports"

mkdir -p "$RUN_DIR" "$REPORT_DIR"

run_predict() {
  local label=$1
  local source_tree=$2
  local split=$3
  local query_json=$4
  local output_dir="$RUN_DIR/${label}_${split}"
  local log_file="$RUN_DIR/${label}_${split}.log"

  local cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "PYTHONPATH=$source_tree"
    "$PIXI"
    run
    --manifest-path
    "$REPO_ROOT"
    -e
    "$PIXI_ENV"
    python
    -m
    openfold3.run_openfold
    predict
    --query-json
    "$query_json"
    --inference-ckpt-path
    "$CKPT"
    --num-diffusion-samples
    "$NUM_DIFFUSION_SAMPLES"
    --num-model-seeds
    "$NUM_MODEL_SEEDS"
    --use-msa-server
    false
    --use-templates
    false
    --output-dir
    "$output_dir"
  )

  echo "== $label $split =="
  echo "source_tree=$source_tree"
  echo "query_json=$query_json"
  echo "output_dir=$output_dir"
  printf '%q ' "${cmd[@]}"
  echo

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  mkdir -p "$output_dir"
  (cd /tmp && "${cmd[@]}") >"$log_file" 2>&1
}

validate_inputs() {
  "$PIXI" run --manifest-path "$REPO_ROOT" -e openfold3-cpu python - <<'PY'
import json
from pathlib import Path

base = Path("/101045/jinyu/PhysPPI/openfold-3/output/openfold3_official_docker_migration")
for split in ["train32", "test32"]:
    path = base / "comparison_inputs" / f"{split}_with_msa.json"
    data = json.loads(path.read_text())
    missing_paths = []
    for query_name, query in data["queries"].items():
        for chain in query["chains"]:
            for key in ["main_msa_file_paths", "paired_msa_file_paths"]:
                for msa_path in chain.get(key) or []:
                    if not Path(msa_path).exists():
                        missing_paths.append((query_name, key, msa_path))
    print(split, "queries", len(data["queries"]), "missing_paths", len(missing_paths))
    if missing_paths:
        raise SystemExit(missing_paths[:10])
PY
}

main() {
  cd "$REPO_ROOT"
  validate_inputs

  run_predict \
    docker_code \
    "$BASE/docker_code" \
    train32 \
    "$INPUT_DIR/train32_with_msa.json"
  run_predict \
    docker_code \
    "$BASE/docker_code" \
    test32 \
    "$INPUT_DIR/test32_with_msa.json"
  run_predict \
    github_original \
    "$BASE/github_original" \
    train32 \
    "$INPUT_DIR/train32_with_msa.json"
  run_predict \
    github_original \
    "$BASE/github_original" \
    test32 \
    "$INPUT_DIR/test32_with_msa.json"
}

main "$@"
