#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/101045/jinyu/PhysPPI/openfold-3}
PIXI=${PIXI:-/101045/jinyu/.pixi/bin/pixi}
ENV_CUDA=${ENV_CUDA:-openfold3-cuda12}
CKPT=${CKPT:-/101045/jinyu/.openfold3/of3-p2-155k.pt}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_DIFFUSION_SAMPLES=${NUM_DIFFUSION_SAMPLES:-1}
NUM_MODEL_SEEDS=${NUM_MODEL_SEEDS:-1}
DRY_RUN=${DRY_RUN:-1}

cd "${REPO}"

BASE=output/physppi_baseline_first/official_predict_32x32
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="${BASE}/run_${STAMP}"
mkdir -p "${RUN_ROOT}/logs"
printf '%s\n' "${RUN_ROOT}" > "${BASE}/current_run.txt"

RUNNER_YAML=output/openfold3_official_docker_migration/comparison_inputs/inference_no_deepspeed_evo.yml
TRAIN_QUERY=output/openfold3_official_docker_migration/comparison_inputs/train32_with_msa.json
TEST_QUERY=output/openfold3_official_docker_migration/comparison_inputs/test32_with_msa.json

for required in "${RUNNER_YAML}" "${TRAIN_QUERY}" "${TEST_QUERY}" "${CKPT}"; do
  test -e "${required}" || {
    echo "missing required input: ${required}" >&2
    exit 2
  }
done

validate_query_paths() {
  "${PIXI}" run -e openfold3-cpu python - "$TRAIN_QUERY" "$TEST_QUERY" <<'PY'
import json
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    path = Path(raw)
    payload = json.loads(path.read_text())
    queries = payload.get("queries", {})
    missing = []
    for query_name, query in queries.items():
        for chain in query.get("chains", []):
            for key in ("main_msa_file_paths", "paired_msa_file_paths"):
                for msa_path in chain.get(key) or []:
                    if not Path(msa_path).is_file():
                        missing.append((query_name, key, msa_path))
    print(path, "queries", len(queries), "missing_msa_paths", len(missing))
    if len(queries) != 32:
        raise SystemExit(f"expected 32 queries in {path}, got {len(queries)}")
    if missing:
        raise SystemExit(missing[:5])
PY
}

run_predict_split() {
  local split=$1
  local query_json=$2
  local output_dir="${RUN_ROOT}/${split}_official_predict"
  local log_file="${RUN_ROOT}/logs/${split}_official_predict.log"
  mkdir -p "${output_dir}"

  local cmd=(
    env
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "${PIXI}"
    run
    -e
    "${ENV_CUDA}"
    python
    -m
    openfold3.run_openfold
    predict
    --query-json
    "${query_json}"
    --runner-yaml
    "${RUNNER_YAML}"
    --inference-ckpt-path
    "${CKPT}"
    --num-diffusion-samples
    "${NUM_DIFFUSION_SAMPLES}"
    --num-model-seeds
    "${NUM_MODEL_SEEDS}"
    --use-msa-server
    false
    --use-templates
    false
    --output-dir
    "${output_dir}"
  )

  {
    echo "split=${split}"
    echo "query_json=${query_json}"
    echo "output_dir=${output_dir}"
    printf '%q ' "${cmd[@]}"
    echo
  } | tee "${log_file}.cmd"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}" >"${log_file}" 2>&1
  ln -sfn . "${output_dir}/predictions"
}

{
  echo "run_root=${RUN_ROOT}"
  echo "started_at=$(date '+%F %T %z')"
  echo "runner_yaml=${RUNNER_YAML}"
  echo "train_query=${TRAIN_QUERY}"
  echo "test_query=${TEST_QUERY}"
  echo "checkpoint=${CKPT}"
  echo "dry_run=${DRY_RUN}"
} > "${RUN_ROOT}/run_info.txt"

validate_query_paths
run_predict_split train "${TRAIN_QUERY}"
run_predict_split test "${TEST_QUERY}"
echo "finished_at=$(date '+%F %T %z')" >> "${RUN_ROOT}/run_info.txt"
touch "${RUN_ROOT}/status.done"
echo "${RUN_ROOT}"
