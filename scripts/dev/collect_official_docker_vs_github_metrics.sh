#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/101045/jinyu/PhysPPI/openfold-3}
PIXI=${PIXI:-/101045/jinyu/.pixi/bin/pixi}
BASE="$REPO_ROOT/output/openfold3_official_docker_migration"
RUN_DIR="$BASE/comparison_runs"
METRICS_DIR="$BASE/comparison_metrics"
EPOCH14="$REPO_ROOT/output/physppi_finetune_validation/phase6/epoch14_eval"

link_native_dir() {
  local split=$1
  local run=$2
  local source_native
  if [[ "$split" == "train32" ]]; then
    source_native="$EPOCH14/train_official_fixed/native"
  elif [[ "$split" == "test32" ]]; then
    source_native="$EPOCH14/test_official_fixed/native"
  else
    echo "unknown split: $split" >&2
    return 2
  fi

  if [[ ! -d "$source_native" ]]; then
    echo "missing native source: $source_native" >&2
    return 2
  fi

  mkdir -p "$run"
  rm -f "$run/native"
  if [[ -e "$run/native" && ! -L "$run/native" ]]; then
    echo "refusing to replace non-symlink native directory: $run/native" >&2
    return 2
  fi
  ln -s "$source_native" "$run/native"
}

require_run_outputs() {
  local run=$1
  if [[ ! -d "$run/predictions" ]]; then
    echo "missing predictions directory: $run/predictions" >&2
    return 2
  fi
}

main() {
  cd "$REPO_ROOT"
  mkdir -p "$METRICS_DIR"

  link_native_dir train32 "$RUN_DIR/docker_code_train32"
  link_native_dir test32 "$RUN_DIR/docker_code_test32"
  link_native_dir train32 "$RUN_DIR/github_original_train32"
  link_native_dir test32 "$RUN_DIR/github_original_test32"

  require_run_outputs "$RUN_DIR/docker_code_train32"
  require_run_outputs "$RUN_DIR/docker_code_test32"
  require_run_outputs "$RUN_DIR/github_original_train32"
  require_run_outputs "$RUN_DIR/github_original_test32"

  "$PIXI" run --manifest-path "$REPO_ROOT" -e openfold3-cpu \
    python scripts/dev/physppi_epoch14_collect_metrics.py \
    --output-dir "$METRICS_DIR" \
    --run train docker_code \
      output/physppi_finetune_validation/phase6/epoch14_eval/sets/train_sample_ids.json \
      "$RUN_DIR/docker_code_train32" \
    --run test docker_code \
      output/physppi_finetune_validation/phase6/epoch14_eval/sets/test_sample_ids.json \
      "$RUN_DIR/docker_code_test32" \
    --run train github_original \
      output/physppi_finetune_validation/phase6/epoch14_eval/sets/train_sample_ids.json \
      "$RUN_DIR/github_original_train32" \
    --run test github_original \
      output/physppi_finetune_validation/phase6/epoch14_eval/sets/test_sample_ids.json \
      "$RUN_DIR/github_original_test32"
}

main "$@"
