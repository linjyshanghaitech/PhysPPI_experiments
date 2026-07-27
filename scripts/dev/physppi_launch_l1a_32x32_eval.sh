#!/usr/bin/env bash
set -euo pipefail

REPO=/101045/jinyu/PhysPPI/openfold-3
PIXI=/101045/jinyu/.pixi/bin/pixi
ENV_CUDA=openfold3-cuda12
ENV_CPU=openfold3-cpu

cd "${REPO}"

BASE=output/physppi_lora_15epoch/L1a_eval_32x32
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="${BASE}/run_${STAMP}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/metrics"
printf '%s\n' "${RUN_ROOT}" > "${BASE}/current_run.txt"

TRAIN_YAML=output/physppi_lora_15epoch/L1a/train.yml
TEST_YAML=output/physppi_lora_15epoch/L1a_eval_32x32/test32_lora_eval.yml
CKPT=output/physppi_lora_15epoch/L1a/run/checkpoints/last.ckpt
TRAIN_IDS=output/physppi_finetune_validation/phase6/epoch14_eval/sets/train_sample_ids.json
TEST_IDS=output/physppi_finetune_validation/phase6/epoch14_eval/sets/test_sample_ids.json
OFFICIAL_BASELINE_RUN=${OFFICIAL_BASELINE_RUN:-}

for required in "${TRAIN_YAML}" "${TEST_YAML}" "${CKPT}" "${TRAIN_IDS}" "${TEST_IDS}"; do
  test -e "${required}" || {
    echo "missing required input: ${required}" >&2
    exit 2
  }
done

run_split() {
  local split=$1
  local yaml=$2
  local ids=$3
  local outdir=$4
  mkdir -p "${outdir}"

  echo "[$(date '+%F %T %z')] launching ${split} L1a sampling"
  local pids=()
  for rank in 0 1 2 3; do
    local log="${RUN_ROOT}/logs/${split}_rank${rank}.log"
    (
      export CUDA_VISIBLE_DEVICES="${rank}"
      "${PIXI}" run -e "${ENV_CUDA}" python scripts/dev/physppi_epoch14_sample_dataset.py \
        --runner-yaml "${yaml}" \
        --sample-ids "${ids}" \
        --checkpoint "${CKPT}" \
        --output-dir "${outdir}" \
        --variant finetuned \
        --seed 42 \
        --rank "${rank}" \
        --world-size 4 \
        --device cuda:0
    ) > "${log}" 2>&1 &
    pids+=("$!")
    echo "${split} rank ${rank} pid ${pids[-1]} log ${log}"
    sleep 3
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done

  if [[ "${status}" != 0 ]]; then
    echo "[$(date '+%F %T %z')] ${split} sampling failed" >&2
    return "${status}"
  fi
  echo "[$(date '+%F %T %z')] ${split} L1a sampling finished"
}

{
  echo "run_root=${RUN_ROOT}"
  echo "checkpoint=${CKPT}"
  echo "train_yaml=${TRAIN_YAML}"
  echo "test_yaml=${TEST_YAML}"
  echo "official_baseline_run=${OFFICIAL_BASELINE_RUN}"
  echo "started_at=$(date '+%F %T %z')"
} > "${RUN_ROOT}/run_info.txt"

if run_split train "${TRAIN_YAML}" "${TRAIN_IDS}" "${RUN_ROOT}/train_l1a"; then
  if run_split test "${TEST_YAML}" "${TEST_IDS}" "${RUN_ROOT}/test_l1a"; then
    if [[ -n "${OFFICIAL_BASELINE_RUN}" ]]; then
      marker="${OFFICIAL_BASELINE_RUN}/metrics/ACCEPTED_OFFICIAL_PREDICT_BASELINE"
      test -e "${marker}" || {
        echo "official baseline run is not accepted: ${marker}" >&2
        exit 2
      }
      printf '%s\n' "${OFFICIAL_BASELINE_RUN}" \
        > "${RUN_ROOT}/metrics/official_baseline_run.txt"
    fi
    "${PIXI}" run -e "${ENV_CPU}" python scripts/dev/physppi_epoch14_collect_metrics.py \
      --output-dir "${RUN_ROOT}/metrics" \
      --run train l1a "${TRAIN_IDS}" "${RUN_ROOT}/train_l1a" \
      --run test l1a "${TEST_IDS}" "${RUN_ROOT}/test_l1a"
    echo "finished_at=$(date '+%F %T %z')" >> "${RUN_ROOT}/run_info.txt"
    touch "${RUN_ROOT}/status.done"
    echo "[$(date '+%F %T %z')] L1a 32+32 evaluation complete"
    exit 0
  fi
fi

echo "failed_at=$(date '+%F %T %z')" >> "${RUN_ROOT}/run_info.txt"
touch "${RUN_ROOT}/status.failed"
exit 1
