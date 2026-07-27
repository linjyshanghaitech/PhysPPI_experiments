#!/usr/bin/env bash
set -euo pipefail

cd /101045/jinyu/PhysPPI/openfold-3

STAGE="${STAGE:-smoke}"
if [[ "${STAGE}" != "smoke" && "${STAGE}" != "full" ]]; then
  echo "STAGE must be smoke or full, got: ${STAGE}" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-output/physppi_q10k_local_msa_training/lora_no_source1}"
DEVICES="${DEVICES:-4}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
CACHE_JSON="${OUTPUT_ROOT}/cache/training_cache_q10k_lora_${STAGE}.json"
RUNNER_YAML="${OUTPUT_ROOT}/${STAGE}/train.yml"
VALIDATION_JSON="${OUTPUT_ROOT}/${STAGE}/validation.json"
NOHUP_LOG="${OUTPUT_ROOT}/${STAGE}/train.nohup.log"
PID_FILE="${OUTPUT_ROOT}/q10k_lora_no_source1_${STAGE}.pid"
LOG_DIR="${OUTPUT_ROOT}/${STAGE}/logs"

mkdir -p "${LOG_DIR}"

echo "[1/4] prepare ${STAGE} train cache"
python3 scripts/dev/physppi_prepare_q10k_lora_train_cache.py \
  --stage "${STAGE}" \
  --output-root "${OUTPUT_ROOT}" \
  | tee "${LOG_DIR}/prepare_cache.log"

echo "[2/4] build no-source1 LoRA YAML"
if [[ "${STAGE}" == "smoke" ]]; then
  python3 scripts/dev/physppi_build_q10k_lora_no_source1_yaml.py \
    --stage smoke \
    --output-root "${OUTPUT_ROOT}" \
    --training-cache "${CACHE_JSON}" \
    --devices "${DEVICES}" \
    --max-steps 2 \
    --max-epochs 1 \
    | tee "${LOG_DIR}/build_yaml.log"
else
  python3 scripts/dev/physppi_build_q10k_lora_no_source1_yaml.py \
    --stage full \
    --output-root "${OUTPUT_ROOT}" \
    --training-cache "${CACHE_JSON}" \
    --devices "${DEVICES}" \
    --max-epochs 3 \
    | tee "${LOG_DIR}/build_yaml.log"
fi

echo "[3/4] validate no-source1 LoRA YAML"
python3 scripts/dev/physppi_validate_q10k_lora_no_source1_yaml.py \
  --runner-yaml "${RUNNER_YAML}" \
  --output-report "${VALIDATION_JSON}" \
  | tee "${LOG_DIR}/validate_yaml.log"

if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
  echo "DRY_RUN_ONLY=1; stop before GPU launch"
  exit 0
fi

echo "[4/4] launch ${STAGE} training"
nohup /101045/jinyu/.pixi/bin/pixi run -e openfold3-cuda12 \
  python -u -m openfold3.run_openfold train \
  --runner-yaml "${RUNNER_YAML}" \
  > "${NOHUP_LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "pid=$(cat "${PID_FILE}")"
echo "nohup_log=${NOHUP_LOG}"
