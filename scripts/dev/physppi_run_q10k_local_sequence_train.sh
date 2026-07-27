#!/usr/bin/env bash
set -euo pipefail

cd /101045/jinyu/PhysPPI/openfold-3

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-output/physppi_q10k_local_msa_training/sequence_${STAMP}}"
FROZEN_JSON="${FROZEN_JSON:-output/physppi_q10k_local_msa_training/frozen/q10k_local_msa_available_20260630.json}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/inputs}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/broad_predict_compatible}"
LOG_DIR="${ROOT}/logs"
Q10K_VARIANTS="${Q10K_VARIANTS:-official_predict}"
VARIANT_ARGS=()
for variant in ${Q10K_VARIANTS}; do
  VARIANT_ARGS+=(--variant "${variant}")
done

mkdir -p "${ROOT}" "${INPUT_DIR}" "${LOG_DIR}"
echo "${ROOT}" > output/physppi_q10k_local_msa_training/latest_sequence_run.txt

echo "[1/5] freeze current Q10k local-MSA set when frozen JSON is absent"
if [[ ! -s "${FROZEN_JSON}" ]]; then
  python3 scripts/dev/physppi_freeze_q10k_local_msa.py \
    --dataset-root /101045/jinyu/PhysPPI/prior_processed/Q-BioLiP-10k \
    --msa-local-root /101045/jinyu/PhysPPI/prior_processed/Q-BioLiP-10k/msa_local \
    --output-json "${FROZEN_JSON}" \
    --frozen-at 2026-06-30T00:00:00+08:00 \
    | tee "${LOG_DIR}/freeze.log"
else
  echo "Using existing frozen JSON: ${FROZEN_JSON}" | tee "${LOG_DIR}/freeze.log"
fi

echo "[2/5] build split and query inputs from frozen JSON"
python3 scripts/dev/physppi_build_q10k_local_training_inputs.py \
  --frozen-json "${FROZEN_JSON}" \
  --output-dir "${INPUT_DIR}" \
  --seed "${SPLIT_SEED:-20260630}" \
  --train-fraction "${TRAIN_FRACTION:-0.8}" \
  --val-fraction "${VAL_FRACTION:-0.1}" \
  | tee "${LOG_DIR}/build_inputs.log"

echo "[3/5] validate frozen IDs, split IDs, and query MSA paths"
python3 scripts/dev/physppi_validate_q10k_local_training_inputs.py \
  --frozen-json "${FROZEN_JSON}" \
  --query-json "${INPUT_DIR}/queries/train_with_msa.json" \
  --query-json "${INPUT_DIR}/queries/val_with_msa.json" \
  --query-json "${INPUT_DIR}/queries/test_with_msa.json" \
  --split-dir "${INPUT_DIR}/splits" \
  --output-report "${INPUT_DIR}/q10k_local_msa_coverage_report.json" \
  | tee "${LOG_DIR}/validate_inputs.log"

echo "[4/5] dry-run broad predict-compatible sequence"
python3 scripts/dev/physppi_broad_predict_compatible_eval.py \
  "${VARIANT_ARGS[@]}" \
  --train-query "${INPUT_DIR}/queries/train_with_msa.json" \
  --test-query "${INPUT_DIR}/queries/test_with_msa.json" \
  --train-ids "${INPUT_DIR}/splits/train_ids.json" \
  --test-ids "${INPUT_DIR}/splits/test_ids.json" \
  --pkl-dir /101045/jinyu/PhysPPI/prior_processed/Q-BioLiP-10k/total_pkl \
  --native-gt-dir /101045/jinyu/PhysPPI/prior_processed/Q-BioLiP-10k/gt_multimer \
  --run-root "${RUN_ROOT}" \
  --dry-run \
  | tee "${LOG_DIR}/broad_eval_dry_run.log"

if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  echo "DRY_RUN_ONLY=1; stop before GPU launch"
  exit 0
fi

echo "[5/5] launch baseline sequence"
python3 scripts/dev/physppi_broad_predict_compatible_eval.py \
  "${VARIANT_ARGS[@]}" \
  --train-query "${INPUT_DIR}/queries/train_with_msa.json" \
  --test-query "${INPUT_DIR}/queries/test_with_msa.json" \
  --train-ids "${INPUT_DIR}/splits/train_ids.json" \
  --test-ids "${INPUT_DIR}/splits/test_ids.json" \
  --pkl-dir /101045/jinyu/PhysPPI/prior_processed/Q-BioLiP-10k/total_pkl \
  --native-gt-dir /101045/jinyu/PhysPPI/prior_processed/Q-BioLiP-10k/gt_multimer \
  --run-root "${RUN_ROOT}" \
  2>&1 | tee "${LOG_DIR}/broad_eval_sequence.log"

echo "${ROOT}" > output/physppi_q10k_local_msa_training/latest_finished_sequence_run.txt
