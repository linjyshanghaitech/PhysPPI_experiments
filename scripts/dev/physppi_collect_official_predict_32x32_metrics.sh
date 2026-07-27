#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/101045/jinyu/PhysPPI/openfold-3}
PIXI=${PIXI:-/101045/jinyu/.pixi/bin/pixi}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT to output/physppi_baseline_first/official_predict_32x32/run_<timestamp>}
SEED=${SEED:-2746317213}

cd "${REPO}"

TRAIN_IDS=output/physppi_finetune_validation/phase6/epoch14_eval/sets/train_sample_ids.json
TEST_IDS=output/physppi_finetune_validation/phase6/epoch14_eval/sets/test_sample_ids.json
TRAIN_NATIVE=output/physppi_finetune_validation/phase6/epoch14_eval/train_official_fixed/native
TEST_NATIVE=output/physppi_finetune_validation/phase6/epoch14_eval/test_official_fixed/native
TRAIN_RUN="${RUN_ROOT}/train_official_predict"
TEST_RUN="${RUN_ROOT}/test_official_predict"
NORMALIZED_ROOT="${RUN_ROOT}/dockq_normalized"
TRAIN_NORMALIZED="${NORMALIZED_ROOT}/train_official_predict"
TEST_NORMALIZED="${NORMALIZED_ROOT}/test_official_predict"
METRICS_DIR="${RUN_ROOT}/metrics"

mkdir -p "${METRICS_DIR}" "${NORMALIZED_ROOT}"

for required in \
  "${TRAIN_IDS}" \
  "${TEST_IDS}" \
  "${TRAIN_NATIVE}" \
  "${TEST_NATIVE}" \
  "${TRAIN_RUN}" \
  "${TEST_RUN}"; do
  test -e "${required}" || {
    echo "missing required input: ${required}" >&2
    exit 2
  }
done

ln -sfn . "${TRAIN_RUN}/predictions"
ln -sfn . "${TEST_RUN}/predictions"

"${PIXI}" run -e openfold3-cpu python \
  scripts/dev/physppi_normalize_cif_for_dockq.py \
  --input-run "${TRAIN_RUN}" \
  --output-run "${TRAIN_NORMALIZED}" \
  --sample-ids-json "${TRAIN_IDS}" \
  --seed "${SEED}"

"${PIXI}" run -e openfold3-cpu python \
  scripts/dev/physppi_normalize_cif_for_dockq.py \
  --input-run "${TEST_RUN}" \
  --output-run "${TEST_NORMALIZED}" \
  --sample-ids-json "${TEST_IDS}" \
  --seed "${SEED}"

ln -sfn "${REPO}/${TRAIN_NATIVE}" "${TRAIN_NORMALIZED}/native"
ln -sfn "${REPO}/${TEST_NATIVE}" "${TEST_NORMALIZED}/native"

"${PIXI}" run -e openfold3-cpu python scripts/dev/physppi_epoch14_collect_metrics.py \
  --output-dir "${METRICS_DIR}" \
  --seed "${SEED}" \
  --run train official_predict "${TRAIN_IDS}" "${TRAIN_NORMALIZED}" \
  --run test official_predict "${TEST_IDS}" "${TEST_NORMALIZED}"

"${PIXI}" run -e openfold3-cpu env RUN_ROOT="${RUN_ROOT}" python - <<'PY'
import json
import os
from pathlib import Path

run_root = os.environ["RUN_ROOT"]
summary = json.loads((Path(run_root) / "metrics" / "metrics_summary.json").read_text())
rows = {(row["split"], row["variant"]): row for row in summary}
for key in [("train", "official_predict"), ("test", "official_predict")]:
    row = rows[key]
    print(key, "count", row["count"], "dockq_ok", row["dockq_ok"], "mean", row["dockq_mean"])
    if row["count"] != 32 or row["dockq_ok"] != 32:
        raise SystemExit(f"official predict metrics not accepted: {key} {row}")
PY

touch "${METRICS_DIR}/ACCEPTED_OFFICIAL_PREDICT_BASELINE"
