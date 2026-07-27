#!/usr/bin/env bash
set -euo pipefail

cd /101045/jinyu/PhysPPI/openfold-3

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="output/physppi_full_adapter_train_${RUN_ID}"
mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

/101045/jinyu/.pixi/bin/pixi run -e openfold3-cuda12 python -m openfold3.run_openfold train \
  --config examples/training_yamls/physppi_training.yml \
  --output-dir "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/train.log"
