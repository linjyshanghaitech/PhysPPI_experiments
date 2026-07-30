#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/101045/jinyu/PhysPPI_jinyu/PhysPPI"
RUN_DIR="${REPO_ROOT}/experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_formal5000"
TRAIN_YAML="${REPO_ROOT}/experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_formal5000.yaml"
TORCHRUN="/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/torchrun"

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

CUDA_LIBS=$(find /101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia -type d -name lib | paste -sd: -)
export LD_LIBRARY_PATH="/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/101045/jinyu/PPI/Prior/src:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

exec "${TORCHRUN}" --standalone --nproc_per_node=8 -m openfold3.run_openfold train --runner-yaml "${TRAIN_YAML}"
