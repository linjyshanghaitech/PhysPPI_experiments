#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/101045/jinyu/PhysPPI_jinyu/PhysPPI
CONDA_ENV=/101045/jinyu/.local/share/mamba/envs/PhysPPI
NVIDIA_LIB=$CONDA_ENV/lib/python3.10/site-packages/nvidia
CUE_LIB=$CONDA_ENV/lib/python3.10/site-packages/cuequivariance_ops/lib

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH="$CUE_LIB:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUTPUT_DIR="$REPO_ROOT/experiments/_runtime/4gpu_model_v1"

echo "=== PhysPPI v1 Training (Regression Head) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$REPO_ROOT"
exec "$CONDA_ENV/bin/python" -u "$CONDA_ENV/bin/run_openfold" train \
    --runner-yaml "$REPO_ROOT/experiments/_runtime/4gpu_train_v1.yaml" \
    --seed 42 --data-seed 42 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
