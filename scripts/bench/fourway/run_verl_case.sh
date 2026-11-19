#!/usr/bin/env bash
# Runs inside the open-mopd container. Env: METHOD BATCH DYN_MIN_TOKENS MICRO RUN_NAME
set -euo pipefail
export RUN_DIR="/study/runs/${RUN_NAME}"
export USER=xxf LOGNAME=xxf XDG_CACHE_HOME=/cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/study/harness:/repo/training/verl
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_DEVICE_MAX_CONNECTIONS=8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=42 OPENMOPD_PINNED_ALIGNMENT_MB=2
export HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=/cache/torch_extensions TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor
export TRITON_CACHE_DIR=/cache/triton VLLM_CACHE_ROOT=/cache/vllm MPLCONFIGDIR=/cache/matplotlib
export HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1 VLLM_USE_V2_MODEL_RUNNER=0
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN_DIR/exit-code.txt"' EXIT
python3 /study/harness/run_verl.py --cfg job --resolve > "$RUN_DIR/resolved-config.yaml"
python3 /study/harness/run_verl.py
