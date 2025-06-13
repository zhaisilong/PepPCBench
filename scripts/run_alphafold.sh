#!/bin/bash

set -e

JSON_PATH=$1
OUTPUT_DIR=$2
RUN_DATA_PIPELINE=$3
RUN_INFERENCE=$4
GPU_ID=$5
AF_ROOT=$HOME/projects/alphafold/alphafold3

source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate af3_tmp

pushd $AF_ROOT
XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_UNIFIED_MEMORY=true XLA_CLIENT_MEM_FRACTION=3.2 XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" CUDA_VISIBLE_DEVICES=$GPU_ID python run_alphafold.py \
--json_path=$JSON_PATH \
    --model_dir=weights \
    --output_dir=$OUTPUT_DIR \
    --jackhmmer_n_cpu=12 \
    --nhmmer_n_cpu=12 \
    --jax_compilation_cache_dir=datasets/public_databases/jax_cache \
    --run_inference=$RUN_INFERENCE \
    --run_data_pipeline=$RUN_DATA_PIPELINE \
    --num_diffusion_samples=5 \
    --max_template_date=2026-01-01 \
    --db_dir=datasets \
    --num_recycles=10
popd
