#!/bin/bash

set -e
source /data/home/yezhuyifan/miniforge3/etc/profile.d/conda.sh
conda activate colabfold

INPUT_PATH=$1
OUTPUT_PATH=$2
CUDA_ID=$3

CUDA_VISIBLE_DEVICES=$CUDA_ID python -m colabfold.relax \
    --use-gpu \
    $INPUT_PATH \
    $OUTPUT_PATH
