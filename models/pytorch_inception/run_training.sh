#!/bin/bash
# 快速训练脚本

DATA_FOLDER="/home/xl/dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
OUTPUT_DIR="./output"

python train.py \
    --datafolder "$DATA_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --task all \
    --batch_size 16 \
    --epochs 15 \
    --lr 0.001 \
    --depth 9 \
    --kernel_size 60 \
    --device cuda

