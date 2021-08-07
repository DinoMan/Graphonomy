#!/bin/bash
GPUS=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo $GPUS
for NUM_DEV in $(seq 1 $GPUS); do
    ID=$(($NUM_DEV-1))
    CUDA_VISIBLE_DEVICES=$ID python exp/inference/inference_videos.py --multiprocess $NUM_DEV $GPUS --model_path model/inference.pth "$@" &> progress_"$NUM_DEV".txt&
done
