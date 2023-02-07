#!/bin/bash

GPUS="1"
#ROOT_DIR="/workspace/input/"
IMG_SIZE=300
WORKERS=4
BATCH_SIZE=64
BACKBONE="b3"
OUTPUT_SIZE=3
CROP="True"
LOG_PATH="/workspace/output/test/log/"
#LOG_NAME="test.log"

#for i in $(seq 5)
#    do
#        ROOT_DIR="/workspace/input/poodle_data_230110/poodle_$i"
#        MODEL_PATH="/workspace/output/train/model/ver$i-40-1_b3_epoch30_batch64_BCEloss_lr0.001_BestEpoch.pth"
#        eval 'python ../src/efficientnet/test.py --gpus=${GPUS} --root_dir=${ROOT_DIR} --workers=${WORKERS} --batch_size=${BATCH_SIZE} --output_size=${OUTPUT_SIZE} --log_path=${LOG_PATH} --model_path=${MODEL_PATH} --crop=${CROP}'
        
#        MODEL_PATH="/workspace/output/train/model/ver$i-40-2_b3_epoch30_batch64_BCEloss_lr0.002_BestEpoch.pth"
#        eval 'python ../src/efficientnet/test.py --gpus=${GPUS} --root_dir=${ROOT_DIR} --workers=${WORKERS} --batch_size=${BATCH_SIZE} --output_size=${OUTPUT_SIZE} --log_path=${LOG_PATH} --model_path=${MODEL_PATH} --crop=${CROP}'
#    done
    
ROOT_DIR="/workspace/poodle_data_230130/"
#MODEL_PATH="/workspace/output/train/model/ver23-01-1_b3_epoch30_batch64_BCEloss_lr0.001_BestEpoch.pth"
MODEL_PATH="/workspace/output/train/model/ver23-01-2_b3_epoch30_batch64_BCEloss_lr0.001_BestEpoch.pth"

eval 'cd /workspace/code/src/efficientnet/'
#eval 'python test.py --gpus=${GPUS} --root_dir=${ROOT_DIR} --workers=${WORKERS} --batch_size=${BATCH_SIZE} --output_size=${OUTPUT_SIZE} --log_path=${LOG_PATH} --model_path=${MODEL_PATH} --crop=${CROP}'
eval 'python val_test.py --gpus=${GPUS} --root_dir=${ROOT_DIR} --workers=${WORKERS} --batch_size=${BATCH_SIZE} --output_size=${OUTPUT_SIZE} --log_path=${LOG_PATH} --model_path=${MODEL_PATH} --crop=${CROP}'
