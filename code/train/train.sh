#!/bin/bash

GPUS="1"
#ROOT_DIR="../../input/"
IMG_SIZE=300
WORKERS=4
BATCH_SIZE=64
EPOCH=30
BACKBONE="b3"
OUTPUT_SIZE=3
CROP="True"
SCHEDULER="True"
PATIENCE=4
LOSS="BCE"
LOG_PATH="../../output/train/log/"
#LOG_NAME="train.log"

#for i in $(seq 5)
#    do
#        ROOT_DIR="../../input/poodle_data_230110/poodle_$i"
#        LR="0.001"
#        VERSION="$i-40-1"
#        eval 'python ../src/efficientnet/train.py --gpus=${GPUS} --root_dir=${ROOT_DIR} --img_size=${IMG_SIZE} --workers=${WORKERS} --batch_size=${BATCH_SIZE} --epoch=${EPOCH} --backbone=${BACKBONE} --patience=${PATIENCE} --output_size=${OUTPUT_SIZE} --loss=${LOSS} --scheduler=${SCHEDULER} --LR=${LR} --version=${VERSION} --crop=${CROP} --log_path=${LOG_PATH}'
        
#        LR="0.002"
#        VERSION="$i-40-2"
#        eval 'python ../src/efficientnet/train.py --gpus=${GPUS} --root_dir=${ROOT_DIR} --img_size=${IMG_SIZE} --workers=${WORKERS} --batch_size=${BATCH_SIZE} --epoch=${EPOCH} --backbone=${BACKBONE} --patience=${PATIENCE} --output_size=${OUTPUT_SIZE} --loss=${LOSS} --scheduler=${SCHEDULER} --LR=${LR} --version=${VERSION} --crop=${CROP} --log_path=${LOG_PATH}'
#  done

ROOT_DIR="../../poodle_data_230130/"
LR="0.001"
VERSION="23-01-2"
eval 'python ../src/efficientnet/train.py --gpus=${GPUS} --root_dir=${ROOT_DIR} --img_size=${IMG_SIZE} --workers=${WORKERS} --batch_size=${BATCH_SIZE} --epoch=${EPOCH} --backbone=${BACKBONE} --patience=${PATIENCE} --output_size=${OUTPUT_SIZE} --loss=${LOSS} --scheduler=${SCHEDULER} --LR=${LR} --version=${VERSION} --crop=${CROP} --log_path=${LOG_PATH}'




