#!/bin/bash

TAGS="dataset230130-TRANSFORM_PERCENTAGE_EXP"
CRITERION="focal"
FOCAL_GAMMA=1.0
MODEL_NAME="resnet50"
TRANSFORM_PERCENTAGE=0.05
OPTIMIZER="adam"
LR=0.0003
TRAIN_DATA_DIR="/workspace/poodle_data_230130/train"  ## poodle_0 Dataset
VAL_DATA_DIR="/workspace/poodle_data_230130/valid"    ## poodle_0 Dataset

#########################################################################

eval 'cd /workspace/code/src/resnet'

eval 'python model_train.py --tags=${TAGS} --train_data_dir=${TRAIN_DATA_DIR} --val_data_dir=${VAL_DATA_DIR} --criterion=${CRITERION} --focal_gamma=${FOCAL_GAMMA} --model_name=${MODEL_NAME} --lr=${LR} --transform_percentage=${TRANSFORM_PERCENTAGE} --optimizer=${OPTIMIZER}'
