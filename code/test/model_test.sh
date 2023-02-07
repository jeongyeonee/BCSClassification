#!/bin/bash

#CHECK_PATH="/workspace/output/train/model/dataset230105-TRANSFORM_PERCENTAGE_EXP-model1673253148.790018.tar"
#CHECK_PATH="/workspace/output/train/model/dataset230119-TRANSFORM_PERCENTAGE_EXP-model1674109864.785069.tar"
CHECK_PATH="/workspace/output/train/model/dataset230130-TRANSFORM_PERCENTAGE_EXP-model1675061367.835669.tar"
TRAIN_DATA_DIR="/workspace/poodle_data_230130/train"  # poodle_0 Dataset
TRAIN_DATA_DIR="/workspace/poodle_data_230130/train"  # poodle_0 Dataset
TEST_DATA_DIR="/workspace/poodle_data_230130/test"    # poodle_0 Dataset
CRITERION="focal"
FOCAL_GAMMA=1.0
MODEL_NAME="resnet50"


eval 'cd /workspace/code/src/resnet'

eval 'python model_test.py --check_path=${CHECK_PATH} --model_name=${MODEL_NAME} --criterion=${CRITERION} --focal_gamma=${FOCAL_GAMMA} --train_data_dir=${TRAIN_DATA_DIR} --test_data_dir=${TEST_DATA_DIR}'
