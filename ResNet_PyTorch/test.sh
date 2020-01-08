#!/bin/sh
#source activate pytorch11_py36
set -eu

N_SAMPLINGS=100
BATCH_SIZE=64

#-------------------
# ResNet-18
#-------------------
mkdir -p ${PWD}/_logs
EXEP_NAME=ResNet18_test_Sampling${N_SAMPLINGS}_191230

python test.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir ../dataset \
    --load_checkpoints_dir checkpoints/ResNet18_train_Epoch10_191230 \
    --dataset mnist --image_size 224 --n_classes 10 \
    --n_samplings ${N_SAMPLINGS} --batch_size ${BATCH_SIZE} \
    --debug > _logs/${EXEP_NAME}.out
