#!/bin/sh
#source activate pytorch11_py36
#nohup sh train.sh > _logs/train.out &
#nohup tensorboard --logdir tensorboard --port 6006 &
set -eu

N_EPOCHES=10
BATCH_SIZE=32
BATCH_SIZE_TEST=256
N_DISPLAY_STEP=10
N_DISPLAY_TEST_STEP=100
N_SAVE_STEP=10000

#-------------------
# ResNet-18
#-------------------
mkdir -p ${PWD}/_logs
EXEP_NAME=ResNet18_train_Epoch${N_EPOCHES}_191230
TENSOR_BOARD_DIR=../tensorboard
if [ -d "${TENSOR_BOARD_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${TENSOR_BOARD_DIR}/${EXEP_NAME}
fi
if [ -d "${TENSOR_BOARD_DIR}/${EXEP_NAME}_test" ] ; then
    rm -r ${TENSOR_BOARD_DIR}/${EXEP_NAME}_test
fi

RESULTS_DIR=results
if [ -d "${RESULTS_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${RESULTS_DIR}/${EXEP_NAME}
fi

python train.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir ../dataset \
    --results_dir ${RESULTS_DIR} \
    --tensorboard_dir ${TENSOR_BOARD_DIR} \
    --save_checkpoints_dir checkpoints --n_save_step ${N_SAVE_STEP} \
    --dataset mnist --image_size 224 \
    --n_test 10000 \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --batch_size_test ${BATCH_SIZE_TEST} \
    --n_display_step ${N_DISPLAY_STEP} --n_display_test_step ${N_DISPLAY_TEST_STEP} \
    --debug > _logs/${EXEP_NAME}.out
