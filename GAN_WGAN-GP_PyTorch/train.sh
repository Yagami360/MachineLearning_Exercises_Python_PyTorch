#!/bin/sh
set -eu

EXEP_NAME=WGAN-GP_train
TENSOR_BOARD_DIR=tensorboard

BATCH_SIZE=512
BATCH_SIZE_TEST=512
N_DISPLAY_STEP=`expr ${BATCH_SIZE} \* 50`
N_DISPLAY_TEST_STEP=`expr ${BATCH_SIZE} \* 500`

if [ -d "${TENSOR_BOARD_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${TENSOR_BOARD_DIR}/${EXEP_NAME}
fi

if [ -d "${TENSOR_BOARD_DIR}/${EXEP_NAME}_test" ] ; then
    rm -r ${TENSOR_BOARD_DIR}/${EXEP_NAME}_test
fi

python train.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir dataset \
    --tensorboard_dir ${TENSOR_BOARD_DIR}
    --dataset mnist --image_size 64 --n_channels 1 \
    --n_test 500 \
    --n_epoches 10 --batch_size ${BATCH_SIZE} --batch_size_test ${BATCH_SIZE_TEST} \
    --n_critic 5 --w_clamp_upper 0.01 --w_clamp_lower -0.01 \
    --n_display_step ${N_DISPLAY_STEP} --n_display_test_step ${N_DISPLAY_TEST_STEP} \
    --debug
