#!/bin/sh
set -eu

EXEP_NAME=WGAN-GP_train
TENSOR_BOARD_DIR=tensorboard

if [ -d "${TENSOR_BOARD_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${TENSOR_BOARD_DIR}/${EXEP_NAME}
fi

python train.py \
    --device cpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir dataset \
    --tensorboard_dir ${TENSOR_BOARD_DIR}
    --n_epoches 10 --batch_size 4 --batch_size_test 4 \
    --dataset mnist --image_size 64 --n_channels 1 \
    --n_critic 5 --w_clamp_upper 0.01 --w_clamp_lower -0.01 \
    --n_display_step 10 --n_display_test_step 10 \
    --debug
