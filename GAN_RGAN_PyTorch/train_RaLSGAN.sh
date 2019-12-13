#!/bin/sh
#source activate pytorch11_py36
#nohup sh train.sh > _logs/train.out &
#nohup tensorboard --logdir tensorboard --port 6006 &
set -e

BATCH_SIZE=256
BATCH_SIZE_TEST=256
N_DISPLAY_STEP=`expr ${BATCH_SIZE} \* 5`
N_DISPLAY_TEST_STEP=`expr ${BATCH_SIZE} \* 50`
N_SAVE_STEP=`expr ${BATCH_SIZE} \* 5000`

#-------------------
# RSGAN
#-------------------
EXEP_NAME=RaLSGAN_train
TENSOR_BOARD_DIR=../tensorboard
if [ -d "${TENSOR_BOARD_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${TENSOR_BOARD_DIR}/${EXEP_NAME}
fi

if [ -d "${TENSOR_BOARD_DIR}/${EXEP_NAME}_test" ] ; then
    rm -r ${TENSOR_BOARD_DIR}/${EXEP_NAME}_test
fi

python train.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir ../dataset \
    --tensorboard_dir ${TENSOR_BOARD_DIR} \
    --save_checkpoints_dir checkpoints --n_save_step ${N_SAVE_STEP} \
    --dataset mnist --image_size 64 --n_channels 1 \
    --n_epoches 10 --batch_size ${BATCH_SIZE} --batch_size_test ${BATCH_SIZE_TEST} \
    --lr 0.0001 --beta1 0.5 --beta2 0.999 \
    --n_test 1000 \
    --n_display_step ${N_DISPLAY_STEP} --n_display_test_step ${N_DISPLAY_TEST_STEP} \
    --gan_type RaLSGAN \
    --debug

#sudo poweroff
