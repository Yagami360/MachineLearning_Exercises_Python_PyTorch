#!/bin/sh
#source activate pytorch11_py36
#nohup sh train.sh > _logs/train.out &
#nohup tensorboard --logdir tensorboard --port 6006 &
set -eu

N_EPOCHES=50
BATCH_SIZE=64
BATCH_SIZE_TEST=256
N_DISPLAY_STEP=5
N_DISPLAY_TEST_STEP=100
N_SAVE_STEP=10000

NETWORK_D_TYPE=vanilla
#NETWORK_D_TYPE=PatchGAN

OPTIMIZER=RMSprop

#-------------------
# RSGAN
#-------------------
mkdir -p ${PWD}/_logs
EXEP_NAME=WGAN_train_D_${NETWORK_D_TYPE}_Opt_${OPTIMIZER}_Epoch${N_EPOCHES}_191228
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
    --dataset mnist --image_size 64 \
    --n_test 10000 \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --batch_size_test ${BATCH_SIZE_TEST} \
    --optimizer ${OPTIMIZER} --lr_G 0.00005 --lr_D 0.00005 --beta1 0.5 --beta2 0.999 \
    --n_display_step ${N_DISPLAY_STEP} --n_display_test_step ${N_DISPLAY_TEST_STEP} \
    --n_critic 5 --w_clamp_upper 0.01 --w_clamp_lower -0.01 \
    --networkD_type ${NETWORK_D_TYPE} \
    --debug > _logs/${EXEP_NAME}.out

sudo poweroff
