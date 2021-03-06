#!/bin/sh
set -e

# for docker
IMAGE_NAME=ml_exercises_pytorch_image
CONTAINER_NAME=ml_exercises_pytorch_container
HOST_DIR=../${PWD}
CONTAINER_DIR=/MachineLearning_Exercises_Python_PyTorch

if [ ! "$(docker image ls -q ${IMAGE_NAME})" ]; then
    docker build ./ -t ${IMAGE_NAME}
fi
if [ ! "$(docker ps -aqf "name=${CONTAINER_NAME}")" ]; then
    docker run -it -d -v ${HOST_DIR}:${CONTAINER_DIR} --name ${CONTAINER_NAME} ${IMAGE_NAME} /bin/bash
fi

# train parmaterters
N_EPOCHES=100
BATCH_SIZE=64
BATCH_SIZE_TEST=256
N_DISPLAY_STEP=50
N_DISPLAY_TEST_STEP=100
N_SAVE_STEP=10000

NETWORK_G_TYPE=vanilla
NETWORK_D_TYPE=vanilla

#-------------------
# DCGAN
#-------------------
mkdir -p ${PWD}/_logs
EXEP_NAME=DCGAN_train_G_${NETWORK_G_TYPE}_D_${NETWORK_D_TYPE}_Epoch${N_EPOCHES}_200119_debug
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

# tarin DCGAN
docker start ${CONTAINER_NAME}
docker exec -it ${CONTAINER_NAME} /bin/bash -c "cd GAN_DCGAN_PyTorch && ls && \
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
        --lr 0.0001 --beta1 0.5 --beta2 0.999 \
        --n_display_step ${N_DISPLAY_STEP} --n_display_test_step ${N_DISPLAY_TEST_STEP} \
        --networkG_type ${NETWORK_G_TYPE} \
        --networkD_type ${NETWORK_D_TYPE} \
        --debug"

#     > _logs/${EXEP_NAME}.out
