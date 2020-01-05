#!/bin/sh
#source activate pytorch11_py36
set -eu

N_SAMPLINGS=100
BATCH_SIZE=64

NETWORK_G_TYPE=vanilla
#NETWORK_D_TYPE=PatchGAN
NETWORK_D_TYPE=vanilla

#-------------------
# RSGAN
#-------------------
EXEP_NAME=DCGAN_test_morphing_G_${NETWORK_G_TYPE}_D_${NETWORK_D_TYPE}_Sampling${N_SAMPLINGS}_191227
RESULTS_DIR=results_test
if [ -d "${RESULTS_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${RESULTS_DIR}/${EXEP_NAME}
fi

python test_morphing.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --results_dir ${RESULTS_DIR} \
    --load_checkpoints_dir checkpoints/DCGAN_train_G_vanilla_D_vanilla_Epoch100_191227 \
    --dataset mnist --image_size 64 \
    --n_samplings ${N_SAMPLINGS} --batch_size ${BATCH_SIZE} \
    --networkG_type ${NETWORK_G_TYPE} \
    --networkD_type ${NETWORK_D_TYPE} \
    --fps 30 --codec gif \
    --debug
