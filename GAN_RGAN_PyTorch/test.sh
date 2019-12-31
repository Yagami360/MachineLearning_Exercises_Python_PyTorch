#!/bin/sh
#source activate pytorch11_py36
set -eu

N_SAMPLINGS=100
BATCH_SIZE=64

#NETWORK_D_TYPE=PatchGAN
NETWORK_D_TYPE=vanilla

#-------------------
# RSGAN
#-------------------
EXEP_NAME=RGAN_test_D_${NETWORK_D_TYPE}_Sampling${N_SAMPLINGS}_191231
RESULTS_DIR=results_test
if [ -d "${RESULTS_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${RESULTS_DIR}/${EXEP_NAME}
fi

python test.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --results_dir ${RESULTS_DIR} \
    --load_checkpoints_dir checkpoints/RaLSGAN_train_D_vanilla_Epoch100_191227 \
    --dataset mnist --image_size 64 \
    --n_samplings ${N_SAMPLINGS} --batch_size ${BATCH_SIZE} \
    --networkD_type ${NETWORK_D_TYPE} \
    --debug

#    --load_checkpoints_dir checkpoints/RSGAN_train_D_vanilla_Epoch100_191227_1 \
#    --load_checkpoints_dir checkpoints/RaSGAN_train_D_vanilla_Epoch100_191229 \
#    --load_checkpoints_dir checkpoints/RaLSGAN_train_D_vanilla_Epoch100_191228 \