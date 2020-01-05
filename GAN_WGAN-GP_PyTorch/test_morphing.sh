#!/bin/sh
#source activate pytorch11_py36
set -eu

N_SAMPLINGS=100
BATCH_SIZE=64

#NETWORK_D_TYPE=vanilla
NETWORK_D_TYPE=NonBatchNorm
#NETWORK_D_TYPE=PatchGAN

#-------------------
# RSGAN
#-------------------
EXEP_NAME=WGANGP_test_morphing_D_${NETWORK_D_TYPE}_Sampling${N_SAMPLINGS}_191230
RESULTS_DIR=results_test
if [ -d "${RESULTS_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${RESULTS_DIR}/${EXEP_NAME}
fi

python test_morphing.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --results_dir ${RESULTS_DIR} \
    --load_checkpoints_dir checkpoints/WGANGP_train_D_NonBatchNorm_Epoch50_191230 \
    --dataset mnist --image_size 64 \
    --n_samplings ${N_SAMPLINGS} --batch_size ${BATCH_SIZE} \
    --networkD_type ${NETWORK_D_TYPE} \
    --fps 30 --codec gif \
    --debug

#    --load_checkpoints_dir checkpoints/WGANGP_train_D_NonBatchNorm_Epoch50_191230 \
#    --load_checkpoints_dir checkpoints/WGANGP_train_D_vanilla_Epoch50_191229 \
