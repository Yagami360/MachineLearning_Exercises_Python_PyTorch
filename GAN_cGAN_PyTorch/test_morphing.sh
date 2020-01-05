#!/bin/sh
#source activate pytorch11_py36
set -eu

N_SAMPLINGS=100
BATCH_SIZE=64

NETWORK_D_TYPE=vanilla
#NETWORK_D_TYPE=PatchGAN

#-------------------
# RSGAN
#-------------------
EXEP_NAME=CGAN_test_morphing_D_${NETWORK_D_TYPE}_Sampling${N_SAMPLINGS}_191227
RESULTS_DIR=results_test
if [ -d "${RESULTS_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${RESULTS_DIR}/${EXEP_NAME}
fi


for ylabel in 0 1 2 3 4 5 6 7 8 9
do
    mkdir -p ${RESULTS_DIR}/${EXEP_NAME}/${ylabel}
    python test_morphing.py \
        --device gpu \
        --exper_name ${EXEP_NAME}/${ylabel} \
        --results_dir ${RESULTS_DIR} \
        --load_checkpoints_dir checkpoints/CGAN_train_gantype_vanilla_D_vanilla_Epoch100_191227 \
        --dataset mnist --image_size 64 \
        --n_samplings ${N_SAMPLINGS} --batch_size ${BATCH_SIZE} \
        --networkD_type ${NETWORK_D_TYPE} \
        --y_label ${ylabel} \
        --fps 30 --codec gif \
        --debug
done
