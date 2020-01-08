#!/bin/sh
#source activate pytorch11_py36
set -eu

N_SAMPLINGS=100
BATCH_SIZE=64

#-------------------
# Pix2Pix
#-------------------
EXEP_NAME=Pix2Pix_test_Sampling${N_SAMPLINGS}_191231
RESULTS_DIR=results_test
if [ -d "${RESULTS_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${RESULTS_DIR}/${EXEP_NAME}
fi

python test.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir ../dataset/maps \
    --results_dir ${RESULTS_DIR} \
    --load_checkpoints_dir checkpoints/UNet_train_Epoch100_191230 \
    --image_size 64 \
    --n_samplings ${N_SAMPLINGS} --batch_size ${BATCH_SIZE} \
    --debug
