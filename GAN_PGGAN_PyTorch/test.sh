#!/bin/sh
#source activate pytorch11_py36
set -eu

N_SAMPLINGS=100
BATCH_SIZE=64

#-------------------
# RSGAN
#-------------------
EXEP_NAME=PGGAN_test_Sampling${N_SAMPLINGS}_191230
RESULTS_DIR=results_test
if [ -d "${RESULTS_DIR}/${EXEP_NAME}" ] ; then
    rm -r ${RESULTS_DIR}/${EXEP_NAME}
fi

python test.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --results_dir ${RESULTS_DIR} \
    --load_checkpoints_dir checkpoints/PGGAN_train_Epoch100_191230 \
    --dataset mnist --init_image_size 4 --final_image_size 32 \
    --n_samplings ${N_SAMPLINGS} --batch_size ${BATCH_SIZE} \
    --debug
