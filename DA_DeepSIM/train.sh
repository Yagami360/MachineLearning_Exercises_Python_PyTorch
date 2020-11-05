#!/bin/sh
#conda activate pytorch11_py36
set -eu
mkdir -p _logs

#N_WORKERS=0
N_WORKERS=4

DATASET_DIR="dataset/zalando_dataset_n20"

#----------------------
# model
#----------------------
N_EPOCHES=200
BATCH_SIZE=4
IMAGE_HIGHT=128
IMAGE_WIDTH=128
NET_G_TYPE=pix2pixhd

#DATA_AUGUMENT_TYPE="none"
#DATA_AUGUMENT_TYPE="affine"
DATA_AUGUMENT_TYPE="affine_tps"
#DATA_AUGUMENT_TYPE="full"

EXPER_NAME=debug
#EXPER_NAME=${NET_G_TYPE}_da-${}_b${BATCH_SIZE}_ep${N_EPOCHES}
rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid

N_DISPLAY_STEP=10
N_DISPLAY_VALID_STEP=10

python train.py \
    --exper_name ${EXPER_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --n_epoches ${N_EPOCHES} \
    --image_height ${IMAGE_HIGHT} --image_width ${IMAGE_WIDTH} --batch_size ${BATCH_SIZE} \
    --n_diaplay_step ${N_DISPLAY_STEP} --n_display_valid_step ${N_DISPLAY_VALID_STEP} \
    --net_G_type ${NET_G_TYPE} \
    --data_augument_type ${DATA_AUGUMENT_TYPE} \
    --n_workers ${N_WORKERS} \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
