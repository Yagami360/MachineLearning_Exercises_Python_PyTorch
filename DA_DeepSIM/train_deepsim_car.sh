#!/bin/sh
#conda activate pytorch11_py36
set -eu
mkdir -p _logs
N_WORKERS=4

DATASET_TYPE="deepsim_car"
DATASET_DIR="dataset/deepsim_dataset"

#----------------------
# model
#----------------------
N_EPOCHES=2000
BATCH_SIZE=1
#IMAGE_HIGHT=512
#IMAGE_WIDTH=1024
IMAGE_HIGHT=320
IMAGE_WIDTH=640

NET_G_TYPE=pix2pixhd
#NET_D_TYPE=patch_gan
NET_D_TYPE=multi_scale

#DATA_AUGUMENT_TYPE="none"
#DATA_AUGUMENT_TYPE="affine"
DATA_AUGUMENT_TYPE="affine_tps"
#DATA_AUGUMENT_TYPE="full"

EXPER_NAME=debug
#EXPER_NAME=deepsim_car_netG-${NET_G_TYPE}_netD-${NET_D_TYPE}_da-${DATA_AUGUMENT_TYPE}_b${BATCH_SIZE}_ep${N_EPOCHES}
rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid

N_DISPLAY_STEP=50
N_DISPLAY_VALID_STEP=50

python train.py \
    --exper_name ${EXPER_NAME} \
    --dataset_dir ${DATASET_DIR} --dataset_type ${DATASET_TYPE} \
    --n_epoches ${N_EPOCHES} \
    --image_height ${IMAGE_HIGHT} --image_width ${IMAGE_WIDTH} \
    --batch_size ${BATCH_SIZE} \
    --n_diaplay_step ${N_DISPLAY_STEP} --n_display_valid_step ${N_DISPLAY_VALID_STEP} \
    --net_G_type ${NET_G_TYPE} --net_D_type ${NET_D_TYPE} \
    --data_augument_type ${DATA_AUGUMENT_TYPE} \
    --n_workers ${N_WORKERS} \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
