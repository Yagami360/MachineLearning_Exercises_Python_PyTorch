#!/bin/sh
#conda activate pytorch11_py36
set -eu
mkdir -p _logs
DATASET_DIR="dataset/dog_vs_cat_dataset_n100"

#----------------------
# model
#----------------------
N_EPOCHES=100
BATCH_SIZE=4
#BATCH_SIZE=32
IMAGE_HIGHT=224
IMAGE_WIDTH=224
NET_G_TYPE=vit-b16
LOAD_CHECKPOINTS_PATH_PRETRAINED="checkpoints/imagenet21k/ViT-B_16.npz"

EXPER_NAME=debug
rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid
if [ ${EXPER_NAME} = "debug" ] ; then
    N_DISPLAY_STEP=10
    N_DISPLAY_VALID_STEP=50
else
    N_DISPLAY_STEP=100
    N_DISPLAY_VALID_STEP=500
fi

python train.py \
    --exper_name ${EXPER_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --n_epoches ${N_EPOCHES} \
    --image_height ${IMAGE_HIGHT} --image_width ${IMAGE_WIDTH} --batch_size ${BATCH_SIZE} \
    --netG_type ${NET_G_TYPE} \
    --load_checkpoints_path_pretrained ${LOAD_CHECKPOINTS_PATH_PRETRAINED} \
    --n_diaplay_step ${N_DISPLAY_STEP} --n_display_valid_step ${N_DISPLAY_VALID_STEP} \
    --data_augument \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
