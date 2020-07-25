#!/bin/sh
#conda activate pytorch11_py36
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
#GEOMETRIC_MODEL=affine
GEOMETRIC_MODEL=tps
#GEOMETRIC_MODEL=hom

N_EPOCHES=5
BATCH_SIZE=4
IMAGE_HIGHT=240
IMAGE_WIDTH=240

EXPER_NAME=debug
#EXPER_NAME=${GEOMETRIC_MODEL}_image${IMAGE_HIGHT}_ep${N_EPOCHES}_b${BATCH_SIZE}_200725
rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid
if [ ${EXPER_NAME} = "debug" ] ; then
    N_DISPLAY_STEP=10
    N_DISPLAY_VALID_STEP=50
    VAL_RATE=0.001
else
    N_DISPLAY_STEP=100
    N_DISPLAY_VALID_STEP=500
    VAL_RATE=0.01
fi

python train.py \
    --exper_name ${EXPER_NAME} \
    --dataset_dir ${HOME}/ML_dataset/VOCdevkit/VOC2012/JPEGImages \
    --geometric_model ${GEOMETRIC_MODEL} \
    --n_epoches ${N_EPOCHES} \
    --image_height ${IMAGE_HIGHT} --image_width ${IMAGE_WIDTH} --batch_size ${BATCH_SIZE} \
    --n_diaplay_step ${N_DISPLAY_STEP} --n_display_valid_step ${N_DISPLAY_VALID_STEP} \
    --val_rate ${VAL_RATE} \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
