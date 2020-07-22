#!/bin/sh
#conda activate pytorch11_py36
#nohup sh train.sh poweroff > graphonomy_img64_b16_200721.out &
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
N_EPOCHES=5
BATCH_SIZE=4
IMAGE_HIGHT=64
IMAGE_WIDTH=64
N_NODE_FEATURES=32

EXPER_NAME=debug
#EXPER_NAME=graphonomy_img${IMAGE_HIGHT}_b${BATCH_SIZE}_200721

rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid

python train.py \
    --exper_name ${EXPER_NAME} \
    --n_epoches ${N_EPOCHES} \
    --image_height ${IMAGE_HIGHT} --image_width ${IMAGE_WIDTH} --n_node_features ${N_NODE_FEATURES} --n_output_channels 1 \
    --batch_size ${BATCH_SIZE} \
    --n_diaplay_step 10 --n_display_valid_step 100 \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi