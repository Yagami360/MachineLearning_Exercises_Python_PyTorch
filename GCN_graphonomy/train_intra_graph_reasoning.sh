#!/bin/sh
#conda activate pytorch11_py36
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
N_EPOCHES=5
BATCH_SIZE=4
IMAGE_HIGHT=64
IMAGE_WIDTH=64
N_CLASSES=20
N_NODE_FEATURES=32

EXPER_NAME=debug
#EXPER_NAME=intra_graph_reasoning_cihp_200719

rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid

python train_intra_graph_reasoning.py \
    --exper_name ${EXPER_NAME} \
    --n_epoches ${N_EPOCHES} \
    --image_height ${IMAGE_HIGHT} --image_width ${IMAGE_WIDTH} --n_classes ${N_CLASSES} --n_node_features ${N_NODE_FEATURES} \
    --batch_size ${BATCH_SIZE} \
    --n_diaplay_step 100 --n_display_valid_step 500 \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi