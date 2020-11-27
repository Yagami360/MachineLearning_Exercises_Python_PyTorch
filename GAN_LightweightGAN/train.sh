#!/bin/sh
#conda activate pytorch11_py36
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
N_EPOCHES=5
BATCH_SIZE=4

IMAGE_SIZE=256
#IMAGE_SIZE=512
#IMAGE_SIZE=1024

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
    --n_epoches ${N_EPOCHES} \
    --image_size ${IMAGE_SIZE} --batch_size ${BATCH_SIZE} \
    --n_diaplay_step ${N_DISPLAY_STEP} --n_display_valid_step ${N_DISPLAY_VALID_STEP} \
    --diaplay_scores \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
