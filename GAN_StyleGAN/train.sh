#!/bin/sh
#conda activate pytorch11_py36
#nohup sh train.sh poweroff > _logs/stylegan_progress8_b4_ep100_201121.out &
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
N_EPOCHES=10
BATCH_SIZE=4
IMAGE_SIZE_INIT=4
IMAGE_SIZE_FINAL=1024

EXPER_NAME=debug
#EXPER_NAME=stylegan_${IMAGE_SIZE_FINAL}_b${BATCH_SIZE}_ep${N_EPOCHES}_201121

rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid
if [ ${EXPER_NAME} = "debug" ] ; then
    N_DISPLAY_STEP=50
    N_DISPLAY_VALID_STEP=50
else
    N_DISPLAY_STEP=100
    N_DISPLAY_VALID_STEP=500
fi

python train.py \
    --exper_name ${EXPER_NAME} \
    --n_epoches ${N_EPOCHES} \
    --image_size_init ${IMAGE_SIZE_INIT} --image_size_final ${IMAGE_SIZE_FINAL} --batch_size ${BATCH_SIZE} \
    --n_diaplay_step ${N_DISPLAY_STEP} --n_display_valid_step ${N_DISPLAY_VALID_STEP} \
    --diaplay_scores \
    --debug

#    --use_amp \

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
