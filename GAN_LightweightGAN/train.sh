#!/bin/sh
#conda activate pytorch11_py36
#nohup sh train.sh > _logs/light-weight-gan_size256_b4_ep1000_201128 &
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
N_EPOCHES=100
BATCH_SIZE=4

#IMAGE_SIZE=256
IMAGE_SIZE=512
#IMAGE_SIZE=1024

EXPER_NAME=debug
#EXPER_NAME=light-weight-gan_size${IMAGE_SIZE}_b${BATCH_SIZE}_ep${N_EPOCHES}_201128
#EXPER_NAME=light-weight-gan_l1-10_vgg-10_adv-1_size${IMAGE_SIZE}_b${BATCH_SIZE}_ep${N_EPOCHES}_201128
rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid
if [ ${EXPER_NAME} = "debug" ] ; then
    N_DISPLAY_STEP=10
    N_DISPLAY_VALID_STEP=50
else
    N_DISPLAY_STEP=100
    N_DISPLAY_VALID_STEP=100
fi

python train.py \
    --exper_name ${EXPER_NAME} \
    --n_epoches ${N_EPOCHES} \
    --image_size ${IMAGE_SIZE} --batch_size ${BATCH_SIZE} \
    --n_diaplay_step ${N_DISPLAY_STEP} --n_display_valid_step ${N_DISPLAY_VALID_STEP} \
    --gan_loss_type hinge --rec_loss_type lpips \
    --lambda_l1 0.0 --lambda_vgg 0.0 --lambda_adv 1.0 \
    --diaplay_scores \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
