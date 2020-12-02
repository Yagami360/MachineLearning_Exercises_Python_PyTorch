#!/bin/sh
#conda activate pytorch11_py36
#nohup sh train.sh > _logs/train_201111_1.out &
#nohup sh train.sh > _logs/train_201111_2.out &
#nohup sh train.sh poweroff > _logs/train_201111_3.out &
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
#N_EPOCHES=10
N_EPOCHES=2000
#N_EPOCHES=5000

BATCH_SIZE=1
IMAGE_HIGHT=186
IMAGE_WIDTH=248

NET_G_TYPE="patch_gan"
NET_D_TYPE="singan_patch_gan"

LAMBDA_L1=10.0
LAMBDA_VGG=10.0
LAMBDA_ADV=1.0
LAMBDA_REC=10.0

EXPER_NAME=debug
#EXPER_NAME=netG-${NET_G_TYPE}_netD-${NET_D_TYPE}_l1-${LAMBDA_L1}_vgg-${LAMBDA_VGG}_adv-${LAMBDA_ADV}_rec-${LAMBDA_REC}__h${IMAGE_HIGHT}_w${IMAGE_WIDTH}_ep${N_EPOCHES}_201111
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
    --image_height ${IMAGE_HIGHT} --image_width ${IMAGE_WIDTH} --batch_size ${BATCH_SIZE} \
    --net_G_type ${NET_G_TYPE} --net_D_type ${NET_D_TYPE} \
    --n_diaplay_step ${N_DISPLAY_STEP} --n_display_valid_step ${N_DISPLAY_VALID_STEP} \
    --lambda_l1 ${LAMBDA_L1} --lambda_vgg ${LAMBDA_VGG} --lambda_adv ${LAMBDA_ADV} --lambda_rec ${LAMBDA_REC} \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
