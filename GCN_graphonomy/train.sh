#!/bin/sh
#conda activate pytorch11_py36
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
N_EPOCHES=200
BATCH_SIZE=2
EXPER_NAME=debug

rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid
#rm -rf tensorboard/${EXPER_NAME}_test

python train.py \
    --exper_name ${EXPER_NAME} \
    --n_epoches ${N_EPOCHES} \
    --batch_size ${BATCH_SIZE} \
    --n_diaplay_step 10 --n_display_valid_step 10 \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi