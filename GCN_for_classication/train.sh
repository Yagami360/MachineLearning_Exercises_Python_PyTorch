#!/bin/sh
#conda activate pytorch11_py36
set -eu
mkdir -p _logs

#----------------------_logs.
# model
#----------------------
N_EPOCHES=200
EXPER_NAME=debug
rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid
#rm -rf tensorboard/${EXPER_NAME}_test

python train.py \
    --exper_name ${EXPER_NAME} \
    --n_epoches ${N_EPOCHES} \
    --n_diaplay_step 5 --n_display_valid_step 5 \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi