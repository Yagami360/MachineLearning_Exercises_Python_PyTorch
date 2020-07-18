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
EXPER_NAME=intra_graph_reasoning_cihp_200719

rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid
#rm -rf tensorboard/${EXPER_NAME}_test

python train_intra_graph_reasoning.py \
    --exper_name ${EXPER_NAME} \
    --n_epoches ${N_EPOCHES} \
    --lr 0.007 --batch_size ${BATCH_SIZE} \
    --n_diaplay_step 100 --n_display_valid_step 100 \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi