#!/bin/sh
#conda activate pytorch11_py36
set -eu
mkdir -p _logs

#----------------------
# model
#----------------------
GEOMETRIC_MODEL_1=affine
GEOMETRIC_MODEL_2=tps

IMAGE_HIGHT=240
IMAGE_WIDTH=240

EXPER_NAME=debug
EXPER_NAME=affine_tps_image240_ep20_b4_200726

LOAD_CHECKPOINTS_PATH_1=checkpoints/affine_image240_ep20_b4_200726/model_G_final.pth
LOAD_CHECKPOINTS_PATH_2=checkpoints/tps_image240_ep20_b4_200726/model_G_final.pth
rm -rf results/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_test
if [ ${EXPER_NAME} = "debug" ] ; then
    N_SAMPLING=5
else
    N_SAMPLING=100000
fi

python test.py \
    --exper_name ${EXPER_NAME} \
    --dataset_dir ${HOME}/ML_dataset/proposal-flow-willow/PF-dataset \
    --load_checkpoints_path_1 ${LOAD_CHECKPOINTS_PATH_1} --load_checkpoints_path_2 ${LOAD_CHECKPOINTS_PATH_2} \
    --geometric_model_1 ${GEOMETRIC_MODEL_1} --geometric_model_2 ${GEOMETRIC_MODEL_2} \
    --n_samplings ${N_SAMPLING} \
    --image_height ${IMAGE_HIGHT} --image_width ${IMAGE_WIDTH} --batch_size_test 1 \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
