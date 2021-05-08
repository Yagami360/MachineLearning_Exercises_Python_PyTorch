#!/bin/sh
#source activate pytorch11_py36
mkdir -p tensorboard
nohup tensorboard --logdir tensorboard --port 6006 --bind_all &