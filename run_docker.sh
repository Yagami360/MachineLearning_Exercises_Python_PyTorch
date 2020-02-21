#!/bin/sh
set -e

IMAGE_NAME=ml_exercises_pytorch_image
CONTAINER_NAME=ml_exercises_pytorch_container
HOST_DIR=${PWD}
CONTAINER_DIR=/home/user/share/MachineLearning_Exercises_Python_PyTorch
PORT=6007

if [ ! "$(docker image ls -q ${IMAGE_NAME})" ]; then
    docker build ./dockerfile -t ${IMAGE_NAME}
    docker rm $(docker ps -aq)
fi

if [ ! "$(docker ps -aqf "name=${CONTAINER_NAME}")" ]; then
    #docker run -it -v ${HOST_DIR}:${CONTAINER_DIR} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -u $(id -u $USER):$(id -g $USER) --name ${CONTAINER_NAME} ${IMAGE_NAME} --runtime nvidia /bin/bash
    docker run -it -v ${HOST_DIR}:${CONTAINER_DIR} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -u $(id -u $USER):$(id -g $USER) --name ${CONTAINER_NAME} ${IMAGE_NAME} /bin/bash
else
    docker start ${CONTAINER_NAME}
    #docker exec -it -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -u $(id -u $USER):$(id -g $USER) --runtime nvidia ${CONTAINER_NAME} /bin/bash
    docker exec -it -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -u $(id -u $USER):$(id -g $USER) ${CONTAINER_NAME} /bin/bash
fi
