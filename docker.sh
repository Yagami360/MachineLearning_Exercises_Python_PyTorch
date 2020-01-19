IMAGE_NAME=ml_exercises_pytorch_image
CONTAINER_NAME=ml_exercises_container
HOST_DIR=${HOME}/MachineLearning_Exercises_Python_PyTorch
CONTAINER_DIR=/workspace/MachineLearning_Exercises_Python_PyTorch

if [ ! "$(docker image ls -q ${IMAGE_NAME})" ]; then
    docker build ./ -t ${IMAGE_NAME}
fi

docker run -it --rm -v ${HOST_DIR}:${CONTAINER_DIR} --name ${CONTAINER_NAME} ${IMAGE_NAME} /bin/bash
