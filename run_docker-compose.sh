#!/bin/sh
set -e
CONTAINER_NAME=ml_exercises_pytorch_container

docker-compose stop
docker-compose up -d
docker-compose logs
docker exec -it ${CONTAINER_NAME} /bin/bash
