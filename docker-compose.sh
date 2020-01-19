CONTAINER_NAME=ml_exercises_container

docker-compose up -d
docker exec -it ${CONTAINER_NAME} /bin/sh
