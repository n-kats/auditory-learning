#!/bin/bash
cd "$(dirname "$(dirname "$0")")" || exit 1
docker build -t "auditory-learning" docker

docker run -it --rm \
  --net=host \
  -u "$(id -u):$(id -g)" \
  -v "$(pwd):/workspace" \
  --env-file .env \
  auditory-learning bash scripts/serve.sh
