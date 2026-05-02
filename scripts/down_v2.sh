#!/bin/bash
cd "$(dirname "$(dirname "$0")")" || exit 1

env_file_args=()
if [ -r .env ]; then
  env_file_args=(--env-file .env)
fi

docker compose "${env_file_args[@]}" -f v2/docker-compose.yml down
