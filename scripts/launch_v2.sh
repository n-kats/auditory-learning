#!/bin/bash
cd "$(dirname "$(dirname "$0")")" || exit 1

usage() {
  cat <<'EOF'
usage: bash scripts/launch_v2.sh
EOF
}

env_file_args=()
if [ -r .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
  env_file_args=(--env-file .env)
elif [ -f .env ]; then
  echo ".env exists but is not readable; continuing without it" >&2
fi

export AUDITORY_LEARNING_V2_HOST="${AUDITORY_LEARNING_V2_HOST:-localhost}"

mkdir -p "${AUDITORY_LEARNING_V2_DATA_DIR_HOST:-$(pwd)/_data/v2_auditory_learning}"
mkdir -p "${AUDITORY_LEARNING_V2_CACHE_DIR_HOST:-$(pwd)/_cache/v2-auditory-learning}/backend/uv_venv"
mkdir -p "${AUDITORY_LEARNING_V2_CACHE_DIR_HOST:-$(pwd)/_cache/v2-auditory-learning}/backend/uv_cache"
mkdir -p "${AUDITORY_LEARNING_V2_CACHE_DIR_HOST:-$(pwd)/_cache/v2-auditory-learning}/frontend/node_modules"
mkdir -p "${AUDITORY_LEARNING_V2_CACHE_DIR_HOST:-$(pwd)/_cache/v2-auditory-learning}/frontend/npm-cache"

docker compose "${env_file_args[@]}" -f v2/docker-compose.yml up -d --build

echo "v2 auditory learning is starting."
echo "backend:  http://${AUDITORY_LEARNING_V2_HOST:-localhost}:${AUDITORY_LEARNING_V2_BACKEND_PORT:-8000}"
echo "frontend: http://${AUDITORY_LEARNING_V2_HOST:-localhost}:${AUDITORY_LEARNING_V2_FRONTEND_PORT:-5174}"
echo "press Ctrl-C to stop log following; use down_v2.sh to stop containers."

docker compose "${env_file_args[@]}" -f v2/docker-compose.yml logs -f backend frontend
