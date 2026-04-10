#!/bin/bash
cd "$(dirname "$(dirname "$0")")" || exit 1

usage() {
  cat <<'EOF'
usage: bash scripts/launch_quick_auditory_learning.sh [--dev]
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

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dev)
      : "${QUICK_AUDITORY_LEARNING_DATA_DIR:=/workspace/_data/quick_auditory_learning}"
      : "${QUICK_AUDITORY_LEARNING_CACHE_DIR:=/workspace/_cache/quick-auditory-learning}"
      : "${QUICK_AUDITORY_LEARNING_JSONL_PATH:=/workspace/_data/quick_auditory_learning/arxiv.jsonl}"
      : "${QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME:=text-embedding-3-large}"
      : "${QUICK_AUDITORY_LEARNING_VOICEVOX_URL:=http://voicevox:50021}"
      export QUICK_AUDITORY_LEARNING_DATA_DIR
      export QUICK_AUDITORY_LEARNING_CACHE_DIR
      export QUICK_AUDITORY_LEARNING_JSONL_PATH
      export QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME
      export QUICK_AUDITORY_LEARNING_VOICEVOX_URL
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

export QUICK_AUDITORY_LEARNING_DATA_DIR_HOST="${QUICK_AUDITORY_LEARNING_DATA_DIR_HOST:-$(pwd)/_data/quick_auditory_learning}"
export QUICK_AUDITORY_LEARNING_CACHE_DIR_HOST="${QUICK_AUDITORY_LEARNING_CACHE_DIR_HOST:-$(pwd)/_cache/quick-auditory-learning}"
export QUICK_AUDITORY_LEARNING_LOG_DIR_HOST="${QUICK_AUDITORY_LEARNING_LOG_DIR_HOST:-$(pwd)/_tmp/quick_auditory_learning/logs}"

jsonl_source_path="${QUICK_AUDITORY_LEARNING_JSONL_PATH:-$(pwd)/_data/quick_auditory_learning/arxiv.jsonl}"
case "$jsonl_source_path" in
  /*) jsonl_source_path_host="$jsonl_source_path" ;;
  *) jsonl_source_path_host="$(pwd)/$jsonl_source_path" ;;
esac
export QUICK_AUDITORY_LEARNING_JSONL_DIR_HOST="${QUICK_AUDITORY_LEARNING_JSONL_DIR_HOST:-${jsonl_source_path_host%/*}}"
export QUICK_AUDITORY_LEARNING_JSONL_PATH="$jsonl_source_path_host"
mkdir -p "$QUICK_AUDITORY_LEARNING_JSONL_DIR_HOST"
if [ -e "$jsonl_source_path_host" ]; then
  :
else
  echo "jsonl source does not exist yet: $jsonl_source_path_host" >&2
fi

mkdir -p "$QUICK_AUDITORY_LEARNING_DATA_DIR_HOST/postgres"
mkdir -p "$QUICK_AUDITORY_LEARNING_CACHE_DIR_HOST/backend/uv_venv"
mkdir -p "$QUICK_AUDITORY_LEARNING_CACHE_DIR_HOST/backend/uv_cache"
mkdir -p "$QUICK_AUDITORY_LEARNING_CACHE_DIR_HOST/frontend/node_modules"
mkdir -p "$QUICK_AUDITORY_LEARNING_CACHE_DIR_HOST/frontend/npm-cache"
mkdir -p "$QUICK_AUDITORY_LEARNING_LOG_DIR_HOST"

docker compose "${env_file_args[@]}" -f quick-auditory-learning/docker-compose.yml up -d --build

echo "quick-auditory-learning is starting."
echo "backend:  http://${QUICK_AUDITORY_LEARNING_HOST:-localhost}:${QUICK_AUDITORY_LEARNING_BACKEND_PORT:-8000}"
echo "frontend: http://${QUICK_AUDITORY_LEARNING_HOST:-localhost}:${QUICK_AUDITORY_LEARNING_FRONTEND_PORT:-5173}"
echo "press Ctrl-C to stop log following; use down_quick_auditory_learning.sh to stop containers."

docker compose "${env_file_args[@]}" -f quick-auditory-learning/docker-compose.yml logs -f backend frontend
