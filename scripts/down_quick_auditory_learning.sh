#!/bin/bash
cd "$(dirname "$(dirname "$0")")" || exit 1

usage() {
  cat <<'EOF'
usage: bash scripts/down_quick_auditory_learning.sh
EOF
}

env_file_args=()
if [ -r .env ]; then
  env_file_args=(--env-file .env)
elif [ -f .env ]; then
  echo ".env exists but is not readable; continuing without it" >&2
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
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

docker compose "${env_file_args[@]}" -f quick-auditory-learning/docker-compose.yml down --remove-orphans

echo "quick-auditory-learning is stopped."
