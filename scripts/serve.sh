#!/bin/bash
cd "$(dirname "$(dirname "$0")")" || exit 1

HOST="${AUDITORY_LEARNING_HOST:-localhost}"
PORT="${AUDITORY_LEARNING_PORT:-8000}"

mkdir -p _tmp/uv-cache
export UV_CACHE_DIR="$(pwd)/_tmp/uv-cache"

py_stamp="$(sha256sum pyproject.toml uv.lock | sha256sum | cut -d' ' -f1)"
venv_stamp=".venv/.sync-stamp"

if [ ! -d .venv ]; then
  uv venv
fi

if [ ! -f "$venv_stamp" ] || [ "$(cat "$venv_stamp")" != "$py_stamp" ]; then
  uv sync --frozen
  mkdir -p .venv
  printf '%s\n' "$py_stamp" > "$venv_stamp"
fi

(
  cd frontend || exit 1
  current_install_stamp="$(sha256sum package.json package-lock.json | sha256sum | cut -d' ' -f1)"
  install_stamp_path="node_modules/.install-stamp"
  build_stamp_path="node_modules/.build-stamp"
  npm_cache_dir="../_tmp/npm-cache"

  if [ ! -f "$install_stamp_path" ] || [ "$(cat "$install_stamp_path")" != "$current_install_stamp" ]; then
    npm install --cache "$npm_cache_dir" --prefer-offline --no-audit --no-fund
    mkdir -p node_modules
    printf '%s\n' "$current_install_stamp" > "$install_stamp_path"
  fi

  current_build_stamp="$(find src public -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
  current_build_stamp="$(printf '%s\n%s\n' "$current_build_stamp" "$(sha256sum index.html vite.config.ts tsconfig.json tsconfig.app.json tsconfig.node.json package.json package-lock.json | sha256sum | cut -d' ' -f1)" | sha256sum | cut -d' ' -f1)"

  if [ ! -f "$build_stamp_path" ] || [ "$(cat "$build_stamp_path")" != "$current_build_stamp" ]; then
    npm run build
    mkdir -p node_modules
    printf '%s\n' "$current_build_stamp" > "$build_stamp_path"
  fi
)

uv run uvicorn auditory_learning.server:app --host "$HOST" --port "$PORT"
