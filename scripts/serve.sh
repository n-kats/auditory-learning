#!/bin/bash
cd "$(dirname "$(dirname "$0")")" || exit 1

HOST="${AUDITORY_LEARNING_HOST:-localhost}"
PORT="${AUDITORY_LEARNING_PORT:-8000}"

uv venv
uv sync

(
  cd frontend || exit 1
  npm install
  npm run build
)

uv run uvicorn auditory_learning.server:app --host "$HOST" --port "$PORT"
