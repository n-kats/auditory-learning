FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/workspace/_cache/quick-auditory-learning/backend/uv_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace/quick-auditory-learning/backend
ENV PYTHONPATH=/workspace/quick-auditory-learning/backend/src

CMD ["bash", "-lc", "uv sync && uv run uvicorn quick_auditory_learning.main:app --host 0.0.0.0 --port 8000"]
