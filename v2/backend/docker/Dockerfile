FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/workspace/_cache/v2-auditory-learning/backend/uv_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl ca-certificates poppler-utils ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace/v2/backend
ENV PYTHONPATH=/workspace/v2/backend/src

CMD ["bash", "-lc", "uv sync && uv run uvicorn v2_auditory_learning.main:app --host 0.0.0.0 --port 8000"]
