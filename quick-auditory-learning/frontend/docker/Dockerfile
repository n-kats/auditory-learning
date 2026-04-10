FROM node:22-slim

ENV NODE_ENV=development

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/quick-auditory-learning/frontend

CMD ["bash", "-lc", "npm install --cache /workspace/_cache/quick-auditory-learning/frontend/npm-cache --prefer-offline --no-audit --no-fund && npm run dev -- --host 0.0.0.0 --port 5173"]
