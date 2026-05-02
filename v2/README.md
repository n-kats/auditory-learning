# v2 auditory learning

v1 と同じく、公開 PDF の URL を入力して内容を解説し、VOICEVOX で音声再生する版です。

v2 は quick-auditory-learning に近い構成を目指して、`backend/` と `frontend/` を分けています。

## 構成

- `backend/`: FastAPI 本体
- `frontend/`: React + Vite の UI
- `docker-compose.yml`: backend, frontend, voicevox の起動定義

## 起動

```bash
bash scripts/launch_v2.sh
```

停止:

```bash
bash scripts/down_v2.sh
```

## 環境変数

- `OPENAI_API_KEY`
- `AUDITORY_LEARNING_V2_DATA_DIR`
- `AUDITORY_LEARNING_V2_PROMPT_PATH`
- `AUDITORY_LEARNING_V2_VOICEVOX_URL`
- `AUDITORY_LEARNING_V2_FRONTEND_URL`
- `AUDITORY_LEARNING_V2_BACKEND_PORT`
- `AUDITORY_LEARNING_V2_FRONTEND_PORT`
- `AUDITORY_LEARNING_V2_HOST`
- `AUDITORY_LEARNING_V2_DATA_DIR_HOST`
- `AUDITORY_LEARNING_V2_CACHE_DIR_HOST`
- `VITE_AUDITORY_LEARNING_V2_API_BASE_URL`

既定値は `v2/docker-compose.yml` と scripts 側にあります。
