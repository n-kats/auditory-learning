# 2026-04-22 v2 create separate quick-like project

- [x] Decide v2 directory layout and startup commands.
- [x] Scaffold v2 backend from v1 PDF app.
- [x] Scaffold v2 frontend with quick-like UI.
- [x] Update docs/directory_structure.md and README links.
- [x] Add or update tests.
- [x] Verify build/test or at least compile for new project.

## 結果
- `v2/` を追加し、`backend/`, `frontend/`, `docker-compose.yml`, `README.md` を分離した。
- backend は `v2_auditory_learning` パッケージで v1 と同じ PDF 解説フローを維持しつつ、quick 風のプロジェクト構成にした。
- frontend は quick に寄せたカードベース UI と、`api`, `pageState`, `audioPreferences`, `objectUrlStore`, `useAudioPlayer` の小モジュール構成にした。
- `scripts/launch_v2.sh` と `scripts/down_v2.sh` を追加し、v2 を独立起動できるようにした。
- v2 の環境変数は `AUDITORY_LEARNING_V2_*` 系に統一し、frontend の API base URL は `VITE_AUDITORY_LEARNING_V2_API_BASE_URL` にした。
- 既存テストと build を確認した。
  - `PYTHONPATH=/workspace/v2/backend/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q /workspace/v2/backend/tests/test_pdf_utils.py`
  - `PYTHONPATH=/workspace/v2/backend/src python -m compileall /workspace/v2/backend/src/v2_auditory_learning /workspace/v2/backend/tests/test_pdf_utils.py`
  - `cd /workspace/v2/frontend && npm test`
  - `cd /workspace/v2/frontend && npm run build`
