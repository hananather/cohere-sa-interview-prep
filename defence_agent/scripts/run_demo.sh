#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/defence_agent/backend:${PYTHONPATH:-}"
export COHERE_CHAT_MODEL="${COHERE_CHAT_MODEL:-command-a-03-2025}"
export COHERE_EMBED_MODEL="${COHERE_EMBED_MODEL:-embed-v4.0}"
export COHERE_RERANK_MODEL="${COHERE_RERANK_MODEL:-rerank-v4.0-pro}"
export GENERATED_CORPUS_DIR="${GENERATED_CORPUS_DIR:-./defence_agent/data/synthetic_corpus}"

if ! python - <<'PY' >/dev/null 2>&1
import fastapi
import streamlit
import defence_agent
PY
then
  python -m pip install --no-build-isolation -e ".[dev]"
fi

mkdir -p defence_agent/data
python -m uvicorn defence_agent.api:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
python -m streamlit run defence_agent/frontend/app.py --server.port=8501 --server.address=127.0.0.1 &
FRONTEND_PID=$!

cat > defence_agent/data/demo-pids.txt <<EOF
$BACKEND_PID
$FRONTEND_PID
EOF

echo "Backend:   http://127.0.0.1:8000"
echo "Frontend:  http://127.0.0.1:8501"
echo "Stop with: defence_agent/scripts/stop_demo.sh"
wait
