#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python -m pip install -e ".[dev]"
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
