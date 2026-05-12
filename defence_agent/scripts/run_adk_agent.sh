#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export DEFTECH_ADK_PERSONA="${DEFTECH_ADK_PERSONA:-clearance_unclassified}"
export DEFTECH_ADK_MODEL="${DEFTECH_ADK_MODEL:-cohere/command-a-03-2025}"

python defence_agent/scripts/build_chroma_index.py

adk web "$ROOT_DIR" \
  --port "${ADK_PORT:-8502}" \
  --session_service_uri memory:// \
  --artifact_service_uri memory://
