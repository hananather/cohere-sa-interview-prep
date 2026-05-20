#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export DEFTECH_ADK_PERSONA="${DEFTECH_ADK_PERSONA:-clearance_unclassified}"
export DEFTECH_ADK_MODEL="${DEFTECH_ADK_MODEL:-cohere/command-a-03-2025}"
export DEFTECH_ADK_CRITIC_MODEL="${DEFTECH_ADK_CRITIC_MODEL:-command-a-reasoning-08-2025}"
export DEFTECH_ADK_CRITIC_THRESHOLD="${DEFTECH_ADK_CRITIC_THRESHOLD:-0.8}"
export DEFTECH_ADK_CRITIC_MAX_CITATIONS="${DEFTECH_ADK_CRITIC_MAX_CITATIONS:-8}"
export DEFENCE_AGENT_PARSER_BACKEND="${DEFENCE_AGENT_PARSER_BACKEND:-local}"
export DEFENCE_AGENT_RETRIEVAL_MODE="${DEFENCE_AGENT_RETRIEVAL_MODE:-hybrid}"
export DEFENCE_AGENT_CHUNK_STRATEGY="${DEFENCE_AGENT_CHUNK_STRATEGY:-page}"

python defence_agent/scripts/build_chroma_index.py

adk web "$ROOT_DIR" \
  --port "${ADK_PORT:-8502}" \
  --session_service_uri memory:// \
  --artifact_service_uri memory://
