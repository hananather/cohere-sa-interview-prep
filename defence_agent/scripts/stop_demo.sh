#!/usr/bin/env bash
set -euo pipefail

PID_FILE="defence_agent/data/demo-pids.txt"
if [[ ! -f "$PID_FILE" ]]; then
  echo "No demo PID file found."
  exit 0
fi

while read -r pid; do
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
  fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "Stopped Defence Agent demo processes."
