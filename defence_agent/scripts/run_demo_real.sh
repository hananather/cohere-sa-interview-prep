#!/usr/bin/env bash
set -euo pipefail

export USE_MOCK_COHERE=false
exec "$(dirname "${BASH_SOURCE[0]}")/run_demo.sh"
