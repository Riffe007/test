#!/usr/bin/env bash
# Thin wrapper around run_pipeline.py. Use from anywhere:
#   ./scripts/convert.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 run_pipeline.py "$@"
