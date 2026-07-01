#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ARGS=()

if [[ "${1:-}" == "--dry-run" ]]; then
  ARGS+=("--dry-run")
fi

"${PYTHON_BIN}" src/phase44_paper_readiness_package.py "${ARGS[@]}"
