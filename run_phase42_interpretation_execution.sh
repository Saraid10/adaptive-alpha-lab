#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

ARGS=("src/phase42_interpretation_execution.py" "--universe" "crypto20")
if [[ "${1:-}" == "--skip-feature-diagnostics" ]]; then
  ARGS+=("--skip-feature-diagnostics")
fi

"${PYTHON}" "${ARGS[@]}"
