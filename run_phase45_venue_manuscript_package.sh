#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
ARGS=("src/phase45_venue_manuscript_package.py")

if [[ "${1:-}" == "--dry-run" ]]; then
  ARGS+=("--dry-run")
fi

"${PYTHON}" "${ARGS[@]}"
