#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-30}"
BATCH_SIZE="${2:-128}"
MAX_WINDOWS="${3:-0}"
MAX_FOLDS="${4:-0}"
PYTHON_BIN="./env/Scripts/python.exe"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

ARGS=(
  src/fold_local_encoder_walkforward.py
  --universe crypto20
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --max-windows "${MAX_WINDOWS}"
  --output-prefix crypto20_fold_local_
)

if [[ "${MAX_FOLDS}" -gt 0 ]]; then
  ARGS+=(--max-folds "${MAX_FOLDS}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"
