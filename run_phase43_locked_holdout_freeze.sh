#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
"${PYTHON}" src/phase43_locked_holdout_freeze.py
