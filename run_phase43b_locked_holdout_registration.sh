#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/phase43b_locked_holdout_registration_v1.json}"
python src/phase43b_locked_holdout_registration.py --config "$CONFIG"
