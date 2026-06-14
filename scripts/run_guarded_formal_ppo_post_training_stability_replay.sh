#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
exec "$PYTHON_BIN" scripts/run_guarded_formal_ppo_post_training_stability_replay.py "$@"
