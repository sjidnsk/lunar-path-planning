#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" scripts/run_return_aligned_guarded_multi_step_ppo_collector_expansion.py "$@"
