#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" scripts/run_quasi_real_guarded_ppo_iterative_miniloop_stability.py "$@"
