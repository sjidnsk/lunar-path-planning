#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/run_generated_sequential_long_horizon_teacher_skill_contract_alignment.py" "$@"
