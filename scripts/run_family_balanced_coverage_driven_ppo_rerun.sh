#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$PYTHON_BIN" "$REPO_ROOT/scripts/run_family_balanced_coverage_driven_ppo_rerun.py" "$@"
