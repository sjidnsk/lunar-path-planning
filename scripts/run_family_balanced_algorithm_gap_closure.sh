#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_family_balanced_algorithm_gap_closure.py" \
  --repo-root "${REPO_ROOT}" \
  "$@"
