#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_expanded_four_family_refined_coverage_driven_ppo_improvement.py" "$@"
