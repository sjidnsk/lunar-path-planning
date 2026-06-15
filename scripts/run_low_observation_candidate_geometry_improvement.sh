#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

"${PYTHON}" "${REPO_ROOT}/scripts/run_low_observation_candidate_geometry_improvement.py" "$@"
