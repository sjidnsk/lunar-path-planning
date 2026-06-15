#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PYTHON}" "${REPO_ROOT}/scripts/run_safe_better_pair_expansion_across_families.py" "$@"
