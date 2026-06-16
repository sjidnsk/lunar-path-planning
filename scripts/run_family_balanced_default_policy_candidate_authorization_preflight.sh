#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

cd "$ROOT"
"$PYTHON_BIN" scripts/run_family_balanced_default_policy_candidate_authorization_preflight.py "$@"
