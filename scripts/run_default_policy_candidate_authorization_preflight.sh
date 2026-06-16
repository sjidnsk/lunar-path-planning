#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
"$PYTHON_BIN" scripts/run_default_policy_candidate_authorization_preflight.py "$@"
