#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_checkpoint_publication_sandbox_install_dry_run_preflight.py" \
  --repo-root "${REPO_ROOT}" \
  "$@"
