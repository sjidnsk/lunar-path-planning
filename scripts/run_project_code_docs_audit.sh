#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
OUTPUT_ROOT="${1:-outputs/project_code_docs_audit_v1}"

"${PYTHON_BIN}" scripts/run_project_code_docs_audit.py --output-root "${OUTPUT_ROOT}"
