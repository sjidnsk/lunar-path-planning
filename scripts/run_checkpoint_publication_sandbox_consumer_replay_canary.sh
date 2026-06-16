#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
"$PYTHON_BIN" scripts/run_checkpoint_publication_sandbox_consumer_replay_canary.py "$@"
