#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}

cd "$REPO_ROOT"

"$PYTHON_BIN" "$SCRIPT_DIR/run_guarded_experimental_policy_staged_release_preflight.py" \
  --shadow-release-trial-root outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1 \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/guarded_experimental_policy_staged_release_preflight_v1.json \
  --output-root outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1
