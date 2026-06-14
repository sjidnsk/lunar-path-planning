#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" scripts/run_guarded_formal_ppo_post_training_stability_replay.py \
  --training-run-root outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1 \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --output-root outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1 \
  --config configs/guarded_formal_ppo_post_training_stability_replay_v1.json
