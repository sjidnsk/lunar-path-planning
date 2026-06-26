# Stage24.5A Repair Hybrid Path Inference Binding

## Goal

Repair the Stage24.5 blocker where Stage21.5 high-fidelity pre/post inference
rows had strong-state joins but were missing selected-candidate Hybrid A* pose
path-cost fields.

## Inputs

- Stage24.5 root:
  `D:\CodexDownloads\lunar-path-planning\stage24_hybrid_astar_pose_planner\outputs\path_feedback_batch_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke_v1`
- Stage24.5 route must be `repair_stage24_5_hybrid_path_inference_binding`.
- Stage24.5 base config:
  `configs/xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke_v1.json`

## Implementation

- High-fidelity trajectory evaluation computes Hybrid A* path cost when
  `hybrid_astar_pose_path_cost_enabled=true`.
- The evaluator reuses `scripts/xunce_hybrid_astar_candidate_path_cost.py`.
- Current pose is `current_cell` center plus the previous executed selected
  theta, with provenance
  `high_fidelity_current_cell_plus_previous_selected_theta/v1`.
- Candidate set hashes remain fixed before Hybrid A* enrichment so pre/post
  strong joins keep the same state/candidate identity semantics.
- Stage24.5A writes a repaired Stage24.5 config and reruns Stage24.5 under
  `s24_5_repaired`.

## Outputs

- `xunce-stage24-5a-summary.json`
- `xunce-stage24-5a-inference-binding-repair-audit.json`
- `xunce-stage24-5a-rerun-stage24-5-summary.json`
- `xunce-stage24-5a-next-stage-routing.json`
- `xunce-stage24-5a-report.md`
- `xunce-stage24-5a-manifest.json`

## Acceptance

- Repaired Stage24.5 has `hybrid_path_inference_required_field_missing_count=0`.
- Repaired Stage24.5 has `path_cost_source_mismatch_count=0`.
- Grid fallback, default-A* replacement, and Ackermann feasibility claim counts are all 0.
- Strong-state join count is greater than 0.
- Boundary fields remain false/0.

## Non-Goals

Stage24.5A does not tune PPO, run training, publish checkpoints, replace the
default policy, connect a real executor, start canary traffic, replace default
A*, introduce continuous theta, or claim performance.
