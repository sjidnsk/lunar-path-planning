# Stage22.1 Theta-Aware Candidate Viewpoint Generation

## Goal

Stage22.1 implements the theta-aware candidate viewpoint contract required by
Stage22.0. It expands point candidates `(x,y)` into viewpoint actions
`(x,y,theta_deg)` so coverage audit, candidate hashes, Stage21.1 collector rows,
and Stage21.3 batch validation can express theta-aware actions.

## Scope

- Keep default A* unchanged; routes still plan to `(x,y)`.
- Keep the current network architecture and checkpoint loading unchanged.
- Preserve `candidate["cell"]` as `[x,y]`; carry theta in metadata fields.
- Expand each base candidate into 8 theta bins by default.
- Emit viewpoint-level candidate hashes, mask semantics, selected viewpoint
  metadata, and theta coverage audit fields.
- Add a Stage21.3 gate that rejects point-only batches when
  `require_theta_aware_viewpoint_contract=true`.

## Implemented Artifacts

- `scripts/xunce_theta_viewpoint_candidates.py`
- `scripts/run_xunce_stage22_1_theta_aware_candidate_viewpoint_generation.py`
- `configs/xunce_stage22_1_theta_aware_candidate_viewpoint_generation_v1.json`
- `tests/test_xunce_theta_viewpoint_candidates.py`
- `tests/test_xunce_stage22_1_theta_aware_candidate_viewpoint_generation.py`

## Current Evidence

Real output root:

`D:\CodexDownloads\lunar-path-planning\stage22_theta_aware_sensor_action_space\outputs\path_feedback_batch_xunce_stage22_1_theta_aware_candidate_viewpoint_generation_v1`

Summary:

- `status=passed`
- `next_required_change=run_stage22_2_theta_aware_coverage_reward_contract`
- `audited_transition_count=720`
- `expanded_viewpoint_row_count=177672`
- `hash_missing_theta_count=0`
- `mask_length_mismatch_count=0`
- `stage21_batch_theta_incompatible_after_expansion=false`

## Boundaries

Stage22.1 does not start PPO, publish checkpoints, replace default policy,
connect an executor, start canary traffic, modify the network architecture, or
modify default A*.

## Next Step

Stage22.2 should define the theta-aware coverage reward contract: reward and
PPO batches must consume viewpoint-level coverage gain rather than point-only
coverage gain.
