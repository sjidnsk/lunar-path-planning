# Quasi-Real Map Domain Gap Evaluation v1

## Summary

Current readiness has reached `policy_training_cuda_device_support_evaluated`.
Generated canary scenes, guarded PPO pilot, collector, one tiny PPO update, and
CUDA smoke are closed. The next stage evaluates the gap between generated
training scenes and quasi-real LOLA south-pole map slices before any real-map
training or larger PPO rollout.

## Interfaces

- `scripts/run_quasi_real_lola_data_prepare.py/.sh`
  - reads the LOLA data manifest,
  - downloads missing raw files into ignored `model-explorer/data/raw/...`,
  - validates file count, bytes, and SHA-256,
  - writes `quasi-real-lola-data-prepare-summary.json`.
- `scripts/run_quasi_real_map_path_feedback_bridge.py/.sh`
  - reads a quasi-real ROI matrix manifest,
  - writes `model-explorer-contract/v1` plus `path-planner-sidecar/v1`,
  - writes `quasi-real-map-slices.jsonl` and
    `quasi-real-map-path-feedback-manifest.json`.
- `scripts/run_quasi_real_map_domain_gap_evaluation.py/.sh`
  - compares quasi-real path-feedback with current generated evidence,
  - writes `quasi-real-map-domain-gap-summary.json`,
    `quasi-real-map-domain-gap-report.md`, and an exclusion report.
- `scripts/run_quasi_real_map_domain_gap_closure.sh`
  - runs prepare, bridge, path-feedback, and domain-gap evaluation.

## Acceptance

- LOLA raw product validation passes with no hash mismatch.
- Bridge covers at least four ROI groups and at least 12 slices.
- `context_id_missing_count=0` and `legacy_identity_fallback_count=0`.
- Path-feedback/domain-gap summary passes with no invalid action mask,
  fallback/open-grid, safety, contract, path/risk, or source-selection
  regression.
- Domain verdict is one of `acceptable_for_next_pilot`,
  `scenario_expansion_required`, or `planner_contract_gap`.
- Readiness may advance only to `quasi_real_map_domain_gap_evaluated`.

## Non-Goals

- No PPO optimizer update.
- No checkpoint publication or default-policy replacement.
- No network/action-space/default-A* change.
- No distance/path-risk/source-selection contract relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic treated as training release evidence.
- No policy performance claim.
