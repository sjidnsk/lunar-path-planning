# Xunce Stage 9 Full Network Static Contract Validation v1 Plan

## Goal

Stage 9 validates the static input/output contract of the research-only `xunce_full_network_v1` implemented in Stage 8. It verifies shape, mask, metadata, missing-indicator handling, finite outputs, and legacy/additive-observation compatibility before any ablation, stress test, training, or release work.

This stage does not train PPO, publish a checkpoint, register a production/default-policy architecture, replace default policy, connect an executor, or claim performance.

## Inputs

- `outputs/path_feedback_batch_xunce_full_network_v1/xunce-full-network-v1-summary.json`
- `scripts/xunce_full_network_common.py`
- Stage 4 topology feature fixture for realistic candidate/edge/memory shapes.

## Interface

Add:

- `configs/xunce_full_network_static_contract_validation_v1.json`
- `scripts/run_xunce_full_network_static_contract_validation.py`
- `scripts/run_xunce_full_network_static_contract_validation.sh`
- `tests/test_xunce_full_network_static_contract_validation.py`

Default output root:

`outputs/path_feedback_batch_xunce_full_network_static_contract_validation_v1/`

Artifacts:

- `xunce-full-network-static-contract-validation-summary.json`
- `xunce-full-network-static-contract-validation-manifest.json`
- `xunce-full-network-static-contract-cases.jsonl`
- `xunce-full-network-static-shape-audit.json`
- `xunce-full-network-static-mask-audit.json`
- `xunce-full-network-static-metadata-audit.json`
- `xunce-full-network-static-compatibility-audit.json`
- `xunce-full-network-static-boundary-audit.json`
- `xunce-full-network-static-rejection-report.json`
- `xunce-full-network-static-contract-validation-report.md`

## Cases

The runner must execute deterministic forward cases:

1. Full topology case: candidates + edges + memory + context + missing indicators.
2. Mask case: at least one invalid candidate stays masked with zero probability.
3. Missing-indicator case: missing indicators may be omitted and default to zeros.
4. Legacy/additive compatibility case: empty edge set and zero memory/context still produce finite masked logits.
5. Metadata case: architecture/component flags and feature counts match the constructed network.

## Decision Contract

Summary fields:

- `status`
- `reason_codes`
- `source_full_network_status`
- `architecture=xunce_full_network_v1`
- `contract_case_count`
- `shape_contract_passed`
- `mask_contract_passed`
- `metadata_contract_passed`
- `missing_indicator_contract_passed`
- `legacy_observation_compatibility_passed`
- `finite_output_contract_passed`
- `static_contract_validation_passed`
- `next_required_change`

Boundary fields must stay false:

- `publishes_checkpoint`
- `replaces_default_policy`
- `connects_real_executor`
- `starts_online_canary`
- `runs_new_ppo_update`
- `modifies_network`
- `modifies_action_space`
- `modifies_default_astar`
- `real_world_release_approved`
- `real_world_performance_claimed`

Passing next gate:

`full_network_ablation_experiments`

Failure next gate:

`fix_full_network_static_contract_validation`

If Stage 8 evidence is missing or not passed:

`fix_xunce_full_network_v1`

## Implementation Steps

1. Write failing tests for default pass, missing Stage 8, mask failure, metadata failure, and boundary violation.
2. Implement runner/config/shell and static case generation.
3. Write all artifacts and markdown report.
4. Update README, architecture report, Global 99 spec, and Xunce design spec.
5. Run Stage 9 tests, Xunce Stage 0-9 tests, Global 99 regression, default runner, docs grep, and `git diff --check`.
6. Commit and push Stage 9 independently.

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_full_network_static_contract_validation.py -q
PYTHON=$PY bash scripts/run_xunce_full_network_static_contract_validation.sh
jq '{status,reason_codes,static_contract_validation_passed,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_xunce_full_network_static_contract_validation_v1/xunce-full-network-static-contract-validation-summary.json
rg -n "Xunce Full Network Static Contract Validation v1|run_xunce_full_network_static_contract_validation|full_network_ablation_experiments" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs
git diff --check
```

## Non-goals

- No PPO update.
- No checkpoint write or publication.
- No production architecture registration.
- No default-policy replacement.
- No executor connection.
- No online canary.
- No performance or real-world claim.
