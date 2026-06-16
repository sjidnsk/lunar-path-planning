# Xunce Stage 8 Full Network v1 Plan

## Goal

Stage 8 implements the first complete research-only Xunce network, `xunce_full_network_v1`. It is the successor to the small prototype and is allowed only because Stage 7 produced a clean `next_required_change=full_xunce_network_v1_design`.

This stage builds a complete forward-pass module and verifies that it can consume topology candidates, pairwise candidate edges, coverage memory, ROI/family/budget context, action masks, and missing indicators. It does not train PPO, register a production architecture, write a checkpoint, replace the default policy, connect an executor, or claim performance.

## Inputs

- `outputs/path_feedback_batch_xunce_architecture_contrast_evaluation_v1/xunce-architecture-contrast-evaluation-summary.json`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-candidates.jsonl`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-edges.jsonl`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-memory.json`
- `scripts/xunce_topology_graph_proto_common.py` for prototype comparison patterns.

## Interface

Add:

- `scripts/xunce_full_network_common.py`
- `scripts/run_xunce_full_network_v1.py`
- `scripts/run_xunce_full_network_v1.sh`
- `configs/xunce_full_network_v1.json`
- `tests/test_xunce_full_network_v1.py`

Default output root:

`outputs/path_feedback_batch_xunce_full_network_v1/`

Artifacts:

- `xunce-full-network-v1-summary.json`
- `xunce-full-network-v1-manifest.json`
- `xunce-full-network-v1-forward-audit.json`
- `xunce-full-network-v1-metadata-audit.json`
- `xunce-full-network-v1-parameter-latency-audit.json`
- `xunce-full-network-v1-logits.jsonl`
- `xunce-full-network-v1-boundary-audit.json`
- `xunce-full-network-v1-rejection-report.json`
- `xunce-full-network-v1-report.md`

## Network Components

The module must include:

- candidate graph encoder;
- edge encoder;
- topology-biased message passing;
- coverage memory token encoder;
- ROI/family/budget fusion encoder;
- masked policy logits head;
- scalar value head;
- explicit metadata with architecture name, feature counts, hidden dim, layer counts, and component flags.

The network may remain research-only under `scripts/`; it must not be registered in `model-explorer` production policy architecture inventory in this stage.

## Decision Contract

Summary fields:

- `status`
- `reason_codes`
- `source_architecture_contrast_status`
- `architecture=xunce_full_network_v1`
- `candidate_count`
- `edge_count`
- `candidate_graph_encoder_used`
- `topology_bias_used`
- `coverage_memory_token_used`
- `roi_budget_fusion_used`
- `masked_logits_valid`
- `value_head_valid`
- `metadata_audit_passed`
- `parameter_count`
- `median_forward_latency_ms`
- `parameter_latency_audit_passed`
- `full_network_v1_passed`
- `next_required_change`

Boundary fields must stay false:

- `publishes_checkpoint`
- `replaces_default_policy`
- `connects_real_executor`
- `starts_online_canary`
- `runs_new_ppo_update`
- `modifies_action_space`
- `modifies_default_astar`
- `real_world_release_approved`
- `real_world_performance_claimed`

`modifies_network=false` means no production/default network is modified. The research-only module itself is created but not installed.

Passing next gate:

`full_network_static_contract_validation`

Failure next gate:

`fix_xunce_full_network_v1`

If Stage 7 evidence is missing or not passed:

`fix_architecture_contrast_evaluation`

## Implementation Steps

1. Write failing tests for default pass, direct forward shape/mask behavior, missing Stage 7, parameter gate failure, and boundary violation.
2. Implement `XunceFullNetworkV1` in a shared script module with deterministic metadata.
3. Implement runner/config/shell to load Stage 7 + Stage 4 evidence, build tensors, run forward passes, and write artifacts.
4. Update README, architecture report, Global 99 spec, and Xunce design spec.
5. Run Stage 8 tests, Xunce Stage 0-8 tests, Global 99 regression, default runner, docs grep, and `git diff --check`.
6. Commit and push Stage 8 independently.

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_full_network_v1.py -q
PYTHON=$PY bash scripts/run_xunce_full_network_v1.sh
jq '{status,reason_codes,architecture,full_network_v1_passed,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_xunce_full_network_v1/xunce-full-network-v1-summary.json
rg -n "Xunce Full Network v1|run_xunce_full_network_v1|full_network_static_contract_validation" \
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
- No action-space or default-A* change.
- No performance or real-world claim.
