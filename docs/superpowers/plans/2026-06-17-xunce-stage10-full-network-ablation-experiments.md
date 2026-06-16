# Xunce Stage 10 Full Network Ablation Experiments v1 Plan

## Goal

Stage 10 runs deterministic ablation experiments for the research-only `xunce_full_network_v1`. It checks whether topology edges, coverage memory, ROI/budget context, and missing-indicator channels measurably influence logits/ranking after Stage 9 confirmed the static contract.

This is still not PPO training, not a performance claim, not checkpoint publication, and not production/default-policy installation.

## Inputs

- `outputs/path_feedback_batch_xunce_full_network_static_contract_validation_v1/xunce-full-network-static-contract-validation-summary.json`
- `scripts/xunce_full_network_common.py`

## Interface

Add:

- `configs/xunce_full_network_ablation_experiments_v1.json`
- `scripts/run_xunce_full_network_ablation_experiments.py`
- `scripts/run_xunce_full_network_ablation_experiments.sh`
- `tests/test_xunce_full_network_ablation_experiments.py`

Default output root:

`outputs/path_feedback_batch_xunce_full_network_ablation_experiments_v1/`

Artifacts:

- `xunce-full-network-ablation-experiments-summary.json`
- `xunce-full-network-ablation-experiments-manifest.json`
- `xunce-full-network-ablation-results.jsonl`
- `xunce-full-network-ablation-module-contribution-audit.json`
- `xunce-full-network-ablation-ranking-delta-audit.json`
- `xunce-full-network-ablation-boundary-audit.json`
- `xunce-full-network-ablation-rejection-report.json`
- `xunce-full-network-ablation-experiments-report.md`

## Ablation Cases

- `full`
- `no_topology_edges`
- `zero_memory`
- `zero_context`
- `zero_missing_indicators`

Each ablation is compared to the full case by:

- max absolute masked-logit delta;
- selected candidate change;
- rank order change;
- finite-output check;
- mask preservation.

## Decision Contract

Summary fields:

- `status`
- `reason_codes`
- `source_static_contract_status`
- `architecture=xunce_full_network_v1`
- `ablation_case_count`
- `max_topology_logit_delta`
- `max_memory_logit_delta`
- `max_context_logit_delta`
- `max_missing_indicator_logit_delta`
- `topology_ablation_effect_detected`
- `memory_ablation_effect_detected`
- `context_ablation_effect_detected`
- `missing_indicator_ablation_effect_detected`
- `mask_preserved`
- `finite_outputs_preserved`
- `ablation_experiments_passed`
- `next_required_change`

Boundary fields must remain false: checkpoint/default-policy/executor/online-canary/PPO/network/action-space/default-A*/real-world claims.

Passing next gate:

`full_network_stress_evaluation`

Failure next gate:

`fix_full_network_ablation_experiments`

If Stage 9 evidence is missing or not passed:

`fix_full_network_static_contract_validation`

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_full_network_ablation_experiments.py -q
PYTHON=$PY bash scripts/run_xunce_full_network_ablation_experiments.sh
jq '{status,reason_codes,ablation_experiments_passed,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_xunce_full_network_ablation_experiments_v1/xunce-full-network-ablation-experiments-summary.json
rg -n "Xunce Full Network Ablation Experiments v1|run_xunce_full_network_ablation_experiments|full_network_stress_evaluation" \
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
