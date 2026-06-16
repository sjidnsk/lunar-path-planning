# Xunce Stage 11 Full Network Stress Evaluation v1 Plan

## Goal

Stage 11 stress-tests the research-only `xunce_full_network_v1` after Stage 10 verified module contribution. It checks candidate-count scaling, edge-count scaling, missing-indicator stress, finite outputs, mask preservation, parameter and latency budgets, and deterministic replay stability.

This stage does not train PPO, publish a checkpoint, register a production/default-policy architecture, replace default policy, connect an executor, or claim real-world performance.

## Inputs

- `outputs/path_feedback_batch_xunce_full_network_ablation_experiments_v1/xunce-full-network-ablation-experiments-summary.json`
- `scripts/xunce_full_network_common.py`

## Interface

Add:

- `configs/xunce_full_network_stress_evaluation_v1.json`
- `scripts/run_xunce_full_network_stress_evaluation.py`
- `scripts/run_xunce_full_network_stress_evaluation.sh`
- `tests/test_xunce_full_network_stress_evaluation.py`

Default output root:

`outputs/path_feedback_batch_xunce_full_network_stress_evaluation_v1/`

Artifacts:

- `xunce-full-network-stress-evaluation-summary.json`
- `xunce-full-network-stress-evaluation-manifest.json`
- `xunce-full-network-stress-case-results.jsonl`
- `xunce-full-network-stress-latency-audit.json`
- `xunce-full-network-stress-determinism-audit.json`
- `xunce-full-network-stress-boundary-audit.json`
- `xunce-full-network-stress-rejection-report.json`
- `xunce-full-network-stress-evaluation-report.md`

## Stress Cases

- candidate counts: small, default, and larger candidate sets;
- edge density: chain and dense synthetic edges;
- missing indicators: zero, partial, all-present;
- numeric range: normal and high-magnitude feature inputs;
- deterministic replay: same seed/input repeated twice must match within tolerance.

## Decision Contract

Summary fields:

- `status`
- `reason_codes`
- `source_ablation_status`
- `architecture=xunce_full_network_v1`
- `stress_case_count`
- `max_candidate_count`
- `max_edge_count`
- `parameter_count`
- `max_forward_latency_ms_observed`
- `non_finite_output_count`
- `mask_preserved`
- `deterministic_replay_max_delta`
- `deterministic_replay_passed`
- `latency_gate_passed`
- `parameter_gate_passed`
- `fallback_rate`
- `stress_evaluation_passed`
- `next_required_change`

Passing next gate:

`guarded_training_candidate_preflight`

Failure next gate:

`fix_full_network_stress_evaluation`

If Stage 10 evidence is missing or not passed:

`fix_full_network_ablation_experiments`

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_full_network_stress_evaluation.py -q
PYTHON=$PY bash scripts/run_xunce_full_network_stress_evaluation.sh
jq '{status,reason_codes,stress_evaluation_passed,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_xunce_full_network_stress_evaluation_v1/xunce-full-network-stress-evaluation-summary.json
rg -n "Xunce Full Network Stress Evaluation v1|run_xunce_full_network_stress_evaluation|guarded_training_candidate_preflight" \
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
