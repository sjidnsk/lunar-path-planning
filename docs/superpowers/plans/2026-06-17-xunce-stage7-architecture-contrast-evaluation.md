# Xunce Stage 7 Architecture Contrast Evaluation v1 Plan

## Goal

Stage 7 turns the passed `Xunce Prototype Mechanism Validation v1` evidence into a deterministic architecture contrast gate. It compares the existing policy architectures (`mlp_v1`, `mlp_missing_v1`, `candidate_attention_v1`) with the research-only `topology_aware_coverage_graph_proto_v1` on the same topology feature fixture.

The stage does not train PPO, register a production architecture, publish a checkpoint, replace the default policy, connect an executor, or claim real-world performance. Its only decision is whether the small Xunce prototype is promising enough to justify building a complete Xunce network v1.

## Inputs

- `outputs/path_feedback_batch_xunce_proto_mechanism_validation_v1/xunce-proto-mechanism-validation-summary.json`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-candidates.jsonl`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-edges.jsonl`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-memory.json`
- `model-explorer/src/model_explorer/policy/architectures.py`
- `scripts/xunce_topology_graph_proto_common.py`

## Interface

Add:

- `configs/xunce_architecture_contrast_evaluation_v1.json`
- `scripts/run_xunce_architecture_contrast_evaluation.py`
- `scripts/run_xunce_architecture_contrast_evaluation.sh`
- `tests/test_xunce_architecture_contrast_evaluation.py`

Default output root:

`outputs/path_feedback_batch_xunce_architecture_contrast_evaluation_v1/`

Artifacts:

- `xunce-architecture-contrast-evaluation-summary.json`
- `xunce-architecture-contrast-evaluation-manifest.json`
- `xunce-architecture-contrast-results.jsonl`
- `xunce-architecture-contrast-parameter-latency-audit.json`
- `xunce-architecture-contrast-ranking-audit.json`
- `xunce-architecture-contrast-boundary-audit.json`
- `xunce-architecture-contrast-rejection-report.json`
- `xunce-architecture-contrast-evaluation-report.md`

## Decision Contract

Summary fields:

- `status`
- `reason_codes`
- `source_mechanism_validation_status`
- `architecture_count`
- `evaluated_architectures`
- `reference_architecture`
- `winner_architecture`
- `xunce_proto_rank`
- `xunce_proto_parameter_count`
- `candidate_attention_parameter_count`
- `xunce_proto_latency_ms`
- `candidate_attention_latency_ms`
- `parameter_efficiency_gate_passed`
- `latency_gate_passed`
- `ranking_quality_gate_passed`
- `architecture_contrast_passed`
- `next_required_change`

Boundary fields must remain false:

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

`full_xunce_network_v1_design`

Failure next gate:

`fix_architecture_contrast_evaluation`

If Stage 6 evidence is missing or not passed:

`fix_xunce_proto_mechanism_validation`

## Scoring

Use a deterministic heuristic target derived from topology fixture features:

`quality = new_coverage - path_cost - revisit_penalty - fallback_risk + budget_alignment`

Each architecture is initialized with a fixed seed and run on the same candidates and mask. The contrast runner records:

- selected candidate
- selected candidate heuristic quality
- regret versus the heuristic best candidate
- agreement with heuristic best candidate
- parameter count
- median forward latency
- finite-output/mask correctness

The Xunce prototype must:

- produce finite logits and probabilities;
- obey action masks;
- keep parameter count within the configured ratio versus `candidate_attention_v1`;
- keep latency within the configured ratio versus `candidate_attention_v1`;
- avoid being worse than all existing architectures on the deterministic ranking quality metric.

This is not a performance proof. It is only a research gate for the complete-network design.

## Implementation Steps

1. Write failing tests for pass, missing Stage 6, parameter gate failure, latency gate failure, ranking failure, and boundary violation.
2. Implement the runner with pure deterministic fixture loading and no training.
3. Add shell entrypoint and default config.
4. Write all artifacts and markdown report.
5. Update README, architecture report, Global 99 spec, and Xunce design spec.
6. Run focused tests, Xunce Stage 0-7 tests, Global 99 regression, default runner, docs grep, and `git diff --check`.
7. Commit and push Stage 7 independently.

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_xunce_architecture_contrast_evaluation.py -q
PYTHON=$PY bash scripts/run_xunce_architecture_contrast_evaluation.sh
jq '{status,reason_codes,winner_architecture,xunce_proto_rank,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_xunce_architecture_contrast_evaluation_v1/xunce-architecture-contrast-evaluation-summary.json
rg -n "Xunce Architecture Contrast Evaluation v1|xunce_architecture_contrast_evaluation|full_xunce_network_v1_design" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs
git diff --check
```

## Non-goals

- No PPO update.
- No checkpoint publication.
- No default-policy replacement.
- No executor connection.
- No online canary.
- No action-space or default-A* change.
- No complete Xunce network implementation yet.
- No real-world performance claim.
