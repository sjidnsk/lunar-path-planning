# Xunce High-Fidelity Real-Map Comparison v1

## Summary

`Xunce High-Fidelity Real-Map Comparison v1` is a post-research-chain evidence
extension for 巡策. It starts only after `Xunce Release Governance Gate v1`
passes with `next_required_change=xunce_research_track_complete`.

The stage has two gates:

1. `Xunce High-Fidelity Real-Map ROI Expansion v1`
2. `Xunce High-Fidelity Real-Map Policy Comparison v1`

The goal is to compare the 巡策 candidate against the incumbent experimental
policy on a larger quasi-real LOLA ROI set before considering any default-policy
candidate authorization preflight.

## Implemented Entry Points

Run ROI expansion:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_high_fidelity_real_map_roi_expansion.sh
```

Run policy comparison:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_high_fidelity_real_map_comparison.sh
```

Default output roots:

- `outputs/path_feedback_batch_xunce_high_fidelity_real_map_roi_expansion_v1/`
- `outputs/path_feedback_batch_xunce_high_fidelity_real_map_comparison_v1/`

## Decision Contract

ROI expansion passes only when it reaches 24 slices across 8 ROI groups,
preserves train/validation/test coverage, keeps context IDs present, keeps
legacy identity fallback at zero, keeps open-grid fallback at zero, and writes
`domain_gap_verdict=acceptable_for_high_fidelity_comparison`.

Policy comparison passes as evidence when the expanded ROI source is valid and
both candidate checkpoints are available for read-only comparison. It establishes
an upgrade advantage only when:

- `failed_required_scenario_count=0`
- `xunce_worse_than_incumbent_count=0`
- `controlled_regression_count=0`
- `xunce_guard_fallback_rate <= 0.05`
- `xunce_better_than_incumbent_count >= 1`
- the parameter and latency budgets pass

If the evidence is valid but advantage is not established, the next gate is
`xunce_research_iteration_required`. If advantage is established, the next gate
is `xunce_default_policy_candidate_authorization_preflight`.

## Non-Goals

This stage does not train PPO, modify the network, publish a checkpoint, replace
default policy, connect a real executor, start online canary traffic, or claim
real-world performance. The real-map scope remains repository-local LOLA
quasi-real ROI and path-feedback sidecar evidence.
