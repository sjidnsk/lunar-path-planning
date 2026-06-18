# Xunce Stage 18I.2 Risk-Aware Frontier-NBV Candidate Repair

## Summary

Stage 18I.2 repairs the candidate-generation evidence chain after Stage 18I.
The goal is not model training. It creates an unbound candidate root that uses
true incumbent binding as the baseline, then requires downstream Stage 18B true
model inference and Stage 18G.0 binding before Stage 18H.0/18F/18C-v2 reuse.

The key correction is to stop using `action_index=0` as a pseudo incumbent.
The runner reads Stage 18G.0 true incumbent binding, matches the incumbent by
candidate cell, and then labels candidates as `incumbent_neighborhood`,
`frontier_boundary`, or `roi_undercovered_boundary`.

## Implementation

- Runner: `scripts/run_xunce_risk_aware_frontier_nbv_candidate_repair.py`
- Config: `configs/xunce_risk_aware_frontier_nbv_candidate_repair_v1.json`
- Stage id: `xunce-risk-aware-frontier-nbv-candidate-repair`
- Output root: `outputs/path_feedback_batch_xunce_risk_aware_frontier_nbv_candidate_repair_v1/`

Default inputs:

- Stage 18A ROI root: `outputs/path_feedback_batch_xunce_high_fidelity_real_map_roi_expansion_v1`
- Stage 18G.0 after-Stage18I true binding root:
  `outputs/path_feedback_batch_xunce_true_incumbent_selection_binding_after_stage18i_v1`
- Stage 18H.0 after-Stage18I quantization root:
  `outputs/path_feedback_batch_xunce_risk_coverage_cost_quantization_audit_after_stage18i_v1`

The runner writes a Stage 18B/18G/18H-compatible root:

- `xunce-high-fidelity-real-map-roi-expansion-summary.json`
- `xunce-high-fidelity-real-map-slices.jsonl`
- `xunce-high-fidelity-path-feedback-audit.json`
- `xunce-candidate-level-coverage-opportunity-summary.json`
- `xunce-risk-aware-frontier-nbv-candidate-repair-summary.json`
- `xunce-risk-aware-frontier-nbv-proposals.jsonl`
- `xunce-risk-aware-frontier-nbv-validated-candidates.jsonl`
- `xunce-risk-aware-frontier-nbv-rejection-audit.json`
- `xunce-risk-aware-frontier-nbv-report.md`

## Rules

- Coverage is geometric path-line plus endpoint provenance:
  `geometric_counterfactual_from_risk_aware_frontier_nbv_repair/v1`.
- `path_cost`, `risk`, `reachable`, and `open_grid_fallback_used` must come from
  path-feedback evidence.
- Proposal-only or unvalidated rows cannot become safe-efficient candidates.
- The output root is unbound and must not copy old model decisions.
- The stage does not train PPO, actor, critic, or evaluator models.
- The stage does not publish checkpoints, replace the default policy, connect
  executors, start online canary, or download new maps.

## Current Evidence Result

Running Stage 18I.2 on the current evidence produced:

```text
status=failed
next_required_change=repair_frontier_nbv_sampling
frontier_boundary_candidate_count=0
roi_undercovered_boundary_candidate_count=32
safe_efficient_candidate_count=0
```

Interpretation:

- The old Stage 18I root no longer falsely looks safe after true incumbent
  binding.
- ROI-undercovered high-coverage candidates exist.
- Those candidates are still cost-regressive relative to the true incumbent.
- No validated frontier candidate currently offers higher coverage with no cost
  or risk regression.

This keeps the bottleneck at candidate generation/path-feedback validation. It
does not yet justify actor/critic training.

## Next Route

If Stage 18I.2 fails as above:

```text
repair_frontier_nbv_sampling
```

The next implementation should generate or validate nearer frontier candidates
whose path cost is no worse than the true incumbent, or widen ROI/map complexity
only if path-feedback cannot expose such candidates.
