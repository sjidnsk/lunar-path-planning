# Xunce Stage 18.4D ROI Metadata Propagation Plan

## Summary

Stage 18.4D dynamic rollout previously resolved ROI group labels in two places:
`xunce_dynamic_frontier_nbv.py` had the newer fallback order, while
`run_xunce_high_fidelity_exploration_coverage_comparison.py` still used a local
legacy expression. This could produce `roi_group=unknown` or inconsistent ROI
labels across proposal, validation, step, episode, and pair artifacts.

## Implemented Direction

- Promote the dynamic helper to `resolve_roi_group(scenario, slice_row)`.
- Keep `_roi_group()` as a compatibility wrapper for existing tests and callers.
- Use the same helper in the Stage 18.4 rollout runner.
- Preserve additive/legacy fallback behavior:
  `scenario.roi_group`, `slice_row.roi_group`, `slice_row.roi_name`,
  `scenario.scenario_group`, `slice_row.metadata.roi_group`, then `unknown`.

## Verification

- Unit tests cover the source order and row-level propagation to formal
  candidates, proposal rows, and validation rows.
- Stage 18.4 integration tests scan dynamic proposals, validation results, steps,
  v2 steps, episodes, comparison pairs, paired decision audit, model inference,
  and ROI breakdown for missing or `unknown` ROI groups.
- A 24x1 dynamic smoke should confirm the real evidence root no longer emits
  `roi_group=unknown` when ROI metadata exists in Stage 18A evidence.

## Non-Goals

This is metadata propagation only. It does not change candidate generation,
A* validation, coverage/risk/cost calculations, model inference, checkpoint
format, action space, executor behavior, PPO, release gates, or online canary.
