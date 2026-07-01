# Stage26.7C Main Coverable Coverage Efficiency Rerun

## Goal

Run Stage26.7 again with `coverage_denominator_mode=main_coverable_cells` and
`coverage_denominator_source=main_coverable_cells/v1`, then judge success by
unit-distance main coverage instead of total path cost.

## Scope

- Reuse Stage26.7B coverable-cell semantics.
- Generate a repaired Stage26.7 config with the main-coverable denominator.
- Re-run Stage26.7 into a Stage26.7C subdirectory.
- Keep Hybrid A* path cost delta as a diagnostic field only.

## Non-Goals

- Do not change reward, PPO, network, candidate generation, Hybrid A*, or synthetic terrain generation.
- Do not publish checkpoints, replace the default policy, connect an executor, or start canary.

## Acceptance

- Stage26.7C confirms Stage26.7B passed before running.
- Stage26.7, Stage26.3, Stage21.5, and high-fidelity configs all carry `main_coverable_cells/v1`.
- Success requires positive coverage and AUC, non-negative `main_coverage_per_100m_delta`, no scenario regression, and no safety regression.
- `hybrid_astar_path_cost_delta` remains visible but does not hard-fail the run by itself.
