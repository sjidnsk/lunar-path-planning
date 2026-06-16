# Global 99% Exploration Coverage Line

## Summary

`Global 99% Exploration Coverage Line` is a new exploration-performance
mainline. Its target is 99% coverage of reachable safe exploration cells inside
a requested ROI on 1km x 1km maps and arbitrary smaller/larger ROI tasks.

This is a roadmap and benchmark contract, not a claim that the current
family-balanced policy or default-policy candidate already reaches 99%
coverage. Current release-governance and family-balanced evidence remains
scoped to offline/shadow/canary claims and must not be widened by this line.

## Contract

The benchmark denominator is `reachable_safe_exploration_cells` inside the ROI,
not all raw map cells. Cells that are unreachable, unsafe, or impossible within
the configured path budget must be reported explicitly through reason codes.
The benchmark must separate policy performance from map infeasibility.

The first output root is:

```text
outputs/path_feedback_batch_global_99_coverage_benchmark_v1/
```

The first-stage runner and config are:

```text
scripts/run_global_99_coverage_benchmark.py
scripts/run_global_99_coverage_benchmark.sh
configs/global_99_coverage_benchmark_v1.json
tests/test_global_99_coverage_benchmark.py
```

The first-stage artifact names are:

- `global-99-coverage-benchmark-summary.json`
- `global-99-coverage-benchmark-manifest.json`
- `global-99-coverage-ledger.jsonl`
- `global-99-coverage-denominator-audit.json`
- `global-99-coverage-rejection-report.json`
- `global-99-coverage-benchmark-report.md`

Expected summary fields:

- `target_coverage_rate=0.99`
- `achieved_coverage_rate`
- `reachable_safe_cell_count`
- `covered_reachable_safe_cell_count`
- `coverage_target_met`
- `infeasible_reason_codes`
- `path_budget_exhausted`
- `coverage_denominator_valid`
- `coverage_ledger_complete`
- `default_policy_replacement_approved=false`
- `real_executor_connection_approved=false`

Suggested failure reason codes:

- `coverage_target_not_met`
- `unreachable_roi_cells`
- `unsafe_roi_cells`
- `insufficient_budget`
- `coverage_denominator_invalid`
- `coverage_ledger_incomplete`
- `path_planner_boundary_violation`
- `release_boundary_violation`

## Development Order

1. `Global 99% Coverage Benchmark v1`
   - Define the 1km x 1km / arbitrary ROI / reachable safe cell / 99% coverage
     evaluation contract.
   - Produce deterministic fixtures and summary artifacts.
   - Do not train PPO or change the policy network.

2. `Frontier Coverage Planner Baseline v1`
   - Add a non-learning baseline using frontier selection, a coverage map,
     revisit penalty, and path budget accounting.
   - This is the "sweeping robot" reference that proves the task loop can run.
   - It should expose whether 99% coverage is blocked by planning, budget,
     map feasibility, or policy choice.

3. `Coverage Memory + Replanning Loop v1`
   - Maintain global coverage memory.
   - Update covered cells after each simulated step or route segment.
   - Replan toward the next uncovered reachable safe frontier until target,
     budget exhaustion, or infeasibility.

4. `Policy-Guided Global Coverage v1`
   - Let the current PPO policy rank frontiers or waypoints at the global
     coverage level.
   - Preserve stable contracts such as `model-explorer-contract/v1`,
     `path-feedback-summary/v1`, and `path-planner-route/v1`.
   - The policy assists global ordering; it does not replace release
     governance or connect a real executor.

5. `99% Multi-Map Generalization v1`
   - Validate across multiple maps, starts, ROI shapes, obstacle layouts, risk
     fields, and budget settings.
   - Report per-family and aggregate coverage, budget usage, fallback,
     controlled regression, and infeasibility rates.

6. `Network Architecture Upgrade v1`
   - Start only after the benchmark, baseline, replanning loop, and
     policy-guided coverage expose a clear network bottleneck.
   - Candidate upgrades include residual MLP, gated MLP, candidate-set encoder,
     lightweight attention, and graph/region encoder variants.
   - Compare performance, parameter count, inference latency, and coverage
     generalization against the existing network.

7. `99% Coverage Release Governance v1`
   - Start only after 99% reachable coverage is evidence-backed in offline
     multi-map shadow/canary validation.
   - Keep it as release governance, not direct default-policy replacement.

## First Target

The first concrete implementation target for this line is:

```text
Global 99% Coverage Benchmark v1
```

Acceptance:

- Defines ROI, reachable safe denominator, coverage ledger, budget model, and
  infeasibility taxonomy.
- Writes summary, manifest, ledger, denominator audit, rejection report, and
  report artifacts under
  `outputs/path_feedback_batch_global_99_coverage_benchmark_v1/`.
- Produces deterministic sample fixtures for at least one 1km x 1km map and
  one arbitrary ROI task.
- Summary reports `coverage_target_met` and all required boundary flags.
- `next_required_change=frontier_coverage_planner_baseline` when the benchmark
  contract is valid.
- Project docs stay aligned with this development order.

## Validation

Implementation validation:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_global_99_coverage_benchmark.py -q
PYTHON=$PY bash scripts/run_global_99_coverage_benchmark.sh
jq '{status,reason_codes,target_coverage_rate,achieved_coverage_rate,coverage_target_met,next_required_change,default_policy_replacement_approved,real_executor_connection_approved}' \
  outputs/path_feedback_batch_global_99_coverage_benchmark_v1/global-99-coverage-benchmark-summary.json
rg -n "Global 99% Coverage Benchmark v1|run_global_99_coverage_benchmark|frontier_coverage_planner_baseline" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md
git diff --check
```

## Non-Goals

No PPO training in benchmark v1, no checkpoint publication, no default policy
replacement, no real executor connection, no guard relaxation, no default A*
replacement, no action-space change, no network architecture change in the
first benchmark stage, no real-world performance claim, no
Ackermann-feasible trajectory claim, and no treating IRIS/GCS/path-planner
diagnostics as release proof.
