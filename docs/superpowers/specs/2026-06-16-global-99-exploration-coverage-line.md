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

The second-stage runner and config are:

```text
scripts/run_frontier_coverage_planner_baseline.py
scripts/run_frontier_coverage_planner_baseline.sh
configs/frontier_coverage_planner_baseline_v1.json
tests/test_frontier_coverage_planner_baseline.py
```

The second-stage output root is:

```text
outputs/path_feedback_batch_frontier_coverage_planner_baseline_v1/
```

The second-stage artifact names are:

- `frontier-coverage-planner-baseline-summary.json`
- `frontier-coverage-planner-baseline-manifest.json`
- `frontier-coverage-plan.jsonl`
- `frontier-coverage-ledger.jsonl`
- `frontier-coverage-budget-audit.json`
- `frontier-coverage-rejection-report.json`
- `frontier-coverage-planner-baseline-report.md`

The third-stage runner and config are:

```text
scripts/run_coverage_memory_replanning_loop.py
scripts/run_coverage_memory_replanning_loop.sh
configs/coverage_memory_replanning_loop_v1.json
tests/test_coverage_memory_replanning_loop.py
```

The third-stage output root is:

```text
outputs/path_feedback_batch_coverage_memory_replanning_loop_v1/
```

The third-stage artifact names are:

- `coverage-memory-replanning-loop-summary.json`
- `coverage-memory-replanning-loop-manifest.json`
- `coverage-memory-replanning-trace.jsonl`
- `coverage-memory-snapshots.jsonl`
- `coverage-memory-ledger.jsonl`
- `coverage-memory-budget-audit.json`
- `coverage-memory-rejection-report.json`
- `coverage-memory-replanning-loop-report.md`

The fourth-stage runner and config are:

```text
scripts/run_policy_guided_global_coverage.py
scripts/run_policy_guided_global_coverage.sh
configs/policy_guided_global_coverage_v1.json
tests/test_policy_guided_global_coverage.py
```

The fourth-stage output root is:

```text
outputs/path_feedback_batch_policy_guided_global_coverage_v1/
```

The fourth-stage artifact names are:

- `policy-guided-global-coverage-summary.json`
- `policy-guided-global-coverage-manifest.json`
- `policy-guided-global-coverage-decisions.jsonl`
- `policy-guided-global-coverage-ledger.jsonl`
- `policy-guided-global-coverage-memory-snapshots.jsonl`
- `policy-guided-global-coverage-policy-score-audit.json`
- `policy-guided-global-coverage-guard-audit.json`
- `policy-guided-global-coverage-budget-audit.json`
- `policy-guided-global-coverage-rejection-report.json`
- `policy-guided-global-coverage-report.md`

The fifth-stage runner and config are:

```text
scripts/run_global_99_multi_map_generalization.py
scripts/run_global_99_multi_map_generalization.sh
configs/global_99_multi_map_generalization_v1.json
tests/test_global_99_multi_map_generalization.py
```

The fifth-stage output root is:

```text
outputs/path_feedback_batch_global_99_multi_map_generalization_v1/
```

The fifth-stage artifact names are:

- `global-99-multi-map-generalization-summary.json`
- `global-99-multi-map-generalization-manifest.json`
- `global-99-multi-map-scenario-results.jsonl`
- `global-99-multi-map-family-summary.json`
- `global-99-multi-map-policy-vs-baseline-audit.json`
- `global-99-multi-map-budget-audit.json`
- `global-99-multi-map-rejection-report.json`
- `global-99-multi-map-generalization-report.md`

The sixth-stage readiness-review runner and config are:

```text
scripts/run_network_architecture_upgrade_readiness_review.py
scripts/run_network_architecture_upgrade_readiness_review.sh
configs/network_architecture_upgrade_readiness_review_v1.json
tests/test_network_architecture_upgrade_readiness_review.py
```

The sixth-stage output root is:

```text
outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1/
```

The sixth-stage artifact names are:

- `network-architecture-upgrade-readiness-summary.json`
- `network-architecture-upgrade-evidence-audit.json`
- `network-architecture-upgrade-bottleneck-attribution.json`
- `network-architecture-upgrade-recommendation-report.md`
- `network-architecture-upgrade-readiness-manifest.json`
- `network-architecture-upgrade-rejection-report.json`

The seventh-stage release-governance-preflight runner and config are:

```text
scripts/run_global_99_release_governance_preflight.py
scripts/run_global_99_release_governance_preflight.sh
configs/global_99_release_governance_preflight_v1.json
tests/test_global_99_release_governance_preflight.py
```

The seventh-stage output root is:

```text
outputs/path_feedback_batch_global_99_release_governance_preflight_v1/
```

The seventh-stage artifact names are:

- `global-99-release-governance-preflight-summary.json`
- `global-99-release-governance-manifest.json`
- `global-99-release-evidence-lineage-audit.json`
- `global-99-release-scope-audit.json`
- `global-99-release-boundary-audit.json`
- `global-99-release-kill-switch-audit.json`
- `global-99-release-rollback-audit.json`
- `global-99-release-telemetry-audit.json`
- `global-99-release-rejection-report.json`
- `global-99-release-governance-preflight-report.md`

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
   - When valid, it sets `next_required_change=coverage_memory_replanning_loop`.

3. `Coverage Memory + Replanning Loop v1`
   - Maintain global coverage memory.
   - Update covered cells after each simulated step or route segment.
   - Replan toward the next uncovered reachable safe frontier until target,
     budget exhaustion, or infeasibility.
   - When valid, it sets `next_required_change=policy_guided_global_coverage`.

4. `Policy-Guided Global Coverage v1`
   - Let the current PPO policy rank frontiers or waypoints at the global
     coverage level.
   - Use read-only checkpoint inference through `ModelExplorerContract` /
     `GoalCandidate` mapping and guarded blended ranking.
   - The policy assists global ordering; it does not run PPO update, publish a
     checkpoint, replace default policy, call path-planner, use NPZ/sidecar
     maps, or connect a real executor.
   - When valid, it sets `next_required_change=global_99_multi_map_generalization`.

5. `99% Multi-Map Generalization v1`
   - Validate across multiple maps, starts, ROI shapes, obstacle layouts, risk
     fields, and budget settings.
   - Report per-family and aggregate coverage, budget usage, fallback,
     controlled regression, and infeasibility rates.
   - When valid, it sets
     `next_required_change=network_architecture_upgrade_readiness_review`.

6. `Network Architecture Upgrade Readiness Review v1`
   - Consume existing multi-map summary, family summary, policy-vs-baseline
     audit, and scenario result artifacts.
   - Decide whether the evidence actually points to a policy-network
     bottleneck.
   - Do not train PPO, add a new architecture, publish a checkpoint, replace
     default policy, call path-planner, use NPZ/sidecar maps, or connect a real
     executor.
   - When evidence is healthy and no network bottleneck is detected, it sets
     `next_required_change=global_99_release_governance_preflight`.
   - When policy regression or excessive guard fallback is present, it may set
     `next_required_change=network_architecture_upgrade_v1`.

7. `Global 99 Release Governance Preflight v1`
   - Consume existing network-readiness, multi-map, and policy-guided summary
     artifacts.
   - Decide whether the synthetic Global 99 evidence is eligible for a
     shadow/canary preflight.
   - Do not publish a checkpoint, replace default policy, connect a real
     executor, run PPO, modify network/action space/default A*, call
     path-planner, or use NPZ/sidecar maps.
   - When valid, it sets
     `next_required_change=global_99_shadow_canary_preflight`.

8. `Global 99 Shadow Canary Preflight v1`
   - Start only after release governance preflight passes.
   - Keep the candidate in shadow/canary governance; do not install it as a
     real default policy.

9. `Network Architecture Upgrade v1`
   - Start only after the benchmark, baseline, replanning loop, and
     policy-guided coverage plus readiness review expose a clear network
     bottleneck.
   - Candidate upgrades include residual MLP, gated MLP, candidate-set encoder,
     lightweight attention, and graph/region encoder variants.
   - Compare performance, parameter count, inference latency, and coverage
     generalization against the existing network.

10. `99% Coverage Release Governance v1`
   - Start only after 99% reachable coverage is evidence-backed in offline
     multi-map shadow/canary validation.
   - Keep it as release governance, not direct default-policy replacement.

## Implemented Targets

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

The second concrete implementation target for this line is:

```text
Frontier Coverage Planner Baseline v1
```

Acceptance:

- Reads `configs/frontier_coverage_planner_baseline_v1.json`.
- Uses `configs/global_99_coverage_benchmark_v1.json` as the source synthetic
  Global 99 contract but ignores source `coverage_events`.
- Generates deterministic frontier baseline plan and ledger rows.
- Writes summary, manifest, plan, ledger, budget audit, rejection report, and
  report artifacts under
  `outputs/path_feedback_batch_frontier_coverage_planner_baseline_v1/`.
- Summary reports `coverage_target_met`, `generated_coverage_event_count`,
  `planned_path_cost_m`, `frontier_plan_complete`, and all required boundary
  flags.
- `next_required_change=coverage_memory_replanning_loop` when the baseline
  closes cleanly.
- Project docs stay aligned with this development order.

The third concrete implementation target for this line is:

```text
Coverage Memory + Replanning Loop v1
```

Acceptance:

- Reads `configs/coverage_memory_replanning_loop_v1.json`.
- Uses the synthetic Global 99 source config and Frontier baseline scoring
  weights, but still does not use PPO, path-planner, NPZ/sidecar maps, or a real
  executor.
- Maintains persistent coverage memory, writes monotonic memory snapshots, and
  supports deterministic resume from a snapshot.
- Writes summary, manifest, trace, snapshots, ledger, budget audit, rejection
  report, and report artifacts under
  `outputs/path_feedback_batch_coverage_memory_replanning_loop_v1/`.
- Summary reports `replanning_cycle_count`, `memory_snapshot_count`,
  `memory_resume_supported`, `memory_resume_verified`,
  `coverage_memory_complete`, and all required boundary flags.
- `next_required_change=policy_guided_global_coverage` when the memory loop
  closes cleanly.
- Project docs stay aligned with this development order.

The fourth concrete implementation target for this line is:

```text
Policy-Guided Global Coverage v1
```

Acceptance:

- Reads `configs/policy_guided_global_coverage_v1.json`.
- Uses deterministic synthetic Global 99 scenarios and the existing Frontier /
  Coverage Memory helper stack.
- Loads the experimental policy candidate checkpoint from
  `outputs/path_feedback_batch_value_stability_candidate_v1/` in read-only
  inference mode.
- Maps frontier candidates into `ModelExplorerContract` / `GoalCandidate`,
  scores candidates with `TorchPolicyScorer`, and blends baseline score with
  normalized policy logits under guard checks.
- Falls back to the baseline candidate when policy scoring is invalid, action
  masks are invalid, policy choices exceed budget, are unreachable, or make no
  coverage progress.
- Writes summary, manifest, decisions, ledger, memory snapshots, policy score
  audit, guard audit, budget audit, rejection report, and report artifacts under
  `outputs/path_feedback_batch_policy_guided_global_coverage_v1/`.
- Summary reports `policy_loaded`, `policy_guidance_applied`,
  `policy_scored_candidate_count`, `policy_guided_decision_count`,
  `policy_selected_decision_count`, `policy_guard_fallback_count`,
  `baseline_agreement_rate`, `controlled_regression_count`, and all required
  boundary flags.
- `next_required_change=global_99_multi_map_generalization` when the guarded
  policy-guided loop closes cleanly.
- Project docs stay aligned with this development order.

The fifth concrete implementation target for this line is:

```text
Global 99 Multi-Map Generalization v1
```

Acceptance:

- Reads `configs/global_99_multi_map_generalization_v1.json`.
- Uses deterministic synthetic scenario families:
  `open_field`, `corridor`, `rooms`, `blocked_roi`, `unsafe_patch`,
  `narrow_passage`, `cells_roi`, and expected-infeasible `budget_limited`.
- Reuses the Policy-Guided Global Coverage runner internals for read-only
  checkpoint loading, frontier candidate scoring, and guarded blended ranking.
- Writes summary, manifest, scenario results, family summary, policy-vs-baseline
  audit, budget audit, rejection report, and report artifacts under
  `outputs/path_feedback_batch_global_99_multi_map_generalization_v1/`.
- Summary reports scenario/family counts, aggregate and minimum required
  scenario coverage, policy guidance counters, policy-vs-baseline counters,
  infeasibility reason codes, and all required boundary flags.
- Expected-infeasible `budget_limited` scenarios may fail with
  `insufficient_budget` / `coverage_target_not_met` without failing the overall
  required-scenario gate.
- `next_required_change=network_architecture_upgrade_readiness_review` when
  all required multi-map scenarios close cleanly.
- Project docs stay aligned with this development order.

The sixth concrete implementation target for this line is:

```text
Network Architecture Upgrade Readiness Review v1
```

Acceptance:

- Reads `configs/network_architecture_upgrade_readiness_review_v1.json`.
- Consumes existing Global 99 multi-map artifacts from
  `outputs/path_feedback_batch_global_99_multi_map_generalization_v1/`.
- Writes readiness summary, evidence audit, bottleneck attribution,
  recommendation report, manifest, and rejection report artifacts under
  `outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1/`.
- Summary reports source multi-map status, required scenario counts, aggregate
  and minimum coverage, policy guidance counters, policy-vs-baseline counters,
  `policy_guard_fallback_rate`, `network_upgrade_recommended`,
  `network_upgrade_readiness_decision`, `network_upgrade_blockers`,
  `bottleneck_attribution`, `candidate_architecture_inventory`, and all required
  boundary flags.
- If source multi-map failed or is missing, the review fails and points back to
  `fix_global_99_multi_map_generalization`.
- If required scenarios are healthy, policy better count is positive, policy
  worse count and controlled regression are zero, and fallback rate is low, the
  review does not recommend a network upgrade and sets
  `next_required_change=global_99_release_governance_preflight`.
- If policy regression or excessive guard fallback appears, the review may set
  `next_required_change=network_architecture_upgrade_v1`.
- Project docs stay aligned with this development order.

The seventh concrete implementation target for this line is:

```text
Global 99 Release Governance Preflight v1
```

Acceptance:

- Reads `configs/global_99_release_governance_preflight_v1.json`.
- Consumes existing network-readiness, multi-map, and policy-guided summary
  artifacts.
- Writes summary, manifest, evidence-lineage audit, scope audit, release
  boundary audit, kill-switch audit, rollback audit, telemetry audit, rejection
  report, and report artifacts under
  `outputs/path_feedback_batch_global_99_release_governance_preflight_v1/`.
- Summary reports source statuses, required scenario counts, aggregate and
  minimum coverage, policy guidance counters, `policy_guard_fallback_rate`,
  `network_upgrade_recommended`, audit pass/fail flags, governance verdict,
  `next_required_change`, and all required boundary flags.
- If network readiness is missing, failed, or not pointing to
  `global_99_release_governance_preflight`, the summary fails and points to
  `fix_network_architecture_upgrade_readiness_review`.
- If required multi-map scenarios are not healthy or coverage is below 0.99,
  the summary fails and points to `fix_global_99_multi_map_generalization`.
- If policy regression appears, the summary fails and points to
  `fix_policy_guided_global_coverage`.
- If any release/default-policy/executor boundary is open, the summary fails
  and points to `resolve_global_99_release_boundary_rejections`.
- If evidence is stable and boundaries are closed, the summary passes with
  `release_governance_verdict=eligible_for_global_99_shadow_canary_preflight`
  and `next_required_change=global_99_shadow_canary_preflight`.
- Project docs stay aligned with this development order.

## Validation

Implementation validation:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py \
  tests/test_global_99_multi_map_generalization.py \
  tests/test_network_architecture_upgrade_readiness_review.py \
  tests/test_global_99_release_governance_preflight.py -q
PYTHON=$PY bash scripts/run_global_99_coverage_benchmark.sh
PYTHON=$PY bash scripts/run_frontier_coverage_planner_baseline.sh
PYTHON=$PY bash scripts/run_coverage_memory_replanning_loop.sh
PYTHON=$PY bash scripts/run_policy_guided_global_coverage.sh
PYTHON=$PY bash scripts/run_global_99_multi_map_generalization.sh
PYTHON=$PY bash scripts/run_network_architecture_upgrade_readiness_review.sh
PYTHON=$PY bash scripts/run_global_99_release_governance_preflight.sh
jq '{status,reason_codes,target_coverage_rate,achieved_coverage_rate,coverage_target_met,next_required_change,default_policy_replacement_approved,real_executor_connection_approved}' \
  outputs/path_feedback_batch_global_99_coverage_benchmark_v1/global-99-coverage-benchmark-summary.json
jq '{status,reason_codes,achieved_coverage_rate,coverage_target_met,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_frontier_coverage_planner_baseline_v1/frontier-coverage-planner-baseline-summary.json
jq '{status,reason_codes,achieved_coverage_rate,coverage_target_met,next_required_change,memory_resume_verified,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_coverage_memory_replanning_loop_v1/coverage-memory-replanning-loop-summary.json
jq '{status,reason_codes,achieved_coverage_rate,coverage_target_met,policy_loaded,policy_guidance_applied,policy_scored_candidate_count,policy_guard_fallback_count,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor,runs_new_ppo_update}' \
  outputs/path_feedback_batch_policy_guided_global_coverage_v1/policy-guided-global-coverage-summary.json
jq '{status,reason_codes,scenario_count,passed_scenario_count,failed_scenario_count,aggregate_achieved_coverage_rate,min_scenario_achieved_coverage_rate,policy_guidance_applied,policy_scored_candidate_count,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor,runs_new_ppo_update}' \
  outputs/path_feedback_batch_global_99_multi_map_generalization_v1/global-99-multi-map-generalization-summary.json
jq '{status,reason_codes,network_upgrade_recommended,network_upgrade_readiness_decision,next_required_change,policy_guard_fallback_rate,baseline_agreement_rate,policy_better_than_baseline_count,policy_worse_than_baseline_count,controlled_regression_count,modifies_network,runs_new_ppo_update,publishes_checkpoint,replaces_default_policy}' \
  outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1/network-architecture-upgrade-readiness-summary.json
jq '{status,reason_codes,release_governance_verdict,next_required_change,release_boundary_audit_passed,kill_switch_audit_passed,rollback_audit_passed,telemetry_audit_passed,publishes_checkpoint,replaces_default_policy,connects_real_executor,runs_new_ppo_update,modifies_network}' \
  outputs/path_feedback_batch_global_99_release_governance_preflight_v1/global-99-release-governance-preflight-summary.json
rg -n "Global 99% Coverage Benchmark v1|run_global_99_coverage_benchmark|Frontier Coverage Planner Baseline v1|run_frontier_coverage_planner_baseline|Coverage Memory \\+ Replanning Loop v1|run_coverage_memory_replanning_loop|Policy-Guided Global Coverage v1|run_policy_guided_global_coverage|Global 99 Multi-Map Generalization v1|run_global_99_multi_map_generalization|Network Architecture Upgrade Readiness Review v1|run_network_architecture_upgrade_readiness_review|Global 99 Release Governance Preflight v1|run_global_99_release_governance_preflight|global_99_shadow_canary_preflight|network_architecture_upgrade_v1" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md
git diff --check
```

## Non-Goals

No PPO training, no checkpoint publication, no default policy replacement, no
real executor connection, no guard relaxation, no default A* replacement, no
action-space change, no network architecture change, no real-world performance
claim, no Ackermann-feasible trajectory claim, and no treating
IRIS/GCS/path-planner diagnostics as release proof. `Policy-Guided Global
Coverage v1` permits read-only experimental checkpoint inference only.
