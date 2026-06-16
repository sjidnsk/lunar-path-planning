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

The eighth-stage shadow-canary-preflight runner and config are:

```text
scripts/run_global_99_shadow_canary_preflight.py
scripts/run_global_99_shadow_canary_preflight.sh
configs/global_99_shadow_canary_preflight_v1.json
tests/test_global_99_shadow_canary_preflight.py
```

The eighth-stage output root is:

```text
outputs/path_feedback_batch_global_99_shadow_canary_preflight_v1/
```

The eighth-stage artifact names are:

- `global-99-shadow-canary-preflight-summary.json`
- `global-99-shadow-canary-manifest.json`
- `global-99-shadow-replay-audit.json`
- `global-99-canary-eligibility-audit.json`
- `global-99-shadow-canary-boundary-audit.json`
- `global-99-shadow-canary-kill-switch-audit.json`
- `global-99-shadow-canary-rollback-audit.json`
- `global-99-shadow-canary-telemetry-audit.json`
- `global-99-shadow-canary-rejection-report.json`
- `global-99-shadow-canary-preflight-report.md`

The ninth-stage shadow-canary-replay runner and config are:

```text
scripts/run_global_99_shadow_canary_replay.py
scripts/run_global_99_shadow_canary_replay.sh
configs/global_99_shadow_canary_replay_v1.json
tests/test_global_99_shadow_canary_replay.py
```

The ninth-stage output root is:

```text
outputs/path_feedback_batch_global_99_shadow_canary_replay_v1/
```

The ninth-stage artifact names are:

- `global-99-shadow-canary-replay-summary.json`
- `global-99-shadow-canary-replay-manifest.json`
- `global-99-shadow-canary-replay-scenario-results.jsonl`
- `global-99-shadow-canary-replay-family-summary.json`
- `global-99-shadow-canary-replay-source-match-audit.json`
- `global-99-shadow-canary-replay-policy-vs-baseline-audit.json`
- `global-99-shadow-canary-replay-boundary-audit.json`
- `global-99-shadow-canary-replay-kill-switch-audit.json`
- `global-99-shadow-canary-replay-rollback-audit.json`
- `global-99-shadow-canary-replay-telemetry-audit.json`
- `global-99-shadow-canary-replay-rejection-report.json`
- `global-99-shadow-canary-replay-report.md`

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
   - Convert synthetic release-governance evidence into offline shadow/canary
     replay eligibility.
   - Keep `canary_traffic_fraction=0.0` and `starts_online_canary=false`.
   - Do not publish a checkpoint, replace default policy, connect a real
     executor, run PPO, modify network/action space/default A*, call
     path-planner, use NPZ/sidecar maps, or claim real-world performance.
   - When valid, it sets
     `next_required_change=global_99_shadow_canary_replay`.

9. `Global 99 Shadow Canary Replay v1`
   - Start only after shadow/canary preflight passes.
   - Rerun deterministic synthetic multi-map policy-guided coverage in offline
     shadow/canary replay mode.
   - Compare replay scenario coverage and path cost against source multi-map
     scenario results by `scenario_id`.
   - Keep `canary_traffic_fraction=0.0` and `starts_online_canary=false`.
   - Do not publish a checkpoint, replace default policy, connect a real
     executor, run PPO, modify network/action space/default A*, call
     path-planner, use NPZ/sidecar maps, or claim real-world performance.
   - When valid, it sets `next_required_change=global_99_real_map_preflight`.

10. `Global 99 Real Map Preflight v1`
   - Start only after synthetic shadow/canary replay passes.
   - Audit existing LOLA quasi-real ROI, domain-gap, path-feedback sidecar, and
     optional semi-real path-feedback validation evidence.
   - Treat "real map" as quasi-real / sidecar evidence preflight only, not
     physical rover execution, online canary traffic, or default-policy
     installation.
   - Normalize upstream null release/executor boundary fields into explicit
     false values in the stage summary.
   - Do not publish a checkpoint, replace default policy, connect a real
     executor, run PPO, modify network/action space/default A*, call
     path-planner, or claim real-world performance.
   - When valid, it sets `next_required_change=global_99_real_map_shadow_replay`.

11. `Global 99 Real Map Shadow Replay v1`
   - Start only after real-map preflight passes.
   - Reuse frozen quasi-real path-feedback manifest, contracts, and sidecars in
     an isolated offline replay output root.
   - Run `model_explorer path-feedback validate/run` against the replay
     manifest and compare source/replay scenario evidence by `scenario_id`.
   - Permit offline path-feedback / path-planner route replay only under
     `path_planner_use_scope=offline_path_feedback_replay_only`.
   - Do not publish a checkpoint, replace default policy, connect a real
     executor, start online canary traffic, run PPO, modify network/action
     space/default A*, or claim real-world performance.
   - When valid, it sets
     `next_required_change=global_99_real_map_release_governance_preflight`.

12. `Global 99 Real Map Release Governance Preflight v1`
   - Start only after real-map shadow replay passes.
   - Audit real-map shadow replay, real-map preflight, quasi-real domain-gap,
     path-feedback, lineage, boundary, kill-switch, rollback, and telemetry.
   - Do not rerun path-planner; only audit upstream offline replay scope as
     `offline_path_feedback_replay_only`.
   - Do not publish a checkpoint, replace default policy, connect a real
     executor, start online canary traffic, run PPO, modify network/action
     space/default A*, or claim real-world performance.
   - When valid, it sets
     `next_required_change=global_99_real_map_shadow_canary_preflight`.

13. `Global 99 Real Map Shadow Canary Preflight v1`
   - Judge offline real-map shadow/canary replay eligibility.
   - Keep `canary_traffic_fraction=0.0` and `starts_online_canary=false`.
   - When valid, it sets
     `next_required_change=global_99_real_map_shadow_canary_replay`.

14. `Global 99 Real Map Shadow Canary Replay v1`
   - Replay real-map shadow/canary evidence offline and compare deterministic
     source/replay evidence, fallback, telemetry, and rollback.
   - When valid, it sets
     `next_required_change=global_99_real_map_evidence_refresh_drift_audit`.

15. `Global 99 Real Map Evidence Refresh / Drift Audit v1`
   - Refresh or re-audit quasi-real manifest, sidecar, context id, and
     source-match evidence for drift.

16. `Global 99 Real Map Multi-ROI Generalization v1`
   - Expand ROI groups, slices, terrain/risk/failure modes, and validate
     required real-map-like scenarios.

17. `Default Policy Candidate Authorization Preflight`
   - Authorize a candidate only; do not install it.

18. `Sandbox Candidate Installation Dry Run`
   - Run sandbox load/hash/provenance/rollback/kill-switch rehearsal only.

19. `Sandbox Consumer Replay / Canary`
   - Validate sandbox consumer replay/canary behavior with fallback.

20. `Controlled Default Policy Candidate Installation Preflight`
   - Decide readiness for a controlled installation review, not replacement.

21. `Network Architecture Upgrade v1`
   - Start only after the benchmark, baseline, replanning loop, and
     policy-guided coverage plus readiness review expose a clear network
     bottleneck.
   - Candidate upgrades include residual MLP, gated MLP, candidate-set encoder,
     lightweight attention, and graph/region encoder variants.
   - Compare performance, parameter count, inference latency, and coverage
     generalization against the existing network.

22. `99% Coverage Release Governance v1`
   - Start only after 99% reachable coverage is evidence-backed in offline
     multi-map shadow/canary validation.
   - Keep it as release governance, not direct default-policy replacement.

### Parallel Network Research Track

`巡策` is the short code name for `Topology-Aware Coverage Policy Network
Design`, documented separately in
`docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`.
It is a research track for a future architecture breakthrough, not a release
gate and not a replacement for the Global 99 governance chain. The primary goal
is higher exploration coverage and stronger generalization; lower parameter
count and faster inference are hard constraints; novelty is a research goal.

The proposed direction is coverage-memory-aware candidate graph ranking. It
keeps the existing candidate action set and guarded policy interface, while
adding topology and memory signals such as frontier cluster, BFS distance, ROI
relation, bottleneck risk, coverage overlap, revisit cost, remaining budget,
and recent coverage-memory trend. The first required stages are
`Network Literature & Project Bottleneck Review v1` and
`Topology-Aware Observation Contract v1`; architecture prototyping and any
training must remain separate gated stages.

The first gate for this parallel track is `Xunce Design Freeze Audit v1`, with
runner `scripts/run_xunce_design_freeze_audit.py`, config
`configs/xunce_design_freeze_v1.json`, and output root
`outputs/path_feedback_batch_xunce_design_freeze_v1/`. It checks that the
formal 巡策 spec records the full 0-17 plan-first stage chain and that current
Global 99 evidence remains compatible with closed checkpoint/default-policy /
executor / online-canary / PPO / network/action-space/default-A* boundaries.
When it passes, its `next_required_change` is `current_head_evidence_refresh`,
not architecture prototyping.

The second gate is `Xunce Current-HEAD Evidence Refresh v1`, with runner
`scripts/run_xunce_current_head_evidence_refresh.py`, config
`configs/xunce_current_head_evidence_refresh_v1.json`, and output root
`outputs/path_feedback_batch_xunce_current_head_evidence_refresh_v1/`. It checks
the Stage 0 freeze summary, controlled-installation evidence, real-map replay,
real-map multi-ROI evidence, and network-readiness evidence against the current
checkout. Declared dirty or mismatched git provenance blocks the research line;
legacy summaries without provenance are counted explicitly and do not silently
become Current-HEAD proof. A passing summary writes
`next_required_change=network_literature_bottleneck_review`.

The third gate is `Xunce Network Literature Bottleneck Review v1`, with runner
`scripts/run_xunce_network_literature_bottleneck_review.py`, config
`configs/xunce_network_literature_bottleneck_review_v1.json`, and output root
`outputs/path_feedback_batch_xunce_network_literature_bottleneck_review_v1/`.
It maps primary literature to actual project bottlenecks and reports whether
the evidence indicates coverage/generalization, fallback/policy regression,
latency/parameter evidence, topology representation, or no current
release-blocking network bottleneck. A passing summary writes
`next_required_change=topology_observation_contract`; this is an additive
observation-contract gate, not a training approval.

This research track must not publish a checkpoint, replace default policy,
connect a real executor, start online canary traffic, run PPO update in the
design/audit stages, modify action space/default A*, claim real-world
performance, or treat path-planner/IRIS/GCS diagnostics as release proof.

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

The eighth concrete implementation target for this line is:

```text
Global 99 Shadow Canary Preflight v1
```

Acceptance:

- Reads `configs/global_99_shadow_canary_preflight_v1.json`.
- Consumes existing release-governance, multi-map, and policy-guided summary
  artifacts.
- Writes summary, manifest, shadow replay audit, canary eligibility audit,
  boundary audit, kill-switch audit, rollback audit, telemetry audit, rejection
  report, and report artifacts under
  `outputs/path_feedback_batch_global_99_shadow_canary_preflight_v1/`.
- Summary reports release-governance status/verdict, required scenario counts,
  aggregate and minimum coverage, policy guidance counters,
  `policy_guard_fallback_rate`, audit pass/fail flags,
  `shadow_canary_preflight_verdict`, `next_required_change`, and all required
  boundary flags.
- If release governance is missing, failed, or not pointing to
  `global_99_shadow_canary_preflight`, the summary fails and points to
  `fix_global_99_release_governance_preflight`.
- If required multi-map scenarios are not healthy or coverage is below 0.99,
  the summary fails and points to `fix_global_99_multi_map_generalization`.
- If policy regression appears, the summary fails and points to
  `fix_policy_guided_global_coverage`.
- If policy guard fallback rate is above threshold, the summary fails and
  points to `fix_global_99_shadow_canary_guard_fallback`.
- If any checkpoint/default-policy/executor/online-canary/traffic boundary is
  open, the summary fails and points to
  `resolve_global_99_shadow_canary_boundary_rejections`.
- If evidence is stable and boundaries are closed, the summary passes with
  `shadow_canary_preflight_verdict=eligible_for_global_99_shadow_canary_replay`
  and `next_required_change=global_99_shadow_canary_replay`.
- Project docs stay aligned with this development order.

The ninth concrete implementation target for this line is:

```text
Global 99 Shadow Canary Replay v1
```

Acceptance:

- Reads `configs/global_99_shadow_canary_replay_v1.json`.
- Consumes existing shadow/canary preflight and multi-map source artifacts.
- Reruns the deterministic synthetic multi-map policy-guided matrix and writes
  replay scenario results, family summary, source-match audit,
  policy-vs-baseline audit, boundary audit, kill-switch audit, rollback audit,
  telemetry audit, rejection report, manifest, summary, and report artifacts
  under `outputs/path_feedback_batch_global_99_shadow_canary_replay_v1/`.
- Summary reports preflight status/verdict, replay scenario counts, aggregate
  and minimum coverage, source-match deltas, policy guidance counters,
  `policy_guard_fallback_rate`, audit pass/fail flags, `next_required_change`,
  and all required boundary flags.
- If preflight is missing, failed, or not pointing to
  `global_99_shadow_canary_replay`, the summary fails and points to
  `fix_global_99_shadow_canary_preflight`.
- If multi-map source evidence is missing or invalid, the summary fails and
  points to `fix_global_99_multi_map_generalization`.
- If replay required scenarios are not healthy or coverage is below 0.99, the
  summary fails and points to `fix_global_99_shadow_canary_replay`.
- If source-match coverage/path-cost deltas exceed tolerance, the summary fails
  and points to `fix_global_99_shadow_canary_replay_determinism`.
- If policy regression appears, the summary fails and points to
  `fix_policy_guided_global_coverage`.
- If policy guard fallback rate is above threshold, the summary fails and
  points to `fix_global_99_shadow_canary_guard_fallback`.
- If any checkpoint/default-policy/executor/online-canary/traffic boundary is
  open, the summary fails and points to
  `resolve_global_99_shadow_canary_replay_boundary_rejections`.
- If replay is deterministic and boundaries are closed, the summary passes with
  `next_required_change=global_99_real_map_preflight`.
- Project docs stay aligned with this development order.

The tenth concrete implementation target for this line is:

```text
Global 99 Real Map Preflight v1
```

Acceptance:

- Reads `configs/global_99_real_map_preflight_v1.json`.
- Consumes existing shadow/canary replay summary from
  `outputs/path_feedback_batch_global_99_shadow_canary_replay_v1/`.
- Consumes existing LOLA quasi-real domain-gap, path-feedback, and bridge
  summaries from `outputs/path_feedback_batch_quasi_real_map_domain_gap_v1/`,
  plus optional semi-real path-feedback validation context from
  `outputs/path_feedback_validation/`.
- Writes summary, manifest, evidence-lineage audit, domain-gap audit,
  path-feedback audit, scope audit, boundary audit, kill-switch audit, rollback
  audit, telemetry audit, rejection report, and report artifacts under
  `outputs/path_feedback_batch_global_99_real_map_preflight_v1/`.
- Summary reports shadow replay status/next change, domain-gap status/verdict,
  slice/ROI counts, context identity counters, open-grid/path-feedback
  regression counters, shadow replay source-match and fallback-rate evidence,
  audit pass/fail flags, `next_required_change`, and explicit normalized
  boundary flags.
- If shadow replay is missing, failed, or not pointing to
  `global_99_real_map_preflight`, the summary fails and points to
  `fix_global_99_shadow_canary_replay`.
- If domain-gap evidence is missing, failed, or not
  `acceptable_for_next_pilot`, the summary fails and points to
  `fix_quasi_real_map_domain_gap_evidence`.
- If slice count or ROI group count is below the configured minimum, the
  summary fails and points to `expand_real_map_roi_coverage`.
- If context IDs are missing or legacy identity fallback appears, the summary
  fails and points to `fix_real_map_context_identity`.
- If open-grid fallback, safety/contract/path-cost/risk/source-selection
  regression, or path-feedback contract gaps appear, the summary fails and
  points to `fix_real_map_path_feedback_contract`.
- If shadow replay fallback rate exceeds threshold, the summary fails and
  points to `fix_global_99_shadow_canary_guard_fallback`.
- If any publication/default-policy/executor/online-canary boundary is open,
  the summary fails and points to
  `resolve_global_99_real_map_preflight_boundary_rejections`.
- If evidence is stable and boundaries are closed, the summary passes with
  `real_map_preflight_verdict=eligible_for_global_99_real_map_shadow_replay`
  and `next_required_change=global_99_real_map_shadow_replay`.
- Project docs stay aligned with this development order.

The eleventh concrete implementation target for this line is:

```text
Global 99 Real Map Shadow Replay v1
```

Acceptance:

- Reads `configs/global_99_real_map_shadow_replay_v1.json`.
- Consumes the real-map preflight summary from
  `outputs/path_feedback_batch_global_99_real_map_preflight_v1/`.
- Consumes existing quasi-real path-feedback manifest, summary, and slice
  evidence from `outputs/path_feedback_batch_quasi_real_map_domain_gap_v1/`.
- Builds an isolated replay path-feedback manifest whose output paths are under
  `outputs/path_feedback_batch_global_99_real_map_shadow_replay_v1/`, so source
  evidence cannot be overwritten.
- Runs `model_explorer path-feedback validate/run` in offline replay mode, then
  compares source and replay scenario evidence by `scenario_id`.
- Writes summary, manifest, replay path-feedback manifest, replay
  path-feedback summary, scenario results, source-match audit, context audit,
  boundary audit, kill-switch audit, rollback audit, telemetry audit, rejection
  report, and report artifacts under
  `outputs/path_feedback_batch_global_99_real_map_shadow_replay_v1/`.
- Summary reports preflight and domain-gap source statuses, source/replay
  scenario counts, slice/ROI/context counters, source-match deltas,
  selected-cell and selection-change mismatch counters, open-grid fallback and
  path-feedback regression counters, audit pass/fail flags,
  `path_planner_use_scope=offline_path_feedback_replay_only`,
  `next_required_change`, and all required boundary flags.
- If preflight is missing, failed, or not pointing to
  `global_99_real_map_shadow_replay`, the summary fails and points to
  `fix_global_99_real_map_preflight`.
- If source manifest, source slices, or source path-feedback summary is missing
  or invalid, the summary fails and points to
  `fix_quasi_real_map_domain_gap_evidence`.
- If replay validation or run fails, the summary fails and points to
  `fix_global_99_real_map_shadow_replay`.
- If source-match deltas exceed tolerance, the summary fails and points to
  `fix_global_99_real_map_shadow_replay_determinism`.
- If context IDs are missing or legacy identity fallback appears, the summary
  fails and points to `fix_real_map_context_identity`.
- If open-grid fallback or safety/contract/path-cost/risk/source-selection
  regressions appear, the summary fails and points to
  `fix_real_map_path_feedback_contract`.
- If any publication/default-policy/executor/online-canary boundary is open,
  the summary fails and points to
  `resolve_global_99_real_map_shadow_replay_boundary_rejections`.
- If evidence is stable and boundaries are closed, the summary passes with
  `real_map_shadow_replay_verdict=eligible_for_global_99_real_map_release_governance_preflight`
  and `next_required_change=global_99_real_map_release_governance_preflight`.
- Project docs stay aligned with this development order.

The twelfth concrete implementation target for this line is:

```text
Global 99 Real Map Release Governance Preflight v1
```

Acceptance:

- Reads `configs/global_99_real_map_release_governance_preflight_v1.json`.
- Consumes Real Map Shadow Replay, Real Map Preflight, and quasi-real
  domain-gap/path-feedback summaries.
- Writes summary, manifest, lineage audit, shadow replay audit, preflight
  audit, domain-gap audit, path-feedback audit, scope audit, boundary audit,
  kill-switch audit, rollback audit, telemetry audit, rejection report, and
  report artifacts under
  `outputs/path_feedback_batch_global_99_real_map_release_governance_preflight_v1/`.
- Summary reports source statuses, source-match deltas, slice/ROI/context
  counters, open-grid/path-feedback regression counters, fallback rate, audit
  pass/fail flags, `audited_path_planner_use_scope`,
  `next_required_change`, and all required boundary flags.
- If shadow replay is missing, failed, or not pointing to this stage, the
  summary fails and points to `fix_global_99_real_map_shadow_replay`.
- If real-map preflight is missing or failed, the summary fails and points to
  `fix_global_99_real_map_preflight`.
- If domain-gap evidence is missing or unacceptable, the summary fails and
  points to `fix_quasi_real_map_domain_gap_evidence`.
- If source-match audit fails, the summary points to
  `fix_global_99_real_map_shadow_replay_determinism`.
- If context IDs or path-feedback contract evidence regress, the summary points
  to `fix_real_map_context_identity` or `fix_real_map_path_feedback_contract`.
- If boundary fields open, the summary points to
  `resolve_global_99_real_map_release_governance_boundary_rejections`.
- If evidence is stable and boundaries are closed, the summary passes with
  `real_map_release_governance_verdict=eligible_for_global_99_real_map_shadow_canary_preflight`
  and `next_required_change=global_99_real_map_shadow_canary_preflight`.
- Project docs stay aligned with this development order.

The thirteenth concrete implementation target for this line is:

```text
Global 99 Real Map Shadow Canary Preflight v1
```

Acceptance:

- Reads `configs/global_99_real_map_shadow_canary_preflight_v1.json`.
- Consumes real-map release-governance and real-map shadow-replay summaries.
- Writes summary, manifest, shadow replay eligibility audit, canary eligibility
  audit, boundary audit, kill-switch audit, rollback audit, telemetry audit,
  rejection report, and report artifacts under
  `outputs/path_feedback_batch_global_99_real_map_shadow_canary_preflight_v1/`.
- Summary reports release-governance status/next change, source-match evidence,
  open-grid fallback, policy guard fallback rate, audit pass/fail flags,
  `canary_traffic_fraction=0.0`, `starts_online_canary=false`,
  `next_required_change`, and all required boundary flags.
- If release-governance evidence is missing, failed, or not pointing to this
  stage, it points to `fix_global_99_real_map_release_governance_preflight`.
- If source-match evidence fails, it points to
  `fix_global_99_real_map_shadow_replay_determinism`.
- If fallback rate exceeds threshold, it points to
  `fix_global_99_real_map_shadow_canary_guard_fallback`.
- If any online-canary/executor/default-policy/checkpoint boundary opens, it
  points to `resolve_global_99_real_map_shadow_canary_boundary_rejections`.
- If evidence is stable and boundaries are closed, the summary passes with
  `real_map_shadow_canary_preflight_verdict=eligible_for_global_99_real_map_shadow_canary_replay`
  and `next_required_change=global_99_real_map_shadow_canary_replay`.
- Project docs stay aligned with this development order.

The fourteenth concrete implementation target for this line is:

```text
Global 99 Real Map Shadow Canary Replay v1
```

Acceptance:

- Reads `configs/global_99_real_map_shadow_canary_replay_v1.json`.
- Consumes real-map shadow/canary preflight, real-map release-governance, and
  real-map shadow-replay summaries.
- Writes summary, manifest, source-match audit, fallback audit, boundary audit,
  kill-switch audit, rollback audit, telemetry audit, rejection report, and
  report artifacts under
  `outputs/path_feedback_batch_global_99_real_map_shadow_canary_replay_v1/`.
- Summary reports all source statuses/next changes, source-match deltas,
  scenario mismatch count, open-grid fallback, policy guard fallback rate,
  audit pass/fail flags, `canary_traffic_fraction=0.0`,
  `starts_online_canary=false`, `next_required_change`, and all required
  boundary flags.
- If shadow/canary preflight evidence is missing, failed, or not pointing to
  this stage, it points to `fix_global_99_real_map_shadow_canary_preflight`.
- If deterministic source-match evidence fails, it points to
  `fix_global_99_real_map_shadow_canary_replay_determinism`.
- If open-grid fallback appears, it points to
  `fix_real_map_path_feedback_contract`.
- If fallback rate exceeds threshold, it points to
  `fix_global_99_real_map_shadow_canary_guard_fallback`.
- If any online-canary/executor/default-policy/checkpoint boundary opens, it
  points to `resolve_global_99_real_map_shadow_canary_replay_boundary_rejections`.
- If evidence is stable and boundaries are closed, the summary passes with
  `real_map_shadow_canary_replay_verdict=eligible_for_global_99_real_map_evidence_refresh_drift_audit`
  and `next_required_change=global_99_real_map_evidence_refresh_drift_audit`.
- Project docs stay aligned with this development order.

The fifteenth concrete implementation target for this line is:

```text
Global 99 Real Map Evidence Refresh / Drift Audit v1
```

Acceptance:

- Reads `configs/global_99_real_map_evidence_refresh_drift_audit_v1.json`.
- Consumes real-map shadow/canary replay, real-map shadow replay, and
  quasi-real domain-gap/path-feedback manifest, summary, bridge, and slice
  artifacts.
- Audits evidence lineage, file fingerprints, manifest/slice scenario IDs,
  contract/sidecar path existence, context IDs, legacy identity fallback,
  source-match deltas, fallback/regression counters, and boundary state.
- Allows upstream `path_planner_use_scope=offline_path_feedback_replay_only`
  evidence, but this stage itself reports `uses_path_planner=false`.
- Writes summary, manifest, lineage audit, fingerprint audit, manifest-sidecar
  audit, context audit, source-match drift audit, boundary audit, kill-switch
  audit, rollback audit, telemetry audit, rejection report, and report
  artifacts under
  `outputs/path_feedback_batch_global_99_real_map_evidence_refresh_drift_audit_v1/`.
- Summary reports source statuses/next changes, domain-gap verdict, slice/ROI
  counts, manifest scenario count, scenario ID mismatch count, missing
  contract/sidecar counts, context counters, source-match deltas, fingerprint
  counts, audit pass/fail flags, `next_required_change`, and all required
  boundary flags.
- If shadow/canary replay evidence is missing, failed, or not pointing to this
  stage, it points to `fix_global_99_real_map_shadow_canary_replay`.
- If domain-gap evidence is missing or unacceptable, it points to
  `fix_quasi_real_map_domain_gap_evidence`.
- If manifest/slice IDs or fingerprints drift, it points to
  `fix_global_99_real_map_evidence_drift`.
- If context IDs are missing or legacy fallback appears, it points to
  `fix_real_map_context_identity`.
- If contract/sidecar paths, open-grid fallback, or path-feedback regression
  fail, it points to `fix_real_map_path_feedback_contract`.
- If boundaries open, it points to
  `resolve_global_99_real_map_evidence_refresh_boundary_rejections`.
- If evidence is stable and boundaries are closed, the summary passes with
  `evidence_refresh_drift_verdict=eligible_for_global_99_real_map_multi_roi_generalization`
  and `next_required_change=global_99_real_map_multi_roi_generalization`.
- Project docs stay aligned with this development order.

The sixteenth concrete implementation target for this line is:

```text
Global 99 Real Map Multi-ROI Generalization v1
```

Acceptance:

- Reads `configs/global_99_real_map_multi_roi_generalization_v1.json`.
- Consumes evidence refresh/drift audit and quasi-real domain-gap slice
  artifacts.
- Audits ROI group coverage, per-ROI split coverage, required scenarios,
  context IDs, legacy identity fallback, contract/sidecar availability,
  fallback/regression counters, and boundary state.
- Writes summary, manifest, scenario results JSONL, ROI-family summary,
  lineage audit, scenario matrix audit, boundary audit, kill-switch audit,
  rollback audit, telemetry audit, rejection report, and report artifacts under
  `outputs/path_feedback_batch_global_99_real_map_multi_roi_generalization_v1/`.
- Summary reports source statuses/next changes, domain-gap verdict, slice/ROI
  counts, passed/failed ROI group counts, passed/failed required scenario
  counts, split coverage, context counters, missing contract/sidecar counters,
  fallback/regression counters, audit pass/fail flags, `next_required_change`,
  and all required boundary flags.
- If drift evidence is missing, failed, or not pointing to this stage, it
  points to `fix_global_99_real_map_evidence_refresh_drift_audit`.
- If domain-gap evidence is missing or unacceptable, it points to
  `fix_quasi_real_map_domain_gap_evidence`.
- If ROI/slice/split coverage is insufficient, it points to
  `expand_real_map_roi_coverage`.
- If context IDs are missing or legacy fallback appears, it points to
  `fix_real_map_context_identity`.
- If contract/sidecar paths, open-grid fallback, or path-feedback regression
  fail, it points to `fix_real_map_path_feedback_contract`.
- If boundaries open, it points to
  `resolve_global_99_real_map_multi_roi_boundary_rejections`.
- If all required ROI groups and scenarios pass, the summary passes with
  `real_map_multi_roi_generalization_verdict=eligible_for_default_policy_candidate_authorization_preflight`
  and `next_required_change=default_policy_candidate_authorization_preflight`.
- Project docs stay aligned with this development order.

The seventeenth concrete implementation target for this line is:

```text
Global 99 Default Policy Candidate Authorization Preflight v1
```

Acceptance:

- Reads `configs/global_99_default_policy_candidate_authorization_preflight_v1.json`.
- Consumes the Global 99 multi-ROI generalization summary.
- Audits candidate read-only status, default-policy read-only status,
  executor isolation, path-planner isolation, kill-switch, rollback, telemetry,
  scope stability, and release boundaries.
- Writes summary, manifest, lineage audit, scope audit, candidate read-only
  audit, default-policy read-only audit, isolation audit, kill-switch audit,
  rollback audit, telemetry audit, rejection report, and report artifacts under
  `outputs/path_feedback_batch_global_99_default_policy_candidate_authorization_preflight_v1/`.
- Summary reports source status/next change, multi-ROI counts, audit pass/fail
  flags, `default_policy_candidate_authorization_preflight_passed`,
  `default_policy_candidate_installation_approved=false`,
  `next_required_change`, and all required boundary flags.
- If multi-ROI evidence is missing, failed, or not pointing to this stage, it
  points to `fix_global_99_real_map_multi_roi_generalization`.
- If candidate/default-policy read-only, executor/path-planner isolation,
  kill-switch, rollback, telemetry, or release boundaries fail, it points to
  `resolve_default_policy_candidate_authorization_rejections`.
- If authorization evidence is stable and boundaries are closed, the summary
  passes with
  `authorization_verdict=eligible_for_sandbox_candidate_installation_dry_run`
  and `next_required_change=sandbox_candidate_installation_dry_run`.
- Project docs stay aligned with this development order.

The eighteenth concrete implementation target for this line is:

```text
Global 99 Sandbox Candidate Installation Dry Run v1
```

Acceptance:

- Reads `configs/global_99_sandbox_candidate_installation_dry_run_v1.json`.
- Consumes Global 99 default-policy candidate authorization preflight summary.
- Writes a sandbox-only candidate descriptor and verifies provenance, SHA-256,
  load, rollback, kill-switch, telemetry, and boundary state.
- Writes summary, manifest, provenance audit, hash audit, load audit, rollback
  audit, kill-switch audit, telemetry audit, boundary audit, rejection report,
  and report artifacts under
  `outputs/path_feedback_batch_global_99_sandbox_candidate_installation_dry_run_v1/`.
- Summary reports source authorization status/next change, candidate hash/size,
  audit pass/fail flags, `default_policy_candidate_installation_approved=false`,
  `next_required_change`, and all required boundary flags.
- If authorization evidence is missing, failed, or not pointing to this stage,
  it points to `fix_default_policy_candidate_authorization_preflight`.
- If sandbox hash/provenance/load/rollback/kill-switch/telemetry fail, it
  points to `fix_sandbox_candidate_installation_dry_run`.
- If boundaries open, it points to
  `resolve_sandbox_candidate_installation_boundary_rejections`.
- If the sandbox dry run is stable and boundaries are closed, the summary
  passes with `sandbox_installation_verdict=eligible_for_sandbox_consumer_replay_canary`
  and `next_required_change=sandbox_consumer_replay_canary`.
- Project docs stay aligned with this development order.

The nineteenth concrete implementation target for this line is:

```text
Global 99 Sandbox Consumer Replay / Canary v1
```

Acceptance:

- Reads `configs/global_99_sandbox_consumer_replay_canary_v1.json`.
- Consumes Global 99 sandbox candidate installation dry-run summary.
- Generates a sandbox replay trace and audits candidate load, fallback,
  telemetry, rollback, controlled regressions, and boundary state.
- Writes summary, manifest, trace JSONL, load audit, fallback audit, telemetry
  audit, rollback audit, boundary audit, rejection report, and report artifacts
  under `outputs/path_feedback_batch_global_99_sandbox_consumer_replay_canary_v1/`.
- Summary reports source install status/next change, consumer step count,
  fallback rate, controlled regression count, audit pass/fail flags,
  `default_policy_candidate_installation_approved=false`,
  `next_required_change`, and all required boundary flags.
- If sandbox install evidence is missing, failed, or not pointing to this
  stage, it points to `fix_sandbox_candidate_installation_dry_run`.
- If consumer load/fallback/telemetry/rollback evidence fails, it points to
  `fix_sandbox_consumer_replay_canary`.
- If boundaries open, it points to
  `resolve_sandbox_consumer_boundary_rejections`.
- If sandbox consumer replay/canary is stable and boundaries are closed, the
  summary passes with
  `sandbox_consumer_verdict=eligible_for_controlled_default_policy_candidate_installation_preflight`
  and `next_required_change=controlled_default_policy_candidate_installation_preflight`.
- Project docs stay aligned with this development order.

The twentieth concrete implementation target for this line is:

```text
Global 99 Controlled Default Policy Candidate Installation Preflight v1
```

Acceptance:

- Reads `configs/global_99_controlled_default_policy_candidate_installation_preflight_v1.json`.
- Consumes Global 99 sandbox consumer replay/canary summary.
- Audits review-only scope, default-policy read-only status, executor
  isolation, online-canary disabled state, kill-switch, rollback, telemetry,
  and boundary state.
- Writes summary, manifest, lineage audit, review-scope audit, boundary audit,
  kill-switch audit, rollback audit, telemetry audit, rejection report, and
  report artifacts under
  `outputs/path_feedback_batch_global_99_controlled_default_policy_candidate_installation_preflight_v1/`.
- Summary reports source consumer status/next change, audit pass/fail flags,
  `controlled_installation_executed=false`,
  `default_policy_replacement_approved=false`, `next_required_change`, and all
  required boundary flags.
- If sandbox consumer evidence is missing, failed, or not pointing to this
  stage, it points to `fix_sandbox_consumer_replay_canary`.
- If review scope, default-policy read-only, executor isolation, online canary,
  kill-switch, rollback, telemetry, or boundaries fail, it points to
  `resolve_controlled_default_policy_candidate_installation_preflight_rejections`.
- If all audits pass, the summary passes with
  `controlled_installation_verdict=eligible_for_controlled_default_policy_candidate_installation_review`
  and `next_required_change=eligible_for_controlled_default_policy_candidate_installation_review`.
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
  tests/test_global_99_release_governance_preflight.py \
  tests/test_global_99_shadow_canary_preflight.py \
  tests/test_global_99_shadow_canary_replay.py \
  tests/test_global_99_real_map_preflight.py \
  tests/test_global_99_real_map_shadow_replay.py \
  tests/test_global_99_real_map_release_governance_preflight.py \
  tests/test_global_99_real_map_shadow_canary_preflight.py \
  tests/test_global_99_real_map_shadow_canary_replay.py \
  tests/test_global_99_real_map_evidence_refresh_drift_audit.py \
  tests/test_global_99_real_map_multi_roi_generalization.py \
  tests/test_global_99_default_policy_candidate_authorization_preflight.py \
  tests/test_global_99_sandbox_candidate_installation_dry_run.py \
  tests/test_global_99_sandbox_consumer_replay_canary.py \
  tests/test_global_99_controlled_default_policy_candidate_installation_preflight.py -q
PYTHON=$PY bash scripts/run_global_99_coverage_benchmark.sh
PYTHON=$PY bash scripts/run_frontier_coverage_planner_baseline.sh
PYTHON=$PY bash scripts/run_coverage_memory_replanning_loop.sh
PYTHON=$PY bash scripts/run_policy_guided_global_coverage.sh
PYTHON=$PY bash scripts/run_global_99_multi_map_generalization.sh
PYTHON=$PY bash scripts/run_network_architecture_upgrade_readiness_review.sh
PYTHON=$PY bash scripts/run_global_99_release_governance_preflight.sh
PYTHON=$PY bash scripts/run_global_99_shadow_canary_preflight.sh
PYTHON=$PY bash scripts/run_global_99_shadow_canary_replay.sh
PYTHON=$PY bash scripts/run_global_99_real_map_preflight.sh
PYTHON=$PY bash scripts/run_global_99_real_map_shadow_replay.sh
PYTHON=$PY bash scripts/run_global_99_real_map_release_governance_preflight.sh
PYTHON=$PY bash scripts/run_global_99_real_map_shadow_canary_preflight.sh
PYTHON=$PY bash scripts/run_global_99_real_map_shadow_canary_replay.sh
PYTHON=$PY bash scripts/run_global_99_real_map_evidence_refresh_drift_audit.sh
PYTHON=$PY bash scripts/run_global_99_real_map_multi_roi_generalization.sh
PYTHON=$PY bash scripts/run_global_99_default_policy_candidate_authorization_preflight.sh
PYTHON=$PY bash scripts/run_global_99_sandbox_candidate_installation_dry_run.sh
PYTHON=$PY bash scripts/run_global_99_sandbox_consumer_replay_canary.sh
PYTHON=$PY bash scripts/run_global_99_controlled_default_policy_candidate_installation_preflight.sh
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
jq '{status,reason_codes,shadow_canary_preflight_verdict,next_required_change,shadow_replay_audit_passed,canary_eligibility_audit_passed,boundary_audit_passed,kill_switch_audit_passed,rollback_audit_passed,telemetry_audit_passed,publishes_checkpoint,replaces_default_policy,connects_real_executor,starts_online_canary,canary_traffic_fraction}' \
  outputs/path_feedback_batch_global_99_shadow_canary_preflight_v1/global-99-shadow-canary-preflight-summary.json
jq '{status,reason_codes,shadow_replay_passed,offline_canary_replay_passed,source_match_audit_passed,next_required_change,max_replay_coverage_delta,policy_guard_fallback_rate,publishes_checkpoint,replaces_default_policy,connects_real_executor,starts_online_canary,canary_traffic_fraction}' \
  outputs/path_feedback_batch_global_99_shadow_canary_replay_v1/global-99-shadow-canary-replay-summary.json
jq '{status,reason_codes,real_map_preflight_verdict,next_required_change,domain_gap_verdict,slice_count,roi_group_count,fallback_or_open_grid_count,boundary_audit_passed,real_world_release_approved,real_world_performance_claimed,replaces_default_policy,connects_real_executor,starts_online_canary,canary_traffic_fraction}' \
  outputs/path_feedback_batch_global_99_real_map_preflight_v1/global-99-real-map-preflight-summary.json
jq '{status,reason_codes,real_map_shadow_replay_verdict,next_required_change,source_match_audit_passed,scenario_mismatch_count,max_replay_coverage_delta,max_replay_path_cost_delta_m,open_grid_fallback_used,connects_real_executor,starts_online_canary,canary_traffic_fraction,replaces_default_policy}' \
  outputs/path_feedback_batch_global_99_real_map_shadow_replay_v1/global-99-real-map-shadow-replay-summary.json
jq '{status,reason_codes,real_map_release_governance_verdict,next_required_change,source_match_audit_passed,scenario_mismatch_count,open_grid_fallback_used,policy_guard_fallback_rate,release_boundary_audit_passed,connects_real_executor,starts_online_canary,replaces_default_policy,uses_path_planner,audited_path_planner_use_scope}' \
  outputs/path_feedback_batch_global_99_real_map_release_governance_preflight_v1/global-99-real-map-release-governance-preflight-summary.json
jq '{status,reason_codes,real_map_shadow_canary_preflight_verdict,next_required_change,source_match_audit_passed,scenario_mismatch_count,open_grid_fallback_used,policy_guard_fallback_rate,boundary_audit_passed,starts_online_canary,canary_traffic_fraction,connects_real_executor,replaces_default_policy}' \
  outputs/path_feedback_batch_global_99_real_map_shadow_canary_preflight_v1/global-99-real-map-shadow-canary-preflight-summary.json
jq '{status,reason_codes,real_map_shadow_canary_replay_verdict,next_required_change,source_match_audit_passed,scenario_mismatch_count,max_replay_coverage_delta,max_replay_path_cost_delta_m,open_grid_fallback_used,policy_guard_fallback_rate,boundary_audit_passed,starts_online_canary,canary_traffic_fraction,connects_real_executor,replaces_default_policy}' \
  outputs/path_feedback_batch_global_99_real_map_shadow_canary_replay_v1/global-99-real-map-shadow-canary-replay-summary.json
jq '{status,reason_codes,evidence_refresh_drift_verdict,next_required_change,source_domain_gap_status,domain_gap_verdict,slice_count,roi_group_count,scenario_id_mismatch_count,missing_contract_count,missing_sidecar_count,context_id_missing_count,legacy_identity_fallback_count,boundary_audit_passed,connects_real_executor,starts_online_canary,replaces_default_policy,uses_path_planner}' \
  outputs/path_feedback_batch_global_99_real_map_evidence_refresh_drift_audit_v1/global-99-real-map-evidence-refresh-drift-audit-summary.json
jq '{status,reason_codes,real_map_multi_roi_generalization_verdict,next_required_change,slice_count,roi_group_count,passed_roi_group_count,failed_roi_group_count,passed_required_scenario_count,failed_required_scenario_count,split_coverage_complete,context_id_missing_count,missing_contract_count,missing_sidecar_count,boundary_audit_passed,connects_real_executor,starts_online_canary,replaces_default_policy}' \
  outputs/path_feedback_batch_global_99_real_map_multi_roi_generalization_v1/global-99-real-map-multi-roi-generalization-summary.json
jq '{status,reason_codes,authorization_verdict,next_required_change,source_multi_roi_status,default_policy_candidate_authorization_preflight_passed,candidate_read_only_audit_passed,default_policy_read_only_audit_passed,executor_isolation_audit_passed,path_planner_isolation_audit_passed,kill_switch_audit_passed,rollback_audit_passed,telemetry_audit_passed,boundary_audit_passed,replaces_default_policy,connects_real_executor,default_policy_candidate_installation_approved}' \
  outputs/path_feedback_batch_global_99_default_policy_candidate_authorization_preflight_v1/global-99-default-policy-candidate-authorization-preflight-summary.json
jq '{status,reason_codes,sandbox_installation_verdict,next_required_change,source_authorization_status,sandbox_candidate_installation_dry_run_passed,sandbox_candidate_hash_audit_passed,sandbox_candidate_load_audit_passed,rollback_audit_passed,kill_switch_audit_passed,telemetry_audit_passed,boundary_audit_passed,replaces_default_policy,connects_real_executor,default_policy_candidate_installation_approved}' \
  outputs/path_feedback_batch_global_99_sandbox_candidate_installation_dry_run_v1/global-99-sandbox-candidate-installation-dry-run-summary.json
jq '{status,reason_codes,sandbox_consumer_verdict,next_required_change,source_sandbox_install_status,sandbox_consumer_replay_canary_passed,consumer_step_count,fallback_rate,controlled_regression_count,candidate_load_audit_passed,fallback_audit_passed,telemetry_audit_passed,rollback_audit_passed,boundary_audit_passed,replaces_default_policy,connects_real_executor,starts_online_canary,default_policy_candidate_installation_approved}' \
  outputs/path_feedback_batch_global_99_sandbox_consumer_replay_canary_v1/global-99-sandbox-consumer-replay-canary-summary.json
jq '{status,reason_codes,controlled_installation_verdict,next_required_change,source_sandbox_consumer_status,controlled_default_policy_candidate_installation_preflight_passed,review_scope_audit_passed,default_policy_read_only_audit_passed,executor_isolation_audit_passed,online_canary_audit_passed,controlled_installation_executed,replaces_default_policy,connects_real_executor,starts_online_canary}' \
  outputs/path_feedback_batch_global_99_controlled_default_policy_candidate_installation_preflight_v1/global-99-controlled-default-policy-candidate-installation-preflight-summary.json
rg -n "Global 99% Coverage Benchmark v1|run_global_99_coverage_benchmark|Frontier Coverage Planner Baseline v1|run_frontier_coverage_planner_baseline|Coverage Memory \\+ Replanning Loop v1|run_coverage_memory_replanning_loop|Policy-Guided Global Coverage v1|run_policy_guided_global_coverage|Global 99 Multi-Map Generalization v1|run_global_99_multi_map_generalization|Network Architecture Upgrade Readiness Review v1|run_network_architecture_upgrade_readiness_review|Global 99 Release Governance Preflight v1|run_global_99_release_governance_preflight|Global 99 Shadow Canary Preflight v1|run_global_99_shadow_canary_preflight|Global 99 Shadow Canary Replay v1|run_global_99_shadow_canary_replay|Global 99 Real Map Preflight v1|run_global_99_real_map_preflight|Global 99 Real Map Shadow Replay v1|run_global_99_real_map_shadow_replay|Global 99 Real Map Release Governance Preflight v1|run_global_99_real_map_release_governance_preflight|Global 99 Real Map Shadow Canary Preflight v1|run_global_99_real_map_shadow_canary_preflight|Global 99 Real Map Shadow Canary Replay v1|run_global_99_real_map_shadow_canary_replay|Global 99 Real Map Evidence Refresh / Drift Audit v1|run_global_99_real_map_evidence_refresh_drift_audit|Global 99 Real Map Multi-ROI Generalization v1|run_global_99_real_map_multi_roi_generalization|Global 99 Default Policy Candidate Authorization Preflight v1|run_global_99_default_policy_candidate_authorization_preflight|Global 99 Sandbox Candidate Installation Dry Run v1|run_global_99_sandbox_candidate_installation_dry_run|Global 99 Sandbox Consumer Replay / Canary v1|run_global_99_sandbox_consumer_replay_canary|Global 99 Controlled Default Policy Candidate Installation Preflight v1|run_global_99_controlled_default_policy_candidate_installation_preflight|eligible_for_controlled_default_policy_candidate_installation_review|network_architecture_upgrade_v1" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md
git diff --check
```

## Non-Goals

No PPO training, no checkpoint publication, no default policy replacement, no
real executor connection, no guard relaxation, no default A* replacement, no
action-space change, no network architecture change, no real-world performance
claim, no Ackermann-feasible trajectory claim, and no treating
IRIS/GCS/path-planner diagnostics as release proof. `Policy-Guided Global
Coverage v1` permits read-only experimental checkpoint inference only. `Global
99 Shadow Canary Preflight v1` permits canary eligibility auditing only; it
does not start online canary traffic. `Global 99 Shadow Canary Replay v1`
permits offline synthetic replay only; it does not use real maps or real
executors. `Global 99 Real Map Preflight v1` audits existing LOLA quasi-real
ROI and path-feedback sidecar evidence only; it does not perform real-map
takeover, run online canary traffic, connect an executor, or claim real-world
performance. `Global 99 Real Map Shadow Replay v1` permits offline quasi-real
path-feedback / path-planner route replay only in an isolated output root; it
does not connect a real executor, start online canary traffic, install default
policy, or claim real-world performance. `Global 99 Real Map Release Governance
Preflight v1` audits that evidence chain and does not invoke path-planner,
connect an executor, start online canary traffic, install default policy, or
claim real-world performance. `Global 99 Real Map Shadow Canary Preflight v1`
checks offline shadow/canary replay eligibility only; canary traffic remains
zero and no online canary starts. `Global 99 Real Map Shadow Canary Replay v1`
replays already approved offline shadow/canary evidence and keeps
`canary_traffic_fraction=0.0`; it does not start online canary traffic, connect
an executor, install default policy, publish a checkpoint, or claim real-world
performance. `Global 99 Real Map Evidence Refresh / Drift Audit v1` re-audits
manifest, sidecar, context ID, fingerprint, and source-match evidence only; it
does not rerun path-planner, connect an executor, start online canary traffic,
install default policy, or claim real-world performance. `Global 99 Real Map
Multi-ROI Generalization v1` audits quasi-real ROI/slice/split coverage only;
it does not install or authorize default-policy replacement, connect an
executor, start online canary traffic, or claim real-world performance. `Global
99 Default Policy Candidate Authorization Preflight v1` authorizes only a
sandbox dry-run candidate path; it does not install the candidate, replace
default policy, connect an executor, publish a checkpoint, or claim real-world
performance. `Global 99 Sandbox Candidate Installation Dry Run v1` writes and
loads a sandbox-only descriptor; it does not touch default policy, connect an
executor, publish a checkpoint, or start online canary traffic. `Global 99
Sandbox Consumer Replay / Canary v1` validates sandbox consumer behavior
offline only; it does not start online canary traffic, install default policy,
connect an executor, or claim real-world performance. `Global 99 Controlled
Default Policy Candidate Installation Preflight v1` is only a review
eligibility gate; it does not execute installation, replace default policy,
connect an executor, publish a checkpoint, or start online canary traffic.
