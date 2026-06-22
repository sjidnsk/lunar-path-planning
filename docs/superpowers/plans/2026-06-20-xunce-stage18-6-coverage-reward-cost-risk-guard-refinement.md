# Xunce Stage 18.6 Coverage Reward Cost-Risk Guard Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. A read-only review subagent must review the guard/routing plan before implementation and again after implementation.

**Goal:** Build a reproducible Stage 18.6 offline guard-refinement loop that prevents Xunce from being treated as better when it gains coverage only by adding path cost or risk.

**Architecture:** Add a read-only Stage 18.6 runner that consumes the Stage 18.5 attribution root and the Stage 18.4E coverage comparison root. The runner replays selected-action and scenario-level guard decisions, calibrates cost/risk-aware coverage advantage criteria, reports whether full candidate-level replay is possible, and routes the next step without training or release side effects. If existing paired audit lacks full per-candidate metrics, Stage 18.6 must say so explicitly and route to an audit-enrichment rerun instead of claiming the guard is fixed.

**Tech Stack:** Python stdlib JSON/JSONL, existing `global_99_coverage_contract` and `global_99_governance_common` helpers, `unittest`/`pytest`, `configs/stage_registry.json`, existing Stage 18.4E and Stage 18.5 artifacts.

---

## Current Evidence Baseline

- Stage 18.5 root: `outputs/path_feedback_batch_xunce_stage18_5_evidence_attribution_review_v1/`
- Stage 18.4E coverage comparison root: `outputs/path_feedback_batch_xunce_high_fidelity_exploration_coverage_comparison_dynamic_stage18_4e_v1/`
- Stage 18.5 status: `passed`
- Evidence authenticity gate: `true`
- Candidate validity gate: `true`
- Guard result: `passed=false`
- Failed guards: `cost_efficiency_regression`, `path_cost_regression`, `risk_regression`
- Observed worst guard metrics:
  - `coverage_delta_cells=150.0`
  - `path_cost_delta_m=232.73179060941447`
  - `risk_delta=4.8335335445279455`
  - `risk_cost_weighted_delta=347.1262232945303`
  - `coverage_per_100m_delta=-54.770272059395296`
  - `coverage_gain_per_path_cost_delta=-0.00013301853688924085`
- Paired decision audit rows: `1920`
- Policy disagreement count: `863`
- Useful disagreement count: `0`
- Same-candidate-set advantage established: `false`
- Regression counts: `24` efficiency regressions, `16` safety regressions
- Candidate exhaustion: diagnostic-only, `24` scenarios
- Stage 18.5 route: `refine_coverage_reward_and_cost_guard`
- Stage 18.6 default route until candidate-level audit exists:
  `rerun_stage18_4e_with_candidate_metric_audit`
- Stage 19: not authorized

---

## File Map

Create:
- `configs/xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1.json` - Stage 18.6 default inputs, guard thresholds, calibration grids, and boundary defaults.
- `scripts/run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py` - read-only Stage 18.6 guard refinement runner.
- `tests/test_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py` - focused unit tests for guard semantics, routing, boundary locks, and candidate-metric readiness.

Modify:
- `configs/stage_registry.json` - add `xunce-stage18-6-coverage-reward-cost-risk-guard-refinement`.
- `scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py` - add optional candidate metric audit output needed for full candidate-level guard replay.
- `tests/test_xunce_high_fidelity_exploration_coverage_comparison.py` - verify candidate metric audit rows are emitted when enabled.
- `scripts/xunce_stage18_pipeline.py` - optionally consume Stage 18.6 root after Stage 18.5 root and expose the guard refinement route.
- `scripts/run_xunce_stage18_research_evidence_pipeline.py` - add CLI/config passthrough for `stage18_6_guard_refinement_root`.
- `tests/test_xunce_stage18_research_evidence_pipeline.py` - verify Stage 18.6 root routing and stale/malformed root rejection.
- `tests/test_platform_stage_runner.py` - include the new stage id in registry coverage.
- `README.md` - document Stage 18.6 purpose, inputs, outputs, route, and non-goals.
- `docs/算法设计与系统架构报告.md` - document Stage 18.6 as a guard-refinement stage, not a training or release stage.
- `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md` - update the Stage 18 route narrative.

---

## Task 1: Stage 18.6 Config And Registry

**Files:**
- Create: `configs/xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1.json`
- Modify: `configs/stage_registry.json`
- Test: `tests/test_platform_stage_runner.py`

- [ ] **Step 1: Add a failing registry/config test**

Add assertions to `tests/test_platform_stage_runner.py` that the stage registry contains:

```python
"xunce-stage18-6-coverage-reward-cost-risk-guard-refinement"
```

Expected stage fields:

```json
{
  "script": "scripts/run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py",
  "default_config": "configs/xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1.json",
  "default_output_root": "outputs/path_feedback_batch_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1"
}
```

- [ ] **Step 2: Run the targeted test and confirm red**

Run:

```powershell
python -m pytest tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage18-6-registry-red
```

Expected: FAIL because the new stage id and script do not exist yet.

- [ ] **Step 3: Create the Stage 18.6 config**

Create `configs/xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1.json`:

```json
{
  "schema_version": "xunce-stage18-6-coverage-reward-cost-risk-guard-refinement-config/v1",
  "stage18_5_attribution_root": "outputs/path_feedback_batch_xunce_stage18_5_evidence_attribution_review_v1",
  "coverage_comparison_root": "outputs/path_feedback_batch_xunce_high_fidelity_exploration_coverage_comparison_dynamic_stage18_4e_v1",
  "canonical_reward_guard_profile_path": "configs/xunce_canonical_reward_guard_profile_v2.json",
  "max_acceptable_path_cost_delta_m": 20.0,
  "max_acceptable_risk_delta": 0.5,
  "max_acceptable_risk_cost_weighted_delta": 25.0,
  "min_coverage_delta_cells": 1.0,
  "min_coverage_per_100m_delta": 0.0,
  "coverage_gain_per_path_cost_delta_mode": "audit_only",
  "cost_weight_grid": [0.0, 0.0001, 0.0005, 0.001, 0.002],
  "risk_weight_grid": [0.0, 0.001, 0.005, 0.01],
  "risk_cost_weight_grid": [0.0, 0.0001, 0.0005, 0.001],
  "require_full_candidate_metric_replay_for_stage19": true,
  "canary_traffic_fraction": 0.0
}
```

- [ ] **Step 4: Register the stage**

Add this object under `configs/stage_registry.json` `stages`:

```json
"xunce-stage18-6-coverage-reward-cost-risk-guard-refinement": {
  "script": "scripts/run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py",
  "default_config": "configs/xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1.json",
  "default_output_root": "outputs/path_feedback_batch_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1",
  "args": ["--config", "{config}", "--output-root", "{output_root}", "--repo-root", "{repo_root}"]
}
```

- [ ] **Step 5: Run the registry test**

Run:

```powershell
python -m pytest tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage18-6-registry-green
```

Expected: PASS once the runner file exists in Task 3; until then the stage-list assertion can pass but dry-run may fail.

---

## Task 2: Stage 18.6 Guard Semantics Tests

**Files:**
- Create: `tests/test_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py`
- Later implementation: `scripts/run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py`

- [ ] **Step 1: Create fixture helpers**

The test fixture must write a minimal root with:

```text
stage18_5/
  xunce-stage18-5-evidence-attribution-summary.json
  xunce-stage18-5-guard-evaluation.json
  xunce-stage18-5-next-stage-routing.json
  xunce-stage18-5-paired-decision-summary.json
  xunce-stage18-5-regression-attribution.jsonl
coverage/
  xunce-exploration-coverage-comparison-summary.json
  xunce-exploration-coverage-comparison-aggregate.json
  xunce-exploration-coverage-comparison-pairs.jsonl
  xunce-exploration-coverage-episodes.jsonl
  xunce-exploration-coverage-paired-decision-audit.jsonl
```

Use schema versions already emitted by Stage 18.5 and Stage 18.4E.

- [ ] **Step 2: Test coverage gain with cost/risk regression routes to guard refinement**

Add:

```python
def test_coverage_gain_with_cost_or_risk_regression_blocks_stage19(tmp_path):
    summary = run_stage18_6_fixture(
        tmp_path,
        coverage_delta=20.0,
        path_cost_delta=5.0,
        risk_delta=0.2,
        risk_cost_weighted_delta=3.0,
        coverage_per_100m_delta=-1.0,
        useful_disagreement_count=0,
    )
    assert summary["status"] == "passed"
    assert summary["guard_refinement_passed"] is False
    assert summary["stage19_readiness"]["authorized"] is False
    assert summary["next_required_change"] in {
        "refine_coverage_reward_and_cost_guard",
        "rerun_stage18_4e_with_candidate_metric_audit",
    }
```

- [ ] **Step 3: Test candidate metric readiness is explicit**

Add:

```python
def test_missing_full_candidate_metrics_prevents_counterfactual_reselection_claim(tmp_path):
    summary = run_stage18_6_fixture(tmp_path, include_candidate_metric_audit=False)
    assert summary["candidate_metric_readiness"]["full_candidate_metric_replay_available"] is False
    assert "missing_candidate_metric_audit" in summary["diagnostic_reason_codes"]
    assert summary["counterfactual_reselection_claimed"] is False
```

- [ ] **Step 4: Test full candidate metric replay can pass only with clean guards**

Add:

```python
def test_full_candidate_replay_requires_clean_cost_risk_and_efficiency(tmp_path):
    summary = run_stage18_6_fixture(
        tmp_path,
        include_candidate_metric_audit=True,
        candidate_metric_rows=[
            {
                "scenario_id": "s0",
                "step_index": 0,
                "candidate_index": 0,
                "expected_new_coverage_cell_count": 9.0,
                "roi_weighted_coverage_delta": 9.0,
                "path_cost": 2.0,
                "risk": 1.0,
                "action_mask_valid": True
            },
            {
                "scenario_id": "s0",
                "step_index": 0,
                "candidate_index": 1,
                "expected_new_coverage_cell_count": 15.0,
                "roi_weighted_coverage_delta": 15.0,
                "path_cost": 20.0,
                "risk": 3.0,
                "action_mask_valid": True
            }
        ],
    )
    assert summary["candidate_metric_readiness"]["full_candidate_metric_replay_available"] is True
    assert summary["guarded_reselection_summary"]["unsafe_high_coverage_candidate_rejected_count"] >= 1
```

- [ ] **Step 5: Test boundary locks hard-fail**

Add:

```python
def test_boundary_flags_or_canary_hard_fail(tmp_path):
    summary = run_stage18_6_fixture(tmp_path, config_overrides={"starts_online_canary": True})
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage18_6_guard_refinement_boundary_rejections"
    assert summary["stage19_readiness"]["authorized"] is False
```

- [ ] **Step 6: Confirm red before implementation**

Run:

```powershell
python -m pytest tests\test_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py -q --basetemp outputs\pytest-stage18-6-runner-red
```

Expected: FAIL with `ModuleNotFoundError` for the new runner.

---

## Task 3: Stage 18.6 Read-Only Runner

**Files:**
- Create: `scripts/run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py`
- Test: `tests/test_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py`

- [ ] **Step 1: Implement CLI and config loading**

The runner CLI must accept:

```text
--config
--output-root
--repo-root
--stage18-5-attribution-root
--coverage-comparison-root
```

Config schema:

```python
CONFIG_SCHEMA_VERSION = "xunce-stage18-6-coverage-reward-cost-risk-guard-refinement-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage18-6-guard-refinement-summary/v1"
```

Boundary fields must remain false and `canary_traffic_fraction` must be exactly `0.0`.

- [ ] **Step 2: Validate Stage 18.5 lineage**

The runner must reject the Stage 18.5 root unless:

```python
summary["schema_version"] == "xunce-stage18-5-evidence-attribution-summary/v1"
summary["next_stage_routing"]["primary_route"] == "rerun_stage18_4e_with_candidate_metric_audit"
summary["next_stage_routing"]["stage19_authorized"] is False
summary["stage19_readiness"]["authorized"] is False
Path(summary["coverage_comparison_root"]).resolve() == Path(config["coverage_comparison_root"]).resolve()
```

Invalid lineage routes to:

```python
"rerun_xunce_stage18_5_evidence_attribution_review"
```

- [ ] **Step 3: Read required Stage 18.4E artifacts**

Require:

```python
COVERAGE_SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
COVERAGE_AGGREGATE_FILE = "xunce-exploration-coverage-comparison-aggregate.json"
COVERAGE_PAIRS_FILE = "xunce-exploration-coverage-comparison-pairs.jsonl"
COVERAGE_EPISODES_FILE = "xunce-exploration-coverage-episodes.jsonl"
PAIRED_DECISION_AUDIT_FILE = "xunce-exploration-coverage-paired-decision-audit.jsonl"
```

Optionally read:

```python
CANDIDATE_METRIC_AUDIT_FILE = "xunce-exploration-coverage-candidate-metric-audit.jsonl"
```

Missing required artifacts route to:

```python
"rerun_xunce_stage18_4e_coverage_comparison_with_required_artifacts"
```

- [ ] **Step 4: Implement selected-action guard replay**

For each paired decision row, compute:

```python
coverage_delta = xunce_selected_expected_new_coverage_cell_count - incumbent_selected_expected_new_coverage_cell_count
roi_delta = xunce_selected_roi_weighted_coverage_delta - incumbent_selected_roi_weighted_coverage_delta
path_cost_delta = xunce_selected_path_cost - incumbent_selected_path_cost
risk_delta = xunce_selected_risk - incumbent_selected_risk
risk_cost_weighted_delta = (
    xunce_selected_path_cost * xunce_selected_risk
    - incumbent_selected_path_cost * incumbent_selected_risk
)
coverage_per_cost_delta = (
    safe_ratio(xunce_selected_expected_new_coverage_cell_count, xunce_selected_path_cost)
    - safe_ratio(incumbent_selected_expected_new_coverage_cell_count, incumbent_selected_path_cost)
)
```

Classify each row as:

```python
"guard_clean_coverage_win"
"false_coverage_win_cost_or_risk_regression"
"neutral_or_loss"
"non_comparable"
```

Guard-clean coverage win requires:

```python
coverage_delta >= min_coverage_delta_cells
path_cost_delta <= max_acceptable_path_cost_delta_m
risk_delta <= max_acceptable_risk_delta
risk_cost_weighted_delta <= max_acceptable_risk_cost_weighted_delta
coverage_per_100m_delta >= min_coverage_per_100m_delta
```

- [ ] **Step 5: Implement scenario-level guard replay**

For each comparison pair row, compute:

```python
scenario_guard_clean = (
    coverage_delta_cells >= min_coverage_delta_cells
    and path_cost_delta_m <= max_acceptable_path_cost_delta_m
    and risk_delta <= max_acceptable_risk_delta
    and risk_cost_weighted_delta <= max_acceptable_risk_cost_weighted_delta
    and coverage_per_100m_delta >= min_coverage_per_100m_delta
)
```

`coverage_gain_per_path_cost_delta` remains audit-only in v2. If the input uses
`coverage_gain_per_path_cost_delta_vs_incumbent` at summary level but pair rows
lack `coverage_gain_per_path_cost_delta`, record the missing field as a
diagnostic instead of turning it into a hard guard.

- [ ] **Step 6: Implement calibration grid**

For every `(cost_weight, risk_weight, risk_cost_weight)` tuple, compute selected-action utility:

```python
utility = (
    roi_weighted_coverage_delta
    - cost_weight * max(0.0, path_cost_delta)
    - risk_weight * max(0.0, risk_delta)
    - risk_cost_weight * max(0.0, risk_cost_weighted_delta)
)
```

Emit grid rows with:

```json
{
  "cost_weight": 0.0005,
  "risk_weight": 0.005,
  "risk_cost_weight": 0.0001,
  "false_coverage_win_count": 0,
  "guard_clean_coverage_win_count": 0,
  "retained_policy_disagreement_count": 0,
  "calibration_viable": false
}
```

Because selected-action replay cannot prove a different policy would choose a better candidate, mark:

```python
calibration_support_level = "selected_action_only"
```

unless candidate metric audit is available.

- [ ] **Step 7: Implement candidate metric readiness**

If `xunce-exploration-coverage-candidate-metric-audit.jsonl` is absent, output:

```json
{
  "schema_version": "xunce-stage18-6-candidate-metric-readiness/v1",
  "full_candidate_metric_replay_available": false,
  "reason_codes": ["missing_candidate_metric_audit"],
  "counterfactual_reselection_claim_allowed": false
}
```

If present, require each row to include:

```text
scenario_id
step_index
candidate_index
expected_new_coverage_cell_count
roi_weighted_coverage_delta
path_cost
risk
action_mask_valid
```

- [ ] **Step 8: Implement routing**

Use this priority:

```python
if boundary_rejected:
    next_required_change = "resolve_stage18_6_guard_refinement_boundary_rejections"
elif required_artifact_missing:
    next_required_change = "rerun_xunce_stage18_4e_coverage_comparison_with_required_artifacts"
elif stage18_5_lineage_invalid:
    next_required_change = "rerun_xunce_stage18_5_evidence_attribution_review"
elif not candidate_metric_readiness["full_candidate_metric_replay_available"]:
    next_required_change = "rerun_stage18_4e_with_candidate_metric_audit"
elif guard_refinement_passed and same_candidate_advantage_established:
    next_required_change = "prepare_stage19_evaluator_critic_preflight"
elif guarded_reselection_summary["safe_candidate_available_count"] == 0:
    next_required_change = "expand_candidate_generation_roi_complexity"
else:
    next_required_change = "refine_coverage_reward_and_cost_guard"
```

For the current baseline, expected route is:

```python
"rerun_stage18_4e_with_candidate_metric_audit"
```

or, if the implementation chooses not to request candidate audit enrichment in the first pass:

```python
"refine_coverage_reward_and_cost_guard"
```

The summary must still make clear that Stage 19 is not authorized.

- [ ] **Step 9: Write outputs**

Write these artifacts under `outputs/path_feedback_batch_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1/`:

```text
xunce-stage18-6-guard-refinement-summary.json
xunce-stage18-6-selected-action-guard-replay.jsonl
xunce-stage18-6-scenario-guard-replay.jsonl
xunce-stage18-6-calibration-grid.jsonl
xunce-stage18-6-candidate-metric-readiness.json
xunce-stage18-6-next-stage-routing.json
xunce-stage18-6-guard-refinement-report.md
xunce-stage18-6-manifest.json
```

- [ ] **Step 10: Run tests and compile**

Run:

```powershell
python -m pytest tests\test_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py -q --basetemp outputs\pytest-stage18-6-runner-green
python -m py_compile scripts\run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py
```

Expected: all pass.

---

## Task 4: Candidate Metric Audit Enrichment For Stage 18.4E

**Files:**
- Modify: `scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py`
- Modify: `configs/xunce_high_fidelity_exploration_coverage_comparison_v1.json`
- Modify: `tests/test_xunce_high_fidelity_exploration_coverage_comparison.py`

- [ ] **Step 1: Add failing test for candidate metric audit output**

Add a test that enables:

```json
"emit_candidate_metric_audit": true
```

Expected output file:

```text
xunce-exploration-coverage-candidate-metric-audit.jsonl
```

Assert every row has:

```python
required = {
    "schema_version",
    "scenario_id",
    "split",
    "roi_group",
    "policy",
    "step_index",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_index",
    "candidate_cell",
    "action_mask_valid",
    "expected_new_coverage_cell_count",
    "roi_weighted_coverage_delta",
    "path_cost",
    "risk",
    "risk_source",
    "coverage_gain_per_path_cost",
}
assert required <= row.keys()
```

- [ ] **Step 2: Implement optional audit rows**

In the rollout loop where `candidates` are already available and selected step rows are emitted, write one candidate metric row per candidate when config flag `emit_candidate_metric_audit` is true.

Do not change:

```text
action space
default A*
policy logits
candidate selection behavior
rollout path execution
```

- [ ] **Step 3: Add output path to manifest/summary**

When enabled, expose:

```json
{
  "candidate_metric_audit": ".../xunce-exploration-coverage-candidate-metric-audit.jsonl",
  "candidate_metric_audit_row_count": 0
}
```

When disabled, expose:

```json
{
  "candidate_metric_audit": null,
  "candidate_metric_audit_row_count": 0
}
```

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests\test_xunce_high_fidelity_exploration_coverage_comparison.py -q --basetemp outputs\pytest-stage18-6-candidate-audit
```

Expected: PASS.

---

## Task 5: Stage 18 Pipeline Optional Consumption

**Files:**
- Modify: `scripts/xunce_stage18_pipeline.py`
- Modify: `scripts/run_xunce_stage18_research_evidence_pipeline.py`
- Modify: `tests/test_xunce_stage18_research_evidence_pipeline.py`

- [ ] **Step 1: Add config and CLI support**

Add optional config key:

```python
"stage18_6_guard_refinement_root"
```

Add CLI flag:

```text
--stage18-6-guard-refinement-root
```

- [ ] **Step 2: Validate Stage 18.6 summary before route adoption**

The pipeline may consume Stage 18.6 only when:

```python
summary["schema_version"] == "xunce-stage18-6-guard-refinement-summary/v1"
summary["stage19_readiness"]["authorized"] is False or summary["next_required_change"] == "prepare_stage19_evaluator_critic_preflight"
summary["coverage_comparison_root"] matches current coverage root
summary["stage18_5_attribution_root"] matches current Stage 18.5 root if provided
summary["next_stage_routing"]["schema_version"] == "xunce-stage18-6-next-stage-routing/v1"
summary["next_stage_routing"]["primary_route"] in ALLOWED_NEXT_REQUIRED_CHANGES
```

Malformed or stale Stage 18.6 root must add blocking reason:

```python
"invalid_stage18_6_guard_refinement_summary"
```

or:

```python
"stale_stage18_6_guard_refinement_root"
```

and must not override the pipeline route.

- [ ] **Step 3: Expose Stage 18.6 fields**

Pipeline summary should include:

```python
"stage18_6_guard_refinement_summary"
"stage18_6_guard_refinement_verdict"
"stage18_6_primary_next_required_change"
"stage18_6_candidate_metric_readiness"
```

- [ ] **Step 4: Add tests**

Add tests for:

```text
valid Stage 18.6 root adopts Stage 18.6 route
stale Stage 18.6 root does not override
malformed Stage 18.6 route does not override
Stage 18.6 cannot authorize Stage 19 unless guard and same-candidate advantage are both clean
CLI override is accepted
```

- [ ] **Step 5: Run pipeline tests**

Run:

```powershell
python -m pytest tests\test_xunce_stage18_research_evidence_pipeline.py -q --basetemp outputs\pytest-stage18-6-pipeline
```

Expected: PASS.

---

## Task 6: Documentation Updates

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] **Step 1: Document the plain-language purpose**

Add this meaning in each doc:

```text
Stage 18.6 checks whether Xunce's extra coverage is worth its extra path cost and risk. Coverage is not treated as an advantage unless cost, risk, risk-cost weighted exposure, and coverage efficiency remain within guard budgets.
```

- [ ] **Step 2: Document current expected route**

Write:

```text
Current Stage 18.5 evidence keeps Stage 19 unauthorized. Stage 18.6 should either request a Stage 18.4E rerun with candidate metric audit or continue `refine_coverage_reward_and_cost_guard`; it must not train, publish, or replace default policy.
```

- [ ] **Step 3: Document artifacts**

List Stage 18.6 config, runner, output root, and all eight output files from Task 3.

- [ ] **Step 4: Document non-goals**

Repeat:

```text
No PPO, no checkpoint publication, no default policy replacement, no executor connection, no canary, no action-space/default-A* changes, no real-world performance claim.
```

---

## Task 7: End-To-End Verification And Subagent Review

**Files:**
- No new files beyond previous tasks.

- [ ] **Step 1: Run the focused verification suite**

Run:

```powershell
python -m pytest `
  tests\test_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py `
  tests\test_xunce_stage18_5_evidence_attribution_review.py `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_xunce_stage18_research_evidence_pipeline.py `
  tests\test_platform_stage_runner.py `
  -q --basetemp outputs\pytest-stage18-6-final
```

- [ ] **Step 2: Compile modified scripts**

Run:

```powershell
python -m py_compile `
  scripts\run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py `
  scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py `
  scripts\xunce_stage18_pipeline.py `
  scripts\run_xunce_stage18_research_evidence_pipeline.py
```

- [ ] **Step 3: Dry-run the registered stage**

Run:

```powershell
python scripts\run_stage.py --stage xunce-stage18-6-coverage-reward-cost-risk-guard-refinement --dry-run
```

Expected: command resolves to the Stage 18.6 runner with config, output root, and repo root.

- [ ] **Step 4: Run Stage 18.6 against current roots**

Run:

```powershell
python scripts\run_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement.py `
  --config configs\xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1.json `
  --output-root outputs\path_feedback_batch_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1 `
  --repo-root .
```

Expected current-result properties:

```text
status=passed
stage19_readiness.authorized=false
guard_refinement_passed=false
counterfactual_reselection_claimed=false unless candidate metric audit exists
next_required_change=rerun_stage18_4e_with_candidate_metric_audit or refine_coverage_reward_and_cost_guard
```

- [ ] **Step 5: Run the research evidence pipeline with Stage 18.5 and Stage 18.6 roots**

Run:

```powershell
python scripts\run_xunce_stage18_research_evidence_pipeline.py `
  --config configs\xunce_stage18_research_evidence_pipeline_v1.json `
  --output-root outputs\path_feedback_batch_xunce_stage18_research_evidence_pipeline_v1 `
  --repo-root . `
  --coverage-comparison-root outputs\path_feedback_batch_xunce_high_fidelity_exploration_coverage_comparison_dynamic_stage18_4e_v1 `
  --stage18-5-attribution-root outputs\path_feedback_batch_xunce_stage18_5_evidence_attribution_review_v1 `
  --stage18-6-guard-refinement-root outputs\path_feedback_batch_xunce_stage18_6_coverage_reward_cost_risk_guard_refinement_v1 `
  --plan-only
```

Expected: pipeline exposes Stage 18.6 fields and does not authorize Stage 19.

- [ ] **Step 6: Request subagent review**

Dispatch a read-only reviewer with this prompt:

```text
Review the Stage 18.6 guard refinement implementation. Confirm that it consumes Stage 18.5 and Stage 18.4E artifacts read-only, keeps all training/release/executor/canary boundaries false, does not claim counterfactual reselection unless candidate metric audit exists, and routes the current baseline away from Stage 19. Report Critical/Important/Minor findings with file and line references.
```

- [ ] **Step 7: Fix all Critical and Important review findings**

Do not proceed with a completion claim while any valid Critical or Important review finding remains open.

---

## Acceptance Criteria

- Stage 18.6 summary says current Stage 18.5 evidence is readable but Xunce advantage remains unproven.
- Coverage gain is treated as valid only when cost, risk, risk-cost weighted exposure, and coverage per 100m pass guard thresholds; coverage gain per path cost remains audit-only in v2.
- The current baseline cannot route to Stage 19 because guard failed and same-candidate-set advantage is not established.
- If full candidate metric audit is absent, Stage 18.6 explicitly blocks counterfactual reselection claims and routes to candidate metric audit enrichment or continued guard refinement.
- If full candidate metric audit is present, unsafe high-coverage candidates are rejected in offline replay.
- All governance fields remain false and `canary_traffic_fraction=0.0`.
- Registry, pipeline, docs, and tests are synchronized.
- A subagent review is completed, and all valid Critical/Important issues are fixed.

## Non-Goals

- Do not start PPO.
- Do not publish a checkpoint.
- Do not replace default policy.
- Do not connect a real executor.
- Do not start online canary.
- Do not modify action space.
- Do not modify default A*.
- Do not claim Ackermann-feasible or real-world performance.
- Do not treat IRIS/GCS/path-planner diagnostics as training or release approval.
