# Stage 3.2 Region Coverage + Cost-Aware Sampled Connector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make passable start/goal cells anchor into project-owned regions and make sampled region candidates cost-aware enough to be selected when they are genuinely better.

**Architecture:** Keep `path-planner-route/v1` and `trajectory_kind=geometric_path` stable. Add only optional diagnostic fields under `sampled_region_path_report/v1`, then aggregate those fields through `model-explorer` and the root batch summary without changing stable required keys.

**Tech Stack:** Python 3.12, NumPy, pytest, unittest, existing `path_planner` and `model_explorer` JSON contracts.

---

### Task 1: Region Coverage Classification And Anchor Regions

**Files:**
- Modify: `path-planner/src/path_planner/search/region_guided.py`
- Test: `path-planner/tests/test_region_guided_search.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
def test_passable_goal_outside_regions_gets_anchor_region_not_broad_missing():
    assert sampled["fallback_reason"] != "goal_region_missing"
    assert sampled["start_goal_anchoring"]["goal_anchor_region_added"] is True
```

```python
def test_blocked_goal_reports_goal_not_passable_without_anchor():
    assert sampled["fallback_reason"] == "goal_not_passable"
    assert sampled["start_goal_anchoring"]["goal_anchor_region_added"] is False
```

```python
def test_passable_goal_anchor_unconnected_reports_specific_reason():
    assert sampled["fallback_reason"] == "goal_anchor_region_unconnected"
```

- [ ] **Step 2: Verify red**

Run:

```bash
cd path-planner && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_region_guided_search.py -k "anchor or passable_goal or blocked_goal" -v
```

Expected: new assertions fail because current code emits broad `goal_region_missing`.

- [ ] **Step 3: Implement minimal coverage logic**

Add internal additive metadata only:

```python
"goal_classification": "goal_not_passable|goal_footprint_unsafe|goal_outside_region_coverage|goal_anchor_region_unconnected|covered",
"goal_anchor_region_added": bool,
"goal_anchor_region_connected": bool,
"goal_anchor_failure_reason": str | None,
```

Create one-cell `grid_box` anchor regions for passable start/goal cells outside existing regions and connect them when they overlap, touch, or can be safely stepped to an existing region cell.

- [ ] **Step 4: Verify green**

Run the targeted path-planner test command above.

### Task 2: Cost-Aware Constrained Connector

**Files:**
- Modify: `path-planner/src/path_planner/search/region_guided.py`
- Test: `path-planner/tests/test_region_guided_search.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
def test_cost_aware_connector_selects_lower_cost_route_inside_region_union():
    assert outcome.report.selected_backend == "sampled_region_path"
    assert sampled["candidate_rankings"][0]["strategy"] == "cost_aware_constrained_astar"
```

```python
def test_higher_cost_constrained_connector_falls_back_with_specific_reason():
    assert sampled["fallback_reason"] == "region_sequence_cost_dominated"
```

- [ ] **Step 2: Verify red**

Run:

```bash
cd path-planner && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_region_guided_search.py -k "cost_aware or cost_dominated" -v
```

Expected: current sampled connector only line-stitches sampled cells and does not produce the requested strategy/reason.

- [ ] **Step 3: Implement minimal constrained A***

Generate a mask from the region sequence plus touching adjacent regions, run A* with the original request policy inside that mask, rank it with existing cost/exposure/tracking metrics, and keep conservative selection rules.

- [ ] **Step 4: Verify green**

Run the targeted path-planner test command above.

### Task 3: Additive Model Explorer And Batch Aggregation

**Files:**
- Modify: `model-explorer/src/model_explorer/policy/path_feedback.py`
- Modify: `model-explorer/tests/test_model_explorer.py`
- Modify: `scripts/run_batch_path_feedback_validation.py`
- Modify: `tests/test_batch_path_feedback_validation.py`

- [ ] **Step 1: Write failing tests**

Assert root and scenario summaries aggregate:

```python
sampled_region_path_anchor_region_added_count
sampled_region_path_anchor_region_connected_count
sampled_region_path_goal_classification_counts
sampled_region_path_connector_attempt_count
```

- [ ] **Step 2: Verify red**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_batch_path_feedback_validation.py -v
cd model-explorer && PYTHONPATH=src /home/kai/anaconda3/envs/lunar-explorer/bin/python -m unittest tests.test_model_explorer -v
```

- [ ] **Step 3: Implement additive aggregation**

Read new fields from `start_goal_anchoring`, `sample_attempts`, and `candidate_rankings`; do not add them to `PATH_FEEDBACK_SUMMARY_REQUIRED_KEYS`.

- [ ] **Step 4: Verify green**

Run both commands from Step 2.

### Task 4: Evidence Refresh And Completion Audit

**Files:**
- Create: `outputs/path_feedback_batch_policy_input_stage3_2_region_connector/stage3-2-region-connector-audit.md`
- Generated: `outputs/path_feedback_batch_policy_input_stage3_2_region_connector/*.json`

- [ ] **Step 1: Run full validation chain**

Run the exact commands from the Stage 3.2 goal.

- [ ] **Step 2: Inspect evidence**

Confirm `open_grid_fallback_used_count == 0`, source summary paths exist, git provenance matches current parent/submodule SHAs, and broad `goal_region_missing` is gone for passable goals or replaced by specific reasons.

- [ ] **Step 3: Write audit**

Record HEAD/submodule SHA, run status, reason codes, open-grid fallback count, sampled selected/fallback counts, anchor classifications, and connector blockers.
