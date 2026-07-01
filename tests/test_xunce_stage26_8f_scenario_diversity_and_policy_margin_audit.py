from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8f_detects_duplicate_scenarios(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as s26

    stage26_8d = _make_stage26_8d_root(tmp_path)
    eval_root = stage26_8d / "h16" / "s260801" / "s26_3"
    _write_stage26_3_eval(eval_root, scenario_ids=("stage26_synthetic_000", "stage26_synthetic_001"), duplicate=True)
    _write_job_state(stage26_8d, [("h16_s260801", eval_root)])
    config = _write_config(tmp_path, stage26_8d_root=stage26_8d, min_completed_eval_count=1)

    summary = s26.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "repair_stage26_synthetic_scenario_diversity"
    audit = _read_json(tmp_path / "out" / "xunce-stage26-8f-scenario-diversity-audit.json")
    assert audit["scenario_diversity_suspect"] is True
    assert audit["scenario_signature_duplicate_group_count"] == 1


def test_stage26_8f_routes_to_audit_binding_when_strong_key_missing(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as s26

    stage26_8d = _make_stage26_8d_root(tmp_path)
    eval_root = stage26_8d / "h16" / "s260801" / "s26_3"
    _write_stage26_3_eval(eval_root, missing_strong_field=True)
    _write_job_state(stage26_8d, [("h16_s260801", eval_root)])
    config = _write_config(tmp_path, stage26_8d_root=stage26_8d, min_completed_eval_count=1)

    summary = s26.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_8f_audit_binding"
    margin = _read_json(tmp_path / "out" / "xunce-stage26-8f-policy-margin-crossing-audit.json")
    assert margin["pre_post_required_field_missing_count"] > 0


def test_stage26_8f_counts_missing_margin_fields_separately(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as s26

    stage26_8d = _make_stage26_8d_root(tmp_path)
    eval_root = stage26_8d / "h16" / "s260801" / "s26_3"
    _write_stage26_3_eval(eval_root, scenario_ids=("a", "b"), duplicate=False, missing_margin_fields=True)
    _write_job_state(stage26_8d, [("h16_s260801", eval_root)])
    config = _write_config(tmp_path, stage26_8d_root=stage26_8d, min_completed_eval_count=1)

    summary = s26.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_8f_audit_binding"
    margin = _read_json(tmp_path / "out" / "xunce-stage26-8f-policy-margin-crossing-audit.json")
    assert margin["strong_state_join_available_count"] > 0
    assert margin["margin_evaluable_join_count"] == 0
    assert margin["margin_required_field_missing_count"] > 0
    assert "pre_action_probs_missing" in margin["margin_required_field_missing_reasons"]


def test_stage26_8f_pending_jobs_do_not_count_as_missing_fields(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as s26

    stage26_8d = _make_stage26_8d_root(tmp_path)
    eval_root = stage26_8d / "h16" / "s260801" / "s26_3"
    _write_stage26_3_eval(eval_root, scenario_ids=("a", "b"), duplicate=False, selected_is_best=True)
    _write_job_state(stage26_8d, [("h16_s260801", eval_root)], pending_jobs=["h20_s260801"])
    config = _write_config(tmp_path, stage26_8d_root=stage26_8d, min_completed_eval_count=1)

    summary = s26.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "repair_stage26_synthetic_policy_update_signal_strength"
    source_index = _read_json(tmp_path / "out" / "xunce-stage26-8f-completed-eval-source-index.json")
    margin = _read_json(tmp_path / "out" / "xunce-stage26-8f-policy-margin-crossing-audit.json")
    assert source_index["pending_job_count"] == 1
    assert margin["pre_post_required_field_missing_count"] == 0
    assert margin["margin_required_field_missing_count"] == 0


def test_stage26_8f_routes_policy_signal_when_delta_too_small_for_margin(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as s26

    stage26_8d = _make_stage26_8d_root(tmp_path)
    eval_root = stage26_8d / "h16" / "s260801" / "s26_3"
    _write_stage26_3_eval(eval_root, scenario_ids=("a", "b"), duplicate=False, selected_is_best=True)
    _write_job_state(stage26_8d, [("h16_s260801", eval_root)])
    config = _write_config(tmp_path, stage26_8d_root=stage26_8d, min_completed_eval_count=1)

    summary = s26.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "repair_stage26_synthetic_policy_update_signal_strength"
    margin = _read_json(tmp_path / "out" / "xunce-stage26-8f-policy-margin-crossing-audit.json")
    assert margin["selected_action_changed_count"] == 0
    assert margin["policy_delta_too_small_for_margin"] is True


def test_stage26_8f_routes_candidate_logging_when_candidate_fields_missing(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as s26

    stage26_8d = _make_stage26_8d_root(tmp_path)
    eval_root = stage26_8d / "h16" / "s260801" / "s26_3"
    _write_stage26_3_eval(eval_root, scenario_ids=("a", "b"), duplicate=False, omit_candidate_fields=True)
    _write_job_state(stage26_8d, [("h16_s260801", eval_root)])
    config = _write_config(tmp_path, stage26_8d_root=stage26_8d, min_completed_eval_count=1)

    summary = s26.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["next_required_change"] == "repair_stage26_8f_candidate_metric_logging"
    opportunity = _read_json(tmp_path / "out" / "xunce-stage26-8f-candidate-opportunity-audit.json")
    assert opportunity["candidate_opportunity_fields_missing_count"] > 0


def test_stage26_8f_requires_more_completed_eval_sources(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as s26

    stage26_8d = _make_stage26_8d_root(tmp_path)
    eval_root = stage26_8d / "h16" / "s260801" / "s26_3"
    _write_stage26_3_eval(eval_root)
    _write_job_state(stage26_8d, [("h16_s260801", eval_root)])
    config = _write_config(tmp_path, stage26_8d_root=stage26_8d, min_completed_eval_count=2)

    summary = s26.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "continue_stage26_8d_seed_horizon_jobs"


def test_stage26_8f_boundary_and_stage_runner_registration(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as s26

    stage26_8d = _make_stage26_8d_root(tmp_path)
    config = _write_config(tmp_path, stage26_8d_root=stage26_8d, starts_online_canary=True)

    summary = s26.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_8f_boundary_rejections"
    registry = _read_json(REPO_ROOT / "configs" / "stage_registry.json")
    assert "xunce-stage26-8f-scenario-diversity-and-policy-margin-audit" in registry["stages"]


def _make_stage26_8d_root(tmp_path: Path) -> Path:
    root = tmp_path / "stage26_8d"
    _write_json(
        root / "xunce-stage26-8d-summary.json",
        {
            "schema_version": "xunce-stage26-8d-summary/v1",
            "stage_id": "xunce-stage26-8d-resumable-seed-horizon-execution",
            "status": "passed",
            "next_required_change": "continue_stage26_8d_seed_horizon_jobs",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return root


def _write_job_state(root: Path, eval_roots: list[tuple[str, Path]], pending_jobs: list[str] | None = None) -> None:
    rows = []
    for index, (job_id, eval_root) in enumerate(eval_roots):
        rows.append(
            {
                "schema_version": "xunce-stage26-8d-job-state/v1",
                "job_id": job_id,
                "horizon_steps": 16,
                "seed": 260801 + index,
                "seed_index": index,
                "phase": "complete",
                "status": "complete",
                "resume_decision": "complete_no_action",
                "stage26_3_root": str(eval_root),
                "stage26_3_source": "stage26_8d_output",
                "summary_path": str(eval_root / "xunce-stage26-3-summary.json"),
                "main_coverage_per_100m_delta": 0.0,
            }
        )
    for job_id in pending_jobs or []:
        rows.append(
            {
                "schema_version": "xunce-stage26-8d-job-state/v1",
                "job_id": job_id,
                "horizon_steps": 20,
                "seed": 260801,
                "seed_index": 0,
                "phase": "stage26_1",
                "status": "pending",
                "resume_decision": "run_stage26_1",
                "summary_path": str(root / "h20" / "s260801" / "s26_1" / "xunce-stage26-1-summary.json"),
            }
        )
    _write_jsonl(root / "xunce-stage26-8d-job-state.jsonl", rows)


def _write_stage26_3_eval(
    root: Path,
    *,
    scenario_ids: tuple[str, ...] = ("stage26_synthetic_000",),
    duplicate: bool = False,
    missing_strong_field: bool = False,
    missing_margin_fields: bool = False,
    omit_candidate_fields: bool = False,
    selected_is_best: bool = False,
) -> None:
    _write_json(
        root / "xunce-stage26-3-summary.json",
        {
            "schema_version": "xunce-stage26-3-summary/v1",
            "stage_id": "xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke",
            "status": "failed",
            "next_required_change": "repair_stage26_synthetic_policy_update_signal_strength",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "post_update_success_metric": "main_coverable_coverage_efficiency/v1",
            "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
            "path_cost_source": "hybrid_astar_pose_path/v1",
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
            "synthetic_terrain_hash": "synthetic-hash",
            "platform_contract_hash": "platform-hash",
            "max_traversable_slope_deg": 30.0,
            "selected_action_changed_count": 0,
            "coverage_per_100m_delta": 0.0,
        },
    )
    _write_json(root / "xunce-stage26-3-synthetic-action-change-audit.json", {"strong_state_join_available_count": 2})
    _write_json(root / "xunce-stage26-3-trajectory-delta-audit.json", {"coverage_per_100m_mean_delta": 0.0})
    pre_rows = []
    post_rows = []
    pre_steps = []
    episodes = []
    for scenario_index, scenario_id in enumerate(scenario_ids):
        for step in range(2):
            selected = 0
            row = _inference_row(
                scenario_id,
                step,
                duplicate=duplicate,
                scenario_index=scenario_index,
                selected=selected,
                selected_is_best=selected_is_best,
                omit_candidate_fields=omit_candidate_fields,
            )
            if missing_strong_field:
                row.pop("covered_cells_hash", None)
            if missing_margin_fields:
                row.pop("action_probs", None)
                row.pop("logits", None)
            pre_rows.append(row)
            post = dict(row)
            if not missing_margin_fields:
                post["action_probs"] = [0.600001, 0.299999, 0.1]
            post_rows.append(post)
            pre_steps.append(
                {
                    "scenario_id": scenario_id,
                    "step_index": step,
                    "candidate_set_hash": row["candidate_set_hash"],
                    "covered_cells_hash": row.get("covered_cells_hash"),
                    "selected_action_index": selected,
                    "new_covered_cell_count": 1,
                }
            )
        episodes.append(
            {
                "scenario_id": scenario_id,
                "final_coverage_rate": 0.1 if duplicate else 0.1 + scenario_index * 0.01,
                "coverage_curve_auc": 0.05 if duplicate else 0.05 + scenario_index * 0.01,
                "coverage_per_100m": 5.0 if duplicate else 5.0 + scenario_index,
                "path_cost_total_m": 100.0 if duplicate else 100.0 + scenario_index,
                "new_covered_cell_count": 10 if duplicate else 10 + scenario_index,
            }
        )
    _write_jsonl(root / "pre" / "xunce-exploration-coverage-model-inference.jsonl", pre_rows)
    _write_jsonl(root / "post" / "xunce-exploration-coverage-model-inference.jsonl", post_rows)
    _write_jsonl(root / "pre" / "xunce-exploration-coverage-steps.jsonl", pre_steps)
    _write_jsonl(root / "post" / "xunce-exploration-coverage-steps.jsonl", pre_steps)
    _write_jsonl(root / "pre" / "xunce-exploration-coverage-episodes.jsonl", episodes)
    _write_jsonl(root / "post" / "xunce-exploration-coverage-episodes.jsonl", episodes)


def _inference_row(
    scenario_id: str,
    step: int,
    *,
    duplicate: bool,
    scenario_index: int,
    selected: int,
    selected_is_best: bool,
    omit_candidate_fields: bool,
) -> dict:
    suffix = 0 if duplicate else scenario_index
    row = {
        "scenario_id": scenario_id,
        "step_index": step,
        "current_cell": [step, suffix],
        "covered_cells_hash": f"covered-{step}-{suffix}",
        "candidate_set_hash": f"candidate-{step}-{suffix}",
        "synthetic_terrain_hash": "synthetic-hash",
        "selected_action_index": selected,
        "selected_viewpoint": [step, suffix, 0],
        "action_probs": [0.6, 0.3, 0.1],
        "logits": [3.0, 2.0, 1.0],
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
    }
    if not omit_candidate_fields:
        row["theta_coverage_gain_per_path_costs"] = [3.0, 2.0, 1.0] if selected_is_best else [1.0, 3.0, 2.0]
        row["theta_new_visible_cell_counts"] = [5, 4, 3] if selected_is_best else [1, 5, 2]
    return row


def _write_config(tmp_path: Path, *, stage26_8d_root: Path, **overrides: object) -> Path:
    path = tmp_path / "stage26_8f_config.json"
    payload = {
        "schema_version": "xunce-stage26-8f-scenario-diversity-and-policy-margin-audit-config/v1",
        "stage_id": "xunce-stage26-8f-scenario-diversity-and-policy-margin-audit",
        "stage26_8d_root": str(stage26_8d_root),
        "min_completed_eval_count": 1,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    _write_json(path, payload)
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
