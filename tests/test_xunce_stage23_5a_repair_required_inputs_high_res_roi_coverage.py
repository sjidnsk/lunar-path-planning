from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "model-explorer" / "src"):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)


def test_stage23_5a_selects_two_valid_windows_and_reruns_chain(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage as s23

    config = _write_config(tmp_path, prior_summary=_prior_stage23_5_summary())
    _patch_roi_precheck(monkeypatch, s23, valid_indexes={0, 1, 2})
    calls = _patch_stage_chain(monkeypatch, s23, final_route=s23.ROUTE_STAGE23_6, final_status="passed")

    summary = s23.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    repaired_2a = json.loads((tmp_path / "out" / s23.REPAIRED_23_2A_CONFIG).read_text(encoding="utf-8"))
    repaired_2b = json.loads((tmp_path / "out" / s23.REPAIRED_23_2B_CONFIG).read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == s23.ROUTE_STAGE23_6
    assert summary["valid_roi_window_count"] == 3
    assert summary["selected_roi_window_count"] == 2
    assert summary["repaired_high_res_roi_expansion_slice_count"] == 2
    assert summary["repaired_high_res_roi_expansion_scenario_count"] == 2
    assert repaired_2a["max_traversable_slope_deg"] == 30.0
    assert repaired_2a["primary_manifest"].endswith("manifest.json")
    assert repaired_2a["download_missing"] is False
    assert repaired_2a["high_res_output_subdir"] == "hr"
    assert repaired_2a["path_planner_sidecar_subdir"] == "ps"
    assert repaired_2a["stage23_1_output_subdir"] == "s1"
    assert repaired_2a["stage23_0a_output_subdir"] == "a"
    assert repaired_2a["stage23_0a_high_fidelity_output_subdir"] == "h"
    assert len(repaired_2a["roi_windows"]) == 2
    assert repaired_2a["roi_windows"][0]["seed"] == 230201
    assert repaired_2a["roi_windows"][1]["seed"] == 230202
    assert repaired_2b["stage23_2a_config"] == str(tmp_path / "out" / s23.REPAIRED_23_2A_CONFIG)
    assert repaired_2b["stage23_2a_default_output_subdir"] == "a30"
    assert repaired_2b["stage23_2a_sensitivity_output_subdir"] == "a20"
    assert calls == ["stage23_2b", "stage23_2", "stage23_3", "stage23_4", "stage23_5"]
    assert summary["publishes_checkpoint"] is False
    assert summary["connects_real_executor"] is False


def test_stage23_5a_routes_expand_when_fewer_than_two_windows_are_valid(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage as s23

    config = _write_config(tmp_path, prior_summary=_prior_stage23_5_summary())
    _patch_roi_precheck(monkeypatch, s23, valid_indexes={0})
    calls = _patch_stage_chain(monkeypatch, s23)

    summary = s23.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_EXPAND_ROI
    assert summary["valid_roi_window_count"] == 1
    assert "valid_high_res_roi_window_count_below_minimum" in summary["blocking_reason_codes"]
    assert calls == []


def test_stage23_5a_preserves_non_required_input_prior_route(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage as s23

    prior = _prior_stage23_5_summary(route="repair_stage23_slope_theta_policy_update_signal_strength", reasons=[])
    config = _write_config(tmp_path, prior_summary=prior)
    _patch_roi_precheck(monkeypatch, s23, valid_indexes={0, 1})
    calls = _patch_stage_chain(monkeypatch, s23)

    summary = s23.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_PRESERVE
    assert "stage23_5_prior_blocker_not_scenario_episode_short" in summary["blocking_reason_codes"]
    assert calls == []


def test_stage23_5a_preserves_required_input_route_with_non_roi_blockers(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage as s23

    prior = _prior_stage23_5_summary(
        reasons=[
            "stage21_5_pre_scenario_count_short",
            "stage21_5_pre_xunce_episode_count_short",
            "slope_theta_inference_required_fields_missing",
        ],
        slope_theta_inference_required_field_missing_count=1,
    )
    config = _write_config(tmp_path, prior_summary=prior)
    _patch_roi_precheck(monkeypatch, s23, valid_indexes={0, 1})
    calls = _patch_stage_chain(monkeypatch, s23)

    summary = s23.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_PRESERVE
    assert "prior_stage23_5_slope_theta_inference_required_fields_missing" in summary["blocking_reason_codes"]
    assert "prior_stage23_5_slope_theta_inference_required_field_missing_count_nonzero" in summary["blocking_reason_codes"]
    assert calls == []


def test_stage23_5a_propagates_failed_upstream_stage_route(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage as s23

    config = _write_config(tmp_path, prior_summary=_prior_stage23_5_summary())
    _patch_roi_precheck(monkeypatch, s23, valid_indexes={0, 1})
    calls = _patch_stage_chain(
        monkeypatch,
        s23,
        stage23_2b_status="failed",
        stage23_2b_route="repair_stage23_0a_candidate_obstacle_source_linkage",
    )

    summary = s23.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_0a_candidate_obstacle_source_linkage"
    assert calls == ["stage23_2b"]


def test_stage23_5a_routes_required_inputs_when_stage23_5_is_still_short(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage as s23

    config = _write_config(tmp_path, prior_summary=_prior_stage23_5_summary())
    _patch_roi_precheck(monkeypatch, s23, valid_indexes={0, 1})
    _patch_stage_chain(monkeypatch, s23, final_route=s23.ROUTE_REQUIRED_INPUTS, final_status="failed", final_short=True)

    summary = s23.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_REQUIRED_INPUTS
    assert summary["stage23_5_action_probability_audit_is_partial_diagnostic"] is True
    assert "stage23_5_stage21_5_pre_scenario_count_short" in summary["blocking_reason_codes"]


def test_stage23_5a_boundary_flags_hard_fail_without_roi_precheck(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage as s23

    config = _write_config(tmp_path, prior_summary=_prior_stage23_5_summary(), publishes_checkpoint=True)
    calls = _patch_stage_chain(monkeypatch, s23)

    summary = s23.run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s23.ROUTE_BOUNDARY
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]
    assert calls == []


def _write_config(tmp_path: Path, *, prior_summary: dict, publishes_checkpoint: bool = False) -> Path:
    prior_root = tmp_path / "prior_stage23_5"
    prior_root.mkdir()
    (prior_root / "xunce-stage23-5-summary.json").write_text(json.dumps(prior_summary), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "fixture_high_res",
                "projection": {"map_scale_meters_per_pixel": 4.0},
                "products": [
                    {
                        "product_id": "dem",
                        "role": "dem",
                        "resolution_m": 4.0,
                        "source_url": "file:///tmp/dem.tif",
                        "source_hash_policy": "runtime_sha256",
                        "files": [{"name": "dem.tif", "url": "file:///tmp/dem.tif"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    s23_2a = tmp_path / "base_23_2a.json"
    _write_json(
        s23_2a,
        {
            "schema_version": "xunce-stage23-2a-high-resolution-terrain-data-ingestion-config/v1",
            "primary_manifest": str(manifest),
            "raw_data_root": str(tmp_path / "raw"),
            "download_missing": False,
            "run_stage23_1_smoke": True,
            "roi_windows": [{"roi_name": "old", "x": 0, "y": 0, "width": 4, "height": 4}],
        },
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "schema_version": "xunce-stage23-5a-repair-required-inputs-high-res-roi-coverage-config/v1",
            "stage23_5_root": str(prior_root),
            "stage23_2a_base_config": str(s23_2a),
            "stage23_2b_base_config": str(tmp_path / "base_23_2b.json"),
            "stage23_2_base_config": str(tmp_path / "base_23_2.json"),
            "stage23_3_base_config": str(tmp_path / "base_23_3.json"),
            "stage23_4_base_config": str(tmp_path / "base_23_4.json"),
            "stage23_5_base_config": str(tmp_path / "base_23_5.json"),
            "primary_manifest": str(manifest),
            "raw_data_root": str(tmp_path / "raw"),
            "download_missing": False,
            "min_valid_slice_count": 2,
            "candidate_count": 8,
            "roi_candidate_windows": [
                {"roi_name": "a", "x": 0, "y": 0, "width": 4, "height": 4, "candidate_count": 8, "seed": 230201, "start_cell": [0, 0]},
                {"roi_name": "b", "x": 4, "y": 0, "width": 4, "height": 4, "candidate_count": 8, "seed": 230202, "start_cell": [0, 0]},
                {"roi_name": "c", "x": 8, "y": 0, "width": 4, "height": 4, "candidate_count": 8, "seed": 230203, "start_cell": [0, 0]},
            ],
            "publishes_checkpoint": publishes_checkpoint,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    for name, schema in (
        ("base_23_2b.json", "xunce-stage23-2b-platform-geometry-sensor-contract-alignment-config/v1"),
        ("base_23_2.json", "xunce-stage23-2-slope-obstacle-aware-theta-reward-contract-config/v1"),
        ("base_23_3.json", "xunce-stage23-3-slope-obstacle-aware-theta-ppo-collector-smoke-config/v1"),
        ("base_23_4.json", "xunce-stage23-4-slope-obstacle-aware-theta-ppo-update-smoke-config/v1"),
        ("base_23_5.json", "xunce-stage23-5-slope-obstacle-aware-theta-post-update-trajectory-eval-smoke-config/v1"),
    ):
        _write_json(tmp_path / name, {"schema_version": schema})
    return config


def _patch_roi_precheck(monkeypatch, s23, *, valid_indexes: set[int]) -> None:
    product = SimpleNamespace(path=Path("dem.tif"), resolution_m=4.0, projection={})

    def fake_prepare_products(*args, **kwargs):
        return {"dem": product}, [], []

    def fake_read_window(*args, **kwargs):
        x = int(kwargs["x"])
        index = x // 4
        all_nodata = index not in valid_indexes
        mask = ((all_nodata, all_nodata), (all_nodata, all_nodata))
        return SimpleNamespace(width=2, height=2, nodata_mask=mask, reader_backend="fixture")

    monkeypatch.setattr(s23.stage23_2a, "_prepare_products", fake_prepare_products)
    monkeypatch.setattr(s23, "read_geotiff_window", fake_read_window)


def _patch_stage_chain(
    monkeypatch,
    s23,
    *,
    stage23_2b_status: str = "passed",
    stage23_2b_route: str = "implement_stage23_2_slope_obstacle_aware_theta_reward_contract",
    final_route: str = "repair_stage23_slope_theta_policy_update_signal_strength",
    final_status: str = "failed",
    final_short: bool = False,
) -> list[str]:
    calls: list[str] = []

    def fake_2b(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage23_2b")
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "xunce-stage23-2b-rerun-stage23-2a-summary.json").write_text(
            json.dumps({"high_res_slice_count": 2, "scenario_count": 2}),
            encoding="utf-8",
        )
        return {
            "status": stage23_2b_status,
            "next_required_change": stage23_2b_route,
            "high_res_slice_count": 2,
            "default_high_res_slice_count": 2,
            "platform_contract_hash": "hash",
            "max_traversable_slope_deg": 30.0,
        }

    def fake_2(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage23_2")
        return {"status": "passed", "next_required_change": "run_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke", "platform_contract_hash": "hash", "max_traversable_slope_deg": 30.0}

    def fake_3(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage23_3")
        return {"status": "passed", "next_required_change": "run_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke", "platform_contract_hash": "hash", "max_traversable_slope_deg": 30.0}

    def fake_4(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage23_4")
        return {"status": "passed", "next_required_change": "run_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke", "platform_contract_hash": "hash", "max_traversable_slope_deg": 30.0}

    def fake_5(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        calls.append("stage23_5")
        return {
            "status": final_status,
            "next_required_change": final_route,
            "stage21_5_execution_incomplete": final_short,
            "stage21_5_execution_reason_codes": ["stage21_5_pre_scenario_count_short"] if final_short else [],
            "blocking_reason_codes": ["stage21_5_pre_scenario_count_short"] if final_short else [],
            "action_probability_audit_is_partial_diagnostic": final_short,
            "strong_state_join_available_count": 8,
            "selected_viewpoint_changed_count": 0,
            "selected_theta_changed_count": 0,
            "selected_action_changed_count": 0,
            "mean_abs_probability_delta": 0.0,
            "final_coverage_delta": 0.0,
            "coverage_auc_delta": 0.0,
            "path_cost_delta": 0.0,
            "hard_risk_violation_count": 0,
            "mask_violation_count": 0,
            "path_planning_failure_count": 0,
            "open_grid_fallback_count": 0,
            "platform_contract_hash": "hash",
        }

    monkeypatch.setattr(s23.stage23_2b, "run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment", fake_2b)
    monkeypatch.setattr(s23.stage23_2, "run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract", fake_2)
    monkeypatch.setattr(s23.stage23_3, "run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke", fake_3)
    monkeypatch.setattr(s23.stage23_4, "run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke", fake_4)
    monkeypatch.setattr(s23.stage23_5, "run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke", fake_5)
    return calls


def _prior_stage23_5_summary(
    *,
    route: str = "rerun_stage23_5_required_inputs",
    reasons: list[str] | None = None,
    slope_theta_inference_required_field_missing_count: int = 0,
) -> dict:
    return {
        "status": "failed",
        "next_required_change": route,
        "stage21_5_execution_incomplete": route == "rerun_stage23_5_required_inputs",
        "stage21_5_execution_reason_codes": reasons
        if reasons is not None
        else [
            "stage21_5_post_scenario_count_short",
            "stage21_5_post_xunce_episode_count_short",
            "stage21_5_pre_scenario_count_short",
            "stage21_5_pre_xunce_episode_count_short",
        ],
        "slope_theta_inference_required_field_missing_count": slope_theta_inference_required_field_missing_count,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
