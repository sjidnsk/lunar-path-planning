from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_7c_generates_main_coverable_stage26_7_config_and_routes_multi_seed(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun as s26

    stage26_7b_root = _write_stage26_7b_root(tmp_path)
    base_config = _write_stage26_7_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_7b_root, base_config)

    def fake_stage26_7(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["coverage_denominator_mode"] == "main_coverable_cells"
        assert cfg["coverage_denominator_source"] == "main_coverable_cells/v1"
        _write_json(output_root / "c2" / "xunce-stage26-7-stage26-3-config.json", cfg)
        _write_json(
            output_root / "c2" / "s26_3" / "xunce-stage26-3-stage21-5-config.json",
            {
                "coverage_denominator_mode": "main_coverable_cells",
                "coverage_denominator_source": "main_coverable_cells/v1",
            },
        )
        _write_json(
            output_root / "c2" / "s26_3" / "xunce-stage26-3-high-fidelity-config.json",
            {
                "coverage_denominator_mode": "main_coverable_cells",
                "coverage_denominator_source": "main_coverable_cells/v1",
            },
        )
        return _stage26_7_summary(status="passed", route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot")

    monkeypatch.setattr(s26.stage26_7, "run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair", fake_stage26_7)
    summary = s26.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    stage26_7_config = json.loads((tmp_path / "out" / "xunce-stage26-7c-stage26-7-config.json").read_text(encoding="utf-8"))
    binding = json.loads((tmp_path / "out" / "xunce-stage26-7c-denominator-binding-audit.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
    assert summary["hybrid_astar_path_cost_delta"] == 12.0
    assert summary["main_coverage_per_100m_delta"] == 0.5
    assert stage26_7_config["coverage_denominator_mode"] == "main_coverable_cells"
    assert binding["denominator_binding_missing_count"] == 0


def test_stage26_7c_rejects_unready_stage26_7b(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun as s26

    stage26_7b_root = _write_stage26_7b_root(tmp_path, status="failed")
    base_config = _write_stage26_7_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_7b_root, base_config)

    called = {"value": False}

    def fake_stage26_7(**_: object) -> dict:
        called["value"] = True
        return {}

    monkeypatch.setattr(s26.stage26_7, "run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair", fake_stage26_7)
    summary = s26.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_7c_required_inputs"
    assert called["value"] is False


def test_stage26_7c_routes_target_score_when_unit_distance_coverage_declines(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun as s26

    stage26_7b_root = _write_stage26_7b_root(tmp_path)
    base_config = _write_stage26_7_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_7b_root, base_config)

    def fake_stage26_7(*, output_root: Path, **_: object) -> dict:
        cfg = {"coverage_denominator_mode": "main_coverable_cells", "coverage_denominator_source": "main_coverable_cells/v1"}
        _write_json(output_root / "c2" / "xunce-stage26-7-stage26-3-config.json", cfg)
        return _stage26_7_summary(status="failed", route="repair_stage26_7_path_efficiency_target_score", c100=-0.1)

    monkeypatch.setattr(s26.stage26_7, "run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair", fake_stage26_7)
    summary = s26.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_7_path_efficiency_target_score"


def test_stage26_7c_blocks_passed_child_when_scenario_regresses(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun as s26

    stage26_7b_root = _write_stage26_7b_root(tmp_path)
    base_config = _write_stage26_7_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_7b_root, base_config)

    def fake_stage26_7(*, output_root: Path, **_: object) -> dict:
        cfg = {"coverage_denominator_mode": "main_coverable_cells", "coverage_denominator_source": "main_coverable_cells/v1"}
        _write_json(output_root / "c2" / "xunce-stage26-7-stage26-3-config.json", cfg)
        summary = _stage26_7_summary(status="passed", route="run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot")
        summary["best_combo_scenario_regression_count"] = 1
        return summary

    monkeypatch.setattr(s26.stage26_7, "run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair", fake_stage26_7)
    summary = s26.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_7_path_efficiency_target_score"
    assert summary["scenario_regression_count"] == 1


def test_stage26_7c_routes_binding_repair_when_denominator_missing(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun as s26

    stage26_7b_root = _write_stage26_7b_root(tmp_path)
    base_config = _write_stage26_7_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_7b_root, base_config)

    def fake_stage26_7(*, output_root: Path, **_: object) -> dict:
        _write_json(output_root / "c2" / "xunce-stage26-7-stage26-3-config.json", {})
        return _stage26_7_summary(status="failed", route="repair_stage26_7_path_efficiency_target_score")

    monkeypatch.setattr(s26.stage26_7, "run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair", fake_stage26_7)
    summary = s26.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_7c_main_coverable_binding"
    assert summary["denominator_binding_missing_count"] > 0


def test_stage26_7c_preserves_child_sampler_blocker(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun as s26

    stage26_7b_root = _write_stage26_7b_root(tmp_path)
    base_config = _write_stage26_7_base_config(tmp_path)
    config = _write_config(tmp_path, stage26_7b_root, base_config)

    def fake_stage26_7(*, output_root: Path, **_: object) -> dict:
        cfg = {"coverage_denominator_mode": "main_coverable_cells", "coverage_denominator_source": "main_coverable_cells/v1"}
        _write_json(output_root / "c2" / "xunce-stage26-7-stage26-3-config.json", cfg)
        return _stage26_7_summary(status="failed", route="repair_stage26_7_path_efficiency_credit_sampler", c100=0.0)

    monkeypatch.setattr(s26.stage26_7, "run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair", fake_stage26_7)
    summary = s26.run_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_7_path_efficiency_credit_sampler"


def _write_stage26_7b_root(tmp_path: Path, *, status: str = "passed") -> Path:
    root = tmp_path / "stage26_7b"
    _write_json(
        root / "xunce-stage26-7b-summary.json",
        {
            "schema_version": "xunce-stage26-7b-summary/v1",
            "stage_id": "xunce-stage26-7b-coverable-cell-semantics-contract",
            "status": status,
            "next_required_change": "rerun_stage26_7_path_efficiency_with_main_coverable_coverage",
        },
    )
    return root


def _write_stage26_7_base_config(tmp_path: Path) -> Path:
    path = tmp_path / "stage26_7_base.json"
    _write_json(
        path,
        {
            "schema_version": "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair-config/v1",
            "stage_id": "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair",
            "stage26_6_root": str(tmp_path / "stage26_6"),
            "stage26_0_root": str(tmp_path / "stage26_0"),
            "score_sweep": [
                {
                    "combo_id": "path_efficiency_v2_strict_cost",
                    "work_dir": "c2",
                    "synthetic_credit_score_version": "path_efficiency_v2",
                    "path_efficiency_max_cost_norm": 0.55,
                }
            ],
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return path


def _write_config(tmp_path: Path, stage26_7b_root: Path, base_config: Path) -> Path:
    path = tmp_path / "stage26_7c.json"
    _write_json(
        path,
        {
            "schema_version": "xunce-stage26-7c-main-coverable-coverage-efficiency-rerun-config/v1",
            "stage_id": "xunce-stage26-7c-main-coverable-coverage-efficiency-rerun",
            "stage26_7b_root": str(stage26_7b_root),
            "stage26_7_base_config": str(base_config),
            "stage26_6_root": str(tmp_path / "stage26_6"),
            "stage26_0_root": str(tmp_path / "stage26_0"),
            "coverage_denominator_mode": "main_coverable_cells",
            "coverage_denominator_source": "main_coverable_cells/v1",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return path


def _stage26_7_summary(*, status: str, route: str, c100: float = 0.5) -> dict:
    return {
        "schema_version": "xunce-stage26-7-summary/v1",
        "stage_id": "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair",
        "status": status,
        "next_required_change": route,
        "coverage_denominator_mode": "main_coverable_cells",
        "coverage_denominator_source": "main_coverable_cells/v1",
        "best_combo_selected_action_changed_count": 1,
        "best_combo_final_coverage_delta": 0.01,
        "best_combo_coverage_auc_delta": 0.02,
        "best_combo_coverage_per_100m_delta": c100,
        "best_combo_hybrid_astar_path_cost_delta": 12.0,
        "best_combo_scenario_regression_count": 0,
        "hard_risk_violation_count": 0,
        "mask_violation_count": 0,
        "path_planning_failure_count": 0,
        "open_grid_fallback_count": 0,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
