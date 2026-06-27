import json
from pathlib import Path


def test_stage26_7b_passes_with_main_coverable_semantics(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7b_coverable_cell_semantics_contract as stage26_7b

    stage_root = _write_stage26_7_root(tmp_path)
    config = _write_config(tmp_path, stage_root)
    summary = stage26_7b.run_xunce_stage26_7b_coverable_cell_semantics_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "rerun_stage26_7_path_efficiency_with_main_coverable_coverage"
    assert summary["main_coverable_denominator_cells_total"] == 13
    assert summary["hazard_observable_denominator_cells_total"] == 3
    assert summary["hard_obstacle_in_main_denominator_count"] == 0
    assert summary["los_only_passable_not_main_count"] == 0
    assert summary["physical_obstacle_payload_count"] == 0
    audit = json.loads((tmp_path / "out" / "xunce-stage26-7b-coverable-cell-semantics-audit.json").read_text(encoding="utf-8"))
    row = audit["rows"][0]
    assert row["synthetic_los_only_blocker_cell_count"] == 1
    assert row["synthetic_los_only_blocker_main_coverable_count"] == 1
    assert row["physical_obstacle_cells_written"] is False
    assert row["physical_obstacle_payload_present"] is False


def test_stage26_7b_rejects_missing_stage26_7_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7b_coverable_cell_semantics_contract as stage26_7b

    config = _write_config(tmp_path, tmp_path / "missing")
    summary = stage26_7b.run_xunce_stage26_7b_coverable_cell_semantics_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_7b_required_inputs"


def test_stage26_7b_rejects_synthetic_physical_pollution(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7b_coverable_cell_semantics_contract as stage26_7b

    stage_root = _write_stage26_7_root(tmp_path, physical_obstacle_cells_written=True)
    config = _write_config(tmp_path, stage_root)
    summary = stage26_7b.run_xunce_stage26_7b_coverable_cell_semantics_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_7b_synthetic_source_semantics"


def test_stage26_7b_rejects_synthetic_physical_payload_even_when_flag_false(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_7b_coverable_cell_semantics_contract as stage26_7b

    stage_root = _write_stage26_7_root(tmp_path, physical_obstacle_cells=[[0, 0]])
    config = _write_config(tmp_path, stage_root)
    summary = stage26_7b.run_xunce_stage26_7b_coverable_cell_semantics_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=Path(__file__).resolve().parents[1],
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_7b_synthetic_source_semantics"
    assert summary["physical_obstacle_payload_count"] == 1


def test_stage26_7b_rejects_wrong_denominator_config(tmp_path: Path) -> None:
    import pytest
    import scripts.run_xunce_stage26_7b_coverable_cell_semantics_contract as stage26_7b

    stage_root = _write_stage26_7_root(tmp_path)
    config = _write_config(tmp_path, stage_root)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["coverage_denominator_mode"] = "roi_valid_cells"
    config.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="coverage_denominator_mode"):
        stage26_7b.run_xunce_stage26_7b_coverable_cell_semantics_contract(
            config_path=config,
            output_root=tmp_path / "out",
            repo_root=Path(__file__).resolve().parents[1],
        )


def _write_stage26_7_root(
    tmp_path: Path,
    *,
    physical_obstacle_cells_written: bool = False,
    physical_obstacle_cells: list[list[int]] | None = None,
) -> Path:
    root = tmp_path / "stage26_7"
    combo = root / "c2"
    sidecar_dir = combo / "s26_1" / "src" / "sc"
    sidecar_dir.mkdir(parents=True)
    sidecar = {
        "passable_mask": [[True, True, True, True] for _ in range(4)],
        "physical_obstacle_cells": physical_obstacle_cells or [],
        "slope_blocked_cells": [[1, 0]],
        "blocked_cells": [[2, 0]],
        "synthetic_hard_obstacle_cells": [[3, 0]],
        "synthetic_los_blocker_cells": [[0, 1]],
        "synthetic_high_risk_cells": [[1, 1]],
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": physical_obstacle_cells_written,
    }
    (sidecar_dir / "s26_1_000.sidecar.json").write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    (root / "xunce-stage26-7-summary.json").write_text(
        json.dumps(
            {
                "stage_id": "xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair",
                "status": "failed",
                "next_required_change": "repair_stage26_7_path_efficiency_target_score",
                "best_combo_id": "path_efficiency_v2_strict_cost",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "xunce-stage26-7-post-update-eval-comparison.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage26-7-post-update-eval-comparison/v1",
                "best_combo": {
                    "combo_id": "path_efficiency_v2_strict_cost",
                    "combo_root": str(combo),
                    "final_coverage_delta": 0.1,
                    "coverage_auc_delta": 0.2,
                    "coverage_per_100m_delta": 0.3,
                    "hybrid_astar_path_cost_delta": 1.0,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def _write_config(tmp_path: Path, stage26_7_root: Path) -> Path:
    config = {
        "schema_version": "xunce-stage26-7b-coverable-cell-semantics-contract-config/v1",
        "stage_id": "xunce-stage26-7b-coverable-cell-semantics-contract",
        "stage26_7_root": str(stage26_7_root),
        "stage26_0_root": str(tmp_path / "stage26_0"),
        "best_combo_work_dir": "c2",
        "coverage_denominator_mode": "main_coverable_cells",
        "coverage_denominator_source": "main_coverable_cells/v1",
        "derive_slope_blocked_cells_from_sidecar_dem": False,
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "stage26_7b_config.json"
    path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return path
