from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "model-explorer" / "src"):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)


def test_stage23_1_routes_material_slope_source_to_stage23_2(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los import (
        SUMMARY_FILE,
        run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        obstacle_source_kind_counts={"slope_blocked_as_obstacle_proxy": 1},
        slope_cell_count=3,
        material=True,
    )
    summary = run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los(
        config_path=_write_config(tmp_path, stage23_0a_root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage23_2_slope_obstacle_aware_theta_reward_contract"
    assert summary["platform_contract_id"] == "agilex_scout_mini_piper"
    assert summary["platform_max_climb_deg"] == 30.0
    assert summary["max_traversable_slope_deg"] == 30.0
    assert summary["slope_sensitivity_thresholds_deg"] == [20.0, 30.0]
    assert summary["slope_source_count"] == 1
    assert summary["slope_blocked_cell_count_total"] == 3
    assert summary["slope_material_occlusion_viewpoint_count"] == 1
    assert (tmp_path / "out" / SUMMARY_FILE).is_file()


def test_stage23_1_routes_nonmaterial_slope_source_to_audit_only(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los import (
        run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        obstacle_source_kind_counts={"slope_blocked_as_obstacle_proxy": 1},
        slope_cell_count=3,
        material=False,
    )
    summary = run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los(
        config_path=_write_config(tmp_path, stage23_0a_root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "document_slope_obstacle_occlusion_audit_only"


def test_stage23_1_does_not_route_to_stage23_2_when_only_physical_source_is_material(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los import (
        run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        obstacle_source_kind_counts={"slope_blocked_as_obstacle_proxy": 1, "physical_obstacle_cells": 1},
        slope_cell_count=3,
        material=True,
        slope_material=False,
        physical_material=True,
    )
    summary = run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los(
        config_path=_write_config(tmp_path, stage23_0a_root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "document_slope_obstacle_occlusion_audit_only"
    assert summary["slope_material_occlusion_viewpoint_count"] == 0


def test_stage23_1_rejects_boundary_flags(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los import (
        run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        obstacle_source_kind_counts={"slope_blocked_as_obstacle_proxy": 1},
        slope_cell_count=3,
        material=True,
    )
    config_path = _write_config(tmp_path, stage23_0a_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["publishes_checkpoint"] = True
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    summary = run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage23_1_boundary_rejections"
    assert "publishes_checkpoint_must_be_false" in summary["blocking_reason_codes"]


def test_stage23_1_rejects_missing_slope_source(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los import (
        run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        obstacle_source_kind_counts={"blocked_as_obstacle_proxy": 1},
        slope_cell_count=0,
        material=True,
    )
    summary = run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los(
        config_path=_write_config(tmp_path, stage23_0a_root),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage23_1_required_dem_sources"
    assert "slope_obstacle_source_absent" in summary["blocking_reason_codes"]


def _write_config(tmp_path: Path, stage23_0a_root: Path) -> Path:
    path = tmp_path / "stage23-1-config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "xunce-stage23-1-slope-derived-obstacle-source-for-endpoint-theta-los-config/v1",
                "run_stage23_0a": False,
                "stage23_0a_root": str(stage23_0a_root),
                "stage23_1_authorized": False,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "canary_traffic_fraction": 0.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_stage23_0a_root(
    root: Path,
    *,
    obstacle_source_kind_counts: dict[str, int],
    slope_cell_count: int,
    material: bool,
    slope_material: bool | None = None,
    physical_material: bool = False,
) -> Path:
    root.mkdir(parents=True)
    source_kind = next(iter(obstacle_source_kind_counts), "slope_blocked_as_obstacle_proxy")
    source = {
        "obstacle_source_id": f"scenario:scenario-001:{source_kind}",
        "obstacle_source_kind": source_kind,
        "obstacle_source_hash": "hash-001",
        "obstacle_source_is_proxy": True,
        "obstacle_cell_count": slope_cell_count,
        "obstacle_cells": [[x, 0] for x in range(slope_cell_count)],
    }
    (root / "xunce-stage23-0a-summary.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "next_required_change": "review_blocked_as_obstacle_proxy_semantics",
                "obstacle_source_count": sum(obstacle_source_kind_counts.values()),
                "obstacle_source_kind_counts": obstacle_source_kind_counts,
                "candidate_obstacle_source_missing_count": 0,
                "candidate_obstacle_source_hash_mismatch_count": 0,
                "stage23_0_status": "passed",
                "stage23_0_los_audit_row_count": 8,
                "stage23_0_obstacle_occlusion_material_to_coverage": material,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "xunce-stage23-0a-obstacle-source-audit.json").write_text(
        json.dumps(
            {
                "source_count": sum(obstacle_source_kind_counts.values()),
                "obstacle_source_kind_counts": obstacle_source_kind_counts,
                "sources": [source],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    s23_0 = root / "s23_0"
    s23_0.mkdir()
    (s23_0 / "xunce-stage23-0-summary.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "next_required_change": "document_obstacle_occlusion_audit_only",
                "los_audit_row_count": 8,
                "obstacle_occlusion_material_to_coverage": material,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    slope_changed = material if slope_material is None else slope_material
    los_rows = [
        {
            "schema_version": "xunce-stage23-0-obstacle-los-audit-row/v1",
            "scenario_id": "scenario-001",
            "step_index": 0,
            "candidate_index": 0,
            "obstacle_source_kind": "slope_blocked_as_obstacle_proxy",
            "coverage_changed_by_obstacle": slope_changed,
            "visibility_removed_by_obstacle_count": 3 if slope_changed else 0,
        }
    ]
    if "physical_obstacle_cells" in obstacle_source_kind_counts:
        los_rows.append(
            {
                "schema_version": "xunce-stage23-0-obstacle-los-audit-row/v1",
                "scenario_id": "scenario-001",
                "step_index": 0,
                "candidate_index": 1,
                "obstacle_source_kind": "physical_obstacle_cells",
                "coverage_changed_by_obstacle": physical_material,
                "visibility_removed_by_obstacle_count": 4 if physical_material else 0,
            }
        )
    (s23_0 / "xunce-stage23-0-obstacle-los-audit.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in los_rows),
        encoding="utf-8",
    )
    return root
