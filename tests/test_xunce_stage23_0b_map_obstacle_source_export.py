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


def test_quasi_real_sidecar_exports_blocked_cells_from_passable_mask_false() -> None:
    from scripts.run_quasi_real_map_path_feedback_bridge import _sidecar_from_roi

    sidecar = _sidecar_from_roi(
        dem_values=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        count_values=[
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        contract={"top_goals": []},
        scenario_id="scenario-001",
        data_manifest={"dataset_id": "fixture", "region": "unit"},
        roi=SimpleNamespace(name="roi-a", split="test", bounds=[0, 0, 3, 3]),
        resolution=1.0,
    )

    assert [1, 1] in sidecar["blocked_cells"]
    assert sidecar["blocked_source_kind"] == "passable_mask_false"
    assert sidecar["metadata"]["blocked_count"] == len(sidecar["blocked_cells"])
    assert "obstacle_cells" not in sidecar


def test_quasi_real_sidecar_exports_slope_blocked_cells_from_physical_slope() -> None:
    from scripts.run_quasi_real_map_path_feedback_bridge import _sidecar_from_roi

    sidecar = _sidecar_from_roi(
        dem_values=[
            [0.0, 10.0],
            [0.0, 0.0],
        ],
        count_values=[
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        contract={"top_goals": []},
        scenario_id="scenario-slope",
        data_manifest={"dataset_id": "fixture", "region": "unit"},
        roi=SimpleNamespace(name="roi-slope", split="test", bounds=[0, 0, 2, 2]),
        resolution=20.0,
        max_traversable_slope_deg=20.0,
    )

    assert "slope_deg" in sidecar["terrain_layers"]
    assert sidecar["max_traversable_slope_deg"] == 20.0
    assert sidecar["slope_blocked_source_kind"] == "slope_gt_max_traversable_deg"
    assert sidecar["slope_blocked_cells"]
    assert sidecar["slope_blocked_cell_count"] == len(sidecar["slope_blocked_cells"])
    assert "obstacle_cells" not in sidecar


def test_stage23_0b_routes_to_proxy_review_when_stage23_0a_rerun_has_only_blocked_proxy(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_0b_map_obstacle_source_export import (
        SUMMARY_FILE,
        run_xunce_stage23_0b_map_obstacle_source_export,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        next_required_change="review_blocked_as_obstacle_proxy_semantics",
        status="passed",
        obstacle_source_count=1,
        kind_counts={"blocked_as_obstacle_proxy": 1},
        candidate_missing=0,
        los_rows=8,
    )
    output = tmp_path / "out"
    summary = run_xunce_stage23_0b_map_obstacle_source_export(
        config_path=_write_config(tmp_path, stage23_0a_root),
        output_root=output,
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "review_blocked_as_obstacle_proxy_semantics"
    assert summary["stage23_0a_obstacle_source_count"] == 1
    assert summary["stage23_0a_candidate_obstacle_source_missing_count"] == 0
    assert summary["stage23_0_los_audit_row_count"] == 8
    assert summary["only_blocked_proxy_sources_available"] is True
    assert summary["only_proxy_sources_available"] is True
    assert (output / SUMMARY_FILE).is_file()


def test_stage23_0b_routes_to_proxy_review_when_stage23_0a_rerun_has_only_slope_proxy(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_0b_map_obstacle_source_export import (
        run_xunce_stage23_0b_map_obstacle_source_export,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        next_required_change="review_blocked_as_obstacle_proxy_semantics",
        status="passed",
        obstacle_source_count=1,
        kind_counts={"slope_blocked_as_obstacle_proxy": 1},
        candidate_missing=0,
        los_rows=8,
    )

    summary = run_xunce_stage23_0b_map_obstacle_source_export(
        config_path=_write_config(tmp_path, stage23_0a_root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "review_blocked_as_obstacle_proxy_semantics"
    assert summary["route_reason"] == "only_proxy_sources_available"
    assert summary["only_blocked_proxy_sources_available"] is False
    assert summary["only_proxy_sources_available"] is True


def test_stage23_0b_routes_to_continue_repair_when_rerun_still_has_no_source(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_0b_map_obstacle_source_export import (
        run_xunce_stage23_0b_map_obstacle_source_export,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        next_required_change="repair_stage23_map_obstacle_source_export",
        status="failed",
        obstacle_source_count=0,
        kind_counts={},
        candidate_missing=12,
        los_rows=0,
    )

    summary = run_xunce_stage23_0b_map_obstacle_source_export(
        config_path=_write_config(tmp_path, stage23_0a_root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "continue_stage23_map_obstacle_source_export_repair"
    assert "stage23_0a_still_missing_obstacle_sources" in summary["blocking_reason_codes"]


def test_stage23_0b_does_not_route_physical_source_without_material_stage23_0(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_0b_map_obstacle_source_export import (
        run_xunce_stage23_0b_map_obstacle_source_export,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        next_required_change="document_obstacle_occlusion_audit_only",
        status="passed",
        obstacle_source_count=1,
        kind_counts={"physical_obstacle_cells": 1},
        candidate_missing=0,
        los_rows=8,
        stage23_0_material=False,
    )

    summary = run_xunce_stage23_0b_map_obstacle_source_export(
        config_path=_write_config(tmp_path, stage23_0a_root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "document_obstacle_occlusion_audit_only"


def test_stage23_0b_routes_existing_source_candidate_missing_to_lineage_repair(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_0b_map_obstacle_source_export import (
        run_xunce_stage23_0b_map_obstacle_source_export,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        next_required_change="repair_stage23_0a_candidate_obstacle_source_linkage",
        status="failed",
        obstacle_source_count=1,
        kind_counts={"blocked_as_obstacle_proxy": 1},
        candidate_missing=4,
        los_rows=0,
    )

    summary = run_xunce_stage23_0b_map_obstacle_source_export(
        config_path=_write_config(tmp_path, stage23_0a_root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_obstacle_source_lineage"
    assert "candidate_obstacle_source_missing" in summary["blocking_reason_codes"]


def test_stage23_0b_rejects_boundary_flags_before_success_route(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_0b_map_obstacle_source_export import (
        run_xunce_stage23_0b_map_obstacle_source_export,
    )

    stage23_0a_root = _write_stage23_0a_root(
        tmp_path / "s23_0a",
        next_required_change="review_blocked_as_obstacle_proxy_semantics",
        status="passed",
        obstacle_source_count=1,
        kind_counts={"blocked_as_obstacle_proxy": 1},
        candidate_missing=0,
        los_rows=8,
    )
    config = json.loads(_write_config(tmp_path, stage23_0a_root).read_text(encoding="utf-8"))
    config["publishes_checkpoint"] = True
    config_path = tmp_path / "bad-boundary.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    summary = run_xunce_stage23_0b_map_obstacle_source_export(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage23_0b_boundary_rejections"
    assert "publishes_checkpoint_must_be_false" in summary["blocking_reason_codes"]


def test_stage23_0a_counts_unknown_candidate_source_id_as_hash_mismatch(tmp_path: Path) -> None:
    from scripts.xunce_obstacle_aware_theta_sensor_coverage import stable_obstacle_source_hash
    from scripts.run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los import (
        run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los,
    )

    source_hash = stable_obstacle_source_hash(
        source_kind="blocked_as_obstacle_proxy",
        obstacle_cells=[[1, 0]],
        no_go_blocks_los=False,
    )
    high_fidelity_root = _write_high_fidelity_root_for_stage23_0a(
        tmp_path,
        source_kind="blocked_as_obstacle_proxy",
        source_hash=source_hash,
        candidate_source_id="scenario:scenario-001:missing-source",
    )

    summary = run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los(
        config_path=_write_stage23_0a_config(tmp_path, high_fidelity_root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_0a_candidate_obstacle_source_linkage"
    assert summary["candidate_obstacle_source_hash_mismatch_count"] == 1


def test_sidecar_source_export_audit_counts_passable_false_and_blocked_cells(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_0b_map_obstacle_source_export import _sidecar_export_audit_from_source_root

    source_root = tmp_path / "source"
    source_root.mkdir()
    sidecar = source_root / "scenario.sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "path-planner-sidecar/v1",
                "passable_mask": [[True, False], [True, True]],
                "blocked_cells": [[1, 0]],
                "blocked_source_kind": "passable_mask_false",
                "slope_blocked_cells": [[0, 1], [1, 1]],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (source_root / "xunce-high-fidelity-real-map-slices.jsonl").write_text(
        json.dumps({"scenario_id": "scenario-001", "sidecar": str(sidecar)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit = _sidecar_export_audit_from_source_root(source_root)

    assert audit["source_roi_expansion_root"] == str(source_root)
    assert audit["sidecar_count"] == 1
    assert audit["passable_mask_false_cell_count_total"] == 1
    assert audit["sidecar_blocked_cells_cell_count_total"] == 1
    assert audit["sidecar_slope_blocked_cells_cell_count_total"] == 2
    assert audit["sidecar_with_blocked_cells_count"] == 1


def _write_config(tmp_path: Path, stage23_0a_root: Path) -> Path:
    payload = {
        "schema_version": "xunce-stage23-0b-map-obstacle-source-export-config/v1",
        "run_stage23_0a": False,
        "stage23_0a_root": str(stage23_0a_root),
        "stage23_0b_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_stage23_0a_root(
    root: Path,
    *,
    next_required_change: str,
    status: str,
    obstacle_source_count: int,
    kind_counts: dict[str, int],
    candidate_missing: int,
    los_rows: int,
    stage23_0_material: bool = True,
) -> Path:
    root.mkdir(parents=True)
    summary = {
        "schema_version": "xunce-stage23-0a-summary/v1",
        "status": status,
        "next_required_change": next_required_change,
        "obstacle_source_count": obstacle_source_count,
        "obstacle_source_kind_counts": kind_counts,
        "only_blocked_proxy_sources_available": kind_counts == {"blocked_as_obstacle_proxy": obstacle_source_count},
        "only_proxy_sources_available": obstacle_source_count > 0
        and set(kind_counts).issubset(
            {
                "blocked_as_obstacle_proxy",
                "slope_blocked_as_obstacle_proxy",
                "no_go_as_obstacle_proxy",
            }
        ),
        "candidate_row_count": 12,
        "candidate_obstacle_source_missing_count": candidate_missing,
        "candidate_obstacle_source_hash_mismatch_count": 0,
        "stage23_0_los_audit_row_count": los_rows,
        "stage23_0_status": "passed" if los_rows else "failed",
        "stage23_0_obstacle_occlusion_material_to_coverage": bool(stage23_0_material),
        "stage23_0_next_required_change": "implement_stage23_1_obstacle_aware_theta_reward_contract"
        if los_rows and "physical_obstacle_cells" in kind_counts and stage23_0_material
        else "document_obstacle_occlusion_audit_only"
        if los_rows and "physical_obstacle_cells" in kind_counts
        else "review_blocked_as_obstacle_proxy_semantics"
        if los_rows
        else "rerun_stage23_0_required_obstacle_sources",
        "stage23_0a_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    (root / "xunce-stage23-0a-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "xunce-stage23-0a-obstacle-source-audit.json").write_text(
        json.dumps(
            {
                "source_count": obstacle_source_count,
                "obstacle_source_kind_counts": kind_counts,
                "only_blocked_proxy_sources_available": summary["only_blocked_proxy_sources_available"],
                "only_proxy_sources_available": summary["only_proxy_sources_available"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def _write_high_fidelity_root_for_stage23_0a(
    tmp_path: Path,
    *,
    source_kind: str,
    source_hash: str,
    candidate_source_id: str,
) -> Path:
    root = tmp_path / "hf"
    root.mkdir()
    source_id = f"scenario:scenario-001:{source_kind}"
    (root / "xunce-exploration-coverage-obstacle-sources.json").write_text(
        json.dumps(
            {
                "schema_version": "xunce-exploration-coverage-obstacle-sources/v1",
                "source_count": 1,
                "sources": [
                    {
                        "scenario_id": "scenario-001",
                        "obstacle_source_id": source_id,
                        "obstacle_source_kind": source_kind,
                        "obstacle_source_hash": source_hash,
                        "obstacle_cells": [[1, 0]],
                        "obstacle_source_is_proxy": True,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    row = {
        "scenario_id": "scenario-001",
        "step_index": 0,
        "candidate_index": 0,
        "candidate_cell": [0, 0],
        "candidate_theta_deg": 0,
        "candidate_viewpoint": [0, 0, 0],
        "sensor_range_cells": 4,
        "sensor_fov_deg": 30.0,
        "obstacle_source_id": candidate_source_id,
        "obstacle_source_hash": source_hash,
        "obstacle_source_kind": source_kind,
        "obstacle_source_missing": False,
    }
    (root / "xunce-exploration-coverage-candidate-metric-audit.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _write_stage23_0a_config(tmp_path: Path, high_fidelity_root: Path) -> Path:
    payload = {
        "schema_version": "xunce-stage23-0a-materialize-obstacle-sources-for-theta-los-config/v1",
        "high_fidelity_root": str(high_fidelity_root),
        "run_high_fidelity_smoke": False,
        "stage23_0a_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    path = tmp_path / "stage23_0a_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
