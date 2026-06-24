from __future__ import annotations

import json
from pathlib import Path

from scripts.xunce_obstacle_aware_theta_sensor_coverage import stable_obstacle_source_hash
from scripts.run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract import (
    LOS_AUDIT_FILE,
    SUMMARY_FILE,
    run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract,
)


def test_stage23_0_routes_to_stage23_1_when_obstacles_change_theta_coverage(tmp_path: Path) -> None:
    root = _write_candidate_root(
        tmp_path,
        [
            _candidate_row(
                obstacle_cells=[[1, 0]],
                sensor_fov_deg=30.0,
                sensor_range_cells=4,
            )
        ],
    )
    output = tmp_path / "out"
    summary = run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract(
        config_path=_write_config(tmp_path, root),
        output_root=output,
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage23_1_obstacle_aware_theta_reward_contract"
    assert summary["obstacle_occlusion_material_to_coverage"] is True
    assert summary["material_occlusion_viewpoint_count"] == 1
    rows = _read_jsonl(output / LOS_AUDIT_FILE)
    assert rows[0]["visibility_removed_by_obstacle_count"] > 0
    assert rows[0]["clear_theta_coverage_hash"] != rows[0]["obstacle_aware_theta_coverage_hash"]


def test_stage23_0_requires_obstacle_source_when_missing(tmp_path: Path) -> None:
    root = _write_candidate_root(tmp_path, [_candidate_row()])
    summary = run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract(
        config_path=_write_config(tmp_path, root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage23_0_required_obstacle_sources"
    assert "obstacle_source_missing" in summary["blocking_reason_codes"]


def test_stage23_0_resolves_obstacle_source_artifact_linkage(tmp_path: Path) -> None:
    obstacle_cells = [[1, 0]]
    source_hash = stable_obstacle_source_hash(
        source_kind="blocked_as_obstacle_proxy",
        obstacle_cells=obstacle_cells,
        no_go_blocks_los=False,
    )
    root = _write_candidate_root(
        tmp_path,
        [
            _candidate_row(
                obstacle_source_id="scenario:scenario-001:blocked_as_obstacle_proxy",
                obstacle_source_hash=source_hash,
                obstacle_source_kind="blocked_as_obstacle_proxy",
                obstacle_source_missing=False,
                sensor_fov_deg=30.0,
                sensor_range_cells=4,
            )
        ],
    )
    _write_obstacle_sources(
        root,
        [
            {
                "obstacle_source_id": "scenario:scenario-001:blocked_as_obstacle_proxy",
                "scenario_id": "scenario-001",
                "obstacle_source_kind": "blocked_as_obstacle_proxy",
                "obstacle_source_hash": source_hash,
                "obstacle_cells": obstacle_cells,
                "obstacle_source_is_proxy": True,
                "obstacle_cell_count": 1,
            }
        ],
    )
    output = tmp_path / "out"
    summary = run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract(
        config_path=_write_config(tmp_path, root),
        output_root=output,
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["obstacle_source_available_count"] == 1
    assert summary["obstacle_source_missing_count"] == 0
    rows = _read_jsonl(output / LOS_AUDIT_FILE)
    assert rows[0]["obstacle_source_id"] == "scenario:scenario-001:blocked_as_obstacle_proxy"
    assert rows[0]["obstacle_source_hash"] == source_hash
    assert rows[0]["visibility_removed_by_obstacle_count"] > 0


def test_stage23_0_documents_audit_only_when_obstacles_do_not_change_coverage(tmp_path: Path) -> None:
    root = _write_candidate_root(
        tmp_path,
        [
            _candidate_row(
                obstacle_cells=[[0, 4]],
                sensor_fov_deg=30.0,
                sensor_range_cells=4,
            )
        ],
    )
    summary = run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract(
        config_path=_write_config(tmp_path, root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "document_obstacle_occlusion_audit_only"
    assert summary["obstacle_occlusion_material_to_coverage"] is False


def test_stage23_0_boundary_flags_hard_fail(tmp_path: Path) -> None:
    root = _write_candidate_root(tmp_path, [_candidate_row(obstacle_cells=[[1, 0]])])
    config = _write_config(tmp_path, root, extra={"publishes_checkpoint": True})
    summary = run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage23_0_boundary_rejections"
    assert "publishes_checkpoint_enabled" in summary["blocking_reason_codes"]


def _candidate_row(**extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "xunce-exploration-coverage-candidate-metric-audit-row/v1",
        "scenario_id": "scenario-001",
        "step_index": 0,
        "candidate_index": 0,
        "candidate_cell": [0, 0],
        "candidate_theta_deg": 0,
        "candidate_viewpoint": [0, 0, 0],
        "sensor_range_cells": 4,
        "sensor_fov_deg": 30.0,
        "theta_coverage_hash": "legacy-clear-hash",
    }
    row.update(extra)
    return row


def _write_candidate_root(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    root = tmp_path / "hf"
    root.mkdir()
    (root / "xunce-exploration-coverage-candidate-metric-audit.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return root


def _write_obstacle_sources(root: Path, sources: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": "xunce-exploration-coverage-obstacle-sources/v1",
        "source_count": len(sources),
        "sources": sources,
    }
    (root / "xunce-exploration-coverage-obstacle-sources.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_config(tmp_path: Path, high_fidelity_root: Path, extra: dict[str, object] | None = None) -> Path:
    payload: dict[str, object] = {
        "schema_version": "xunce-stage23-0-endpoint-obstacle-aware-theta-sensor-coverage-contract-config/v1",
        "high_fidelity_root": str(high_fidelity_root),
        "max_audit_rows": 100,
        "stage23_0_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    if extra:
        payload.update(extra)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
