from __future__ import annotations

import json
from pathlib import Path

from scripts.xunce_obstacle_aware_theta_sensor_coverage import stable_obstacle_source_hash
from scripts.run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los import (
    SUMMARY_FILE,
    _route,
    run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los,
)


def test_stage23_0a_reruns_stage23_0_with_materialized_physical_source(tmp_path: Path) -> None:
    source_hash = stable_obstacle_source_hash(
        source_kind="physical_obstacle_cells",
        obstacle_cells=[[1, 0]],
        no_go_blocks_los=False,
    )
    high_fidelity_root = _write_high_fidelity_root(
        tmp_path,
        source_kind="physical_obstacle_cells",
        source_hash=source_hash,
        obstacle_cells=[[1, 0]],
    )
    output = tmp_path / "out"

    summary = run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los(
        config_path=_write_config(tmp_path, high_fidelity_root),
        output_root=output,
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "implement_stage23_1_obstacle_aware_theta_reward_contract"
    assert summary["stage23_0_status"] == "passed"
    assert summary["candidate_obstacle_source_missing_count"] == 0
    assert summary["candidate_obstacle_source_hash_mismatch_count"] == 0
    assert summary["obstacle_source_kind_counts"]["physical_obstacle_cells"] == 1
    assert (output / SUMMARY_FILE).is_file()
    rerun_summary = _read_json(output / "s23_0" / "xunce-stage23-0-summary.json")
    assert rerun_summary["los_audit_row_count"] == 1


def test_stage23_0a_routes_to_proxy_review_when_only_blocked_proxy_exists(tmp_path: Path) -> None:
    source_hash = stable_obstacle_source_hash(
        source_kind="blocked_as_obstacle_proxy",
        obstacle_cells=[[1, 0]],
        no_go_blocks_los=False,
    )
    high_fidelity_root = _write_high_fidelity_root(
        tmp_path,
        source_kind="blocked_as_obstacle_proxy",
        source_hash=source_hash,
        obstacle_cells=[[1, 0]],
    )

    summary = run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los(
        config_path=_write_config(tmp_path, high_fidelity_root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "review_blocked_as_obstacle_proxy_semantics"
    assert summary["only_blocked_proxy_sources_available"] is True


def test_stage23_0a_routes_to_proxy_review_when_only_slope_proxy_exists(tmp_path: Path) -> None:
    source_hash = stable_obstacle_source_hash(
        source_kind="slope_blocked_as_obstacle_proxy",
        obstacle_cells=[[1, 0]],
        no_go_blocks_los=False,
    )
    high_fidelity_root = _write_high_fidelity_root(
        tmp_path,
        source_kind="slope_blocked_as_obstacle_proxy",
        source_hash=source_hash,
        obstacle_cells=[[1, 0]],
    )

    summary = run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los(
        config_path=_write_config(tmp_path, high_fidelity_root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "review_blocked_as_obstacle_proxy_semantics"
    assert summary["only_blocked_proxy_sources_available"] is False
    assert summary["only_proxy_sources_available"] is True


def test_stage23_0a_flags_candidate_source_hash_mismatch(tmp_path: Path) -> None:
    source_hash = stable_obstacle_source_hash(
        source_kind="physical_obstacle_cells",
        obstacle_cells=[[1, 0]],
        no_go_blocks_los=False,
    )
    high_fidelity_root = _write_high_fidelity_root(
        tmp_path,
        source_kind="physical_obstacle_cells",
        source_hash=source_hash,
        obstacle_cells=[[1, 0]],
        candidate_source_hash="bad-hash",
    )

    summary = run_xunce_stage23_0a_materialize_obstacle_sources_for_theta_los(
        config_path=_write_config(tmp_path, high_fidelity_root),
        output_root=tmp_path / "out",
        repo_root=Path.cwd(),
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage23_0a_candidate_obstacle_source_linkage"
    assert summary["candidate_obstacle_source_hash_mismatch_count"] == 1


def test_stage23_0a_does_not_pass_proxy_review_when_stage23_0_rerun_failed() -> None:
    status, route, reason = _route(
        [],
        {
            "source_count": 1,
            "only_blocked_proxy_sources_available": True,
        },
        {
            "candidate_obstacle_source_missing_count": 0,
            "candidate_obstacle_source_hash_mismatch_count": 0,
        },
        {
            "status": "failed",
            "next_required_change": "rerun_stage23_0_required_obstacle_sources",
        },
    )

    assert status == "failed"
    assert route == "repair_stage23_map_obstacle_source_export"
    assert reason == "stage23_0_rerun_failed"


def _write_high_fidelity_root(
    tmp_path: Path,
    *,
    source_kind: str,
    source_hash: str,
    obstacle_cells: list[list[int]],
    candidate_source_hash: str | None = None,
) -> Path:
    root = tmp_path / "hf"
    root.mkdir()
    source_id = f"scenario:scenario-001:{source_kind}"
    source_payload = {
        "schema_version": "xunce-exploration-coverage-obstacle-sources/v1",
        "source_count": 1,
        "proxy_source_count": 1 if source_kind.endswith("_proxy") else 0,
        "physical_source_count": 1 if source_kind == "physical_obstacle_cells" else 0,
        "sources": [
            {
                "schema_version": "xunce-exploration-coverage-obstacle-source/v1",
                "scenario_id": "scenario-001",
                "obstacle_source_id": source_id,
                "obstacle_source_kind": source_kind,
                "obstacle_source_hash": source_hash,
                "obstacle_cells": obstacle_cells,
                "obstacle_cell_count": len(obstacle_cells),
                "obstacle_source_is_proxy": source_kind.endswith("_proxy"),
            }
        ],
    }
    (root / "xunce-exploration-coverage-obstacle-sources.json").write_text(
        json.dumps(source_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    row = {
        "schema_version": "xunce-exploration-coverage-candidate-metric-audit-row/v1",
        "scenario_id": "scenario-001",
        "step_index": 0,
        "candidate_index": 0,
        "candidate_cell": [0, 0],
        "candidate_theta_deg": 0,
        "candidate_viewpoint": [0, 0, 0],
        "sensor_range_cells": 4,
        "sensor_fov_deg": 30.0,
        "obstacle_source_id": source_id,
        "obstacle_source_hash": candidate_source_hash or source_hash,
        "obstacle_source_kind": source_kind,
        "obstacle_source_missing": False,
    }
    (root / "xunce-exploration-coverage-candidate-metric-audit.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _write_config(tmp_path: Path, high_fidelity_root: Path) -> Path:
    payload = {
        "schema_version": "xunce-stage23-0a-materialize-obstacle-sources-for-theta-los-config/v1",
        "high_fidelity_root": str(high_fidelity_root),
        "run_high_fidelity_smoke": False,
        "max_audit_rows": 100,
        "stage23_0a_authorized": False,
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


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
