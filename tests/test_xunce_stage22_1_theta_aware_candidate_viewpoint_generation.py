from __future__ import annotations

import json
from pathlib import Path

import scripts.run_xunce_stage22_1_theta_aware_candidate_viewpoint_generation as s22


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _fixture_roots(tmp_path: Path, *, bad_mask: bool = False) -> tuple[Path, Path]:
    stage22_0 = tmp_path / "stage22_0"
    _write_json(
        stage22_0 / "xunce-stage22-0-summary.json",
        {
            "schema_version": "xunce-stage22-0-summary/v1",
            "status": "passed",
            "next_required_change": "implement_stage22_1_theta_aware_candidate_viewpoint_generation",
            "theta_material_to_coverage": True,
        },
    )
    stage21_18 = tmp_path / "stage21_18"
    stage21_6 = tmp_path / "stage21_6"
    stage21_3 = stage21_6 / "seed_2101" / "stage21_3"
    _write_jsonl(
        stage21_18 / "xunce-stage21-18-round-results.jsonl",
        [{"round_index": 1, "stage21_6_roots": [str(stage21_6)]}],
    )
    _write_jsonl(
        stage21_6 / "xunce-stage21-6-seed-results.jsonl",
        [{"seed": 2101, "stage21_3_root": str(stage21_3)}],
    )
    action_mask = [True] if bad_mask else [True, True]
    _write_jsonl(
        stage21_3 / "xunce-stage21-3-ppo-trainable-batch.jsonl",
        [
            {
                "scenario_id": "s0",
                "step_index": 0,
                "action_index": 0,
                "observation": {
                    "candidate_cells": [[2, 0], [0, 2]],
                    "candidate_feature_names": ["path_cost"],
                    "candidate_features": [[2.0], [3.0]],
                    "action_mask": action_mask,
                },
                "info": {
                    "scenario_id": "s0",
                    "step_index": 0,
                    "current_cell_before": [0, 0],
                    "candidate_cells": [[2, 0], [0, 2]],
                    "candidate_set_hash": "point-only-hash",
                    "action_mask": action_mask,
                    "sampling_mask": action_mask,
                },
            }
        ],
    )
    return stage22_0, stage21_18


def _config(tmp_path: Path, stage22_0: Path, stage21_18: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": s22.CONFIG_SCHEMA_VERSION,
        "stage22_0_root": str(stage22_0),
        "stage21_18_root": str(stage21_18),
        "stage21_19_root": "",
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_model_id": "theta-fov-90-range-radius/v1",
        "sensor_fov_deg": 90,
        "sensor_range_cells": 2,
        "coverage_denominator_cells": 100,
        "max_viewpoint_candidate_audit_rows": 100,
        "stage22_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage22_1_config.json"
    _write_json(path, payload)
    return path


def test_stage22_1_expands_point_candidates_to_viewpoints_and_routes_stage22_2(tmp_path: Path) -> None:
    stage22_0, stage21_18 = _fixture_roots(tmp_path)

    summary = s22.run_xunce_stage22_1_theta_aware_candidate_viewpoint_generation(
        config_path=_config(tmp_path, stage22_0, stage21_18),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "xunce-stage22-1-viewpoint-candidate-audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == s22.ROUTE_STAGE22_2
    assert summary["expanded_viewpoint_row_count"] == 16
    assert len(rows) == 16
    assert rows[0]["candidate_viewpoint"] == [2, 0, 0]
    assert rows[1]["candidate_viewpoint"] == [2, 0, 45]
    assert rows[0]["candidate_set_hash"] != "point-only-hash"


def test_stage22_1_routes_to_contract_repair_when_mask_length_mismatches(tmp_path: Path) -> None:
    stage22_0, stage21_18 = _fixture_roots(tmp_path, bad_mask=True)

    summary = s22.run_xunce_stage22_1_theta_aware_candidate_viewpoint_generation(
        config_path=_config(tmp_path, stage22_0, stage21_18),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s22.ROUTE_VIEWPOINT_CONTRACT
    assert summary["mask_length_mismatch_count"] == 1


def test_stage22_1_boundary_flags_hard_fail(tmp_path: Path) -> None:
    stage22_0, stage21_18 = _fixture_roots(tmp_path)

    summary = s22.run_xunce_stage22_1_theta_aware_candidate_viewpoint_generation(
        config_path=_config(tmp_path, stage22_0, stage21_18, publishes_checkpoint=True),
        output_root=tmp_path / "out",
        repo_root=tmp_path,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == s22.ROUTE_BOUNDARY
