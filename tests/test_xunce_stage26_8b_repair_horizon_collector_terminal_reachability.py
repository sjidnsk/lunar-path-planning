from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_stage26_8b_rejects_non_horizon_binding_stage26_8a_root(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability as s26

    root = _make_stage26_8a_root(tmp_path, route="repair_stage26_synthetic_policy_update_signal_strength")
    config = _write_config(tmp_path, root)

    summary = s26.run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage26_8b_required_inputs"
    assert "stage26_8a_route_not_horizon_binding_or_safety" in summary["blocking_reason_codes"]


def test_stage26_8b_reruns_h16_seed0_and_routes_resume_when_stage26_1_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability as s26

    root = _make_stage26_8a_root(tmp_path)
    config = _write_config(tmp_path, root)

    def fake_stage26_1(*, config_path: Path, output_root: Path, repo_root: Path) -> dict:
        assert config_path == root / "h16" / "s26_8" / "s0" / "xunce-stage26-8-stage26-1-config.json"
        output_root.mkdir(parents=True, exist_ok=True)
        summary = _stage26_1_summary(status="passed", route="run_stage26_2_synthetic_terrain_ppo_update_smoke")
        _write_json(output_root / s26.stage26_1.SUMMARY_FILE, summary)
        _write_json(output_root / s26.stage26_1.AUDIT_FILE, _stage26_1_audit(source_match=True))
        s21_1 = output_root / "s21_1"
        s21_1.mkdir()
        _write_json(
            s21_1 / s26.stage26_1.stage21_1.SUMMARY_FILE,
            {
                "status": "passed",
                "next_required_change": "implement_stage21_2_coverage_first_ppo_reward_contract",
                "trainable_transition_count": 45,
                "min_trainable_transition_count": 12,
                "blocking_reason_codes": [],
                "no_hybrid_reachable_candidate_terminal_count": 3,
            },
        )
        _write_jsonl(
            s21_1 / s26.stage26_1.stage21_1.REJECTION_FILE,
            [
                {
                    "reason": "no_hybrid_reachable_candidate_terminal",
                    "hybrid_astar_reachable_mask": [False, False],
                }
            ],
        )
        return summary

    monkeypatch.setattr(
        s26.stage26_1,
        "run_xunce_stage26_1_synthetic_terrain_collector_smoke",
        fake_stage26_1,
    )

    summary = s26.run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    terminal = json.loads((tmp_path / "out" / "xunce-stage26-8b-terminal-reachability-audit.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "resume_stage26_8a_from_h16_h20"
    assert summary["h16_repaired_reward_row_count"] == 45
    assert terminal["repaired_no_hybrid_reachable_candidate_terminal_count"] == 3


def test_stage26_8b_routes_real_source_mismatch_to_map_binding(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability as s26

    root = _make_stage26_8a_root(tmp_path, h16_source_match=False)
    config = _write_config(tmp_path, root, run_repair_chain=False)

    summary = s26.run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "repair_stage26_1_collector_synthetic_map_binding"
    assert "h16_source_roi_expansion_root_mismatch" in summary["blocking_reason_codes"]


def test_stage26_8b_boundary_rejection(tmp_path: Path) -> None:
    import scripts.run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability as s26

    root = _make_stage26_8a_root(tmp_path)
    config = _write_config(tmp_path, root, publishes_checkpoint=True)

    summary = s26.run_xunce_stage26_8b_repair_horizon_collector_terminal_reachability(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "resolve_stage26_8b_boundary_rejections"
    assert "publishes_checkpoint" in summary["blocking_reason_codes"]


def _make_stage26_8a_root(
    tmp_path: Path,
    *,
    route: str = "repair_stage26_8a_horizon_eval_binding_or_safety",
    h16_source_match: bool = True,
) -> Path:
    root = tmp_path / "stage26_8a"
    _write_json(
        root / "xunce-stage26-8a-summary.json",
        {
            "schema_version": "xunce-stage26-8a-summary/v1",
            "stage_id": "xunce-stage26-8a-expand-seed-or-horizon-budget",
            "status": "failed",
            "next_required_change": route,
        },
    )
    h12 = root / "h12" / "s26_8" / "s0" / "s26_1"
    h16 = root / "h16" / "s26_8" / "s0" / "s26_1"
    _write_json(h12 / "xunce-stage26-1-summary.json", _stage26_1_summary(status="passed"))
    _write_json(h16 / "xunce-stage26-1-summary.json", _stage26_1_summary(status="failed"))
    _write_json(h16 / "xunce-stage26-1-synthetic-transition-contract-audit.json", _stage26_1_audit(source_match=h16_source_match))
    _write_json(
        h16 / "s21_1" / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json",
        {
            "status": "failed",
            "blocking_reason_codes": ["no_hybrid_reachable_candidate_terminal"],
            "trainable_transition_count": 45,
        },
    )
    _write_jsonl(
        h16 / "s21_1" / "xunce-stage21-1-rejection-report.jsonl",
        [{"reason": "no_hybrid_reachable_candidate_terminal", "hybrid_astar_reachable_mask": [False]}],
    )
    _write_json(root / "h16" / "s26_8" / "s0" / "xunce-stage26-8-stage26-1-config.json", {"stage": "stage26-1"})
    return root


def _stage26_1_summary(*, status: str, route: str = "repair_stage26_1_collector_synthetic_map_binding") -> dict:
    return {
        "schema_version": "xunce-stage26-1-summary/v1",
        "stage_id": "xunce-stage26-1-synthetic-terrain-collector-smoke",
        "status": status,
        "next_required_change": route,
        "transition_count": 45,
        "reward_row_count": 45 if status == "passed" else 0,
        "batch_row_count": 45 if status == "passed" else 0,
        "transition_missing_reward_count": 0 if status == "passed" else 45,
        "transition_missing_batch_count": 0 if status == "passed" else 45,
        "synthetic_transition_contract_missing_count": 0,
        "synthetic_terrain_hash": "stage26-synthetic-hash",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "max_traversable_slope_deg": 30.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _stage26_1_audit(*, source_match: bool) -> dict:
    return {
        "schema_version": "xunce-stage26-1-synthetic-transition-contract-audit/v1",
        "stage21_1_source_roi_expansion_root_match": source_match,
        "synthetic_transition_contract_missing_count": 0,
        "transition_missing_reward_count": 0,
        "transition_missing_batch_count": 0,
        "synthetic_reward_provenance_missing_count": 0,
        "synthetic_batch_contract_missing_count": 0,
    }


def _write_config(tmp_path: Path, root: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "xunce-stage26-8b-repair-horizon-collector-terminal-reachability-config/v1",
        "stage_id": "xunce-stage26-8b-repair-horizon-collector-terminal-reachability",
        "stage26_8a_root": str(root),
        "run_repair_chain": True,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage26_8b_config.json"
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
