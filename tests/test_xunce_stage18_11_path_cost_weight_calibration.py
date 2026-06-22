import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
MODEL_EXPLORER_SRC = str(REPO_ROOT / "model-explorer" / "src")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
if MODEL_EXPLORER_SRC not in sys.path:
    sys.path.insert(0, MODEL_EXPLORER_SRC)


def _profile(path: str = "configs/xunce_canonical_reward_guard_profile_v3.json"):
    from model_explorer.policy.canonical_reward import load_canonical_reward_profile

    return load_canonical_reward_profile(REPO_ROOT / path)


def _run(config_path: Path, output_root: Path):
    from scripts.run_xunce_stage18_11_path_cost_weight_calibration import (
        run_xunce_stage18_11_path_cost_weight_calibration,
    )

    return run_xunce_stage18_11_path_cost_weight_calibration(
        config_path=config_path,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )


def test_profile_variants_only_change_path_cost_weight() -> None:
    base = _profile().to_content_dict()
    paths = [
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w000.json",
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w005.json",
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w010.json",
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w020.json",
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w030.json",
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w040.json",
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w060.json",
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w080.json",
    ]

    seen_hashes = set()
    for path in paths:
        profile = _profile(path)
        payload = profile.to_content_dict()
        seen_hashes.add(profile.profile_hash)
        assert profile.profile_version == "v3"
        for key, value in base["soft_reward_components"].items():
            if key == "path_cost_weight":
                continue
            assert payload["soft_reward_components"][key] == value
        assert payload["normalizers"] == base["normalizers"]
        assert payload["risk_policy"] == base["risk_policy"]
        assert payload["trajectory_guards"] == base["trajectory_guards"]
    assert len(seen_hashes) == len(paths)


def test_one_step_replay_path_cost_weight_changes_rerank_selection(tmp_path: Path) -> None:
    profile = _profile()
    root = _write_stage18_4e(tmp_path, profile, candidate_count=6, final_coverage=0.55)
    config = _write_config(tmp_path, [root], profile_paths=[
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w000.json",
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w080.json",
    ])

    summary = _run(config, tmp_path / "out")

    assert summary["status"] == "partial"
    assert summary["next_required_change"] == "run_stage18_11_reward_rerank_diagnostic_rollouts"
    rows = _read_jsonl(tmp_path / "out" / "xunce-stage18-11-weight-replay-results.jsonl")
    by_weight = {row["path_cost_weight"]: row for row in rows}
    assert by_weight[0.0]["reward_rerank_selected_action_index"] == 0
    assert by_weight[0.8]["reward_rerank_selected_action_index"] == 1
    assert by_weight[0.8]["does_not_claim_trajectory_outcome"] is True
    assert by_weight[0.8]["does_not_claim_xunce_policy_changed"] is True


def test_final_coverage_below_99_routes_to_stage18_12_after_diagnostics(tmp_path: Path) -> None:
    profile = _profile()
    roots = [
        _write_stage18_4e(tmp_path, profile, candidate_count=6, final_coverage=0.55),
        _write_stage18_4e(tmp_path, profile, candidate_count=24, final_coverage=0.34),
    ]
    config = _write_config(tmp_path, roots, profile_paths=[
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w000.json",
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w020.json",
    ])
    _write_diagnostic_rollout(tmp_path, candidate_count=6, weight_label="w000", final_coverage=0.60)
    _write_diagnostic_rollout(tmp_path, candidate_count=24, weight_label="w020", final_coverage=0.50)
    _write_diagnostic_rollout(tmp_path, candidate_count=6, weight_label="w020", final_coverage=0.58)
    _write_diagnostic_rollout(tmp_path, candidate_count=24, weight_label="w000", final_coverage=0.57)

    summary = _run(config, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "stage18_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage"
    assert summary["target_final_coverage_rate"] == 0.99
    assert summary["stage19_authorized"] is False


def test_hard_risk_routes_before_coverage_horizon(tmp_path: Path) -> None:
    profile = _profile()
    root = _write_stage18_4e(tmp_path, profile, candidate_count=6, final_coverage=0.55, hard_risk=True)
    config = _write_config(tmp_path, [root], profile_paths=[
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w000.json",
    ])
    _write_diagnostic_rollout(tmp_path, candidate_count=6, weight_label="w000", final_coverage=0.60)

    summary = _run(config, tmp_path / "out")

    assert summary["next_required_change"] == "repair_path_risk_boundary_filtering"


def test_boundary_flag_hard_fails(tmp_path: Path) -> None:
    profile = _profile()
    root = _write_stage18_4e(tmp_path, profile, candidate_count=6, final_coverage=0.55)
    config = _write_config(tmp_path, [root], profile_paths=[
        "configs/xunce_canonical_reward_guard_profile_v3_path_cost_w000.json",
    ])
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["starts_online_canary"] = True
    config.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _run(config, tmp_path / "out")

    assert summary["next_required_change"] == "resolve_stage18_11_boundary_rejections"
    assert "starts_online_canary" in summary["blocking_reason_codes"]


def _write_config(tmp_path: Path, roots: list[Path], *, profile_paths: list[str]) -> Path:
    sweeps = []
    for root in roots:
        candidate_count = int(root.parent.name.split("_")[1])
        sweeps.append(
            {
                "candidate_count": candidate_count,
                "proposal_pool_limit": candidate_count * 8,
                "coverage_comparison_root": str(root),
            }
        )
    config = {
        "schema_version": "xunce-stage18-11-path-cost-weight-calibration-config/v1",
        "artifact_workspace_root": str(tmp_path),
        "canonical_observed_profile": str(REPO_ROOT / "configs" / "xunce_canonical_reward_guard_profile_v3.json"),
        "profile_variants": [str(REPO_ROOT / path) for path in profile_paths],
        "sweeps": sweeps,
        "target_final_coverage_rate": 0.99,
        "rollout_steps": 40,
        "required_scenario_count": 24,
        "diagnostic_rollout_workspace_root": str(tmp_path / "diagnostic_rollouts"),
        "canary_traffic_fraction": 0.0,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "runs_new_ppo_update": False,
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "modifies_network": False,
        "modifies_action_space": False,
        "modifies_default_astar": False,
    }
    path = tmp_path / "stage18_11_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage18_4e(
    tmp_path: Path,
    profile,
    *,
    candidate_count: int,
    final_coverage: float,
    hard_risk: bool = False,
) -> Path:
    root = tmp_path / f"count_{candidate_count:03d}" / "stage18_4e"
    root.mkdir(parents=True, exist_ok=True)
    lineage = {
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
    }
    summary = {
        "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
        **lineage,
        "coverage_denominator_cells": 1000,
        "stage19_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
    }
    aggregate = {
        "schema_version": "xunce-exploration-coverage-comparison-aggregate/v1",
        **lineage,
        "scenario_count": 1,
    }
    manifest = {
        "schema_version": "xunce-exploration-coverage-comparison-manifest/v1",
        **lineage,
        "normalized_config": {"coverage_denominator_cells": 1000},
    }
    episodes = [
        {"policy": "xunce", "final_coverage_rate": final_coverage, "hard_risk_violation_count": int(hard_risk)},
        {"policy": "incumbent", "final_coverage_rate": 0.2, "hard_risk_violation_count": 0},
    ]
    pairs = [{"scenario_id": "s1", "coverage_delta_cells": 10.0}]
    paired = [
        {
            "scenario_id": "s1",
            "executing_policy": "xunce",
            "step_index": 0,
            "candidate_set_hash": "set-a",
            "covered_cells_hash": "covered-a",
            "xunce_selected_action_index": 0,
            "incumbent_selected_action_index": 1,
        }
    ]
    candidates = [
        _candidate(profile, index=0, coverage=10.0, cost=100.0, hard_risk=hard_risk),
        _candidate(profile, index=1, coverage=9.8, cost=1.0),
    ]
    for filename, payload in (
        ("xunce-exploration-coverage-comparison-summary.json", summary),
        ("xunce-exploration-coverage-comparison-aggregate.json", aggregate),
        ("xunce-exploration-coverage-comparison-manifest.json", manifest),
    ):
        (root / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(root / "xunce-exploration-coverage-episodes.jsonl", episodes)
    _write_jsonl(root / "xunce-exploration-coverage-comparison-pairs.jsonl", pairs)
    _write_jsonl(root / "xunce-exploration-coverage-paired-decision-audit.jsonl", paired)
    _write_jsonl(root / "xunce-exploration-coverage-candidate-metric-audit.jsonl", candidates)
    return root


def _candidate(profile, *, index: int, coverage: float, cost: float, hard_risk: bool = False) -> dict:
    return {
        "schema_version": "xunce-exploration-coverage-candidate-metric-audit-row/v1",
        "scenario_id": "s1",
        "policy": "xunce",
        "executing_policy": "xunce",
        "step_index": 0,
        "candidate_set_hash": "set-a",
        "covered_cells_hash": "covered-a",
        "candidate_index": index,
        "action_mask_valid": True,
        "expected_new_coverage_cell_count": coverage,
        "roi_weighted_coverage_delta": coverage,
        "path_cost": cost,
        "risk": 1.0,
        "risk_cost_weighted": cost,
        "path_allowed_by_risk": not hard_risk,
        "hard_risk_flags": ["blocked"] if hard_risk else [],
        "soft_risk_exposure": 0.0,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
    }


def _write_diagnostic_rollout(tmp_path: Path, *, candidate_count: int, weight_label: str, final_coverage: float) -> None:
    root = tmp_path / "diagnostic_rollouts" / f"count_{candidate_count:03d}_{weight_label}" / "stage18_4e"
    root.mkdir(parents=True, exist_ok=True)
    (root / "xunce-exploration-coverage-comparison-summary.json").write_text(
        json.dumps({"schema_version": "xunce-exploration-coverage-comparison-summary/v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(
        root / "xunce-exploration-coverage-episodes.jsonl",
        [
            {
                "policy": "canonical_reward_rerank_oracle",
                "final_coverage_rate": final_coverage,
                "path_cost_total_m": 10.0,
                "soft_risk_exposure_total": 1.0,
                "hard_risk_violation_count": 0,
            }
        ],
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
