import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _run(config_path: Path, stage19_root: Path, output_root: Path, stage20_1_root: Path | None = None):
    from scripts.run_xunce_stage20_reward_rerank_oracle_imitation_dataset import (
        run_xunce_stage20_reward_rerank_oracle_imitation_dataset,
    )

    return run_xunce_stage20_reward_rerank_oracle_imitation_dataset(
        config_path=config_path,
        stage19_evaluator_critic_preflight_root=stage19_root,
        stage20_1_same_candidate_oracle_imitation_root=stage20_1_root,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )


def test_stage20_generates_strict_same_candidate_teacher_samples_and_exclusions(tmp_path: Path) -> None:
    stage19_root = _write_stage19_fixture(tmp_path, same_candidate_xunce_count=24)
    config = _write_config(tmp_path)

    summary = _run(config, stage19_root, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["trainable_pair_count"] == 24
    assert summary["excluded_pair_count"] == 3
    assert summary["dry_run_summary"]["dry_run_executed"] is True
    assert summary["dry_run_summary"]["teacher_action_accuracy"] >= 0.0
    assert summary["next_required_change"] == "collect_more_reward_rerank_same_candidate_preference_evidence"
    assert summary["stage20_authorized"] is False
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False
    samples = _read_jsonl(tmp_path / "out" / "xunce-stage20-oracle-imitation-teacher-samples.jsonl")
    assert len(samples) == 24
    assert samples[0]["teacher_action_index"] == 3
    assert samples[0]["xunce_action_index"] == 1
    assert samples[0]["training_signal_type"] == "teacher_imitation_label"
    assert samples[0]["sample_weight"] == 1.0
    exclusions = _read_jsonl(tmp_path / "out" / "xunce-stage20-oracle-imitation-exclusion-report.jsonl")
    reason_codes = {row["reason_code"] for row in exclusions}
    assert "different_candidate_set" in reason_codes
    assert "baseline_policy_not_xunce" in reason_codes
    assert "hard_risk_not_clean" in reason_codes


def test_stage20_weights_equal_coverage_lower_cost_samples(tmp_path: Path) -> None:
    stage19_root = _write_stage19_fixture(
        tmp_path,
        same_candidate_xunce_count=24,
        oracle_coverage=10.0,
        xunce_coverage=10.0,
        oracle_cost=8.0,
        xunce_cost=12.0,
    )
    config = _write_config(tmp_path)

    summary = _run(config, stage19_root, tmp_path / "out")

    assert summary["trainable_pair_count"] == 24
    sample = _read_jsonl(tmp_path / "out" / "xunce-stage20-oracle-imitation-teacher-samples.jsonl")[0]
    assert sample["sample_weight"] == 0.5


def test_stage20_routes_to_collect_more_when_below_dry_run_minimum(tmp_path: Path) -> None:
    stage19_root = _write_stage19_fixture(tmp_path, same_candidate_xunce_count=23)
    config = _write_config(tmp_path)

    summary = _run(config, stage19_root, tmp_path / "out")

    assert summary["trainable_pair_count"] == 23
    assert summary["dry_run_summary"]["dry_run_executed"] is False
    assert summary["next_required_change"] == "collect_more_reward_rerank_same_candidate_preference_evidence"


def test_stage20_routes_to_checkpoint_preflight_when_enough_samples_and_dry_run_passes(tmp_path: Path) -> None:
    stage19_root = _write_stage19_fixture(tmp_path, same_candidate_xunce_count=200)
    config = _write_config(tmp_path)

    summary = _run(config, stage19_root, tmp_path / "out")

    assert summary["trainable_pair_count"] == 200
    assert summary["dry_run_summary"]["dry_run_executed"] is True
    assert summary["next_required_change"] == "stage20_1_supervised_oracle_imitation_checkpoint_preflight"
    assert summary["stage20_authorized"] is False


def test_stage20_merges_stage20_1_on_policy_teacher_labels(tmp_path: Path) -> None:
    stage19_root = _write_stage19_fixture(tmp_path, same_candidate_xunce_count=24)
    stage20_1_root = _write_stage20_1_fixture(tmp_path, label_count=200)
    config = _write_config(tmp_path)

    summary = _run(config, stage19_root, tmp_path / "out", stage20_1_root)

    assert summary["trainable_pair_count"] == 224
    assert summary["dataset_stats"]["stage20_1_trainable_label_count"] == 200
    assert summary["next_required_change"] == "stage20_1_supervised_oracle_imitation_checkpoint_preflight"
    samples = _read_jsonl(tmp_path / "out" / "xunce-stage20-oracle-imitation-teacher-samples.jsonl")
    source_types = {row["source_type"] for row in samples}
    assert "stage19_preference_pair_audit" in source_types
    assert "stage20_1_on_policy_teacher_label" in source_types
    assert summary["stage20_authorized"] is False


def test_stage20_rejects_stage20_1_profile_mismatch(tmp_path: Path) -> None:
    stage19_root = _write_stage19_fixture(tmp_path, same_candidate_xunce_count=24)
    stage20_1_root = _write_stage20_1_fixture(tmp_path, label_count=24, teacher_profile_hash="other-profile-hash")
    config = _write_config(tmp_path)

    summary = _run(config, stage19_root, tmp_path / "out", stage20_1_root)

    assert summary["next_required_change"] == "rerun_stage19_evaluator_critic_preflight"
    assert "stage20_1_teacher_profile_hash_mismatch" in summary["blocking_reason_codes"]


def test_stage20_boundary_flag_hard_fails(tmp_path: Path) -> None:
    stage19_root = _write_stage19_fixture(tmp_path, same_candidate_xunce_count=24)
    config = _write_config(tmp_path, starts_online_canary=True)

    summary = _run(config, stage19_root, tmp_path / "out")

    assert summary["next_required_change"] == "resolve_stage20_oracle_imitation_boundary_rejections"
    assert "starts_online_canary" in summary["blocking_reason_codes"]


def _write_stage20_1_fixture(
    tmp_path: Path,
    *,
    label_count: int,
    teacher_profile_hash: str = "fixture-rerank-profile-hash",
) -> Path:
    root = tmp_path / "stage20_1"
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "xunce-stage20-1-same-candidate-oracle-imitation-summary/v1",
        "status": "passed",
        "next_required_change": "stage20_1_supervised_oracle_imitation_checkpoint_preflight",
        "teacher_profile_id": "xunce-coverage-cost-risk-boundary-v3-path-cost-w010",
        "teacher_profile_hash": teacher_profile_hash,
        "trainable_label_count": label_count,
        "stage20_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    labels = [
        {
            "schema_version": "xunce-stage20-1-on-policy-teacher-label/v1",
            "source_index": index,
            "source_type": "stage20_1_on_policy_teacher_label",
            "scenario_id": f"s20-1-{index}",
            "step_index": index,
            "baseline_policy": "xunce",
            "teacher_policy": "canonical_reward_rerank_oracle",
            "same_candidate_set": True,
            "candidate_set_hash": f"stage20-1-set-{index}",
            "current_cell": [0, 0],
            "covered_cells_hash": f"covered-{index}",
            "teacher_action_index": 3,
            "xunce_action_index": 1,
            "teacher_profile_id": "xunce-coverage-cost-risk-boundary-v3-path-cost-w010",
            "teacher_profile_hash": teacher_profile_hash,
            "oracle_new_covered_cell_count": 18.0,
            "xunce_new_covered_cell_count": 7.0,
            "oracle_path_cost": 24.0,
            "xunce_path_cost": 20.0,
            "oracle_soft_risk_exposure": 2.0,
            "xunce_soft_risk_exposure": 1.0,
            "hard_risk_clean_pair": True,
            "sample_weight": 1.0,
            "training_signal_type": "teacher_imitation_label",
        }
        for index in range(label_count)
    ]
    (root / "xunce-stage20-1-same-candidate-oracle-imitation-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(root / "xunce-stage20-1-on-policy-teacher-labels.jsonl", labels)
    return root


def _write_config(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage20-oracle-imitation-config/v1",
        "min_trainable_pair_count_for_dry_run": 24,
        "min_trainable_pair_count_for_checkpoint_training": 200,
        "require_same_candidate_set": True,
        "require_hard_risk_clean_pair": True,
        "stage20_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "training_or_release_authorized": False,
    }
    payload.update(overrides)
    path = tmp_path / "stage20_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage19_fixture(
    tmp_path: Path,
    *,
    same_candidate_xunce_count: int,
    oracle_coverage: float = 20.0,
    xunce_coverage: float = 5.0,
    oracle_cost: float = 50.0,
    xunce_cost: float = 20.0,
) -> Path:
    root = tmp_path / "stage19"
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "xunce-stage19-evaluator-critic-preflight-summary/v1",
        "status": "passed",
        "next_required_change": "stage20_reward_rerank_oracle_preference_dataset_preparation",
        "profile_id": "xunce-coverage-cost-risk-boundary-v3",
        "profile_version": "v3",
        "profile_hash": "fixture-profile-hash",
        "stage20_authorized": False,
        "training_or_release_authorized": False,
        "oracle_target_feasible": True,
        "xunce_checkpoint_advantage_established": False,
        "critic_target_readiness": {"critic_target_ready": True},
    }
    practical = {
        "schema_version": "xunce-stage19-practical-target-selection/v1",
        "selected_candidate_count": 36,
        "selected_path_cost_weight": 0.1,
        "primary": {
            "canonical_reward_rerank_profile_id": "xunce-coverage-cost-risk-boundary-v3-path-cost-w010",
            "canonical_reward_rerank_profile_hash": "fixture-rerank-profile-hash",
        },
    }
    critic = {
        "schema_version": "xunce-stage19-critic-target-readiness/v1",
        "critic_target_ready": True,
        "training_data_published": False,
    }
    rows = [
        _preference_row(
            index,
            same_candidate=True,
            baseline_policy="xunce",
            hard_clean=True,
            oracle_coverage=oracle_coverage,
            xunce_coverage=xunce_coverage,
            oracle_cost=oracle_cost,
            xunce_cost=xunce_cost,
        )
        for index in range(same_candidate_xunce_count)
    ]
    rows.append(_preference_row(1000, same_candidate=False, baseline_policy="xunce", hard_clean=True))
    rows.append(_preference_row(1001, same_candidate=True, baseline_policy="incumbent", hard_clean=True))
    rows.append(_preference_row(1002, same_candidate=True, baseline_policy="xunce", hard_clean=False))
    (root / "xunce-stage19-evaluator-critic-preflight-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "xunce-stage19-practical-target-selection.json").write_text(
        json.dumps(practical, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "xunce-stage19-critic-target-readiness.json").write_text(
        json.dumps(critic, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(root / "xunce-stage19-preference-pair-audit.jsonl", rows)
    return root


def _preference_row(
    index: int,
    *,
    same_candidate: bool,
    baseline_policy: str,
    hard_clean: bool,
    oracle_coverage: float = 20.0,
    xunce_coverage: float = 5.0,
    oracle_cost: float = 50.0,
    xunce_cost: float = 20.0,
) -> dict:
    return {
        "schema_version": "xunce-stage19-preference-pair-audit-row/v1",
        "scenario_id": f"s{index}",
        "step_index": index,
        "baseline_policy": baseline_policy,
        "same_candidate_set": same_candidate,
        "oracle_selected_action_index": 3,
        "baseline_selected_action_index": 1,
        "oracle_candidate_set_hash": f"set-{index}",
        "baseline_candidate_set_hash": f"set-{index}" if same_candidate else f"other-{index}",
        "oracle_new_covered_cell_count": oracle_coverage,
        "baseline_new_covered_cell_count": xunce_coverage,
        "oracle_path_cost": oracle_cost,
        "baseline_path_cost": xunce_cost,
        "oracle_soft_risk_exposure": 10.0,
        "baseline_soft_risk_exposure": 8.0,
        "oracle_hard_risk_violation_count": 0.0 if hard_clean else 1.0,
        "baseline_hard_risk_violation_count": 0.0,
        "oracle_selected_higher_coverage": oracle_coverage > xunce_coverage,
        "oracle_selected_lower_or_acceptable_cost": oracle_cost <= xunce_cost * 1.25 or oracle_coverage > xunce_coverage,
        "hard_risk_clean_pair": hard_clean,
        "audit_only": True,
        "not_training_data": True,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
