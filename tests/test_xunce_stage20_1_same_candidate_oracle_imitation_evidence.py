import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = str(REPO_ROOT / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def _run(config_path: Path, stage19_root: Path, stage20_root: Path, coverage_root: Path, output_root: Path):
    from scripts.run_xunce_stage20_1_same_candidate_oracle_imitation_evidence import (
        run_xunce_stage20_1_same_candidate_oracle_imitation_evidence,
    )

    return run_xunce_stage20_1_same_candidate_oracle_imitation_evidence(
        config_path=config_path,
        stage19_evaluator_critic_preflight_root=stage19_root,
        stage20_oracle_imitation_dataset_root=stage20_root,
        coverage_comparison_root=coverage_root,
        output_root=output_root,
        repo_root=REPO_ROOT,
    )


def test_stage20_1_converts_on_policy_teacher_labels_and_exclusions(tmp_path: Path) -> None:
    coverage_root = _write_coverage_fixture(tmp_path, trainable_count=24)
    stage19_root = _write_stage19_fixture(tmp_path, selected_root=coverage_root)
    stage20_root = _write_stage20_fixture(tmp_path)
    config = _write_config(tmp_path)

    summary = _run(config, stage19_root, stage20_root, coverage_root, tmp_path / "out")

    assert summary["status"] == "passed"
    assert summary["trainable_label_count"] == 24
    assert summary["excluded_label_count"] == 3
    assert summary["next_required_change"] == "rerun_stage20_oracle_imitation_dataset_with_expanded_labels"
    assert summary["stage20_authorized"] is False
    assert summary["runs_new_ppo_update"] is False
    assert summary["publishes_checkpoint"] is False
    labels = _read_jsonl(tmp_path / "out" / "xunce-stage20-1-on-policy-teacher-labels.jsonl")
    assert len(labels) == 24
    assert labels[0]["source_type"] == "stage20_1_on_policy_teacher_label"
    assert labels[0]["teacher_action_index"] == 3
    assert labels[0]["xunce_action_index"] == 1
    assert labels[0]["sample_weight"] == 1.0
    exclusions = _read_jsonl(tmp_path / "out" / "xunce-stage20-1-teacher-label-exclusion-report.jsonl")
    reason_codes = {row["reason_code"] for row in exclusions}
    assert "teacher_and_xunce_selected_same_action" in reason_codes
    assert "hard_risk_not_clean" in reason_codes
    assert "baseline_policy_not_xunce" in reason_codes


def test_stage20_1_profile_hash_mismatch_hard_fails(tmp_path: Path) -> None:
    coverage_root = _write_coverage_fixture(tmp_path, trainable_count=24, teacher_hash="bad-hash")
    stage19_root = _write_stage19_fixture(tmp_path, selected_root=coverage_root, teacher_hash="expected-hash")
    stage20_root = _write_stage20_fixture(tmp_path)
    config = _write_config(tmp_path)

    summary = _run(config, stage19_root, stage20_root, coverage_root, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage20_1_required_inputs"
    assert "teacher_profile_hash_mismatch" in summary["blocking_reason_codes"]


def test_stage20_1_routes_to_checkpoint_preflight_when_enough_labels(tmp_path: Path) -> None:
    coverage_root = _write_coverage_fixture(tmp_path, trainable_count=200)
    stage19_root = _write_stage19_fixture(tmp_path, selected_root=coverage_root)
    stage20_root = _write_stage20_fixture(tmp_path)
    config = _write_config(tmp_path)

    summary = _run(config, stage19_root, stage20_root, coverage_root, tmp_path / "out")

    assert summary["trainable_label_count"] == 200
    assert summary["next_required_change"] == "stage20_1_supervised_oracle_imitation_checkpoint_preflight"
    assert summary["stage20_authorized"] is False
    assert summary["training_or_release_authorized"] is False


def test_stage20_1_boundary_flag_hard_fails(tmp_path: Path) -> None:
    coverage_root = _write_coverage_fixture(tmp_path, trainable_count=24)
    stage19_root = _write_stage19_fixture(tmp_path, selected_root=coverage_root)
    stage20_root = _write_stage20_fixture(tmp_path)
    config = _write_config(tmp_path, starts_online_canary=True)

    summary = _run(config, stage19_root, stage20_root, coverage_root, tmp_path / "out")

    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "rerun_stage20_1_required_inputs"
    assert "starts_online_canary" in summary["blocking_reason_codes"]


def _write_config(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": "xunce-stage20-1-same-candidate-oracle-imitation-evidence-config/v1",
        "min_trainable_label_count_for_checkpoint_preflight": 200,
        "stage20_authorized": False,
        "training_or_release_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    payload.update(overrides)
    path = tmp_path / "stage20_1_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_stage19_fixture(tmp_path: Path, *, selected_root: Path, teacher_hash: str = "expected-hash") -> Path:
    root = tmp_path / "stage19"
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "xunce-stage19-evaluator-critic-preflight-summary.json",
        {
            "schema_version": "xunce-stage19-evaluator-critic-preflight-summary/v1",
            "status": "passed",
            "next_required_change": "stage20_reward_rerank_oracle_preference_dataset_preparation",
            "stage20_authorized": False,
            "training_or_release_authorized": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    _write_json(
        root / "xunce-stage19-practical-target-selection.json",
        {
            "schema_version": "xunce-stage19-practical-target-selection/v1",
            "selected_root": str(selected_root.resolve()),
            "primary": {
                "root": str(selected_root.resolve()),
                "canonical_reward_rerank_profile_id": "xunce-coverage-cost-risk-boundary-v3-path-cost-w010",
                "canonical_reward_rerank_profile_hash": teacher_hash,
            },
        },
    )
    return root


def _write_stage20_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "stage20"
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "xunce-stage20-oracle-imitation-summary.json",
        {
            "schema_version": "xunce-stage20-oracle-imitation-summary/v1",
            "status": "passed",
            "trainable_pair_count": 24,
            "stage20_authorized": False,
            "training_or_release_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    return root


def _write_coverage_fixture(tmp_path: Path, *, trainable_count: int, teacher_hash: str = "expected-hash") -> Path:
    root = tmp_path / "coverage"
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "xunce-exploration-coverage-comparison-summary.json",
        {
            "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
            "status": "passed",
            "include_canonical_reward_rerank_oracle": True,
            "on_policy_oracle_teacher_label_row_count": trainable_count + 3,
            "on_policy_oracle_teacher_profile_id": "xunce-coverage-cost-risk-boundary-v3-path-cost-w010",
            "on_policy_oracle_teacher_profile_hash": teacher_hash,
            "canonical_reward_rerank_profile_id": "xunce-coverage-cost-risk-boundary-v3-path-cost-w010",
            "canonical_reward_rerank_profile_hash": teacher_hash,
            "stage20_authorized": False,
            "training_or_release_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        },
    )
    _write_json(root / "xunce-exploration-coverage-comparison-manifest.json", {"schema_version": "manifest/v1"})
    _write_jsonl(
        root / "xunce-exploration-coverage-on-policy-oracle-teacher-labels.jsonl",
        [_label_row(index, teacher_hash=teacher_hash) for index in range(trainable_count)]
        + [
            _label_row(1000, teacher_hash=teacher_hash, same_action=True),
            _label_row(1001, teacher_hash=teacher_hash, hard_clean=False),
            _label_row(1002, teacher_hash=teacher_hash, baseline_policy="incumbent"),
        ],
    )
    return root


def _label_row(
    index: int,
    *,
    teacher_hash: str,
    same_action: bool = False,
    hard_clean: bool = True,
    baseline_policy: str = "xunce",
) -> dict:
    return {
        "schema_version": "xunce-on-policy-oracle-teacher-label/v1",
        "scenario_id": f"s{index}",
        "step_index": index,
        "baseline_policy": baseline_policy,
        "teacher_policy": "canonical_reward_rerank_oracle",
        "same_candidate_set": True,
        "candidate_set_hash": f"set-{index}",
        "current_cell": [0, 0],
        "covered_cells_hash": f"covered-{index}",
        "teacher_action_index": 1 if same_action else 3,
        "xunce_action_index": 1,
        "teacher_profile_id": "xunce-coverage-cost-risk-boundary-v3-path-cost-w010",
        "teacher_profile_hash": teacher_hash,
        "teacher_reward_components": {"coverage_component": 0.2},
        "xunce_candidate_metrics": {
            "new_covered_cell_count": 5.0,
            "path_cost": 20.0,
            "soft_risk_exposure": 8.0,
        },
        "teacher_candidate_metrics": {
            "new_covered_cell_count": 20.0,
            "path_cost": 50.0,
            "soft_risk_exposure": 10.0,
        },
        "hard_risk_clean_pair": hard_clean,
        "teacher_selected_higher_coverage": True,
        "teacher_selected_lower_or_acceptable_cost": True,
        "sample_weight": 1.0,
        "training_signal_type": "teacher_imitation_label",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
