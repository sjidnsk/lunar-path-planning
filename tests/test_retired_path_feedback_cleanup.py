"""Regression contract for the retired path-feedback route cleanup."""

import json
from pathlib import Path


RETIRED_PATH_FEEDBACK_PATHS = (
    "configs/path_feedback_batch_anchor_projection_candidate_generation_v1.json",
    "configs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1.json",
    "configs/path_feedback_batch_anchor_projection_distance_contract_relaxation_safety_audit_v1.json",
    "configs/path_feedback_batch_anchor_projection_nontrainable_context_reduction_v1.json",
    "configs/path_feedback_batch_dataset_v1.json",
    "configs/path_feedback_batch_fresh_holdout_policy_candidate_evaluation_v1.json",
    "configs/path_feedback_batch_planner_validated_trainable_target_mining_v1.json",
    "configs/path_feedback_batch_policy_gated_canary_diversity_v1.json",
    "configs/path_feedback_batch_policy_gated_canary_full_family_opportunity_v1.json",
    "configs/path_feedback_batch_policy_gated_canary_opportunity_quality_v1.json",
    "configs/path_feedback_batch_policy_gated_canary_rollout_v1.json",
    "configs/path_feedback_batch_policy_gated_canary_value_stability_v1.json",
    "configs/path_feedback_batch_raw_policy_generalization_test_v1.json",
    "configs/path_feedback_batch_raw_policy_generalization_train_v1.json",
    "configs/path_feedback_batch_raw_policy_generalization_val_v1.json",
    "configs/path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1.json",
    "configs/path_feedback_batch_sequential_multi_step_opportunity_v1.json",
    "scripts/run_batch_path_feedback_validation.py",
    "scripts/run_batch_path_feedback_validation.sh",
    "scripts/run_path_feedback_stability_analysis.py",
    "scripts/run_path_feedback_stability_analysis.sh",
    "scripts/run_path_feedback_validation.py",
    "scripts/run_path_feedback_validation.sh",
    "scripts/run_quasi_real_map_path_feedback_bridge.py",
    "scripts/run_quasi_real_map_path_feedback_bridge.sh",
    "tests/test_batch_path_feedback_validation.py",
    "tests/test_path_feedback_stability_analysis.py",
    "tests/test_path_feedback_validation_script.py",
    "tests/test_path_feedback_windows_compat.py",
)


def test_manifest_listed_path_feedback_paths_are_absent() -> None:
    assert [path for path in RETIRED_PATH_FEEDBACK_PATHS if Path(path).exists()] == []


def test_stage_registry_and_platform_surface_do_not_expose_path_feedback() -> None:
    registry = json.loads(Path("configs/stage_registry.json").read_text(encoding="utf-8"))
    assert not any("path-feedback" in stage for stage in registry["stages"])
    assert "run_path_feedback_" not in Path("scripts/run_platform_validation_matrix.py").read_text(
        encoding="utf-8"
    )
    assert "path_feedback" not in Path(".github/workflows/platform-compatibility.yml").read_text(
        encoding="utf-8"
    )
