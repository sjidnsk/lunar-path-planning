from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot as _git_snapshot
    from git_provenance import git_snapshots_match as _git_snapshots_match
    from git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
    from git_provenance import public_git as _public_git
except ModuleNotFoundError:  # pragma: no cover - import path used by unit tests
    from scripts.git_provenance import git_snapshot as _git_snapshot
    from scripts.git_provenance import git_snapshots_match as _git_snapshots_match
    from scripts.git_provenance import inspect_source_git_provenance as _inspect_source_git_provenance
    from scripts.git_provenance import public_git as _public_git


CONFIG_SCHEMA_VERSION = "policy-training-readiness-review-config/v1"
SUMMARY_SCHEMA_VERSION = "policy-training-readiness-review-summary/v1"
SMOKE_SCHEMA_VERSION = "calibrated-policy-application-smoke-summary/v1"
READINESS_SCHEMA_VERSION = "channel-aware-training-readiness-summary/v1"
COVERAGE_SCHEMA_VERSION = "channel-aware-contrast-coverage-summary/v1"
CALIBRATION_SCHEMA_VERSION = "channel-aware-selection-contrast-calibration-summary/v1"
ANCHOR_CANDIDATE_SCHEMA_VERSION = "anchor-projection-candidate-generation-summary/v1"
ANCHOR_CONTRACT_SCHEMA_VERSION = "anchor-projection-evidence-contract-summary/v1"
CONTRACT_AWARE_TARGET_SCHEMA_VERSION = "anchor-projection-contract-aware-trainable-target-summary/v1"
PLANNER_VALIDATED_MINING_SCHEMA_VERSION = "planner-validated-trainable-target-mining-summary/v1"
HYBRID_TRAINING_DRY_RUN_SCHEMA_VERSION = "hybrid-policy-training-dry-run-summary/v1"
CONTROLLED_HYBRID_CANDIDATE_SCHEMA_VERSION = (
    "controlled-hybrid-policy-training-candidate-summary/v1"
)
CONTROLLED_HYBRID_HOLDOUT_SCHEMA_VERSION = (
    "controlled-hybrid-policy-holdout-evaluation-summary/v1"
)
FRESH_HOLDOUT_SCHEMA_VERSION = "fresh-holdout-policy-candidate-evaluation-summary/v1"
SCENARIO_DISJOINT_ROLLOUT_SCHEMA_VERSION = (
    "scenario-disjoint-policy-rollout-evaluation-summary/v1"
)
RAW_POLICY_STRICT_ROLLOUT_SCHEMA_VERSION = "raw-policy-strict-rollout-evaluation-summary/v1"
RAW_POLICY_GENERALIZATION_SCHEMA_VERSION = "raw-policy-generalization-evaluation-summary/v1"
POLICY_GATED_CANARY_SCHEMA_VERSION = "policy-gated-canary-rollout-summary/v1"
POLICY_GATED_SEQUENTIAL_CANARY_SCHEMA_VERSION = (
    "policy-gated-sequential-canary-rollout-summary/v1"
)
SUBMODULES = ("dev-platform-constraints", "model-explorer", "path-planner")
READY_SMOKE_ACTION = "ready_for_policy_training_readiness_review"
READY_DRY_RUN_ACTION = "ready_for_limited_policy_training_dry_run"
HYBRID_DRY_RUN_COMPLETED_ACTION = "hybrid_training_dry_run_completed"
CONTROLLED_HYBRID_CANDIDATE_EVALUATED_ACTION = (
    "controlled_hybrid_training_candidate_evaluated"
)
FRESH_HOLDOUT_CANDIDATE_EVALUATED_ACTION = "fresh_holdout_policy_candidate_evaluated"
SCENARIO_DISJOINT_POLICY_CANDIDATE_EVALUATED_ACTION = (
    "scenario_disjoint_policy_candidate_evaluated"
)
SCENARIO_DISJOINT_POLICY_ROLLOUT_EVALUATED_ACTION = (
    "scenario_disjoint_policy_rollout_evaluated"
)
RAW_POLICY_DECISION_ALIGNMENT_EVALUATED_ACTION = "raw_policy_decision_alignment_evaluated"
RAW_POLICY_GENERALIZATION_EVALUATED_ACTION = "raw_policy_generalization_evaluated"
POLICY_GATED_CANARY_ROLLOUT_EVALUATED_ACTION = "policy_gated_canary_rollout_evaluated"
POLICY_GATED_CANARY_DIVERSITY_EVALUATED_ACTION = "policy_gated_canary_diversity_evaluated"
POLICY_GATED_CANARY_OPPORTUNITY_QUALITY_EVALUATED_ACTION = (
    "policy_gated_canary_opportunity_quality_evaluated"
)
POLICY_GATED_CANARY_FULL_FAMILY_OPPORTUNITY_EVALUATED_ACTION = (
    "policy_gated_canary_full_family_opportunity_evaluated"
)
POLICY_GATED_CANARY_VALUE_STABILITY_EVALUATED_ACTION = (
    "policy_gated_canary_value_stability_evaluated"
)
POLICY_GATED_SEQUENTIAL_CANARY_ROLLOUT_EVALUATED_ACTION = (
    "policy_gated_sequential_canary_rollout_evaluated"
)
POLICY_GATED_SEQUENTIAL_SAFE_CHOICE_CALIBRATED_ACTION = (
    "policy_gated_sequential_safe_choice_calibrated"
)
POLICY_GATED_SEQUENTIAL_MULTI_STEP_OPPORTUNITY_EVALUATED_ACTION = (
    "policy_gated_sequential_multi_step_opportunity_evaluated"
)
PPO_ROLLOUT_COLLECTOR_DRY_RUN_EVALUATED_ACTION = "ppo_rollout_collector_dry_run_evaluated"
PPO_ROLLOUT_COLLECTOR_SCHEMA_VERSION = "ppo-rollout-collector-summary/v1"
LIMITED_PPO_UPDATE_SMOKE_EVALUATED_ACTION = "limited_ppo_update_smoke_evaluated"
LIMITED_PPO_UPDATE_SMOKE_SCHEMA_VERSION = "limited-ppo-update-smoke-summary/v1"
LIMITED_QUASI_REAL_PPO_UPDATE_SMOKE_EVALUATED_ACTION = (
    "limited_quasi_real_ppo_update_smoke_evaluated"
)
LIMITED_QUASI_REAL_PPO_UPDATE_SMOKE_SCHEMA_VERSION = (
    "limited-quasi-real-ppo-update-smoke-summary/v1"
)
GENERATED_SEQUENTIAL_GATE_METRIC_ACCOUNTING_AUDIT_SCHEMA_VERSION = (
    "generated-sequential-gate-metric-accounting-audit-summary/v1"
)
GENERATED_SEQUENTIAL_LONG_HORIZON_TEACHER_SKILL_CONTRACT_SCHEMA_VERSION = (
    "generated-sequential-long-horizon-teacher-skill-contract-summary/v1"
)
ITERATIVE_PPO_MINI_LOOP_STABILITY_EVALUATED_ACTION = (
    "iterative_ppo_mini_loop_stability_evaluated"
)
ITERATIVE_PPO_MINI_LOOP_STABILITY_SCHEMA_VERSION = (
    "iterative-ppo-mini-loop-stability-summary/v1"
)
RETURN_ALIGNED_GUARDED_MULTISTEP_COLLECTOR_EVALUATED_ACTION = (
    "return_aligned_guarded_multistep_collector_evaluated"
)
RETURN_ALIGNED_GUARDED_MULTISTEP_COLLECTOR_SCHEMA_VERSION = (
    "return-aligned-guarded-multistep-collector-summary/v1"
)
RETURN_ALIGNED_GUARDED_PPO_UPDATE_SMOKE_EVALUATED_ACTION = (
    "return_aligned_guarded_ppo_update_smoke_evaluated"
)
RETURN_ALIGNED_GUARDED_PPO_UPDATE_SMOKE_SCHEMA_VERSION = (
    "return-aligned-guarded-ppo-update-smoke-summary/v1"
)
GUARDED_PPO_ROLLOUT_PILOT_EVALUATED_ACTION = "guarded_ppo_rollout_pilot_evaluated"
GUARDED_PPO_ROLLOUT_PILOT_SCHEMA_VERSION = "guarded-ppo-rollout-pilot-summary/v1"
QUASI_REAL_GUARDED_PPO_ROLLOUT_PILOT_EVALUATED_ACTION = (
    "quasi_real_guarded_ppo_rollout_pilot_evaluated"
)
QUASI_REAL_GUARDED_PPO_ROLLOUT_PILOT_SCHEMA_VERSION = (
    "quasi-real-guarded-ppo-rollout-pilot-summary/v1"
)
QUASI_REAL_GUARDED_PPO_STABILITY_REPLAY_EVALUATED_ACTION = (
    "quasi_real_guarded_ppo_stability_replay_evaluated"
)
QUASI_REAL_GUARDED_PPO_STABILITY_REPLAY_SCHEMA_VERSION = (
    "quasi-real-guarded-ppo-stability-replay-summary/v1"
)
QUASI_REAL_GUARDED_PPO_HORIZON5_BATCH_EXPANSION_EVALUATED_ACTION = (
    "quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated"
)
QUASI_REAL_GUARDED_PPO_HORIZON5_BATCH_EXPANSION_SCHEMA_VERSION = (
    "quasi-real-guarded-ppo-horizon5-batch-expansion-summary/v1"
)
QUASI_REAL_GUARDED_PPO_SCALE512_MULTISEED_PREFLIGHT_EVALUATED_ACTION = (
    "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated"
)
QUASI_REAL_GUARDED_PPO_SCALE512_MULTISEED_PREFLIGHT_SCHEMA_VERSION = (
    "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary/v1"
)
QUASI_REAL_GUARDED_PPO_ITERATIVE_MINILOOP_STABILITY_EVALUATED_ACTION = (
    "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated"
)
QUASI_REAL_GUARDED_PPO_ITERATIVE_MINILOOP_STABILITY_SCHEMA_VERSION = (
    "quasi-real-guarded-ppo-iterative-miniloop-stability-summary/v1"
)
QUASI_REAL_GUARDED_FORMAL_PPO_PREFLIGHT_EVALUATED_ACTION = (
    "quasi_real_guarded_formal_ppo_preflight_evaluated"
)
QUASI_REAL_GUARDED_FORMAL_PPO_PREFLIGHT_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-preflight-summary/v1"
)
QUASI_REAL_GUARDED_FORMAL_PPO_ROLLOUT_CANARY_EVALUATED_ACTION = (
    "quasi_real_guarded_formal_ppo_rollout_canary_evaluated"
)
QUASI_REAL_GUARDED_FORMAL_PPO_ROLLOUT_CANARY_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-rollout-canary-summary/v1"
)
QUASI_REAL_GUARDED_FORMAL_PPO_STABILITY_HOLDOUT_VALIDATED_ACTION = (
    "quasi_real_guarded_formal_ppo_stability_holdout_validated"
)
QUASI_REAL_GUARDED_FORMAL_PPO_STABILITY_HOLDOUT_SCHEMA_VERSION = (
    "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1"
)
POLICY_TRAINING_CUDA_DEVICE_SUPPORT_EVALUATED_ACTION = (
    "policy_training_cuda_device_support_evaluated"
)
POLICY_TRAINING_CUDA_DEVICE_SUPPORT_SCHEMA_VERSION = (
    "policy-training-cuda-device-support-summary/v1"
)
QUASI_REAL_MAP_DOMAIN_GAP_EVALUATED_ACTION = "quasi_real_map_domain_gap_evaluated"
QUASI_REAL_MAP_DOMAIN_GAP_SCHEMA_VERSION = "quasi-real-map-domain-gap-summary/v1"
QUASI_REAL_SHADOW_POLICY_BEHAVIOR_AUDITED_ACTION = (
    "quasi_real_shadow_policy_behavior_audited"
)
QUASI_REAL_SHADOW_POLICY_BEHAVIOR_SCHEMA_VERSION = (
    "quasi-real-shadow-policy-behavior-summary/v1"
)
QUASI_REAL_SHADOW_ALIGNMENT_EVALUATED_ACTION = "quasi_real_shadow_alignment_evaluated"
QUASI_REAL_SHADOW_ALIGNMENT_SCHEMA_VERSION = "quasi-real-shadow-alignment-summary/v1"
QUASI_REAL_GUARDED_POLICY_PILOT_EVALUATED_ACTION = (
    "quasi_real_guarded_policy_pilot_evaluated"
)
QUASI_REAL_GUARDED_POLICY_PILOT_SCHEMA_VERSION = (
    "quasi-real-guarded-policy-pilot-summary/v1"
)
QUASI_REAL_SAFE_ALTERNATIVE_OPPORTUNITY_DIAGNOSED_ACTION = (
    "quasi_real_safe_alternative_opportunity_diagnosed"
)
QUASI_REAL_SAFE_ALTERNATIVE_OPPORTUNITY_SCHEMA_VERSION = (
    "quasi-real-safe-alternative-opportunity-summary/v1"
)
QUASI_REAL_SAFE_BETTER_OPPORTUNITY_EXPANDED_ACTION = (
    "quasi_real_safe_better_opportunity_expanded"
)
QUASI_REAL_SAFE_BETTER_OPPORTUNITY_EXPANSION_SCHEMA_VERSION = (
    "quasi-real-safe-better-opportunity-expansion-summary/v1"
)
QUASI_REAL_TEACHER_EQUIVALENT_VALIDATED_ACTION = (
    "quasi_real_teacher_equivalent_validated"
)
QUASI_REAL_TEACHER_EQUIVALENT_SCHEMA_VERSION = (
    "quasi-real-teacher-equivalent-summary/v1"
)
QUASI_REAL_TEACHER_DISTILLATION_EVALUATED_ACTION = (
    "quasi_real_teacher_distillation_robustness_evaluated"
)
QUASI_REAL_TEACHER_DISTILLATION_SCHEMA_VERSION = (
    "quasi-real-teacher-distillation-summary/v1"
)
QUASI_REAL_GUARDED_TEACHER_FOLLOWING_PILOT_EVALUATED_ACTION = (
    "quasi_real_guarded_teacher_following_pilot_evaluated"
)
QUASI_REAL_GUARDED_TEACHER_FOLLOWING_PILOT_SCHEMA_VERSION = (
    "quasi-real-guarded-teacher-following-pilot-summary/v1"
)
CONTROLLED_HYBRID_NEXT_REQUIRED_CHANGE = (
    "training_objective_or_sample_weight_refinement_required"
)
RAW_POLICY_ALIGNMENT_NEXT_REQUIRED_CHANGE = (
    "policy_objective_or_feature_refinement_required"
)
RAW_POLICY_GENERALIZATION_NEXT_REQUIRED_CHANGE = (
    "scenario_distribution_gap_requires_more_holdout_coverage"
)
CONTRACT_GUARD_FIELDS = (
    "does_not_modify_default_astar",
    "does_not_modify_ppo",
    "does_not_modify_network",
    "does_not_modify_action_space",
    "does_not_modify_model_explorer_contract",
    "does_not_modify_path_planner_route_contract",
    "does_not_modify_path_planner_sidecar_contract",
    "no_ackermann_feasible_trajectory_claim",
)
FALLBACK_COUNT_FIELDS = (
    "open_grid_fallback_used_count",
    "open_grid_fallback_count",
    "fallback_used_count",
    "fallback_or_open_grid_count",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Review calibrated policy target selection evidence before any policy training."
    )
    parser.add_argument("--batch-root", required=True, help="Batch root containing channel-aware summaries.")
    parser.add_argument(
        "--calibrated-policy-application-smoke-summary",
        help="calibrated-policy-application-smoke-summary/v1 JSON. Defaults to <batch-root>/calibrated-policy-application-smoke-summary.json.",
    )
    parser.add_argument(
        "--readiness-summary",
        help="channel-aware-training-readiness-summary/v1 JSON. Defaults to <batch-root>/channel-aware-training-readiness-summary.json.",
    )
    parser.add_argument(
        "--contrast-coverage-summary",
        help="channel-aware-contrast-coverage-summary/v1 JSON. Defaults to <batch-root>/channel-aware-contrast-coverage-summary.json.",
    )
    parser.add_argument(
        "--selection-contrast-calibration-summary",
        help="channel-aware-selection-contrast-calibration-summary/v1 JSON. Defaults to <batch-root>/channel-aware-selection-contrast-calibration-summary.json.",
    )
    parser.add_argument(
        "--anchor-projection-candidate-generation-summary",
        help="Optional anchor-projection-candidate-generation-summary/v1 JSON. Defaults to <batch-root>/anchor-projection-candidate-generation-summary.json when present.",
    )
    parser.add_argument(
        "--anchor-projection-evidence-contract-summary",
        help="Optional anchor-projection-evidence-contract-summary/v1 JSON. Defaults to <batch-root>/anchor-projection-evidence-contract-summary.json when present.",
    )
    parser.add_argument(
        "--contract-aware-trainable-target-summary",
        help="Optional anchor-projection-contract-aware-trainable-target-summary/v1 JSON. Defaults to <batch-root>/anchor-projection-contract-aware-trainable-target-summary.json when present.",
    )
    parser.add_argument(
        "--planner-validated-trainable-target-mining-summary",
        help="Optional planner-validated-trainable-target-mining-summary/v1 JSON. Defaults to <batch-root>/planner-validated-trainable-target-mining-summary.json when present.",
    )
    parser.add_argument(
        "--hybrid-policy-training-dry-run-summary",
        help="Optional hybrid-policy-training-dry-run-summary/v1 JSON. Defaults to <batch-root>/hybrid-policy-training-dry-run-summary.json when present.",
    )
    parser.add_argument(
        "--controlled-hybrid-policy-training-candidate-summary",
        help="Optional controlled-hybrid-policy-training-candidate-summary/v1 JSON.",
    )
    parser.add_argument(
        "--controlled-hybrid-policy-holdout-evaluation-summary",
        help="Optional controlled-hybrid-policy-holdout-evaluation-summary/v1 JSON.",
    )
    parser.add_argument(
        "--fresh-holdout-policy-candidate-evaluation-summary",
        help="Optional fresh-holdout-policy-candidate-evaluation-summary/v1 JSON.",
    )
    parser.add_argument(
        "--scenario-disjoint-policy-rollout-evaluation-summary",
        help="Optional scenario-disjoint-policy-rollout-evaluation-summary/v1 JSON.",
    )
    parser.add_argument(
        "--raw-policy-strict-rollout-evaluation-summary",
        help="Optional raw-policy-strict-rollout-evaluation-summary/v1 JSON.",
    )
    parser.add_argument(
        "--raw-policy-generalization-evaluation-summary",
        help="Optional raw-policy-generalization-evaluation-summary/v1 JSON.",
    )
    parser.add_argument(
        "--policy-gated-canary-rollout-summary",
        help="Optional policy-gated-canary-rollout-summary/v1 JSON.",
    )
    parser.add_argument(
        "--policy-gated-sequential-canary-rollout-summary",
        help="Optional policy-gated-sequential-canary-rollout-summary/v1 JSON.",
    )
    parser.add_argument(
        "--ppo-rollout-collector-summary",
        help="Optional ppo-rollout-collector-summary/v1 JSON.",
    )
    parser.add_argument(
        "--limited-ppo-update-smoke-summary",
        help="Optional limited-ppo-update-smoke-summary/v1 JSON.",
    )
    parser.add_argument(
        "--limited-quasi-real-ppo-update-smoke-summary",
        help="Optional limited-quasi-real-ppo-update-smoke-summary/v1 JSON.",
    )
    parser.add_argument(
        "--generated-sequential-gate-metric-accounting-audit-summary",
        help="Optional generated-sequential-gate-metric-accounting-audit-summary/v1 JSON.",
    )
    parser.add_argument(
        "--generated-sequential-long-horizon-teacher-skill-contract-summary",
        help="Optional generated-sequential-long-horizon-teacher-skill-contract-summary/v1 JSON.",
    )
    parser.add_argument(
        "--iterative-ppo-mini-loop-stability-summary",
        help="Optional iterative-ppo-mini-loop-stability-summary/v1 JSON.",
    )
    parser.add_argument(
        "--return-aligned-guarded-multistep-collector-summary",
        help="Optional return-aligned-guarded-multistep-collector-summary/v1 JSON.",
    )
    parser.add_argument(
        "--return-aligned-guarded-ppo-update-smoke-summary",
        help="Optional return-aligned-guarded-ppo-update-smoke-summary/v1 JSON.",
    )
    parser.add_argument(
        "--guarded-ppo-rollout-pilot-summary",
        help="Optional guarded-ppo-rollout-pilot-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-ppo-rollout-pilot-summary",
        help="Optional quasi-real-guarded-ppo-rollout-pilot-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-ppo-stability-replay-summary",
        help="Optional quasi-real-guarded-ppo-stability-replay-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-ppo-horizon5-batch-expansion-summary",
        help="Optional quasi-real-guarded-ppo-horizon5-batch-expansion-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-ppo-scale512-multiseed-preflight-summary",
        help="Optional quasi-real-guarded-ppo-scale512-multiseed-preflight-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-ppo-iterative-miniloop-stability-summary",
        help="Optional quasi-real-guarded-ppo-iterative-miniloop-stability-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-formal-ppo-preflight-summary",
        help="Optional quasi-real-guarded-formal-ppo-preflight-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-formal-ppo-rollout-canary-summary",
        help="Optional quasi-real-guarded-formal-ppo-rollout-canary-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-formal-ppo-stability-holdout-validation-summary",
        help="Optional quasi-real-guarded-formal-ppo-stability-holdout-validation-summary/v1 JSON.",
    )
    parser.add_argument(
        "--policy-training-cuda-device-support-summary",
        help="Optional policy-training-cuda-device-support-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-map-domain-gap-summary",
        help="Optional quasi-real-map-domain-gap-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-shadow-policy-behavior-summary",
        help="Optional quasi-real-shadow-policy-behavior-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-shadow-alignment-summary",
        help="Optional quasi-real-shadow-alignment-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-policy-pilot-summary",
        help="Optional quasi-real-guarded-policy-pilot-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-safe-alternative-opportunity-summary",
        help="Optional quasi-real-safe-alternative-opportunity-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-safe-better-opportunity-expansion-summary",
        help="Optional quasi-real-safe-better-opportunity-expansion-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-teacher-equivalent-validation-summary",
        help="Optional quasi-real-teacher-equivalent-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-teacher-distillation-summary",
        help="Optional quasi-real-teacher-distillation-summary/v1 JSON.",
    )
    parser.add_argument(
        "--quasi-real-guarded-teacher-following-pilot-summary",
        help="Optional quasi-real-guarded-teacher-following-pilot-summary/v1 JSON.",
    )
    parser.add_argument(
        "--config",
        default="configs/policy_training_readiness_review_v1.json",
        help="Policy training readiness review config JSON. Defaults to configs/policy_training_readiness_review_v1.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned output paths.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without writing outputs.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    batch_root = _resolve_path(args.batch_root, repo_root)
    smoke_path = (
        _resolve_path(args.calibrated_policy_application_smoke_summary, repo_root)
        if args.calibrated_policy_application_smoke_summary
        else batch_root / "calibrated-policy-application-smoke-summary.json"
    )
    readiness_path = (
        _resolve_path(args.readiness_summary, repo_root)
        if args.readiness_summary
        else batch_root / "channel-aware-training-readiness-summary.json"
    )
    coverage_path = (
        _resolve_path(args.contrast_coverage_summary, repo_root)
        if args.contrast_coverage_summary
        else batch_root / "channel-aware-contrast-coverage-summary.json"
    )
    calibration_path = (
        _resolve_path(args.selection_contrast_calibration_summary, repo_root)
        if args.selection_contrast_calibration_summary
        else batch_root / "channel-aware-selection-contrast-calibration-summary.json"
    )
    anchor_candidate_path = (
        _resolve_path(args.anchor_projection_candidate_generation_summary, repo_root)
        if args.anchor_projection_candidate_generation_summary
        else batch_root / "anchor-projection-candidate-generation-summary.json"
    )
    anchor_contract_path = (
        _resolve_path(args.anchor_projection_evidence_contract_summary, repo_root)
        if args.anchor_projection_evidence_contract_summary
        else batch_root / "anchor-projection-evidence-contract-summary.json"
    )
    contract_aware_target_path = (
        _resolve_path(args.contract_aware_trainable_target_summary, repo_root)
        if args.contract_aware_trainable_target_summary
        else batch_root / "anchor-projection-contract-aware-trainable-target-summary.json"
    )
    planner_validated_mining_path = (
        _resolve_path(args.planner_validated_trainable_target_mining_summary, repo_root)
        if args.planner_validated_trainable_target_mining_summary
        else batch_root / "planner-validated-trainable-target-mining-summary.json"
    )
    hybrid_training_dry_run_path = (
        _resolve_path(args.hybrid_policy_training_dry_run_summary, repo_root)
        if args.hybrid_policy_training_dry_run_summary
        else batch_root / "hybrid-policy-training-dry-run-summary.json"
    )
    controlled_candidate_path = (
        _resolve_path(args.controlled_hybrid_policy_training_candidate_summary, repo_root)
        if args.controlled_hybrid_policy_training_candidate_summary
        else batch_root / "controlled-hybrid-policy-training-candidate-summary.json"
    )
    controlled_holdout_path = (
        _resolve_path(args.controlled_hybrid_policy_holdout_evaluation_summary, repo_root)
        if args.controlled_hybrid_policy_holdout_evaluation_summary
        else batch_root / "controlled-hybrid-policy-holdout-evaluation-summary.json"
    )
    fresh_holdout_path = (
        _resolve_path(args.fresh_holdout_policy_candidate_evaluation_summary, repo_root)
        if args.fresh_holdout_policy_candidate_evaluation_summary
        else batch_root / "fresh-holdout-policy-candidate-evaluation-summary.json"
    )
    scenario_rollout_path = (
        _resolve_path(args.scenario_disjoint_policy_rollout_evaluation_summary, repo_root)
        if args.scenario_disjoint_policy_rollout_evaluation_summary
        else batch_root / "scenario-disjoint-policy-rollout-evaluation-summary.json"
    )
    raw_strict_rollout_path = (
        _resolve_path(args.raw_policy_strict_rollout_evaluation_summary, repo_root)
        if args.raw_policy_strict_rollout_evaluation_summary
        else batch_root / "raw-policy-strict-rollout-evaluation-summary.json"
    )
    raw_generalization_path = (
        _resolve_path(args.raw_policy_generalization_evaluation_summary, repo_root)
        if args.raw_policy_generalization_evaluation_summary
        else batch_root / "raw-policy-generalization-evaluation-summary.json"
    )
    policy_canary_path = (
        _resolve_path(args.policy_gated_canary_rollout_summary, repo_root)
        if args.policy_gated_canary_rollout_summary
        else batch_root / "policy-gated-canary-rollout-summary.json"
    )
    sequential_canary_path = (
        _resolve_path(args.policy_gated_sequential_canary_rollout_summary, repo_root)
        if args.policy_gated_sequential_canary_rollout_summary
        else batch_root / "policy-gated-sequential-canary-rollout-summary.json"
    )
    ppo_collector_path = (
        _resolve_path(args.ppo_rollout_collector_summary, repo_root)
        if args.ppo_rollout_collector_summary
        else batch_root / "ppo-rollout-collector-summary.json"
    )
    limited_ppo_update_smoke_path = (
        _resolve_path(args.limited_ppo_update_smoke_summary, repo_root)
        if args.limited_ppo_update_smoke_summary
        else batch_root / "limited-ppo-update-smoke-summary.json"
    )
    limited_quasi_real_ppo_update_smoke_path = (
        _resolve_path(args.limited_quasi_real_ppo_update_smoke_summary, repo_root)
        if args.limited_quasi_real_ppo_update_smoke_summary
        else batch_root / "limited-quasi-real-ppo-update-smoke-summary.json"
    )
    generated_sequential_gate_metric_accounting_audit_path = (
        _resolve_path(args.generated_sequential_gate_metric_accounting_audit_summary, repo_root)
        if args.generated_sequential_gate_metric_accounting_audit_summary
        else batch_root / "generated-sequential-gate-metric-accounting-audit-summary.json"
    )
    generated_sequential_long_horizon_teacher_skill_contract_path = (
        _resolve_path(args.generated_sequential_long_horizon_teacher_skill_contract_summary, repo_root)
        if args.generated_sequential_long_horizon_teacher_skill_contract_summary
        else batch_root / "long-horizon-teacher-skill-contract-summary.json"
    )
    iterative_ppo_mini_loop_path = (
        _resolve_path(args.iterative_ppo_mini_loop_stability_summary, repo_root)
        if args.iterative_ppo_mini_loop_stability_summary
        else batch_root / "iterative-ppo-mini-loop-stability-summary.json"
    )
    return_aligned_guarded_multistep_collector_path = (
        _resolve_path(args.return_aligned_guarded_multistep_collector_summary, repo_root)
        if args.return_aligned_guarded_multistep_collector_summary
        else batch_root / "return-aligned-collector-summary.json"
    )
    return_aligned_guarded_ppo_update_smoke_path = (
        _resolve_path(args.return_aligned_guarded_ppo_update_smoke_summary, repo_root)
        if args.return_aligned_guarded_ppo_update_smoke_summary
        else batch_root / "return-aligned-guarded-ppo-update-smoke-summary.json"
    )
    guarded_ppo_rollout_pilot_path = (
        _resolve_path(args.guarded_ppo_rollout_pilot_summary, repo_root)
        if args.guarded_ppo_rollout_pilot_summary
        else batch_root / "guarded-ppo-rollout-pilot-summary.json"
    )
    quasi_real_guarded_ppo_rollout_pilot_path = (
        _resolve_path(args.quasi_real_guarded_ppo_rollout_pilot_summary, repo_root)
        if args.quasi_real_guarded_ppo_rollout_pilot_summary
        else batch_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"
    )
    quasi_real_guarded_ppo_stability_replay_path = (
        _resolve_path(args.quasi_real_guarded_ppo_stability_replay_summary, repo_root)
        if args.quasi_real_guarded_ppo_stability_replay_summary
        else batch_root / "quasi-real-guarded-ppo-stability-replay-summary.json"
    )
    quasi_real_guarded_ppo_horizon5_batch_expansion_path = (
        _resolve_path(
            args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary,
            repo_root,
        )
        if args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary
        else batch_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json"
    )
    quasi_real_guarded_ppo_scale512_multiseed_preflight_path = (
        _resolve_path(
            args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary,
            repo_root,
        )
        if args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary
        else batch_root
        / "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json"
    )
    quasi_real_guarded_ppo_iterative_miniloop_stability_path = (
        _resolve_path(
            args.quasi_real_guarded_ppo_iterative_miniloop_stability_summary,
            repo_root,
        )
        if args.quasi_real_guarded_ppo_iterative_miniloop_stability_summary
        else batch_root
        / "quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json"
    )
    quasi_real_guarded_formal_ppo_preflight_path = (
        _resolve_path(
            args.quasi_real_guarded_formal_ppo_preflight_summary,
            repo_root,
        )
        if args.quasi_real_guarded_formal_ppo_preflight_summary
        else batch_root / "quasi-real-guarded-formal-ppo-preflight-summary.json"
    )
    quasi_real_guarded_formal_ppo_rollout_canary_path = (
        _resolve_path(
            args.quasi_real_guarded_formal_ppo_rollout_canary_summary,
            repo_root,
        )
        if args.quasi_real_guarded_formal_ppo_rollout_canary_summary
        else batch_root / "quasi-real-guarded-formal-ppo-rollout-canary-summary.json"
    )
    quasi_real_guarded_formal_ppo_stability_holdout_path = (
        _resolve_path(
            args.quasi_real_guarded_formal_ppo_stability_holdout_validation_summary,
            repo_root,
        )
        if args.quasi_real_guarded_formal_ppo_stability_holdout_validation_summary
        else batch_root
        / "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json"
    )
    policy_training_cuda_device_support_path = (
        _resolve_path(args.policy_training_cuda_device_support_summary, repo_root)
        if args.policy_training_cuda_device_support_summary
        else batch_root / "policy-training-cuda-device-support-summary.json"
    )
    quasi_real_map_domain_gap_path = (
        _resolve_path(args.quasi_real_map_domain_gap_summary, repo_root)
        if args.quasi_real_map_domain_gap_summary
        else batch_root / "quasi-real-map-domain-gap-summary.json"
    )
    quasi_real_shadow_policy_behavior_path = (
        _resolve_path(args.quasi_real_shadow_policy_behavior_summary, repo_root)
        if args.quasi_real_shadow_policy_behavior_summary
        else batch_root / "quasi-real-shadow-policy-behavior-summary.json"
    )
    quasi_real_shadow_alignment_path = (
        _resolve_path(args.quasi_real_shadow_alignment_summary, repo_root)
        if args.quasi_real_shadow_alignment_summary
        else batch_root / "quasi-real-shadow-alignment-summary.json"
    )
    quasi_real_guarded_policy_pilot_path = (
        _resolve_path(args.quasi_real_guarded_policy_pilot_summary, repo_root)
        if args.quasi_real_guarded_policy_pilot_summary
        else batch_root / "quasi-real-guarded-policy-pilot-summary.json"
    )
    quasi_real_safe_alternative_opportunity_path = (
        _resolve_path(args.quasi_real_safe_alternative_opportunity_summary, repo_root)
        if args.quasi_real_safe_alternative_opportunity_summary
        else batch_root / "quasi-real-safe-alternative-opportunity-summary.json"
    )
    quasi_real_safe_better_opportunity_expansion_path = (
        _resolve_path(args.quasi_real_safe_better_opportunity_expansion_summary, repo_root)
        if args.quasi_real_safe_better_opportunity_expansion_summary
        else batch_root / "quasi-real-safe-better-opportunity-expansion-summary.json"
    )
    quasi_real_teacher_equivalent_validation_path = (
        _resolve_path(args.quasi_real_teacher_equivalent_validation_summary, repo_root)
        if args.quasi_real_teacher_equivalent_validation_summary
        else batch_root / "quasi-real-teacher-equivalent-summary.json"
    )
    quasi_real_teacher_distillation_path = (
        _resolve_path(args.quasi_real_teacher_distillation_summary, repo_root)
        if args.quasi_real_teacher_distillation_summary
        else batch_root / "quasi-real-teacher-distillation-summary.json"
    )
    quasi_real_guarded_teacher_following_pilot_path = (
        _resolve_path(args.quasi_real_guarded_teacher_following_pilot_summary, repo_root)
        if args.quasi_real_guarded_teacher_following_pilot_summary
        else batch_root / "quasi-real-guarded-teacher-following-pilot-summary.json"
    )
    explicit_iterative_only_summary = bool(args.iterative_ppo_mini_loop_stability_summary) and not any(
        (
            args.anchor_projection_candidate_generation_summary,
            args.anchor_projection_evidence_contract_summary,
            args.contract_aware_trainable_target_summary,
            args.planner_validated_trainable_target_mining_summary,
            args.hybrid_policy_training_dry_run_summary,
            args.controlled_hybrid_policy_training_candidate_summary,
            args.controlled_hybrid_policy_holdout_evaluation_summary,
            args.fresh_holdout_policy_candidate_evaluation_summary,
            args.scenario_disjoint_policy_rollout_evaluation_summary,
            args.raw_policy_strict_rollout_evaluation_summary,
            args.raw_policy_generalization_evaluation_summary,
            args.policy_gated_canary_rollout_summary,
            args.policy_gated_sequential_canary_rollout_summary,
            args.ppo_rollout_collector_summary,
            args.limited_ppo_update_smoke_summary,
            args.limited_quasi_real_ppo_update_smoke_summary,
            args.generated_sequential_gate_metric_accounting_audit_summary,
            args.generated_sequential_long_horizon_teacher_skill_contract_summary,
            args.return_aligned_guarded_multistep_collector_summary,
            args.return_aligned_guarded_ppo_update_smoke_summary,
            args.guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_stability_replay_summary,
            args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary,
            args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary,
            args.policy_training_cuda_device_support_summary,
            args.quasi_real_map_domain_gap_summary,
            args.quasi_real_shadow_policy_behavior_summary,
            args.quasi_real_shadow_alignment_summary,
            args.quasi_real_guarded_policy_pilot_summary,
            args.quasi_real_safe_alternative_opportunity_summary,
            args.quasi_real_safe_better_opportunity_expansion_summary,
            args.quasi_real_teacher_equivalent_validation_summary,
            args.quasi_real_teacher_distillation_summary,
            args.quasi_real_guarded_teacher_following_pilot_summary,
        )
    )
    explicit_guarded_only_summary = bool(args.guarded_ppo_rollout_pilot_summary) and not any(
        (
            args.anchor_projection_candidate_generation_summary,
            args.anchor_projection_evidence_contract_summary,
            args.contract_aware_trainable_target_summary,
            args.planner_validated_trainable_target_mining_summary,
            args.hybrid_policy_training_dry_run_summary,
            args.controlled_hybrid_policy_training_candidate_summary,
            args.controlled_hybrid_policy_holdout_evaluation_summary,
            args.fresh_holdout_policy_candidate_evaluation_summary,
            args.scenario_disjoint_policy_rollout_evaluation_summary,
            args.raw_policy_strict_rollout_evaluation_summary,
            args.raw_policy_generalization_evaluation_summary,
            args.policy_gated_canary_rollout_summary,
            args.policy_gated_sequential_canary_rollout_summary,
            args.ppo_rollout_collector_summary,
            args.limited_ppo_update_smoke_summary,
            args.limited_quasi_real_ppo_update_smoke_summary,
            args.generated_sequential_gate_metric_accounting_audit_summary,
            args.generated_sequential_long_horizon_teacher_skill_contract_summary,
            args.iterative_ppo_mini_loop_stability_summary,
            args.return_aligned_guarded_multistep_collector_summary,
            args.return_aligned_guarded_ppo_update_smoke_summary,
            args.quasi_real_guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_stability_replay_summary,
            args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary,
            args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary,
            args.policy_training_cuda_device_support_summary,
            args.quasi_real_map_domain_gap_summary,
            args.quasi_real_shadow_policy_behavior_summary,
            args.quasi_real_shadow_alignment_summary,
            args.quasi_real_guarded_policy_pilot_summary,
            args.quasi_real_safe_alternative_opportunity_summary,
            args.quasi_real_safe_better_opportunity_expansion_summary,
            args.quasi_real_teacher_equivalent_validation_summary,
            args.quasi_real_teacher_distillation_summary,
            args.quasi_real_guarded_teacher_following_pilot_summary,
        )
    )
    explicit_return_aligned_only_summary = bool(
        args.return_aligned_guarded_multistep_collector_summary
    ) and not any(
        (
            args.anchor_projection_candidate_generation_summary,
            args.anchor_projection_evidence_contract_summary,
            args.contract_aware_trainable_target_summary,
            args.planner_validated_trainable_target_mining_summary,
            args.hybrid_policy_training_dry_run_summary,
            args.controlled_hybrid_policy_training_candidate_summary,
            args.controlled_hybrid_policy_holdout_evaluation_summary,
            args.fresh_holdout_policy_candidate_evaluation_summary,
            args.scenario_disjoint_policy_rollout_evaluation_summary,
            args.raw_policy_strict_rollout_evaluation_summary,
            args.raw_policy_generalization_evaluation_summary,
            args.policy_gated_canary_rollout_summary,
            args.policy_gated_sequential_canary_rollout_summary,
            args.ppo_rollout_collector_summary,
            args.limited_ppo_update_smoke_summary,
            args.limited_quasi_real_ppo_update_smoke_summary,
            args.generated_sequential_gate_metric_accounting_audit_summary,
            args.generated_sequential_long_horizon_teacher_skill_contract_summary,
            args.iterative_ppo_mini_loop_stability_summary,
            args.return_aligned_guarded_ppo_update_smoke_summary,
            args.guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_stability_replay_summary,
            args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary,
            args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary,
            args.policy_training_cuda_device_support_summary,
            args.quasi_real_map_domain_gap_summary,
            args.quasi_real_shadow_policy_behavior_summary,
            args.quasi_real_shadow_alignment_summary,
            args.quasi_real_guarded_policy_pilot_summary,
            args.quasi_real_safe_alternative_opportunity_summary,
            args.quasi_real_safe_better_opportunity_expansion_summary,
            args.quasi_real_teacher_equivalent_validation_summary,
            args.quasi_real_teacher_distillation_summary,
            args.quasi_real_guarded_teacher_following_pilot_summary,
        )
    )
    explicit_quasi_real_guarded_rollout_only_summary = bool(
        args.quasi_real_guarded_ppo_rollout_pilot_summary
    ) and not any(
        (
            args.anchor_projection_candidate_generation_summary,
            args.anchor_projection_evidence_contract_summary,
            args.contract_aware_trainable_target_summary,
            args.planner_validated_trainable_target_mining_summary,
            args.hybrid_policy_training_dry_run_summary,
            args.controlled_hybrid_policy_training_candidate_summary,
            args.controlled_hybrid_policy_holdout_evaluation_summary,
            args.fresh_holdout_policy_candidate_evaluation_summary,
            args.scenario_disjoint_policy_rollout_evaluation_summary,
            args.raw_policy_strict_rollout_evaluation_summary,
            args.raw_policy_generalization_evaluation_summary,
            args.policy_gated_canary_rollout_summary,
            args.policy_gated_sequential_canary_rollout_summary,
            args.ppo_rollout_collector_summary,
            args.limited_ppo_update_smoke_summary,
            args.limited_quasi_real_ppo_update_smoke_summary,
            args.generated_sequential_gate_metric_accounting_audit_summary,
            args.generated_sequential_long_horizon_teacher_skill_contract_summary,
            args.iterative_ppo_mini_loop_stability_summary,
            args.return_aligned_guarded_multistep_collector_summary,
            args.return_aligned_guarded_ppo_update_smoke_summary,
            args.guarded_ppo_rollout_pilot_summary,
            args.policy_training_cuda_device_support_summary,
            args.quasi_real_guarded_ppo_stability_replay_summary,
            args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary,
            args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary,
            args.quasi_real_map_domain_gap_summary,
            args.quasi_real_shadow_policy_behavior_summary,
            args.quasi_real_shadow_alignment_summary,
            args.quasi_real_guarded_policy_pilot_summary,
            args.quasi_real_safe_alternative_opportunity_summary,
            args.quasi_real_safe_better_opportunity_expansion_summary,
            args.quasi_real_teacher_equivalent_validation_summary,
            args.quasi_real_teacher_distillation_summary,
            args.quasi_real_guarded_teacher_following_pilot_summary,
        )
    )
    explicit_quasi_real_guarded_stability_only_summary = bool(
        args.quasi_real_guarded_ppo_stability_replay_summary
    ) and not any(
        (
            args.anchor_projection_candidate_generation_summary,
            args.anchor_projection_evidence_contract_summary,
            args.contract_aware_trainable_target_summary,
            args.planner_validated_trainable_target_mining_summary,
            args.hybrid_policy_training_dry_run_summary,
            args.controlled_hybrid_policy_training_candidate_summary,
            args.controlled_hybrid_policy_holdout_evaluation_summary,
            args.fresh_holdout_policy_candidate_evaluation_summary,
            args.scenario_disjoint_policy_rollout_evaluation_summary,
            args.raw_policy_strict_rollout_evaluation_summary,
            args.raw_policy_generalization_evaluation_summary,
            args.policy_gated_canary_rollout_summary,
            args.policy_gated_sequential_canary_rollout_summary,
            args.ppo_rollout_collector_summary,
            args.limited_ppo_update_smoke_summary,
            args.limited_quasi_real_ppo_update_smoke_summary,
            args.generated_sequential_gate_metric_accounting_audit_summary,
            args.generated_sequential_long_horizon_teacher_skill_contract_summary,
            args.iterative_ppo_mini_loop_stability_summary,
            args.return_aligned_guarded_multistep_collector_summary,
            args.return_aligned_guarded_ppo_update_smoke_summary,
            args.guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary,
            args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary,
            args.policy_training_cuda_device_support_summary,
            args.quasi_real_map_domain_gap_summary,
            args.quasi_real_shadow_policy_behavior_summary,
            args.quasi_real_shadow_alignment_summary,
            args.quasi_real_guarded_policy_pilot_summary,
            args.quasi_real_safe_alternative_opportunity_summary,
            args.quasi_real_safe_better_opportunity_expansion_summary,
            args.quasi_real_teacher_equivalent_validation_summary,
            args.quasi_real_teacher_distillation_summary,
            args.quasi_real_guarded_teacher_following_pilot_summary,
        )
    )
    explicit_quasi_real_guarded_horizon5_only_summary = bool(
        args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary
    ) and not any(
        (
            args.anchor_projection_candidate_generation_summary,
            args.anchor_projection_evidence_contract_summary,
            args.contract_aware_trainable_target_summary,
            args.planner_validated_trainable_target_mining_summary,
            args.hybrid_policy_training_dry_run_summary,
            args.controlled_hybrid_policy_training_candidate_summary,
            args.controlled_hybrid_policy_holdout_evaluation_summary,
            args.fresh_holdout_policy_candidate_evaluation_summary,
            args.scenario_disjoint_policy_rollout_evaluation_summary,
            args.raw_policy_strict_rollout_evaluation_summary,
            args.raw_policy_generalization_evaluation_summary,
            args.policy_gated_canary_rollout_summary,
            args.policy_gated_sequential_canary_rollout_summary,
            args.ppo_rollout_collector_summary,
            args.limited_ppo_update_smoke_summary,
            args.limited_quasi_real_ppo_update_smoke_summary,
            args.generated_sequential_gate_metric_accounting_audit_summary,
            args.generated_sequential_long_horizon_teacher_skill_contract_summary,
            args.iterative_ppo_mini_loop_stability_summary,
            args.return_aligned_guarded_multistep_collector_summary,
            args.return_aligned_guarded_ppo_update_smoke_summary,
            args.guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_stability_replay_summary,
            args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary,
            args.policy_training_cuda_device_support_summary,
            args.quasi_real_map_domain_gap_summary,
            args.quasi_real_shadow_policy_behavior_summary,
            args.quasi_real_shadow_alignment_summary,
            args.quasi_real_guarded_policy_pilot_summary,
            args.quasi_real_safe_alternative_opportunity_summary,
            args.quasi_real_safe_better_opportunity_expansion_summary,
            args.quasi_real_teacher_equivalent_validation_summary,
            args.quasi_real_teacher_distillation_summary,
            args.quasi_real_guarded_teacher_following_pilot_summary,
        )
    )
    explicit_quasi_real_guarded_scale512_only_summary = bool(
        args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary
    ) and not any(
        (
            args.anchor_projection_candidate_generation_summary,
            args.anchor_projection_evidence_contract_summary,
            args.contract_aware_trainable_target_summary,
            args.planner_validated_trainable_target_mining_summary,
            args.hybrid_policy_training_dry_run_summary,
            args.controlled_hybrid_policy_training_candidate_summary,
            args.controlled_hybrid_policy_holdout_evaluation_summary,
            args.fresh_holdout_policy_candidate_evaluation_summary,
            args.scenario_disjoint_policy_rollout_evaluation_summary,
            args.raw_policy_strict_rollout_evaluation_summary,
            args.raw_policy_generalization_evaluation_summary,
            args.policy_gated_canary_rollout_summary,
            args.policy_gated_sequential_canary_rollout_summary,
            args.ppo_rollout_collector_summary,
            args.limited_ppo_update_smoke_summary,
            args.limited_quasi_real_ppo_update_smoke_summary,
            args.generated_sequential_gate_metric_accounting_audit_summary,
            args.generated_sequential_long_horizon_teacher_skill_contract_summary,
            args.iterative_ppo_mini_loop_stability_summary,
            args.return_aligned_guarded_multistep_collector_summary,
            args.return_aligned_guarded_ppo_update_smoke_summary,
            args.guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_stability_replay_summary,
            args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary,
            args.policy_training_cuda_device_support_summary,
            args.quasi_real_map_domain_gap_summary,
            args.quasi_real_shadow_policy_behavior_summary,
            args.quasi_real_shadow_alignment_summary,
            args.quasi_real_guarded_policy_pilot_summary,
            args.quasi_real_safe_alternative_opportunity_summary,
            args.quasi_real_safe_better_opportunity_expansion_summary,
            args.quasi_real_teacher_equivalent_validation_summary,
            args.quasi_real_teacher_distillation_summary,
            args.quasi_real_guarded_teacher_following_pilot_summary,
        )
    )
    explicit_quasi_real_guarded_formal_stability_holdout_only_summary = bool(
        args.quasi_real_guarded_formal_ppo_stability_holdout_validation_summary
    ) and not any(
        (
            args.anchor_projection_candidate_generation_summary,
            args.anchor_projection_evidence_contract_summary,
            args.contract_aware_trainable_target_summary,
            args.planner_validated_trainable_target_mining_summary,
            args.hybrid_policy_training_dry_run_summary,
            args.controlled_hybrid_policy_training_candidate_summary,
            args.controlled_hybrid_policy_holdout_evaluation_summary,
            args.fresh_holdout_policy_candidate_evaluation_summary,
            args.scenario_disjoint_policy_rollout_evaluation_summary,
            args.raw_policy_strict_rollout_evaluation_summary,
            args.raw_policy_generalization_evaluation_summary,
            args.policy_gated_canary_rollout_summary,
            args.policy_gated_sequential_canary_rollout_summary,
            args.ppo_rollout_collector_summary,
            args.limited_ppo_update_smoke_summary,
            args.limited_quasi_real_ppo_update_smoke_summary,
            args.generated_sequential_gate_metric_accounting_audit_summary,
            args.generated_sequential_long_horizon_teacher_skill_contract_summary,
            args.iterative_ppo_mini_loop_stability_summary,
            args.return_aligned_guarded_multistep_collector_summary,
            args.return_aligned_guarded_ppo_update_smoke_summary,
            args.guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_rollout_pilot_summary,
            args.quasi_real_guarded_ppo_stability_replay_summary,
            args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary,
            args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary,
            args.quasi_real_guarded_ppo_iterative_miniloop_stability_summary,
            args.quasi_real_guarded_formal_ppo_preflight_summary,
            args.quasi_real_guarded_formal_ppo_rollout_canary_summary,
            args.policy_training_cuda_device_support_summary,
            args.quasi_real_map_domain_gap_summary,
            args.quasi_real_shadow_policy_behavior_summary,
            args.quasi_real_shadow_alignment_summary,
            args.quasi_real_guarded_policy_pilot_summary,
            args.quasi_real_safe_alternative_opportunity_summary,
            args.quasi_real_safe_better_opportunity_expansion_summary,
            args.quasi_real_teacher_equivalent_validation_summary,
            args.quasi_real_teacher_distillation_summary,
            args.quasi_real_guarded_teacher_following_pilot_summary,
        )
    )
    anchor_only_defaults_available = (
        not explicit_iterative_only_summary
        and not explicit_guarded_only_summary
        and not explicit_return_aligned_only_summary
        and not explicit_quasi_real_guarded_rollout_only_summary
        and not explicit_quasi_real_guarded_stability_only_summary
        and not explicit_quasi_real_guarded_horizon5_only_summary
        and not explicit_quasi_real_guarded_scale512_only_summary
        and not explicit_quasi_real_guarded_formal_stability_holdout_only_summary
        and not args.quasi_real_guarded_ppo_iterative_miniloop_stability_summary
        and not args.return_aligned_guarded_ppo_update_smoke_summary
        and not args.quasi_real_guarded_ppo_rollout_pilot_summary
        and not args.quasi_real_guarded_ppo_stability_replay_summary
        and not args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary
        and not args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary
        and not args.quasi_real_guarded_ppo_iterative_miniloop_stability_summary
        and not args.quasi_real_guarded_formal_ppo_preflight_summary
        and not args.quasi_real_guarded_formal_ppo_rollout_canary_summary
        and not args.quasi_real_guarded_formal_ppo_stability_holdout_validation_summary
        and
        anchor_candidate_path.is_file()
        and anchor_contract_path.is_file()
        and not any(path.is_file() for path in (smoke_path, readiness_path, coverage_path, calibration_path))
    )
    config_path = _resolve_path(args.config, repo_root)
    try:
        config = _load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    summary = analyze_policy_training_readiness_review(
        batch_root=batch_root,
        smoke_path=smoke_path,
        readiness_path=readiness_path,
        coverage_path=coverage_path,
        calibration_path=calibration_path,
        anchor_candidate_path=anchor_candidate_path,
        anchor_contract_path=anchor_contract_path,
        contract_aware_target_path=contract_aware_target_path,
        planner_validated_mining_path=planner_validated_mining_path,
        hybrid_training_dry_run_path=hybrid_training_dry_run_path,
        controlled_candidate_path=controlled_candidate_path,
        controlled_holdout_path=controlled_holdout_path,
        fresh_holdout_path=fresh_holdout_path,
        scenario_rollout_path=scenario_rollout_path,
        raw_strict_rollout_path=raw_strict_rollout_path,
        raw_generalization_path=raw_generalization_path,
        policy_canary_path=policy_canary_path,
        sequential_canary_path=sequential_canary_path,
        ppo_collector_path=ppo_collector_path,
        limited_ppo_update_smoke_path=limited_ppo_update_smoke_path,
        limited_quasi_real_ppo_update_smoke_path=limited_quasi_real_ppo_update_smoke_path,
        generated_sequential_gate_metric_accounting_audit_path=(
            generated_sequential_gate_metric_accounting_audit_path
        ),
        generated_sequential_long_horizon_teacher_skill_contract_path=(
            generated_sequential_long_horizon_teacher_skill_contract_path
        ),
        iterative_ppo_mini_loop_path=iterative_ppo_mini_loop_path,
        return_aligned_guarded_multistep_collector_path=(
            return_aligned_guarded_multistep_collector_path
        ),
        return_aligned_guarded_ppo_update_smoke_path=(
            return_aligned_guarded_ppo_update_smoke_path
        ),
        guarded_ppo_rollout_pilot_path=guarded_ppo_rollout_pilot_path,
        quasi_real_guarded_ppo_rollout_pilot_path=(
            quasi_real_guarded_ppo_rollout_pilot_path
        ),
        quasi_real_guarded_ppo_stability_replay_path=(
            quasi_real_guarded_ppo_stability_replay_path
        ),
        quasi_real_guarded_ppo_horizon5_batch_expansion_path=(
            quasi_real_guarded_ppo_horizon5_batch_expansion_path
        ),
        quasi_real_guarded_ppo_scale512_multiseed_preflight_path=(
            quasi_real_guarded_ppo_scale512_multiseed_preflight_path
        ),
        quasi_real_guarded_ppo_iterative_miniloop_stability_path=(
            quasi_real_guarded_ppo_iterative_miniloop_stability_path
        ),
        quasi_real_guarded_formal_ppo_preflight_path=(
            quasi_real_guarded_formal_ppo_preflight_path
        ),
        quasi_real_guarded_formal_ppo_rollout_canary_path=(
            quasi_real_guarded_formal_ppo_rollout_canary_path
        ),
        quasi_real_guarded_formal_ppo_stability_holdout_path=(
            quasi_real_guarded_formal_ppo_stability_holdout_path
        ),
        policy_training_cuda_device_support_path=policy_training_cuda_device_support_path,
        quasi_real_map_domain_gap_path=quasi_real_map_domain_gap_path,
        quasi_real_shadow_policy_behavior_path=quasi_real_shadow_policy_behavior_path,
        quasi_real_shadow_alignment_path=quasi_real_shadow_alignment_path,
        quasi_real_guarded_policy_pilot_path=quasi_real_guarded_policy_pilot_path,
        quasi_real_safe_alternative_opportunity_path=(
            quasi_real_safe_alternative_opportunity_path
        ),
        quasi_real_safe_better_opportunity_expansion_path=(
            quasi_real_safe_better_opportunity_expansion_path
        ),
        quasi_real_teacher_equivalent_validation_path=(
            quasi_real_teacher_equivalent_validation_path
        ),
        quasi_real_teacher_distillation_path=quasi_real_teacher_distillation_path,
        quasi_real_guarded_teacher_following_pilot_path=(
            quasi_real_guarded_teacher_following_pilot_path
        ),
        anchor_candidate_required=bool(args.anchor_projection_candidate_generation_summary)
        or anchor_only_defaults_available,
        anchor_contract_required=bool(args.anchor_projection_evidence_contract_summary)
        or anchor_only_defaults_available,
        contract_aware_target_required=bool(args.contract_aware_trainable_target_summary),
        planner_validated_mining_required=bool(
            args.planner_validated_trainable_target_mining_summary
        ),
        hybrid_training_dry_run_required=bool(args.hybrid_policy_training_dry_run_summary),
        controlled_candidate_required=bool(
            args.controlled_hybrid_policy_training_candidate_summary
        ),
        controlled_holdout_required=bool(
            args.controlled_hybrid_policy_holdout_evaluation_summary
        ),
        fresh_holdout_required=bool(args.fresh_holdout_policy_candidate_evaluation_summary),
        scenario_rollout_required=bool(args.scenario_disjoint_policy_rollout_evaluation_summary),
        raw_strict_rollout_required=bool(args.raw_policy_strict_rollout_evaluation_summary),
        raw_generalization_required=bool(args.raw_policy_generalization_evaluation_summary),
        policy_canary_required=bool(args.policy_gated_canary_rollout_summary),
        sequential_canary_required=bool(args.policy_gated_sequential_canary_rollout_summary),
        ppo_collector_required=bool(args.ppo_rollout_collector_summary),
        limited_ppo_update_smoke_required=bool(args.limited_ppo_update_smoke_summary),
        generated_sequential_gate_metric_accounting_audit_required=bool(
            args.generated_sequential_gate_metric_accounting_audit_summary
        ),
        generated_sequential_long_horizon_teacher_skill_contract_required=bool(
            args.generated_sequential_long_horizon_teacher_skill_contract_summary
        ),
        iterative_ppo_mini_loop_required=bool(args.iterative_ppo_mini_loop_stability_summary),
        return_aligned_guarded_multistep_collector_required=bool(
            args.return_aligned_guarded_multistep_collector_summary
        ),
        return_aligned_guarded_ppo_update_smoke_required=bool(
            args.return_aligned_guarded_ppo_update_smoke_summary
        ),
        guarded_ppo_rollout_pilot_required=bool(args.guarded_ppo_rollout_pilot_summary),
        quasi_real_guarded_ppo_rollout_pilot_required=bool(
            args.quasi_real_guarded_ppo_rollout_pilot_summary
        ),
        quasi_real_guarded_ppo_stability_replay_required=bool(
            args.quasi_real_guarded_ppo_stability_replay_summary
        ),
        quasi_real_guarded_ppo_horizon5_batch_expansion_required=bool(
            args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary
        ),
        quasi_real_guarded_ppo_scale512_multiseed_preflight_required=bool(
            args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary
        ),
        quasi_real_guarded_ppo_iterative_miniloop_stability_required=bool(
            args.quasi_real_guarded_ppo_iterative_miniloop_stability_summary
        ),
        quasi_real_guarded_formal_ppo_preflight_required=bool(
            args.quasi_real_guarded_formal_ppo_preflight_summary
        ),
        quasi_real_guarded_formal_ppo_rollout_canary_required=bool(
            args.quasi_real_guarded_formal_ppo_rollout_canary_summary
        ),
        quasi_real_guarded_formal_ppo_stability_holdout_required=bool(
            args.quasi_real_guarded_formal_ppo_stability_holdout_validation_summary
        ),
        policy_training_cuda_device_support_required=bool(
            args.policy_training_cuda_device_support_summary
        ),
        quasi_real_map_domain_gap_required=bool(args.quasi_real_map_domain_gap_summary),
        quasi_real_shadow_policy_behavior_required=bool(
            args.quasi_real_shadow_policy_behavior_summary
        ),
        quasi_real_shadow_alignment_required=bool(args.quasi_real_shadow_alignment_summary),
        quasi_real_guarded_policy_pilot_required=bool(
            args.quasi_real_guarded_policy_pilot_summary
        ),
        quasi_real_safe_alternative_opportunity_required=bool(
            args.quasi_real_safe_alternative_opportunity_summary
        ),
        quasi_real_safe_better_opportunity_expansion_required=bool(
            args.quasi_real_safe_better_opportunity_expansion_summary
        ),
        quasi_real_teacher_equivalent_validation_required=bool(
            args.quasi_real_teacher_equivalent_validation_summary
        ),
        quasi_real_teacher_distillation_required=bool(
            args.quasi_real_teacher_distillation_summary
        ),
        quasi_real_guarded_teacher_following_pilot_required=bool(
            args.quasi_real_guarded_teacher_following_pilot_summary
        ),
        config=config,
        repo_root=repo_root,
    )
    output_file = _output_file(batch_root, config)
    validation_message = {
        "status": "config validated" if summary["status"] == "passed" else "validation failed",
        "batch_root": _display_path(batch_root, repo_root),
        "calibrated_policy_application_smoke_summary": _display_path(smoke_path, repo_root),
        "readiness_summary": _display_path(readiness_path, repo_root),
        "contrast_coverage_summary": _display_path(coverage_path, repo_root),
        "selection_contrast_calibration_summary": _display_path(calibration_path, repo_root),
        "anchor_projection_candidate_generation_summary": (
            _display_path(anchor_candidate_path, repo_root)
            if anchor_candidate_path.is_file() or args.anchor_projection_candidate_generation_summary
            else None
        ),
        "anchor_projection_evidence_contract_summary": (
            _display_path(anchor_contract_path, repo_root)
            if anchor_contract_path.is_file() or args.anchor_projection_evidence_contract_summary
            else None
        ),
        "contract_aware_trainable_target_summary": (
            _display_path(contract_aware_target_path, repo_root)
            if contract_aware_target_path.is_file() or args.contract_aware_trainable_target_summary
            else None
        ),
        "planner_validated_trainable_target_mining_summary": (
            _display_path(planner_validated_mining_path, repo_root)
            if planner_validated_mining_path.is_file()
            or args.planner_validated_trainable_target_mining_summary
            else None
        ),
        "hybrid_policy_training_dry_run_summary": (
            _display_path(hybrid_training_dry_run_path, repo_root)
            if hybrid_training_dry_run_path.is_file()
            or args.hybrid_policy_training_dry_run_summary
            else None
        ),
        "controlled_hybrid_policy_training_candidate_summary": (
            _display_path(controlled_candidate_path, repo_root)
            if controlled_candidate_path.is_file()
            or args.controlled_hybrid_policy_training_candidate_summary
            else None
        ),
        "controlled_hybrid_policy_holdout_evaluation_summary": (
            _display_path(controlled_holdout_path, repo_root)
            if controlled_holdout_path.is_file()
            or args.controlled_hybrid_policy_holdout_evaluation_summary
            else None
        ),
        "fresh_holdout_policy_candidate_evaluation_summary": (
            _display_path(fresh_holdout_path, repo_root)
            if fresh_holdout_path.is_file()
            or args.fresh_holdout_policy_candidate_evaluation_summary
            else None
        ),
        "scenario_disjoint_policy_rollout_evaluation_summary": (
            _display_path(scenario_rollout_path, repo_root)
            if scenario_rollout_path.is_file()
            or args.scenario_disjoint_policy_rollout_evaluation_summary
            else None
        ),
        "raw_policy_strict_rollout_evaluation_summary": (
            _display_path(raw_strict_rollout_path, repo_root)
            if raw_strict_rollout_path.is_file()
            or args.raw_policy_strict_rollout_evaluation_summary
            else None
        ),
        "raw_policy_generalization_evaluation_summary": (
            _display_path(raw_generalization_path, repo_root)
            if raw_generalization_path.is_file()
            or args.raw_policy_generalization_evaluation_summary
            else None
        ),
        "policy_gated_canary_rollout_summary": (
            _display_path(policy_canary_path, repo_root)
            if policy_canary_path.is_file() or args.policy_gated_canary_rollout_summary
            else None
        ),
        "policy_gated_sequential_canary_rollout_summary": (
            _display_path(sequential_canary_path, repo_root)
            if sequential_canary_path.is_file()
            or args.policy_gated_sequential_canary_rollout_summary
            else None
        ),
        "ppo_rollout_collector_summary": (
            _display_path(ppo_collector_path, repo_root)
            if ppo_collector_path.is_file() or args.ppo_rollout_collector_summary
            else None
        ),
        "limited_ppo_update_smoke_summary": (
            _display_path(limited_ppo_update_smoke_path, repo_root)
            if limited_ppo_update_smoke_path.is_file() or args.limited_ppo_update_smoke_summary
            else None
        ),
        "limited_quasi_real_ppo_update_smoke_summary": (
            _display_path(limited_quasi_real_ppo_update_smoke_path, repo_root)
            if limited_quasi_real_ppo_update_smoke_path.is_file()
            or args.limited_quasi_real_ppo_update_smoke_summary
            else None
        ),
        "generated_sequential_gate_metric_accounting_audit_summary": (
            _display_path(generated_sequential_gate_metric_accounting_audit_path, repo_root)
            if generated_sequential_gate_metric_accounting_audit_path.is_file()
            or args.generated_sequential_gate_metric_accounting_audit_summary
            else None
        ),
        "generated_sequential_long_horizon_teacher_skill_contract_summary": (
            _display_path(
                generated_sequential_long_horizon_teacher_skill_contract_path,
                repo_root,
            )
            if generated_sequential_long_horizon_teacher_skill_contract_path.is_file()
            or args.generated_sequential_long_horizon_teacher_skill_contract_summary
            else None
        ),
        "iterative_ppo_mini_loop_stability_summary": (
            _display_path(iterative_ppo_mini_loop_path, repo_root)
            if iterative_ppo_mini_loop_path.is_file() or args.iterative_ppo_mini_loop_stability_summary
            else None
        ),
        "return_aligned_guarded_multistep_collector_summary": (
            _display_path(return_aligned_guarded_multistep_collector_path, repo_root)
            if return_aligned_guarded_multistep_collector_path.is_file()
            or args.return_aligned_guarded_multistep_collector_summary
            else None
        ),
        "return_aligned_guarded_ppo_update_smoke_summary": (
            _display_path(return_aligned_guarded_ppo_update_smoke_path, repo_root)
            if return_aligned_guarded_ppo_update_smoke_path.is_file()
            or args.return_aligned_guarded_ppo_update_smoke_summary
            else None
        ),
        "guarded_ppo_rollout_pilot_summary": (
            _display_path(guarded_ppo_rollout_pilot_path, repo_root)
            if guarded_ppo_rollout_pilot_path.is_file() or args.guarded_ppo_rollout_pilot_summary
            else None
        ),
        "quasi_real_guarded_ppo_rollout_pilot_summary": (
            _display_path(quasi_real_guarded_ppo_rollout_pilot_path, repo_root)
            if quasi_real_guarded_ppo_rollout_pilot_path.is_file()
            or args.quasi_real_guarded_ppo_rollout_pilot_summary
            else None
        ),
        "quasi_real_guarded_ppo_stability_replay_summary": (
            _display_path(quasi_real_guarded_ppo_stability_replay_path, repo_root)
            if quasi_real_guarded_ppo_stability_replay_path.is_file()
            or args.quasi_real_guarded_ppo_stability_replay_summary
            else None
        ),
        "quasi_real_guarded_ppo_horizon5_batch_expansion_summary": (
            _display_path(
                quasi_real_guarded_ppo_horizon5_batch_expansion_path,
                repo_root,
            )
            if quasi_real_guarded_ppo_horizon5_batch_expansion_path.is_file()
            or args.quasi_real_guarded_ppo_horizon5_batch_expansion_summary
            else None
        ),
        "quasi_real_guarded_ppo_scale512_multiseed_preflight_summary": (
            _display_path(
                quasi_real_guarded_ppo_scale512_multiseed_preflight_path,
                repo_root,
            )
            if quasi_real_guarded_ppo_scale512_multiseed_preflight_path.is_file()
            or args.quasi_real_guarded_ppo_scale512_multiseed_preflight_summary
            else None
        ),
        "quasi_real_guarded_ppo_iterative_miniloop_stability_summary": (
            _display_path(
                quasi_real_guarded_ppo_iterative_miniloop_stability_path,
                repo_root,
            )
            if quasi_real_guarded_ppo_iterative_miniloop_stability_path.is_file()
            or args.quasi_real_guarded_ppo_iterative_miniloop_stability_summary
            else None
        ),
        "quasi_real_guarded_formal_ppo_preflight_summary": (
            _display_path(
                quasi_real_guarded_formal_ppo_preflight_path,
                repo_root,
            )
            if quasi_real_guarded_formal_ppo_preflight_path.is_file()
            or args.quasi_real_guarded_formal_ppo_preflight_summary
            else None
        ),
        "quasi_real_guarded_formal_ppo_rollout_canary_summary": (
            _display_path(
                quasi_real_guarded_formal_ppo_rollout_canary_path,
                repo_root,
            )
            if quasi_real_guarded_formal_ppo_rollout_canary_path.is_file()
            or args.quasi_real_guarded_formal_ppo_rollout_canary_summary
            else None
        ),
        "quasi_real_guarded_formal_ppo_stability_holdout_validation_summary": (
            _display_path(
                quasi_real_guarded_formal_ppo_stability_holdout_path,
                repo_root,
            )
            if quasi_real_guarded_formal_ppo_stability_holdout_path.is_file()
            or args.quasi_real_guarded_formal_ppo_stability_holdout_validation_summary
            else None
        ),
        "policy_training_cuda_device_support_summary": (
            _display_path(policy_training_cuda_device_support_path, repo_root)
            if policy_training_cuda_device_support_path.is_file()
            or args.policy_training_cuda_device_support_summary
            else None
        ),
        "quasi_real_map_domain_gap_summary": (
            _display_path(quasi_real_map_domain_gap_path, repo_root)
            if quasi_real_map_domain_gap_path.is_file() or args.quasi_real_map_domain_gap_summary
            else None
        ),
        "quasi_real_shadow_policy_behavior_summary": (
            _display_path(quasi_real_shadow_policy_behavior_path, repo_root)
            if quasi_real_shadow_policy_behavior_path.is_file()
            or args.quasi_real_shadow_policy_behavior_summary
            else None
        ),
        "quasi_real_shadow_alignment_summary": (
            _display_path(quasi_real_shadow_alignment_path, repo_root)
            if quasi_real_shadow_alignment_path.is_file()
            or args.quasi_real_shadow_alignment_summary
            else None
        ),
        "quasi_real_guarded_policy_pilot_summary": (
            _display_path(quasi_real_guarded_policy_pilot_path, repo_root)
            if quasi_real_guarded_policy_pilot_path.is_file()
            or args.quasi_real_guarded_policy_pilot_summary
            else None
        ),
        "quasi_real_safe_alternative_opportunity_summary": (
            _display_path(quasi_real_safe_alternative_opportunity_path, repo_root)
            if quasi_real_safe_alternative_opportunity_path.is_file()
            or args.quasi_real_safe_alternative_opportunity_summary
            else None
        ),
        "quasi_real_safe_better_opportunity_expansion_summary": (
            _display_path(quasi_real_safe_better_opportunity_expansion_path, repo_root)
            if quasi_real_safe_better_opportunity_expansion_path.is_file()
            or args.quasi_real_safe_better_opportunity_expansion_summary
            else None
        ),
        "quasi_real_teacher_equivalent_validation_summary": (
            _display_path(quasi_real_teacher_equivalent_validation_path, repo_root)
            if quasi_real_teacher_equivalent_validation_path.is_file()
            or args.quasi_real_teacher_equivalent_validation_summary
            else None
        ),
        "quasi_real_teacher_distillation_summary": (
            _display_path(quasi_real_teacher_distillation_path, repo_root)
            if quasi_real_teacher_distillation_path.is_file()
            or args.quasi_real_teacher_distillation_summary
            else None
        ),
        "quasi_real_guarded_teacher_following_pilot_summary": (
            _display_path(quasi_real_guarded_teacher_following_pilot_path, repo_root)
            if quasi_real_guarded_teacher_following_pilot_path.is_file()
            or args.quasi_real_guarded_teacher_following_pilot_summary
            else None
        ),
        "config": _display_path(config_path, repo_root),
        "reason_codes": summary["reason_codes"],
        "training_readiness_status": summary["training_readiness_status"],
        "training_blockers": summary["training_blockers"],
        "recommended_next_action": summary["recommended_next_action"],
        "quasi_real_guarded_ppo_horizon5_batch_expansion_readiness": summary.get(
            "quasi_real_guarded_ppo_horizon5_batch_expansion_readiness"
        ),
        "quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness": summary.get(
            "quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness"
        ),
        "quasi_real_guarded_ppo_iterative_miniloop_stability_readiness": summary.get(
            "quasi_real_guarded_ppo_iterative_miniloop_stability_readiness"
        ),
        "quasi_real_guarded_formal_ppo_preflight_readiness": summary.get(
            "quasi_real_guarded_formal_ppo_preflight_readiness"
        ),
        "quasi_real_guarded_formal_ppo_rollout_canary_readiness": summary.get(
            "quasi_real_guarded_formal_ppo_rollout_canary_readiness"
        ),
        "quasi_real_guarded_formal_ppo_stability_holdout_validation_readiness": summary.get(
            "quasi_real_guarded_formal_ppo_stability_holdout_validation_readiness"
        ),
        "policy_training_readiness_review_summary": _display_path(output_file, repo_root),
    }
    print(json.dumps(validation_message, ensure_ascii=False))

    if args.validate_only or args.dry_run:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "would_write": {
                            "policy_training_readiness_review_summary": _display_path(
                                output_file,
                                repo_root,
                            ),
                        },
                        "recommended_next_action": summary["recommended_next_action"],
                    },
                    ensure_ascii=False,
                )
            )
        return 1 if summary["status"] == "failed" else 0

    _write_json(output_file, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "training_readiness_status": summary["training_readiness_status"],
                "policy_training_readiness_review_summary": _display_path(output_file, repo_root),
                "recommended_next_action": summary["recommended_next_action"],
                "failure_reason_code_counts": summary["failure_reason_code_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def analyze_policy_training_readiness_review(
    *,
    batch_root: Path,
    smoke_path: Path,
    readiness_path: Path,
    coverage_path: Path,
    calibration_path: Path,
    anchor_candidate_path: Path,
    anchor_contract_path: Path,
    contract_aware_target_path: Path,
    planner_validated_mining_path: Path,
    hybrid_training_dry_run_path: Path,
    controlled_candidate_path: Path,
    controlled_holdout_path: Path,
    fresh_holdout_path: Path,
    scenario_rollout_path: Path,
    raw_strict_rollout_path: Path,
    raw_generalization_path: Path,
    policy_canary_path: Path,
    sequential_canary_path: Path,
    ppo_collector_path: Path,
    limited_ppo_update_smoke_path: Path,
    iterative_ppo_mini_loop_path: Path,
    guarded_ppo_rollout_pilot_path: Path,
    policy_training_cuda_device_support_path: Path,
    quasi_real_map_domain_gap_path: Path,
    quasi_real_shadow_policy_behavior_path: Path,
    quasi_real_shadow_alignment_path: Path,
    quasi_real_guarded_policy_pilot_path: Path,
    quasi_real_safe_alternative_opportunity_path: Path,
    quasi_real_safe_better_opportunity_expansion_path: Path,
    quasi_real_teacher_equivalent_validation_path: Path | None = None,
    quasi_real_teacher_distillation_path: Path | None = None,
    quasi_real_guarded_teacher_following_pilot_path: Path | None = None,
    limited_quasi_real_ppo_update_smoke_path: Path | None = None,
    generated_sequential_gate_metric_accounting_audit_path: Path | None = None,
    generated_sequential_long_horizon_teacher_skill_contract_path: Path | None = None,
    return_aligned_guarded_multistep_collector_path: Path | None = None,
    return_aligned_guarded_ppo_update_smoke_path: Path | None = None,
    quasi_real_guarded_ppo_rollout_pilot_path: Path | None = None,
    quasi_real_guarded_ppo_stability_replay_path: Path | None = None,
    quasi_real_guarded_ppo_horizon5_batch_expansion_path: Path | None = None,
    quasi_real_guarded_ppo_scale512_multiseed_preflight_path: Path | None = None,
    quasi_real_guarded_ppo_iterative_miniloop_stability_path: Path | None = None,
    quasi_real_guarded_formal_ppo_preflight_path: Path | None = None,
    quasi_real_guarded_formal_ppo_rollout_canary_path: Path | None = None,
    quasi_real_guarded_formal_ppo_stability_holdout_path: Path | None = None,
    anchor_candidate_required: bool = False,
    anchor_contract_required: bool = False,
    contract_aware_target_required: bool = False,
    planner_validated_mining_required: bool = False,
    hybrid_training_dry_run_required: bool = False,
    controlled_candidate_required: bool = False,
    controlled_holdout_required: bool = False,
    fresh_holdout_required: bool = False,
    scenario_rollout_required: bool = False,
    raw_strict_rollout_required: bool = False,
    raw_generalization_required: bool = False,
    policy_canary_required: bool = False,
    sequential_canary_required: bool = False,
    ppo_collector_required: bool = False,
    limited_ppo_update_smoke_required: bool = False,
    limited_quasi_real_ppo_update_smoke_required: bool = False,
    generated_sequential_gate_metric_accounting_audit_required: bool = False,
    generated_sequential_long_horizon_teacher_skill_contract_required: bool = False,
    iterative_ppo_mini_loop_required: bool = False,
    return_aligned_guarded_multistep_collector_required: bool = False,
    return_aligned_guarded_ppo_update_smoke_required: bool = False,
    guarded_ppo_rollout_pilot_required: bool = False,
    quasi_real_guarded_ppo_rollout_pilot_required: bool = False,
    quasi_real_guarded_ppo_stability_replay_required: bool = False,
    quasi_real_guarded_ppo_horizon5_batch_expansion_required: bool = False,
    quasi_real_guarded_ppo_scale512_multiseed_preflight_required: bool = False,
    quasi_real_guarded_ppo_iterative_miniloop_stability_required: bool = False,
    quasi_real_guarded_formal_ppo_preflight_required: bool = False,
    quasi_real_guarded_formal_ppo_rollout_canary_required: bool = False,
    quasi_real_guarded_formal_ppo_stability_holdout_required: bool = False,
    policy_training_cuda_device_support_required: bool = False,
    quasi_real_map_domain_gap_required: bool = False,
    quasi_real_shadow_policy_behavior_required: bool = False,
    quasi_real_shadow_alignment_required: bool = False,
    quasi_real_guarded_policy_pilot_required: bool = False,
    quasi_real_safe_alternative_opportunity_required: bool = False,
    quasi_real_safe_better_opportunity_expansion_required: bool = False,
    quasi_real_teacher_equivalent_validation_required: bool = False,
    quasi_real_teacher_distillation_required: bool = False,
    quasi_real_guarded_teacher_following_pilot_required: bool = False,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    source_summaries: dict[str, Any] = {}
    if quasi_real_teacher_equivalent_validation_path is None:
        quasi_real_teacher_equivalent_validation_path = (
            batch_root / "quasi-real-teacher-equivalent-summary.json"
        )
    if quasi_real_teacher_distillation_path is None:
        quasi_real_teacher_distillation_path = (
            batch_root / "quasi-real-teacher-distillation-summary.json"
        )
    if quasi_real_guarded_teacher_following_pilot_path is None:
        quasi_real_guarded_teacher_following_pilot_path = (
            batch_root / "quasi-real-guarded-teacher-following-pilot-summary.json"
        )
    if limited_quasi_real_ppo_update_smoke_path is None:
        limited_quasi_real_ppo_update_smoke_path = (
            batch_root / "limited-quasi-real-ppo-update-smoke-summary.json"
        )
    if generated_sequential_gate_metric_accounting_audit_path is None:
        generated_sequential_gate_metric_accounting_audit_path = (
            batch_root / "generated-sequential-gate-metric-accounting-audit-summary.json"
        )
    if generated_sequential_long_horizon_teacher_skill_contract_path is None:
        generated_sequential_long_horizon_teacher_skill_contract_path = (
            batch_root / "long-horizon-teacher-skill-contract-summary.json"
        )
    if return_aligned_guarded_multistep_collector_path is None:
        return_aligned_guarded_multistep_collector_path = (
            batch_root / "return-aligned-collector-summary.json"
        )
    if return_aligned_guarded_ppo_update_smoke_path is None:
        return_aligned_guarded_ppo_update_smoke_path = (
            batch_root / "return-aligned-guarded-ppo-update-smoke-summary.json"
        )
    if quasi_real_guarded_ppo_rollout_pilot_path is None:
        quasi_real_guarded_ppo_rollout_pilot_path = (
            batch_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"
        )
    if quasi_real_guarded_ppo_stability_replay_path is None:
        quasi_real_guarded_ppo_stability_replay_path = (
            batch_root / "quasi-real-guarded-ppo-stability-replay-summary.json"
        )
    if quasi_real_guarded_ppo_horizon5_batch_expansion_path is None:
        quasi_real_guarded_ppo_horizon5_batch_expansion_path = (
            batch_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json"
        )
    if quasi_real_guarded_ppo_scale512_multiseed_preflight_path is None:
        quasi_real_guarded_ppo_scale512_multiseed_preflight_path = (
            batch_root / "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json"
        )
    if quasi_real_guarded_ppo_iterative_miniloop_stability_path is None:
        quasi_real_guarded_ppo_iterative_miniloop_stability_path = (
            batch_root / "quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json"
        )
    if quasi_real_guarded_formal_ppo_preflight_path is None:
        quasi_real_guarded_formal_ppo_preflight_path = (
            batch_root / "quasi-real-guarded-formal-ppo-preflight-summary.json"
        )
    if quasi_real_guarded_formal_ppo_rollout_canary_path is None:
        quasi_real_guarded_formal_ppo_rollout_canary_path = (
            batch_root / "quasi-real-guarded-formal-ppo-rollout-canary-summary.json"
        )
    if quasi_real_guarded_formal_ppo_stability_holdout_path is None:
        quasi_real_guarded_formal_ppo_stability_holdout_path = (
            batch_root
            / "quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json"
        )
    anchor_only_mode = (
        anchor_candidate_required
        and anchor_contract_required
        and anchor_candidate_path.is_file()
        and anchor_contract_path.is_file()
        and not any(path.is_file() for path in (smoke_path, readiness_path, coverage_path, calibration_path))
    )
    iterative_only_mode = iterative_ppo_mini_loop_required and not any(
        (
            anchor_candidate_required,
            anchor_contract_required,
            contract_aware_target_required,
            planner_validated_mining_required,
            hybrid_training_dry_run_required,
            controlled_candidate_required,
            controlled_holdout_required,
            fresh_holdout_required,
            scenario_rollout_required,
            raw_strict_rollout_required,
            raw_generalization_required,
            policy_canary_required,
            sequential_canary_required,
            ppo_collector_required,
            limited_ppo_update_smoke_required,
            limited_quasi_real_ppo_update_smoke_required,
            generated_sequential_gate_metric_accounting_audit_required,
            generated_sequential_long_horizon_teacher_skill_contract_required,
            return_aligned_guarded_multistep_collector_required,
            return_aligned_guarded_ppo_update_smoke_required,
            guarded_ppo_rollout_pilot_required,
            quasi_real_guarded_ppo_rollout_pilot_required,
            quasi_real_guarded_ppo_stability_replay_required,
            quasi_real_guarded_ppo_horizon5_batch_expansion_required,
            quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
            policy_training_cuda_device_support_required,
            quasi_real_map_domain_gap_required,
            quasi_real_shadow_policy_behavior_required,
            quasi_real_shadow_alignment_required,
            quasi_real_guarded_policy_pilot_required,
            quasi_real_safe_alternative_opportunity_required,
            quasi_real_safe_better_opportunity_expansion_required,
            quasi_real_teacher_equivalent_validation_required,
            quasi_real_teacher_distillation_required,
            quasi_real_guarded_teacher_following_pilot_required,
        )
    )
    guarded_only_mode = guarded_ppo_rollout_pilot_required and not any(
        (
            anchor_candidate_required,
            anchor_contract_required,
            contract_aware_target_required,
            planner_validated_mining_required,
            hybrid_training_dry_run_required,
            controlled_candidate_required,
            controlled_holdout_required,
            fresh_holdout_required,
            scenario_rollout_required,
            raw_strict_rollout_required,
            raw_generalization_required,
            policy_canary_required,
            sequential_canary_required,
            ppo_collector_required,
            limited_ppo_update_smoke_required,
            limited_quasi_real_ppo_update_smoke_required,
            generated_sequential_gate_metric_accounting_audit_required,
            generated_sequential_long_horizon_teacher_skill_contract_required,
            iterative_ppo_mini_loop_required,
            return_aligned_guarded_multistep_collector_required,
            return_aligned_guarded_ppo_update_smoke_required,
            quasi_real_guarded_ppo_rollout_pilot_required,
            quasi_real_guarded_ppo_stability_replay_required,
            quasi_real_guarded_ppo_horizon5_batch_expansion_required,
            quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
            policy_training_cuda_device_support_required,
            quasi_real_map_domain_gap_required,
            quasi_real_shadow_policy_behavior_required,
            quasi_real_shadow_alignment_required,
            quasi_real_guarded_policy_pilot_required,
            quasi_real_safe_alternative_opportunity_required,
            quasi_real_safe_better_opportunity_expansion_required,
            quasi_real_teacher_equivalent_validation_required,
            quasi_real_teacher_distillation_required,
            quasi_real_guarded_teacher_following_pilot_required,
        )
    )
    return_aligned_only_mode = return_aligned_guarded_multistep_collector_required and not any(
        (
            anchor_candidate_required,
            anchor_contract_required,
            contract_aware_target_required,
            planner_validated_mining_required,
            hybrid_training_dry_run_required,
            controlled_candidate_required,
            controlled_holdout_required,
            fresh_holdout_required,
            scenario_rollout_required,
            raw_strict_rollout_required,
            raw_generalization_required,
            policy_canary_required,
            sequential_canary_required,
            ppo_collector_required,
            limited_ppo_update_smoke_required,
            limited_quasi_real_ppo_update_smoke_required,
            generated_sequential_gate_metric_accounting_audit_required,
            generated_sequential_long_horizon_teacher_skill_contract_required,
            iterative_ppo_mini_loop_required,
            guarded_ppo_rollout_pilot_required,
            quasi_real_guarded_ppo_rollout_pilot_required,
            quasi_real_guarded_ppo_stability_replay_required,
            quasi_real_guarded_ppo_horizon5_batch_expansion_required,
            quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
            policy_training_cuda_device_support_required,
            quasi_real_map_domain_gap_required,
            quasi_real_shadow_policy_behavior_required,
            quasi_real_shadow_alignment_required,
            quasi_real_guarded_policy_pilot_required,
            quasi_real_safe_alternative_opportunity_required,
            quasi_real_safe_better_opportunity_expansion_required,
            quasi_real_teacher_equivalent_validation_required,
            quasi_real_teacher_distillation_required,
            quasi_real_guarded_teacher_following_pilot_required,
        )
    )
    return_aligned_update_only_mode = return_aligned_guarded_ppo_update_smoke_required and not any(
        (
            anchor_candidate_required,
            anchor_contract_required,
            contract_aware_target_required,
            planner_validated_mining_required,
            hybrid_training_dry_run_required,
            controlled_candidate_required,
            controlled_holdout_required,
            fresh_holdout_required,
            scenario_rollout_required,
            raw_strict_rollout_required,
            raw_generalization_required,
            policy_canary_required,
            sequential_canary_required,
            ppo_collector_required,
            limited_ppo_update_smoke_required,
            limited_quasi_real_ppo_update_smoke_required,
            generated_sequential_gate_metric_accounting_audit_required,
            generated_sequential_long_horizon_teacher_skill_contract_required,
            iterative_ppo_mini_loop_required,
            return_aligned_guarded_multistep_collector_required,
            guarded_ppo_rollout_pilot_required,
            quasi_real_guarded_ppo_rollout_pilot_required,
            quasi_real_guarded_ppo_stability_replay_required,
            quasi_real_guarded_ppo_horizon5_batch_expansion_required,
            quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
            policy_training_cuda_device_support_required,
            quasi_real_map_domain_gap_required,
            quasi_real_shadow_policy_behavior_required,
            quasi_real_shadow_alignment_required,
            quasi_real_guarded_policy_pilot_required,
            quasi_real_safe_alternative_opportunity_required,
            quasi_real_safe_better_opportunity_expansion_required,
            quasi_real_teacher_equivalent_validation_required,
            quasi_real_teacher_distillation_required,
            quasi_real_guarded_teacher_following_pilot_required,
        )
    )
    quasi_real_guarded_rollout_only_mode = (
        quasi_real_guarded_ppo_rollout_pilot_required
        and not any(
            (
                anchor_candidate_required,
                anchor_contract_required,
                contract_aware_target_required,
                planner_validated_mining_required,
                hybrid_training_dry_run_required,
                controlled_candidate_required,
                controlled_holdout_required,
                fresh_holdout_required,
                scenario_rollout_required,
                raw_strict_rollout_required,
                raw_generalization_required,
                policy_canary_required,
                sequential_canary_required,
                ppo_collector_required,
                limited_ppo_update_smoke_required,
                limited_quasi_real_ppo_update_smoke_required,
                generated_sequential_gate_metric_accounting_audit_required,
                generated_sequential_long_horizon_teacher_skill_contract_required,
                iterative_ppo_mini_loop_required,
                return_aligned_guarded_multistep_collector_required,
                return_aligned_guarded_ppo_update_smoke_required,
                guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_stability_replay_required,
                quasi_real_guarded_ppo_horizon5_batch_expansion_required,
                quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
                policy_training_cuda_device_support_required,
                quasi_real_map_domain_gap_required,
                quasi_real_shadow_policy_behavior_required,
                quasi_real_shadow_alignment_required,
                quasi_real_guarded_policy_pilot_required,
                quasi_real_safe_alternative_opportunity_required,
                quasi_real_safe_better_opportunity_expansion_required,
                quasi_real_teacher_equivalent_validation_required,
                quasi_real_teacher_distillation_required,
                quasi_real_guarded_teacher_following_pilot_required,
            )
        )
    )
    quasi_real_guarded_stability_only_mode = (
        quasi_real_guarded_ppo_stability_replay_required
        and not any(
            (
                anchor_candidate_required,
                anchor_contract_required,
                contract_aware_target_required,
                planner_validated_mining_required,
                hybrid_training_dry_run_required,
                controlled_candidate_required,
                controlled_holdout_required,
                fresh_holdout_required,
                scenario_rollout_required,
                raw_strict_rollout_required,
                raw_generalization_required,
                policy_canary_required,
                sequential_canary_required,
                ppo_collector_required,
                limited_ppo_update_smoke_required,
                limited_quasi_real_ppo_update_smoke_required,
                generated_sequential_gate_metric_accounting_audit_required,
                generated_sequential_long_horizon_teacher_skill_contract_required,
                iterative_ppo_mini_loop_required,
                return_aligned_guarded_multistep_collector_required,
                return_aligned_guarded_ppo_update_smoke_required,
                guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_horizon5_batch_expansion_required,
                quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
                policy_training_cuda_device_support_required,
                quasi_real_map_domain_gap_required,
                quasi_real_shadow_policy_behavior_required,
                quasi_real_shadow_alignment_required,
                quasi_real_guarded_policy_pilot_required,
                quasi_real_safe_alternative_opportunity_required,
                quasi_real_safe_better_opportunity_expansion_required,
                quasi_real_teacher_equivalent_validation_required,
                quasi_real_teacher_distillation_required,
                quasi_real_guarded_teacher_following_pilot_required,
            )
        )
    )
    quasi_real_guarded_horizon5_only_mode = (
        quasi_real_guarded_ppo_horizon5_batch_expansion_required
        and not any(
            (
                anchor_candidate_required,
                anchor_contract_required,
                contract_aware_target_required,
                planner_validated_mining_required,
                hybrid_training_dry_run_required,
                controlled_candidate_required,
                controlled_holdout_required,
                fresh_holdout_required,
                scenario_rollout_required,
                raw_strict_rollout_required,
                raw_generalization_required,
                policy_canary_required,
                sequential_canary_required,
                ppo_collector_required,
                limited_ppo_update_smoke_required,
                limited_quasi_real_ppo_update_smoke_required,
                generated_sequential_gate_metric_accounting_audit_required,
                generated_sequential_long_horizon_teacher_skill_contract_required,
                iterative_ppo_mini_loop_required,
                return_aligned_guarded_multistep_collector_required,
                return_aligned_guarded_ppo_update_smoke_required,
                guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_stability_replay_required,
                quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
                policy_training_cuda_device_support_required,
                quasi_real_map_domain_gap_required,
                quasi_real_shadow_policy_behavior_required,
                quasi_real_shadow_alignment_required,
                quasi_real_guarded_policy_pilot_required,
                quasi_real_safe_alternative_opportunity_required,
                quasi_real_safe_better_opportunity_expansion_required,
                quasi_real_teacher_equivalent_validation_required,
                quasi_real_teacher_distillation_required,
                quasi_real_guarded_teacher_following_pilot_required,
            )
        )
    )
    quasi_real_guarded_scale512_only_mode = (
        quasi_real_guarded_ppo_scale512_multiseed_preflight_required
        and not any(
            (
                anchor_candidate_required,
                anchor_contract_required,
                contract_aware_target_required,
                planner_validated_mining_required,
                hybrid_training_dry_run_required,
                controlled_candidate_required,
                controlled_holdout_required,
                fresh_holdout_required,
                scenario_rollout_required,
                raw_strict_rollout_required,
                raw_generalization_required,
                policy_canary_required,
                sequential_canary_required,
                ppo_collector_required,
                limited_ppo_update_smoke_required,
                limited_quasi_real_ppo_update_smoke_required,
                generated_sequential_gate_metric_accounting_audit_required,
                generated_sequential_long_horizon_teacher_skill_contract_required,
                iterative_ppo_mini_loop_required,
                return_aligned_guarded_multistep_collector_required,
                return_aligned_guarded_ppo_update_smoke_required,
                guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_stability_replay_required,
                quasi_real_guarded_ppo_horizon5_batch_expansion_required,
                policy_training_cuda_device_support_required,
                quasi_real_map_domain_gap_required,
                quasi_real_shadow_policy_behavior_required,
                quasi_real_shadow_alignment_required,
                quasi_real_guarded_policy_pilot_required,
                quasi_real_safe_alternative_opportunity_required,
                quasi_real_safe_better_opportunity_expansion_required,
                quasi_real_teacher_equivalent_validation_required,
                quasi_real_teacher_distillation_required,
                quasi_real_guarded_teacher_following_pilot_required,
            )
        )
    )
    quasi_real_guarded_iterative_miniloop_only_mode = (
        quasi_real_guarded_ppo_iterative_miniloop_stability_required
        and not any(
            (
                anchor_candidate_required,
                anchor_contract_required,
                contract_aware_target_required,
                planner_validated_mining_required,
                hybrid_training_dry_run_required,
                controlled_candidate_required,
                controlled_holdout_required,
                fresh_holdout_required,
                scenario_rollout_required,
                raw_strict_rollout_required,
                raw_generalization_required,
                policy_canary_required,
                sequential_canary_required,
                ppo_collector_required,
                limited_ppo_update_smoke_required,
                limited_quasi_real_ppo_update_smoke_required,
                generated_sequential_gate_metric_accounting_audit_required,
                generated_sequential_long_horizon_teacher_skill_contract_required,
                iterative_ppo_mini_loop_required,
                return_aligned_guarded_multistep_collector_required,
                return_aligned_guarded_ppo_update_smoke_required,
                guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_stability_replay_required,
                quasi_real_guarded_ppo_horizon5_batch_expansion_required,
                quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
                policy_training_cuda_device_support_required,
                quasi_real_map_domain_gap_required,
                quasi_real_shadow_policy_behavior_required,
                quasi_real_shadow_alignment_required,
                quasi_real_guarded_policy_pilot_required,
                quasi_real_safe_alternative_opportunity_required,
                quasi_real_safe_better_opportunity_expansion_required,
                quasi_real_teacher_equivalent_validation_required,
                quasi_real_teacher_distillation_required,
                quasi_real_guarded_teacher_following_pilot_required,
            )
        )
    )
    quasi_real_guarded_formal_preflight_only_mode = (
        quasi_real_guarded_formal_ppo_preflight_required
        and not any(
            (
                anchor_candidate_required,
                anchor_contract_required,
                contract_aware_target_required,
                planner_validated_mining_required,
                hybrid_training_dry_run_required,
                controlled_candidate_required,
                controlled_holdout_required,
                fresh_holdout_required,
                scenario_rollout_required,
                raw_strict_rollout_required,
                raw_generalization_required,
                policy_canary_required,
                sequential_canary_required,
                ppo_collector_required,
                limited_ppo_update_smoke_required,
                limited_quasi_real_ppo_update_smoke_required,
                generated_sequential_gate_metric_accounting_audit_required,
                generated_sequential_long_horizon_teacher_skill_contract_required,
                iterative_ppo_mini_loop_required,
                return_aligned_guarded_multistep_collector_required,
                return_aligned_guarded_ppo_update_smoke_required,
                guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_stability_replay_required,
                quasi_real_guarded_ppo_horizon5_batch_expansion_required,
                quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
                quasi_real_guarded_ppo_iterative_miniloop_stability_required,
                quasi_real_guarded_formal_ppo_stability_holdout_required,
                policy_training_cuda_device_support_required,
                quasi_real_map_domain_gap_required,
                quasi_real_shadow_policy_behavior_required,
                quasi_real_shadow_alignment_required,
                quasi_real_guarded_policy_pilot_required,
                quasi_real_safe_alternative_opportunity_required,
                quasi_real_safe_better_opportunity_expansion_required,
                quasi_real_teacher_equivalent_validation_required,
                quasi_real_teacher_distillation_required,
                quasi_real_guarded_teacher_following_pilot_required,
            )
        )
    )
    quasi_real_guarded_formal_canary_only_mode = (
        quasi_real_guarded_formal_ppo_rollout_canary_required
        and not any(
            (
                anchor_candidate_required,
                anchor_contract_required,
                contract_aware_target_required,
                planner_validated_mining_required,
                hybrid_training_dry_run_required,
                controlled_candidate_required,
                controlled_holdout_required,
                fresh_holdout_required,
                scenario_rollout_required,
                raw_strict_rollout_required,
                raw_generalization_required,
                policy_canary_required,
                sequential_canary_required,
                ppo_collector_required,
                limited_ppo_update_smoke_required,
                limited_quasi_real_ppo_update_smoke_required,
                generated_sequential_gate_metric_accounting_audit_required,
                generated_sequential_long_horizon_teacher_skill_contract_required,
                iterative_ppo_mini_loop_required,
                return_aligned_guarded_multistep_collector_required,
                return_aligned_guarded_ppo_update_smoke_required,
                guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_stability_replay_required,
                quasi_real_guarded_ppo_horizon5_batch_expansion_required,
                quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
                quasi_real_guarded_ppo_iterative_miniloop_stability_required,
                quasi_real_guarded_formal_ppo_preflight_required,
                quasi_real_guarded_formal_ppo_stability_holdout_required,
                policy_training_cuda_device_support_required,
                quasi_real_map_domain_gap_required,
                quasi_real_shadow_policy_behavior_required,
                quasi_real_shadow_alignment_required,
                quasi_real_guarded_policy_pilot_required,
                quasi_real_safe_alternative_opportunity_required,
                quasi_real_safe_better_opportunity_expansion_required,
                quasi_real_teacher_equivalent_validation_required,
                quasi_real_teacher_distillation_required,
                quasi_real_guarded_teacher_following_pilot_required,
            )
        )
    )
    quasi_real_guarded_formal_stability_holdout_only_mode = (
        quasi_real_guarded_formal_ppo_stability_holdout_required
        and not any(
            (
                anchor_candidate_required,
                anchor_contract_required,
                contract_aware_target_required,
                planner_validated_mining_required,
                hybrid_training_dry_run_required,
                controlled_candidate_required,
                controlled_holdout_required,
                fresh_holdout_required,
                scenario_rollout_required,
                raw_strict_rollout_required,
                raw_generalization_required,
                policy_canary_required,
                sequential_canary_required,
                ppo_collector_required,
                limited_ppo_update_smoke_required,
                limited_quasi_real_ppo_update_smoke_required,
                generated_sequential_gate_metric_accounting_audit_required,
                generated_sequential_long_horizon_teacher_skill_contract_required,
                iterative_ppo_mini_loop_required,
                return_aligned_guarded_multistep_collector_required,
                return_aligned_guarded_ppo_update_smoke_required,
                guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_rollout_pilot_required,
                quasi_real_guarded_ppo_stability_replay_required,
                quasi_real_guarded_ppo_horizon5_batch_expansion_required,
                quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
                quasi_real_guarded_ppo_iterative_miniloop_stability_required,
                quasi_real_guarded_formal_ppo_preflight_required,
                quasi_real_guarded_formal_ppo_rollout_canary_required,
                policy_training_cuda_device_support_required,
                quasi_real_map_domain_gap_required,
                quasi_real_shadow_policy_behavior_required,
                quasi_real_shadow_alignment_required,
                quasi_real_guarded_policy_pilot_required,
                quasi_real_safe_alternative_opportunity_required,
                quasi_real_safe_better_opportunity_expansion_required,
                quasi_real_teacher_equivalent_validation_required,
                quasi_real_teacher_distillation_required,
                quasi_real_guarded_teacher_following_pilot_required,
            )
        )
    )
    stage_isolated_mode = (
        anchor_only_mode
        or iterative_only_mode
        or guarded_only_mode
        or return_aligned_only_mode
        or return_aligned_update_only_mode
        or quasi_real_guarded_rollout_only_mode
        or quasi_real_guarded_stability_only_mode
        or quasi_real_guarded_horizon5_only_mode
        or quasi_real_guarded_scale512_only_mode
        or quasi_real_guarded_iterative_miniloop_only_mode
        or quasi_real_guarded_formal_preflight_only_mode
        or quasi_real_guarded_formal_canary_only_mode
        or quasi_real_guarded_formal_stability_holdout_only_mode
    )
    if (
        guarded_only_mode
        or return_aligned_only_mode
        or return_aligned_update_only_mode
        or quasi_real_guarded_rollout_only_mode
        or quasi_real_guarded_stability_only_mode
        or quasi_real_guarded_horizon5_only_mode
        or quasi_real_guarded_scale512_only_mode
        or quasi_real_guarded_iterative_miniloop_only_mode
        or quasi_real_guarded_formal_preflight_only_mode
        or quasi_real_guarded_formal_canary_only_mode
        or quasi_real_guarded_formal_stability_holdout_only_mode
    ):
        isolated_label = (
            "quasi_real_guarded_formal_ppo_stability_holdout_validation_only"
            if quasi_real_guarded_formal_stability_holdout_only_mode
            else
            "quasi_real_guarded_formal_ppo_rollout_canary_only"
            if quasi_real_guarded_formal_canary_only_mode
            else
            "quasi_real_guarded_formal_ppo_preflight_only"
            if quasi_real_guarded_formal_preflight_only_mode
            else
            "quasi_real_guarded_ppo_iterative_miniloop_stability_only"
            if quasi_real_guarded_iterative_miniloop_only_mode
            else
            "quasi_real_guarded_ppo_scale512_multiseed_preflight_only"
            if quasi_real_guarded_scale512_only_mode
            else
            "quasi_real_guarded_ppo_horizon5_batch_expansion_only"
            if quasi_real_guarded_horizon5_only_mode
            else
            "quasi_real_guarded_ppo_stability_replay_only"
            if quasi_real_guarded_stability_only_mode
            else
            "quasi_real_guarded_ppo_rollout_pilot_only"
            if quasi_real_guarded_rollout_only_mode
            else (
                "return_aligned_guarded_ppo_update_smoke_only"
                if return_aligned_update_only_mode
                else (
                    "return_aligned_guarded_multistep_collector_only"
                    if return_aligned_only_mode
                    else "guarded_ppo_rollout_pilot_only"
                )
            )
        )
        isolated_root = batch_root / ".stage-isolated" / isolated_label

        def _isolated_path(path: Path, label: str) -> Path:
            return isolated_root / label / path.name

        smoke_path = _isolated_path(smoke_path, "calibrated_policy_application_smoke_summary")
        readiness_path = _isolated_path(readiness_path, "channel_aware_training_readiness_summary")
        coverage_path = _isolated_path(coverage_path, "channel_aware_contrast_coverage_summary")
        calibration_path = _isolated_path(calibration_path, "channel_aware_selection_contrast_calibration_summary")
        anchor_candidate_path = _isolated_path(anchor_candidate_path, "anchor_projection_candidate_generation_summary")
        anchor_contract_path = _isolated_path(anchor_contract_path, "anchor_projection_evidence_contract_summary")
        contract_aware_target_path = _isolated_path(contract_aware_target_path, "contract_aware_trainable_target_summary")
        planner_validated_mining_path = _isolated_path(
            planner_validated_mining_path,
            "planner_validated_trainable_target_mining_summary",
        )
        hybrid_training_dry_run_path = _isolated_path(
            hybrid_training_dry_run_path,
            "hybrid_policy_training_dry_run_summary",
        )
        controlled_candidate_path = _isolated_path(
            controlled_candidate_path,
            "controlled_hybrid_policy_training_candidate_summary",
        )
        controlled_holdout_path = _isolated_path(
            controlled_holdout_path,
            "controlled_hybrid_policy_holdout_evaluation_summary",
        )
        fresh_holdout_path = _isolated_path(fresh_holdout_path, "fresh_holdout_policy_candidate_evaluation_summary")
        scenario_rollout_path = _isolated_path(
            scenario_rollout_path,
            "scenario_disjoint_policy_rollout_evaluation_summary",
        )
        raw_strict_rollout_path = _isolated_path(
            raw_strict_rollout_path,
            "raw_policy_strict_rollout_evaluation_summary",
        )
        raw_generalization_path = _isolated_path(
            raw_generalization_path,
            "raw_policy_generalization_evaluation_summary",
        )
        policy_canary_path = _isolated_path(policy_canary_path, "policy_gated_canary_rollout_summary")
        sequential_canary_path = _isolated_path(
            sequential_canary_path,
            "policy_gated_sequential_canary_rollout_summary",
        )
        ppo_collector_path = _isolated_path(ppo_collector_path, "ppo_rollout_collector_summary")
        limited_ppo_update_smoke_path = _isolated_path(
            limited_ppo_update_smoke_path,
            "limited_ppo_update_smoke_summary",
        )
        limited_quasi_real_ppo_update_smoke_path = _isolated_path(
            limited_quasi_real_ppo_update_smoke_path,
            "limited_quasi_real_ppo_update_smoke_summary",
        )
        generated_sequential_gate_metric_accounting_audit_path = _isolated_path(
            generated_sequential_gate_metric_accounting_audit_path,
            "generated_sequential_gate_metric_accounting_audit_summary",
        )
        generated_sequential_long_horizon_teacher_skill_contract_path = _isolated_path(
            generated_sequential_long_horizon_teacher_skill_contract_path,
            "generated_sequential_long_horizon_teacher_skill_contract_summary",
        )
        iterative_ppo_mini_loop_path = _isolated_path(
            iterative_ppo_mini_loop_path,
            "iterative_ppo_mini_loop_stability_summary",
        )
        if not return_aligned_only_mode:
            return_aligned_guarded_multistep_collector_path = _isolated_path(
                return_aligned_guarded_multistep_collector_path,
                "return_aligned_guarded_multistep_collector_summary",
            )
        if not return_aligned_update_only_mode:
            return_aligned_guarded_ppo_update_smoke_path = _isolated_path(
                return_aligned_guarded_ppo_update_smoke_path,
                "return_aligned_guarded_ppo_update_smoke_summary",
            )
        if not guarded_only_mode:
            guarded_ppo_rollout_pilot_path = _isolated_path(
                guarded_ppo_rollout_pilot_path,
                "guarded_ppo_rollout_pilot_summary",
            )
        if not quasi_real_guarded_rollout_only_mode:
            quasi_real_guarded_ppo_rollout_pilot_path = _isolated_path(
                quasi_real_guarded_ppo_rollout_pilot_path,
                "quasi_real_guarded_ppo_rollout_pilot_summary",
            )
        if not quasi_real_guarded_stability_only_mode:
            quasi_real_guarded_ppo_stability_replay_path = _isolated_path(
                quasi_real_guarded_ppo_stability_replay_path,
                "quasi_real_guarded_ppo_stability_replay_summary",
            )
        if not quasi_real_guarded_horizon5_only_mode:
            quasi_real_guarded_ppo_horizon5_batch_expansion_path = _isolated_path(
                quasi_real_guarded_ppo_horizon5_batch_expansion_path,
                "quasi_real_guarded_ppo_horizon5_batch_expansion_summary",
            )
        if not quasi_real_guarded_scale512_only_mode:
            quasi_real_guarded_ppo_scale512_multiseed_preflight_path = _isolated_path(
                quasi_real_guarded_ppo_scale512_multiseed_preflight_path,
                "quasi_real_guarded_ppo_scale512_multiseed_preflight_summary",
            )
        if not quasi_real_guarded_iterative_miniloop_only_mode:
            quasi_real_guarded_ppo_iterative_miniloop_stability_path = _isolated_path(
                quasi_real_guarded_ppo_iterative_miniloop_stability_path,
                "quasi_real_guarded_ppo_iterative_miniloop_stability_summary",
            )
        if not quasi_real_guarded_formal_preflight_only_mode:
            quasi_real_guarded_formal_ppo_preflight_path = _isolated_path(
                quasi_real_guarded_formal_ppo_preflight_path,
                "quasi_real_guarded_formal_ppo_preflight_summary",
            )
        if not quasi_real_guarded_formal_canary_only_mode:
            quasi_real_guarded_formal_ppo_rollout_canary_path = _isolated_path(
                quasi_real_guarded_formal_ppo_rollout_canary_path,
                "quasi_real_guarded_formal_ppo_rollout_canary_summary",
            )
        if not quasi_real_guarded_formal_stability_holdout_only_mode:
            quasi_real_guarded_formal_ppo_stability_holdout_path = _isolated_path(
                quasi_real_guarded_formal_ppo_stability_holdout_path,
                "quasi_real_guarded_formal_ppo_stability_holdout_validation_summary",
            )
        policy_training_cuda_device_support_path = _isolated_path(
            policy_training_cuda_device_support_path,
            "policy_training_cuda_device_support_summary",
        )
        quasi_real_map_domain_gap_path = _isolated_path(
            quasi_real_map_domain_gap_path,
            "quasi_real_map_domain_gap_summary",
        )
        quasi_real_shadow_policy_behavior_path = _isolated_path(
            quasi_real_shadow_policy_behavior_path,
            "quasi_real_shadow_policy_behavior_summary",
        )
        quasi_real_shadow_alignment_path = _isolated_path(
            quasi_real_shadow_alignment_path,
            "quasi_real_shadow_alignment_summary",
        )
        quasi_real_guarded_policy_pilot_path = _isolated_path(
            quasi_real_guarded_policy_pilot_path,
            "quasi_real_guarded_policy_pilot_summary",
        )
        quasi_real_safe_alternative_opportunity_path = _isolated_path(
            quasi_real_safe_alternative_opportunity_path,
            "quasi_real_safe_alternative_opportunity_summary",
        )
        quasi_real_safe_better_opportunity_expansion_path = _isolated_path(
            quasi_real_safe_better_opportunity_expansion_path,
            "quasi_real_safe_better_opportunity_expansion_summary",
        )
        quasi_real_teacher_equivalent_validation_path = _isolated_path(
            quasi_real_teacher_equivalent_validation_path,
            "quasi_real_teacher_equivalent_validation_summary",
        )
        quasi_real_teacher_distillation_path = _isolated_path(
            quasi_real_teacher_distillation_path,
            "quasi_real_teacher_distillation_summary",
        )
        quasi_real_guarded_teacher_following_pilot_path = _isolated_path(
            quasi_real_guarded_teacher_following_pilot_path,
            "quasi_real_guarded_teacher_following_pilot_summary",
        )
    if stage_isolated_mode:
        smoke = _load_optional_source(
            smoke_path,
            label="calibrated_policy_application_smoke_summary",
            expected_schema=SMOKE_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        readiness = _load_optional_source(
            readiness_path,
            label="channel_aware_training_readiness_summary",
            expected_schema=READINESS_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        coverage = _load_optional_source(
            coverage_path,
            label="channel_aware_contrast_coverage_summary",
            expected_schema=COVERAGE_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        calibration = _load_optional_source(
            calibration_path,
            label="channel_aware_selection_contrast_calibration_summary",
            expected_schema=CALIBRATION_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
    else:
        smoke = _load_source(
            smoke_path,
            label="calibrated_policy_application_smoke_summary",
            expected_schema=SMOKE_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        readiness = _load_source(
            readiness_path,
            label="channel_aware_training_readiness_summary",
            expected_schema=READINESS_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        coverage = _load_source(
            coverage_path,
            label="channel_aware_contrast_coverage_summary",
            expected_schema=COVERAGE_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
        calibration = _load_source(
            calibration_path,
            label="channel_aware_selection_contrast_calibration_summary",
            expected_schema=CALIBRATION_SCHEMA_VERSION,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
    anchor_candidate = _load_optional_source(
        anchor_candidate_path,
        label="anchor_projection_candidate_generation_summary",
        expected_schema=ANCHOR_CANDIDATE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=anchor_candidate_required,
    )
    anchor_contract = _load_optional_source(
        anchor_contract_path,
        label="anchor_projection_evidence_contract_summary",
        expected_schema=ANCHOR_CONTRACT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=anchor_contract_required,
    )
    contract_aware_target = _load_optional_source(
        contract_aware_target_path,
        label="contract_aware_trainable_target_summary",
        expected_schema=CONTRACT_AWARE_TARGET_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=contract_aware_target_required,
    )
    planner_validated_mining = _load_optional_source(
        planner_validated_mining_path,
        label="planner_validated_trainable_target_mining_summary",
        expected_schema=PLANNER_VALIDATED_MINING_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=planner_validated_mining_required,
    )
    hybrid_training_dry_run = _load_optional_source(
        hybrid_training_dry_run_path,
        label="hybrid_policy_training_dry_run_summary",
        expected_schema=HYBRID_TRAINING_DRY_RUN_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=hybrid_training_dry_run_required,
    )
    controlled_candidate = _load_optional_source(
        controlled_candidate_path,
        label="controlled_hybrid_policy_training_candidate_summary",
        expected_schema=CONTROLLED_HYBRID_CANDIDATE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=controlled_candidate_required,
    )
    controlled_holdout = _load_optional_source(
        controlled_holdout_path,
        label="controlled_hybrid_policy_holdout_evaluation_summary",
        expected_schema=CONTROLLED_HYBRID_HOLDOUT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=controlled_holdout_required,
    )
    fresh_holdout = _load_optional_source(
        fresh_holdout_path,
        label="fresh_holdout_policy_candidate_evaluation_summary",
        expected_schema=FRESH_HOLDOUT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=fresh_holdout_required,
    )
    scenario_rollout = _load_optional_source(
        scenario_rollout_path,
        label="scenario_disjoint_policy_rollout_evaluation_summary",
        expected_schema=SCENARIO_DISJOINT_ROLLOUT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=scenario_rollout_required,
    )
    raw_strict_rollout = _load_optional_source(
        raw_strict_rollout_path,
        label="raw_policy_strict_rollout_evaluation_summary",
        expected_schema=RAW_POLICY_STRICT_ROLLOUT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=raw_strict_rollout_required,
    )
    raw_generalization = _load_optional_source(
        raw_generalization_path,
        label="raw_policy_generalization_evaluation_summary",
        expected_schema=RAW_POLICY_GENERALIZATION_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=raw_generalization_required,
    )
    policy_canary = _load_optional_source(
        policy_canary_path,
        label="policy_gated_canary_rollout_summary",
        expected_schema=POLICY_GATED_CANARY_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=policy_canary_required,
    )
    sequential_canary = _load_optional_source(
        sequential_canary_path,
        label="policy_gated_sequential_canary_rollout_summary",
        expected_schema=POLICY_GATED_SEQUENTIAL_CANARY_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=sequential_canary_required,
    )
    ppo_collector = _load_optional_source(
        ppo_collector_path,
        label="ppo_rollout_collector_summary",
        expected_schema=PPO_ROLLOUT_COLLECTOR_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=ppo_collector_required,
    )
    limited_ppo_update_smoke = _load_optional_source(
        limited_ppo_update_smoke_path,
        label="limited_ppo_update_smoke_summary",
        expected_schema=LIMITED_PPO_UPDATE_SMOKE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=limited_ppo_update_smoke_required,
    )
    limited_quasi_real_ppo_update_smoke = _load_optional_source(
        limited_quasi_real_ppo_update_smoke_path,
        label="limited_quasi_real_ppo_update_smoke_summary",
        expected_schema=LIMITED_QUASI_REAL_PPO_UPDATE_SMOKE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=limited_quasi_real_ppo_update_smoke_required,
    )
    generated_sequential_gate_metric_accounting_audit = _load_optional_source(
        generated_sequential_gate_metric_accounting_audit_path,
        label="generated_sequential_gate_metric_accounting_audit_summary",
        expected_schema=GENERATED_SEQUENTIAL_GATE_METRIC_ACCOUNTING_AUDIT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=generated_sequential_gate_metric_accounting_audit_required,
    )
    generated_sequential_long_horizon_teacher_skill_contract = _load_optional_source(
        generated_sequential_long_horizon_teacher_skill_contract_path,
        label="generated_sequential_long_horizon_teacher_skill_contract_summary",
        expected_schema=GENERATED_SEQUENTIAL_LONG_HORIZON_TEACHER_SKILL_CONTRACT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=generated_sequential_long_horizon_teacher_skill_contract_required,
    )
    iterative_ppo_mini_loop = _load_optional_source(
        iterative_ppo_mini_loop_path,
        label="iterative_ppo_mini_loop_stability_summary",
        expected_schema=ITERATIVE_PPO_MINI_LOOP_STABILITY_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=iterative_ppo_mini_loop_required,
    )
    return_aligned_guarded_multistep_collector = _load_optional_source(
        return_aligned_guarded_multistep_collector_path,
        label="return_aligned_guarded_multistep_collector_summary",
        expected_schema=RETURN_ALIGNED_GUARDED_MULTISTEP_COLLECTOR_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=return_aligned_guarded_multistep_collector_required,
    )
    return_aligned_guarded_ppo_update_smoke = _load_optional_source(
        return_aligned_guarded_ppo_update_smoke_path,
        label="return_aligned_guarded_ppo_update_smoke_summary",
        expected_schema=RETURN_ALIGNED_GUARDED_PPO_UPDATE_SMOKE_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=return_aligned_guarded_ppo_update_smoke_required,
    )
    guarded_ppo_rollout_pilot = _load_optional_source(
        guarded_ppo_rollout_pilot_path,
        label="guarded_ppo_rollout_pilot_summary",
        expected_schema=GUARDED_PPO_ROLLOUT_PILOT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=guarded_ppo_rollout_pilot_required,
    )
    quasi_real_guarded_ppo_rollout_pilot = _load_optional_source(
        quasi_real_guarded_ppo_rollout_pilot_path,
        label="quasi_real_guarded_ppo_rollout_pilot_summary",
        expected_schema=QUASI_REAL_GUARDED_PPO_ROLLOUT_PILOT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_ppo_rollout_pilot_required,
    )
    quasi_real_guarded_ppo_stability_replay = _load_optional_source(
        quasi_real_guarded_ppo_stability_replay_path,
        label="quasi_real_guarded_ppo_stability_replay_summary",
        expected_schema=QUASI_REAL_GUARDED_PPO_STABILITY_REPLAY_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_ppo_stability_replay_required,
    )
    quasi_real_guarded_ppo_horizon5_batch_expansion = _load_optional_source(
        quasi_real_guarded_ppo_horizon5_batch_expansion_path,
        label="quasi_real_guarded_ppo_horizon5_batch_expansion_summary",
        expected_schema=QUASI_REAL_GUARDED_PPO_HORIZON5_BATCH_EXPANSION_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_ppo_horizon5_batch_expansion_required,
    )
    quasi_real_guarded_ppo_scale512_multiseed_preflight = _load_optional_source(
        quasi_real_guarded_ppo_scale512_multiseed_preflight_path,
        label="quasi_real_guarded_ppo_scale512_multiseed_preflight_summary",
        expected_schema=QUASI_REAL_GUARDED_PPO_SCALE512_MULTISEED_PREFLIGHT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_ppo_scale512_multiseed_preflight_required,
    )
    quasi_real_guarded_ppo_iterative_miniloop_stability = _load_optional_source(
        quasi_real_guarded_ppo_iterative_miniloop_stability_path,
        label="quasi_real_guarded_ppo_iterative_miniloop_stability_summary",
        expected_schema=QUASI_REAL_GUARDED_PPO_ITERATIVE_MINILOOP_STABILITY_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_ppo_iterative_miniloop_stability_required,
    )
    quasi_real_guarded_formal_ppo_preflight = _load_optional_source(
        quasi_real_guarded_formal_ppo_preflight_path,
        label="quasi_real_guarded_formal_ppo_preflight_summary",
        expected_schema=QUASI_REAL_GUARDED_FORMAL_PPO_PREFLIGHT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_formal_ppo_preflight_required,
    )
    quasi_real_guarded_formal_ppo_rollout_canary = _load_optional_source(
        quasi_real_guarded_formal_ppo_rollout_canary_path,
        label="quasi_real_guarded_formal_ppo_rollout_canary_summary",
        expected_schema=QUASI_REAL_GUARDED_FORMAL_PPO_ROLLOUT_CANARY_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_formal_ppo_rollout_canary_required,
    )
    quasi_real_guarded_formal_ppo_stability_holdout = _load_optional_source(
        quasi_real_guarded_formal_ppo_stability_holdout_path,
        label="quasi_real_guarded_formal_ppo_stability_holdout_validation_summary",
        expected_schema=QUASI_REAL_GUARDED_FORMAL_PPO_STABILITY_HOLDOUT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_formal_ppo_stability_holdout_required,
    )
    policy_training_cuda_device_support = _load_optional_source(
        policy_training_cuda_device_support_path,
        label="policy_training_cuda_device_support_summary",
        expected_schema=POLICY_TRAINING_CUDA_DEVICE_SUPPORT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=policy_training_cuda_device_support_required,
    )
    quasi_real_map_domain_gap = _load_optional_source(
        quasi_real_map_domain_gap_path,
        label="quasi_real_map_domain_gap_summary",
        expected_schema=QUASI_REAL_MAP_DOMAIN_GAP_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_map_domain_gap_required,
    )
    quasi_real_shadow_policy_behavior = _load_optional_source(
        quasi_real_shadow_policy_behavior_path,
        label="quasi_real_shadow_policy_behavior_summary",
        expected_schema=QUASI_REAL_SHADOW_POLICY_BEHAVIOR_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_shadow_policy_behavior_required,
    )
    quasi_real_shadow_alignment = _load_optional_source(
        quasi_real_shadow_alignment_path,
        label="quasi_real_shadow_alignment_summary",
        expected_schema=QUASI_REAL_SHADOW_ALIGNMENT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_shadow_alignment_required,
    )
    quasi_real_guarded_policy_pilot = _load_optional_source(
        quasi_real_guarded_policy_pilot_path,
        label="quasi_real_guarded_policy_pilot_summary",
        expected_schema=QUASI_REAL_GUARDED_POLICY_PILOT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_policy_pilot_required,
    )
    quasi_real_safe_alternative_opportunity = _load_optional_source(
        quasi_real_safe_alternative_opportunity_path,
        label="quasi_real_safe_alternative_opportunity_summary",
        expected_schema=QUASI_REAL_SAFE_ALTERNATIVE_OPPORTUNITY_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_safe_alternative_opportunity_required,
    )
    quasi_real_safe_better_opportunity_expansion = _load_optional_source(
        quasi_real_safe_better_opportunity_expansion_path,
        label="quasi_real_safe_better_opportunity_expansion_summary",
        expected_schema=QUASI_REAL_SAFE_BETTER_OPPORTUNITY_EXPANSION_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_safe_better_opportunity_expansion_required,
    )
    quasi_real_teacher_equivalent_validation = _load_optional_source(
        quasi_real_teacher_equivalent_validation_path,
        label="quasi_real_teacher_equivalent_validation_summary",
        expected_schema=QUASI_REAL_TEACHER_EQUIVALENT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_teacher_equivalent_validation_required,
    )
    quasi_real_teacher_distillation = _load_optional_source(
        quasi_real_teacher_distillation_path,
        label="quasi_real_teacher_distillation_summary",
        expected_schema=QUASI_REAL_TEACHER_DISTILLATION_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_teacher_distillation_required,
    )
    quasi_real_guarded_teacher_following_pilot = _load_optional_source(
        quasi_real_guarded_teacher_following_pilot_path,
        label="quasi_real_guarded_teacher_following_pilot_summary",
        expected_schema=QUASI_REAL_GUARDED_TEACHER_FOLLOWING_PILOT_SCHEMA_VERSION,
        repo_root=repo_root,
        reason_codes=reason_codes,
        source_summaries=source_summaries,
        required=quasi_real_guarded_teacher_following_pilot_required,
    )
    if _fail_on_input_failure(config):
        for label, payload in (
            ("calibrated_policy_application_smoke_summary", smoke),
            ("channel_aware_training_readiness_summary", readiness),
            ("channel_aware_contrast_coverage_summary", coverage),
            ("channel_aware_selection_contrast_calibration_summary", calibration),
            ("anchor_projection_candidate_generation_summary", anchor_candidate),
            ("anchor_projection_evidence_contract_summary", anchor_contract),
            ("contract_aware_trainable_target_summary", contract_aware_target),
            ("planner_validated_trainable_target_mining_summary", planner_validated_mining),
            ("hybrid_policy_training_dry_run_summary", hybrid_training_dry_run),
            ("controlled_hybrid_policy_training_candidate_summary", controlled_candidate),
            ("controlled_hybrid_policy_holdout_evaluation_summary", controlled_holdout),
            ("fresh_holdout_policy_candidate_evaluation_summary", fresh_holdout),
            ("scenario_disjoint_policy_rollout_evaluation_summary", scenario_rollout),
            ("raw_policy_strict_rollout_evaluation_summary", raw_strict_rollout),
            ("raw_policy_generalization_evaluation_summary", raw_generalization),
            ("policy_gated_canary_rollout_summary", policy_canary),
            ("policy_gated_sequential_canary_rollout_summary", sequential_canary),
            ("ppo_rollout_collector_summary", ppo_collector),
            ("limited_ppo_update_smoke_summary", limited_ppo_update_smoke),
            (
                "generated_sequential_gate_metric_accounting_audit_summary",
                generated_sequential_gate_metric_accounting_audit,
            ),
            (
                "generated_sequential_long_horizon_teacher_skill_contract_summary",
                generated_sequential_long_horizon_teacher_skill_contract,
            ),
            ("iterative_ppo_mini_loop_stability_summary", iterative_ppo_mini_loop),
            ("guarded_ppo_rollout_pilot_summary", guarded_ppo_rollout_pilot),
            (
                "quasi_real_guarded_ppo_rollout_pilot_summary",
                quasi_real_guarded_ppo_rollout_pilot,
            ),
            (
                "quasi_real_guarded_ppo_stability_replay_summary",
                quasi_real_guarded_ppo_stability_replay,
            ),
            (
                "quasi_real_guarded_ppo_horizon5_batch_expansion_summary",
                quasi_real_guarded_ppo_horizon5_batch_expansion,
            ),
            (
                "quasi_real_guarded_ppo_scale512_multiseed_preflight_summary",
                quasi_real_guarded_ppo_scale512_multiseed_preflight,
            ),
            (
                "quasi_real_guarded_ppo_iterative_miniloop_stability_summary",
                quasi_real_guarded_ppo_iterative_miniloop_stability,
            ),
            ("policy_training_cuda_device_support_summary", policy_training_cuda_device_support),
            ("quasi_real_map_domain_gap_summary", quasi_real_map_domain_gap),
            (
                "quasi_real_shadow_policy_behavior_summary",
                quasi_real_shadow_policy_behavior
                if not quasi_real_shadow_alignment
                else {},
            ),
            ("quasi_real_shadow_alignment_summary", quasi_real_shadow_alignment),
            (
                "quasi_real_guarded_policy_pilot_summary",
                quasi_real_guarded_policy_pilot
                if not quasi_real_safe_alternative_opportunity
                else {},
            ),
            (
                "quasi_real_safe_alternative_opportunity_summary",
                quasi_real_safe_alternative_opportunity,
            ),
            (
                "quasi_real_safe_better_opportunity_expansion_summary",
                quasi_real_safe_better_opportunity_expansion
                if not (
                    quasi_real_teacher_equivalent_validation
                    or quasi_real_teacher_distillation
                    or quasi_real_guarded_teacher_following_pilot
                )
                else {},
            ),
            (
                "quasi_real_teacher_equivalent_validation_summary",
                quasi_real_teacher_equivalent_validation
                if not (
                    quasi_real_teacher_distillation
                    or quasi_real_guarded_teacher_following_pilot
                )
                else {},
            ),
            (
                "quasi_real_teacher_distillation_summary",
                quasi_real_teacher_distillation,
            ),
            (
                "quasi_real_guarded_teacher_following_pilot_summary",
                quasi_real_guarded_teacher_following_pilot,
            ),
        ):
            if payload.get("status") == "failed":
                _append_reason(reason_codes, f"{label}_failed")

    current_git = _git_snapshot(repo_root)
    source_git_matches = []
    if not stage_isolated_mode:
        source_git_matches.extend(
            [
                _inspect_git(smoke, label="calibrated_policy_application_smoke_summary", current_git=current_git, config=config, reason_codes=reason_codes),
                _inspect_git(readiness, label="channel_aware_training_readiness_summary", current_git=current_git, config=config, reason_codes=reason_codes),
                _inspect_git(coverage, label="channel_aware_contrast_coverage_summary", current_git=current_git, config=config, reason_codes=reason_codes),
                _inspect_git(calibration, label="channel_aware_selection_contrast_calibration_summary", current_git=current_git, config=config, reason_codes=reason_codes),
            ]
        )
    if anchor_candidate:
        source_git_matches.append(
            _inspect_git(
                anchor_candidate,
                label="anchor_projection_candidate_generation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if anchor_contract:
        source_git_matches.append(
            _inspect_git(
                anchor_contract,
                label="anchor_projection_evidence_contract_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if contract_aware_target:
        source_git_matches.append(
            _inspect_git(
                contract_aware_target,
                label="contract_aware_trainable_target_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if planner_validated_mining:
        source_git_matches.append(
            _inspect_git(
                planner_validated_mining,
                label="planner_validated_trainable_target_mining_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if hybrid_training_dry_run:
        source_git_matches.append(
            _inspect_git(
                hybrid_training_dry_run,
                label="hybrid_policy_training_dry_run_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if controlled_candidate:
        source_git_matches.append(
            _inspect_git(
                controlled_candidate,
                label="controlled_hybrid_policy_training_candidate_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if controlled_holdout:
        source_git_matches.append(
            _inspect_git(
                controlled_holdout,
                label="controlled_hybrid_policy_holdout_evaluation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if fresh_holdout:
        source_git_matches.append(
            _inspect_git(
                fresh_holdout,
                label="fresh_holdout_policy_candidate_evaluation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if scenario_rollout:
        source_git_matches.append(
            _inspect_git(
                scenario_rollout,
                label="scenario_disjoint_policy_rollout_evaluation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if raw_strict_rollout:
        source_git_matches.append(
            _inspect_git(
                raw_strict_rollout,
                label="raw_policy_strict_rollout_evaluation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if raw_generalization:
        source_git_matches.append(
            _inspect_git(
                raw_generalization,
                label="raw_policy_generalization_evaluation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if policy_canary:
        source_git_matches.append(
            _inspect_git(
                policy_canary,
                label="policy_gated_canary_rollout_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if sequential_canary:
        source_git_matches.append(
            _inspect_git(
                sequential_canary,
                label="policy_gated_sequential_canary_rollout_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if ppo_collector:
        source_git_matches.append(
            _inspect_git(
                ppo_collector,
                label="ppo_rollout_collector_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if limited_ppo_update_smoke:
        source_git_matches.append(
            _inspect_git(
                limited_ppo_update_smoke,
                label="limited_ppo_update_smoke_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if iterative_ppo_mini_loop:
        source_git_matches.append(
            _inspect_git(
                iterative_ppo_mini_loop,
                label="iterative_ppo_mini_loop_stability_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if return_aligned_guarded_multistep_collector:
        source_git_matches.append(
            _inspect_git(
                return_aligned_guarded_multistep_collector,
                label="return_aligned_guarded_multistep_collector_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if return_aligned_guarded_ppo_update_smoke:
        source_git_matches.append(
            _inspect_git(
                return_aligned_guarded_ppo_update_smoke,
                label="return_aligned_guarded_ppo_update_smoke_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if guarded_ppo_rollout_pilot:
        source_git_matches.append(
            _inspect_git(
                guarded_ppo_rollout_pilot,
                label="guarded_ppo_rollout_pilot_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_ppo_rollout_pilot:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_ppo_rollout_pilot,
                label="quasi_real_guarded_ppo_rollout_pilot_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_ppo_stability_replay:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_ppo_stability_replay,
                label="quasi_real_guarded_ppo_stability_replay_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_ppo_horizon5_batch_expansion:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_ppo_horizon5_batch_expansion,
                label="quasi_real_guarded_ppo_horizon5_batch_expansion_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_ppo_scale512_multiseed_preflight:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_ppo_scale512_multiseed_preflight,
                label="quasi_real_guarded_ppo_scale512_multiseed_preflight_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_ppo_iterative_miniloop_stability:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_ppo_iterative_miniloop_stability,
                label="quasi_real_guarded_ppo_iterative_miniloop_stability_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_formal_ppo_preflight:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_formal_ppo_preflight,
                label="quasi_real_guarded_formal_ppo_preflight_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_formal_ppo_rollout_canary:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_formal_ppo_rollout_canary,
                label="quasi_real_guarded_formal_ppo_rollout_canary_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_formal_ppo_stability_holdout:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_formal_ppo_stability_holdout,
                label="quasi_real_guarded_formal_ppo_stability_holdout_validation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if policy_training_cuda_device_support:
        source_git_matches.append(
            _inspect_git(
                policy_training_cuda_device_support,
                label="policy_training_cuda_device_support_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_map_domain_gap:
        source_git_matches.append(
            _inspect_git(
                quasi_real_map_domain_gap,
                label="quasi_real_map_domain_gap_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_shadow_policy_behavior:
        source_git_matches.append(
            _inspect_git(
                quasi_real_shadow_policy_behavior,
                label="quasi_real_shadow_policy_behavior_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_shadow_alignment:
        source_git_matches.append(
            _inspect_git(
                quasi_real_shadow_alignment,
                label="quasi_real_shadow_alignment_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_policy_pilot:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_policy_pilot,
                label="quasi_real_guarded_policy_pilot_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_safe_alternative_opportunity:
        source_git_matches.append(
            _inspect_git(
                quasi_real_safe_alternative_opportunity,
                label="quasi_real_safe_alternative_opportunity_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_safe_better_opportunity_expansion:
        source_git_matches.append(
            _inspect_git(
                quasi_real_safe_better_opportunity_expansion,
                label="quasi_real_safe_better_opportunity_expansion_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_teacher_equivalent_validation:
        source_git_matches.append(
            _inspect_git(
                quasi_real_teacher_equivalent_validation,
                label="quasi_real_teacher_equivalent_validation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_teacher_distillation:
        source_git_matches.append(
            _inspect_git(
                quasi_real_teacher_distillation,
                label="quasi_real_teacher_distillation_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    if quasi_real_guarded_teacher_following_pilot:
        source_git_matches.append(
            _inspect_git(
                quasi_real_guarded_teacher_following_pilot,
                label="quasi_real_guarded_teacher_following_pilot_summary",
                current_git=current_git,
                config=config,
                reason_codes=reason_codes,
            )
        )
    reason_codes = _filter_stale_source_git_for_quasi_real_collector(
        reason_codes,
        ppo_collector=ppo_collector,
        quasi_real_guarded_teacher_following_pilot=quasi_real_guarded_teacher_following_pilot,
    )
    reason_codes = _filter_stale_source_git_for_iterative_only(
        reason_codes,
        iterative_ppo_mini_loop=iterative_ppo_mini_loop,
        iterative_only_mode=iterative_only_mode,
    )

    review = _review_metrics(
        smoke=smoke,
        readiness=readiness,
        coverage=coverage,
        calibration=calibration,
        anchor_candidate=anchor_candidate,
        anchor_contract=anchor_contract,
        contract_aware_target=contract_aware_target,
        planner_validated_mining=planner_validated_mining,
        hybrid_training_dry_run=hybrid_training_dry_run,
        controlled_candidate=controlled_candidate,
        controlled_holdout=controlled_holdout,
        fresh_holdout=fresh_holdout,
        scenario_rollout=scenario_rollout,
        raw_strict_rollout=raw_strict_rollout,
        raw_generalization=raw_generalization,
        policy_canary=policy_canary,
        sequential_canary=sequential_canary,
        ppo_collector=ppo_collector,
        limited_ppo_update_smoke=limited_ppo_update_smoke,
        limited_quasi_real_ppo_update_smoke=limited_quasi_real_ppo_update_smoke,
        generated_sequential_gate_metric_accounting_audit=(
            generated_sequential_gate_metric_accounting_audit
        ),
        generated_sequential_long_horizon_teacher_skill_contract=(
            generated_sequential_long_horizon_teacher_skill_contract
        ),
        iterative_ppo_mini_loop=iterative_ppo_mini_loop,
        return_aligned_guarded_multistep_collector=(
            return_aligned_guarded_multistep_collector
        ),
        return_aligned_guarded_ppo_update_smoke=(
            return_aligned_guarded_ppo_update_smoke
        ),
        guarded_ppo_rollout_pilot=guarded_ppo_rollout_pilot,
        quasi_real_guarded_ppo_rollout_pilot=quasi_real_guarded_ppo_rollout_pilot,
        quasi_real_guarded_ppo_stability_replay=quasi_real_guarded_ppo_stability_replay,
        quasi_real_guarded_ppo_horizon5_batch_expansion=(
            quasi_real_guarded_ppo_horizon5_batch_expansion
        ),
        quasi_real_guarded_ppo_scale512_multiseed_preflight=(
            quasi_real_guarded_ppo_scale512_multiseed_preflight
        ),
        quasi_real_guarded_ppo_iterative_miniloop_stability=(
            quasi_real_guarded_ppo_iterative_miniloop_stability
        ),
        quasi_real_guarded_formal_ppo_preflight=(
            quasi_real_guarded_formal_ppo_preflight
        ),
        quasi_real_guarded_formal_ppo_rollout_canary=(
            quasi_real_guarded_formal_ppo_rollout_canary
        ),
        quasi_real_guarded_formal_ppo_stability_holdout=(
            quasi_real_guarded_formal_ppo_stability_holdout
        ),
        policy_training_cuda_device_support=policy_training_cuda_device_support,
        quasi_real_map_domain_gap=quasi_real_map_domain_gap,
        quasi_real_shadow_policy_behavior=quasi_real_shadow_policy_behavior,
        quasi_real_shadow_alignment=quasi_real_shadow_alignment,
        quasi_real_guarded_policy_pilot=quasi_real_guarded_policy_pilot,
        quasi_real_safe_alternative_opportunity=(
            quasi_real_safe_alternative_opportunity
        ),
        quasi_real_safe_better_opportunity_expansion=(
            quasi_real_safe_better_opportunity_expansion
        ),
        quasi_real_teacher_equivalent_validation=(
            quasi_real_teacher_equivalent_validation
        ),
        quasi_real_teacher_distillation=quasi_real_teacher_distillation,
        quasi_real_guarded_teacher_following_pilot=(
            quasi_real_guarded_teacher_following_pilot
        ),
        validation_reason_codes=reason_codes,
        anchor_only_mode=stage_isolated_mode,
        config=config,
    )
    status = "failed" if reason_codes else "passed"
    failure_reason_counts = Counter(reason_codes)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "failure_reason_code_counts": dict(sorted(failure_reason_counts.items())),
        "batch_root": _display_path(batch_root, repo_root),
        "calibrated_policy_application_smoke_summary_path": _display_path(smoke_path, repo_root),
        "readiness_summary_path": _display_path(readiness_path, repo_root),
        "contrast_coverage_summary_path": _display_path(coverage_path, repo_root),
        "selection_contrast_calibration_summary_path": _display_path(calibration_path, repo_root),
        "anchor_projection_candidate_generation_summary_path": (
            _display_path(anchor_candidate_path, repo_root) if anchor_candidate else None
        ),
        "anchor_projection_evidence_contract_summary_path": (
            _display_path(anchor_contract_path, repo_root) if anchor_contract else None
        ),
        "contract_aware_trainable_target_summary_path": (
            _display_path(contract_aware_target_path, repo_root) if contract_aware_target else None
        ),
        "planner_validated_trainable_target_mining_summary_path": (
            _display_path(planner_validated_mining_path, repo_root)
            if planner_validated_mining
            else None
        ),
        "hybrid_policy_training_dry_run_summary_path": (
            _display_path(hybrid_training_dry_run_path, repo_root)
            if hybrid_training_dry_run
            else None
        ),
        "controlled_hybrid_policy_training_candidate_summary_path": (
            _display_path(controlled_candidate_path, repo_root)
            if controlled_candidate
            else None
        ),
        "controlled_hybrid_policy_holdout_evaluation_summary_path": (
            _display_path(controlled_holdout_path, repo_root)
            if controlled_holdout
            else None
        ),
        "fresh_holdout_policy_candidate_evaluation_summary_path": (
            _display_path(fresh_holdout_path, repo_root)
            if fresh_holdout
            else None
        ),
        "scenario_disjoint_policy_rollout_evaluation_summary_path": (
            _display_path(scenario_rollout_path, repo_root)
            if scenario_rollout
            else None
        ),
        "raw_policy_strict_rollout_evaluation_summary_path": (
            _display_path(raw_strict_rollout_path, repo_root)
            if raw_strict_rollout
            else None
        ),
        "raw_policy_generalization_evaluation_summary_path": (
            _display_path(raw_generalization_path, repo_root)
            if raw_generalization
            else None
        ),
        "policy_gated_canary_rollout_summary_path": (
            _display_path(policy_canary_path, repo_root)
            if policy_canary
            else None
        ),
        "policy_gated_sequential_canary_rollout_summary_path": (
            _display_path(sequential_canary_path, repo_root)
            if sequential_canary
            else None
        ),
        "ppo_rollout_collector_summary_path": (
            _display_path(ppo_collector_path, repo_root)
            if ppo_collector
            else None
        ),
        "limited_ppo_update_smoke_summary_path": (
            _display_path(limited_ppo_update_smoke_path, repo_root)
            if limited_ppo_update_smoke
            else None
        ),
        "iterative_ppo_mini_loop_stability_summary_path": (
            _display_path(iterative_ppo_mini_loop_path, repo_root)
            if iterative_ppo_mini_loop
            else None
        ),
        "return_aligned_guarded_multistep_collector_summary_path": (
            _display_path(return_aligned_guarded_multistep_collector_path, repo_root)
            if return_aligned_guarded_multistep_collector
            else None
        ),
        "return_aligned_guarded_ppo_update_smoke_summary_path": (
            _display_path(return_aligned_guarded_ppo_update_smoke_path, repo_root)
            if return_aligned_guarded_ppo_update_smoke
            else None
        ),
        "guarded_ppo_rollout_pilot_summary_path": (
            _display_path(guarded_ppo_rollout_pilot_path, repo_root)
            if guarded_ppo_rollout_pilot
            else None
        ),
        "quasi_real_guarded_ppo_rollout_pilot_summary_path": (
            _display_path(quasi_real_guarded_ppo_rollout_pilot_path, repo_root)
            if quasi_real_guarded_ppo_rollout_pilot
            else None
        ),
        "quasi_real_guarded_ppo_stability_replay_summary_path": (
            _display_path(quasi_real_guarded_ppo_stability_replay_path, repo_root)
            if quasi_real_guarded_ppo_stability_replay
            else None
        ),
        "quasi_real_guarded_ppo_horizon5_batch_expansion_summary_path": (
            _display_path(
                quasi_real_guarded_ppo_horizon5_batch_expansion_path,
                repo_root,
            )
            if quasi_real_guarded_ppo_horizon5_batch_expansion
            else None
        ),
        "quasi_real_guarded_ppo_scale512_multiseed_preflight_summary_path": (
            _display_path(
                quasi_real_guarded_ppo_scale512_multiseed_preflight_path,
                repo_root,
            )
            if quasi_real_guarded_ppo_scale512_multiseed_preflight
            else None
        ),
        "quasi_real_guarded_ppo_iterative_miniloop_stability_summary_path": (
            _display_path(
                quasi_real_guarded_ppo_iterative_miniloop_stability_path,
                repo_root,
            )
            if quasi_real_guarded_ppo_iterative_miniloop_stability
            else None
        ),
        "quasi_real_guarded_formal_ppo_preflight_summary_path": (
            _display_path(
                quasi_real_guarded_formal_ppo_preflight_path,
                repo_root,
            )
            if quasi_real_guarded_formal_ppo_preflight
            else None
        ),
        "quasi_real_guarded_formal_ppo_rollout_canary_summary_path": (
            _display_path(
                quasi_real_guarded_formal_ppo_rollout_canary_path,
                repo_root,
            )
            if quasi_real_guarded_formal_ppo_rollout_canary
            else None
        ),
        "quasi_real_guarded_formal_ppo_stability_holdout_validation_summary_path": (
            _display_path(
                quasi_real_guarded_formal_ppo_stability_holdout_path,
                repo_root,
            )
            if quasi_real_guarded_formal_ppo_stability_holdout
            else None
        ),
        "policy_training_cuda_device_support_summary_path": (
            _display_path(policy_training_cuda_device_support_path, repo_root)
            if policy_training_cuda_device_support
            else None
        ),
        "quasi_real_map_domain_gap_summary_path": (
            _display_path(quasi_real_map_domain_gap_path, repo_root)
            if quasi_real_map_domain_gap
            else None
        ),
        "quasi_real_shadow_policy_behavior_summary_path": (
            _display_path(quasi_real_shadow_policy_behavior_path, repo_root)
            if quasi_real_shadow_policy_behavior
            else None
        ),
        "quasi_real_shadow_alignment_summary_path": (
            _display_path(quasi_real_shadow_alignment_path, repo_root)
            if quasi_real_shadow_alignment
            else None
        ),
        "quasi_real_guarded_policy_pilot_summary_path": (
            _display_path(quasi_real_guarded_policy_pilot_path, repo_root)
            if quasi_real_guarded_policy_pilot
            else None
        ),
        "quasi_real_safe_alternative_opportunity_summary_path": (
            _display_path(quasi_real_safe_alternative_opportunity_path, repo_root)
            if quasi_real_safe_alternative_opportunity
            else None
        ),
        "quasi_real_safe_better_opportunity_expansion_summary_path": (
            _display_path(quasi_real_safe_better_opportunity_expansion_path, repo_root)
            if quasi_real_safe_better_opportunity_expansion
            else None
        ),
        "quasi_real_teacher_equivalent_validation_summary_path": (
            _display_path(quasi_real_teacher_equivalent_validation_path, repo_root)
            if quasi_real_teacher_equivalent_validation
            else None
        ),
        "quasi_real_teacher_distillation_summary_path": (
            _display_path(quasi_real_teacher_distillation_path, repo_root)
            if quasi_real_teacher_distillation
            else None
        ),
        "quasi_real_guarded_teacher_following_pilot_summary_path": (
            _display_path(quasi_real_guarded_teacher_following_pilot_path, repo_root)
            if quasi_real_guarded_teacher_following_pilot
            else None
        ),
        "generated_sequential_gate_metric_accounting_audit_summary_path": (
            _display_path(generated_sequential_gate_metric_accounting_audit_path, repo_root)
            if generated_sequential_gate_metric_accounting_audit
            else None
        ),
        "generated_sequential_long_horizon_teacher_skill_contract_summary_path": (
            _display_path(
                generated_sequential_long_horizon_teacher_skill_contract_path,
                repo_root,
            )
            if generated_sequential_long_horizon_teacher_skill_contract
            else None
        ),
        "application_scope": (
            "anchor_projection_readiness_contract_review_only"
            if anchor_only_mode
            else "calibrated_policy_training_readiness_review_audit_only"
        ),
        "quality_signal_use": "calibrated_policy_target_training_contract_review_only",
        "source_summaries": source_summaries,
        "config": _public_config(config),
        "git_provenance": {
            "current": current_git,
            "calibrated_policy_application_smoke": _public_git(smoke),
            "training_readiness": _public_git(readiness),
            "contrast_coverage": _public_git(coverage),
            "selection_contrast_calibration": _public_git(calibration),
            "anchor_projection_candidate_generation": _public_git(anchor_candidate),
            "anchor_projection_evidence_contract": _public_git(anchor_contract),
            "contract_aware_trainable_target": _public_git(contract_aware_target),
            "planner_validated_trainable_target_mining": _public_git(planner_validated_mining),
            "hybrid_policy_training_dry_run": _public_git(hybrid_training_dry_run),
            "controlled_hybrid_policy_training_candidate": _public_git(controlled_candidate),
            "controlled_hybrid_policy_holdout_evaluation": _public_git(controlled_holdout),
            "fresh_holdout_policy_candidate_evaluation": _public_git(fresh_holdout),
            "scenario_disjoint_policy_rollout_evaluation": _public_git(scenario_rollout),
            "raw_policy_strict_rollout_evaluation": _public_git(raw_strict_rollout),
            "raw_policy_generalization_evaluation": _public_git(raw_generalization),
            "policy_gated_canary_rollout": _public_git(policy_canary),
            "policy_gated_sequential_canary_rollout": _public_git(sequential_canary),
            "ppo_rollout_collector": _public_git(ppo_collector),
            "limited_ppo_update_smoke": _public_git(limited_ppo_update_smoke),
            "generated_sequential_gate_metric_accounting_audit": _public_git(
                generated_sequential_gate_metric_accounting_audit
            ),
            "generated_sequential_long_horizon_teacher_skill_contract": _public_git(
                generated_sequential_long_horizon_teacher_skill_contract
            ),
            "iterative_ppo_mini_loop_stability": _public_git(iterative_ppo_mini_loop),
            "return_aligned_guarded_multistep_collector": _public_git(
                return_aligned_guarded_multistep_collector
            ),
            "return_aligned_guarded_ppo_update_smoke": _public_git(
                return_aligned_guarded_ppo_update_smoke
            ),
            "guarded_ppo_rollout_pilot": _public_git(guarded_ppo_rollout_pilot),
            "quasi_real_guarded_ppo_rollout_pilot": _public_git(
                quasi_real_guarded_ppo_rollout_pilot
            ),
            "quasi_real_guarded_ppo_stability_replay": _public_git(
                quasi_real_guarded_ppo_stability_replay
            ),
            "quasi_real_guarded_ppo_horizon5_batch_expansion": _public_git(
                quasi_real_guarded_ppo_horizon5_batch_expansion
            ),
            "quasi_real_guarded_ppo_scale512_multiseed_preflight": _public_git(
                quasi_real_guarded_ppo_scale512_multiseed_preflight
            ),
            "quasi_real_guarded_ppo_iterative_miniloop_stability": _public_git(
                quasi_real_guarded_ppo_iterative_miniloop_stability
            ),
            "quasi_real_guarded_formal_ppo_preflight": _public_git(
                quasi_real_guarded_formal_ppo_preflight
            ),
            "quasi_real_guarded_formal_ppo_rollout_canary": _public_git(
                quasi_real_guarded_formal_ppo_rollout_canary
            ),
            "quasi_real_guarded_formal_ppo_stability_holdout_validation": (
                _public_git(quasi_real_guarded_formal_ppo_stability_holdout)
            ),
            "policy_training_cuda_device_support": _public_git(
                policy_training_cuda_device_support
            ),
            "quasi_real_map_domain_gap": _public_git(quasi_real_map_domain_gap),
            "quasi_real_shadow_policy_behavior": _public_git(
                quasi_real_shadow_policy_behavior
            ),
            "quasi_real_shadow_alignment": _public_git(quasi_real_shadow_alignment),
            "quasi_real_guarded_policy_pilot": _public_git(quasi_real_guarded_policy_pilot),
            "quasi_real_safe_alternative_opportunity": _public_git(
                quasi_real_safe_alternative_opportunity
            ),
            "quasi_real_safe_better_opportunity_expansion": _public_git(
                quasi_real_safe_better_opportunity_expansion
            ),
            "quasi_real_teacher_equivalent_validation": _public_git(
                quasi_real_teacher_equivalent_validation
            ),
            "quasi_real_teacher_distillation": _public_git(
                quasi_real_teacher_distillation
            ),
            "quasi_real_guarded_teacher_following_pilot": _public_git(
                quasi_real_guarded_teacher_following_pilot
            ),
            "current_matches_sources": all(source_git_matches),
        },
        **review,
        "runs_training": False,
        "audit_only": True,
        "no_ppo_training": True,
        "no_large_scale_training": True,
        "no_real_world_performance_claim": True,
        "channel_aware_backend_opt_in": True,
        "does_not_modify_default_astar": True,
        "does_not_modify_ppo": True,
        "does_not_modify_network": True,
        "does_not_modify_action_space": True,
        "does_not_modify_model_explorer_contract": True,
        "does_not_modify_path_planner_route_contract": True,
        "does_not_modify_path_planner_sidecar_contract": True,
        "no_ackermann_feasible_trajectory_claim": True,
        "non_goals": list(config.get("non_goals", [])),
    }


def _review_metrics(
    *,
    smoke: dict[str, Any],
    readiness: dict[str, Any],
    coverage: dict[str, Any],
    calibration: dict[str, Any],
    anchor_candidate: dict[str, Any],
    anchor_contract: dict[str, Any],
    contract_aware_target: dict[str, Any],
    planner_validated_mining: dict[str, Any],
    hybrid_training_dry_run: dict[str, Any],
    controlled_candidate: dict[str, Any],
    controlled_holdout: dict[str, Any],
    fresh_holdout: dict[str, Any],
    scenario_rollout: dict[str, Any],
    raw_strict_rollout: dict[str, Any],
    raw_generalization: dict[str, Any],
    policy_canary: dict[str, Any],
    sequential_canary: dict[str, Any],
    ppo_collector: dict[str, Any],
    limited_ppo_update_smoke: dict[str, Any],
    limited_quasi_real_ppo_update_smoke: dict[str, Any],
    generated_sequential_gate_metric_accounting_audit: dict[str, Any],
    generated_sequential_long_horizon_teacher_skill_contract: dict[str, Any],
    iterative_ppo_mini_loop: dict[str, Any],
    return_aligned_guarded_multistep_collector: dict[str, Any],
    return_aligned_guarded_ppo_update_smoke: dict[str, Any],
    guarded_ppo_rollout_pilot: dict[str, Any],
    quasi_real_guarded_ppo_rollout_pilot: dict[str, Any],
    quasi_real_guarded_ppo_stability_replay: dict[str, Any],
    quasi_real_guarded_ppo_horizon5_batch_expansion: dict[str, Any],
    quasi_real_guarded_ppo_scale512_multiseed_preflight: dict[str, Any],
    quasi_real_guarded_ppo_iterative_miniloop_stability: dict[str, Any],
    quasi_real_guarded_formal_ppo_preflight: dict[str, Any],
    quasi_real_guarded_formal_ppo_rollout_canary: dict[str, Any],
    quasi_real_guarded_formal_ppo_stability_holdout: dict[str, Any],
    policy_training_cuda_device_support: dict[str, Any],
    quasi_real_map_domain_gap: dict[str, Any],
    quasi_real_shadow_policy_behavior: dict[str, Any],
    quasi_real_shadow_alignment: dict[str, Any],
    quasi_real_guarded_policy_pilot: dict[str, Any],
    quasi_real_safe_alternative_opportunity: dict[str, Any],
    quasi_real_safe_better_opportunity_expansion: dict[str, Any],
    quasi_real_teacher_equivalent_validation: dict[str, Any],
    quasi_real_teacher_distillation: dict[str, Any],
    quasi_real_guarded_teacher_following_pilot: dict[str, Any],
    validation_reason_codes: list[str],
    anchor_only_mode: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config["readiness_thresholds"]
    source_rate = _source_rate(smoke, readiness, coverage, calibration)
    calibrated_rate = _calibrated_rate(smoke, readiness, coverage, calibration)
    applied_count = _int_value_or_default(smoke.get("applied_calibrated_candidate_count"), 0)
    rejected_goal_blocked_count = max(
        _int_value_or_default(smoke.get("rejected_goal_blocked_count"), 0),
        _int_value_or_default(calibration.get("goal_blocked_count"), 0),
    )
    platform_goal_contract_mismatch_count = max(
        _int_value_or_default(smoke.get("platform_goal_contract_mismatch_count"), 0),
        _int_value_or_default(calibration.get("platform_goal_contract_mismatch_count"), 0),
    )
    platform_goal_trainable_anchor_projection_count = max(
        _int_value_or_default(smoke.get("platform_goal_trainable_anchor_projection_count"), 0),
        _int_value_or_default(calibration.get("platform_goal_trainable_anchor_projection_count"), 0),
    )
    platform_goal_nontrainable_blocked_target_count = max(
        _int_value_or_default(
            smoke.get("platform_goal_nontrainable_blocked_target_count"),
            platform_goal_contract_mismatch_count - platform_goal_trainable_anchor_projection_count,
        ),
        _int_value_or_default(
            calibration.get("platform_goal_nontrainable_blocked_target_count"),
            platform_goal_contract_mismatch_count - platform_goal_trainable_anchor_projection_count,
        ),
    )
    platform_goal_anchor_available_count = max(
        _int_value_or_default(smoke.get("platform_goal_anchor_available_count"), 0),
        _int_value_or_default(calibration.get("platform_goal_anchor_available_count"), 0),
    )
    platform_goal_unresolved_count = max(
        _int_value_or_default(smoke.get("platform_goal_unresolved_count"), 0),
        _int_value_or_default(calibration.get("platform_goal_unresolved_count"), 0),
    )
    platform_goal_class_counts = _max_counter_dict(
        smoke.get("platform_goal_feasibility_class_counts"),
        calibration.get("platform_goal_feasibility_class_counts"),
    )
    safety_regression_count = max(
        _int_value_or_default(smoke.get("safety_regression_count"), 0),
        _int_value_or_default(readiness.get("calibration_safety_regression_count"), 0),
        _int_value_or_default(coverage.get("safety_regression_count"), 0),
        _int_value_or_default(calibration.get("safety_regression_count"), 0),
        _int_value_or_default(anchor_candidate.get("safety_regression_count"), 0),
        _int_value_or_default(anchor_contract.get("safety_regression_count"), 0),
        _int_value_or_default(contract_aware_target.get("safety_regression_count"), 0),
        _int_value_or_default(planner_validated_mining.get("safety_regression_count"), 0),
    )
    fallback_or_open_grid_count = max(
        _fallback_or_open_grid_count(smoke),
        _fallback_or_open_grid_count(readiness),
        _fallback_or_open_grid_count(coverage),
        _fallback_or_open_grid_count(calibration),
        _fallback_or_open_grid_count(anchor_candidate),
        _fallback_or_open_grid_count(anchor_contract),
        _fallback_or_open_grid_count(contract_aware_target),
        _fallback_or_open_grid_count(planner_validated_mining),
    )
    changed_scenario_ids = _unique(
        _string_list(smoke.get("changed_scenario_ids"))
        or _string_list(coverage.get("changed_scenario_ids"))
        or _string_list(calibration.get("changed_scenario_ids"))
    )
    contract_mutations = _contract_mutations(
        {
            "calibrated_policy_application_smoke": smoke,
            "training_readiness": readiness,
            "contrast_coverage": coverage,
            "selection_contrast_calibration": calibration,
            "anchor_projection_candidate_generation": anchor_candidate,
            "anchor_projection_evidence_contract": anchor_contract,
            "contract_aware_trainable_target": contract_aware_target,
            "planner_validated_trainable_target_mining": planner_validated_mining,
            "hybrid_policy_training_dry_run": hybrid_training_dry_run,
            "controlled_hybrid_policy_training_candidate": controlled_candidate,
            "controlled_hybrid_policy_holdout_evaluation": controlled_holdout,
            "fresh_holdout_policy_candidate_evaluation": fresh_holdout,
            "scenario_disjoint_policy_rollout_evaluation": scenario_rollout,
            "raw_policy_strict_rollout_evaluation": raw_strict_rollout,
            "raw_policy_generalization_evaluation": raw_generalization,
            "policy_gated_canary_rollout": policy_canary,
            "policy_gated_sequential_canary_rollout": sequential_canary,
            "ppo_rollout_collector": ppo_collector,
            "limited_ppo_update_smoke": limited_ppo_update_smoke,
            "iterative_ppo_mini_loop_stability": iterative_ppo_mini_loop,
            "return_aligned_guarded_multistep_collector": (
                return_aligned_guarded_multistep_collector
            ),
            "return_aligned_guarded_ppo_update_smoke": (
                return_aligned_guarded_ppo_update_smoke
            ),
            "guarded_ppo_rollout_pilot": guarded_ppo_rollout_pilot,
            "quasi_real_guarded_ppo_rollout_pilot": quasi_real_guarded_ppo_rollout_pilot,
            "quasi_real_guarded_ppo_stability_replay": quasi_real_guarded_ppo_stability_replay,
            "quasi_real_guarded_ppo_horizon5_batch_expansion": (
                quasi_real_guarded_ppo_horizon5_batch_expansion
            ),
            "quasi_real_guarded_ppo_scale512_multiseed_preflight": (
                quasi_real_guarded_ppo_scale512_multiseed_preflight
            ),
            "quasi_real_guarded_formal_ppo_preflight": (
                quasi_real_guarded_formal_ppo_preflight
            ),
            "quasi_real_guarded_formal_ppo_rollout_canary": (
                quasi_real_guarded_formal_ppo_rollout_canary
            ),
            "quasi_real_guarded_formal_ppo_stability_holdout_validation": (
                quasi_real_guarded_formal_ppo_stability_holdout
            ),
            "policy_training_cuda_device_support": policy_training_cuda_device_support,
            "quasi_real_map_domain_gap": quasi_real_map_domain_gap,
            "quasi_real_shadow_policy_behavior": quasi_real_shadow_policy_behavior,
            "quasi_real_shadow_alignment": quasi_real_shadow_alignment,
            "quasi_real_guarded_policy_pilot": quasi_real_guarded_policy_pilot,
            "quasi_real_safe_alternative_opportunity": quasi_real_safe_alternative_opportunity,
            "quasi_real_safe_better_opportunity_expansion": (
                quasi_real_safe_better_opportunity_expansion
            ),
            "quasi_real_teacher_equivalent_validation": (
                quasi_real_teacher_equivalent_validation
            ),
            "quasi_real_teacher_distillation": quasi_real_teacher_distillation,
            "quasi_real_guarded_teacher_following_pilot": (
                quasi_real_guarded_teacher_following_pilot
            ),
        }
    )
    anchor_projection_readiness = _anchor_projection_readiness(
        candidate=anchor_candidate,
        contract=anchor_contract,
        contract_aware_target=contract_aware_target,
        planner_validated_mining=planner_validated_mining,
        thresholds=thresholds,
    )
    hybrid_training_readiness = _hybrid_training_readiness(hybrid_training_dry_run)
    fresh_holdout_readiness = _fresh_holdout_policy_candidate_readiness(fresh_holdout)
    scenario_rollout_readiness = _scenario_disjoint_policy_rollout_readiness(scenario_rollout)
    raw_strict_rollout_readiness = _raw_policy_strict_rollout_readiness(raw_strict_rollout)
    raw_generalization_readiness = _raw_policy_generalization_readiness(raw_generalization)
    policy_canary_readiness = _policy_gated_canary_rollout_readiness(policy_canary)
    sequential_canary_readiness = _policy_gated_sequential_canary_rollout_readiness(
        sequential_canary
    )
    ppo_collector_readiness = _ppo_rollout_collector_readiness(ppo_collector)
    limited_ppo_update_smoke_readiness = _limited_ppo_update_smoke_readiness(
        limited_ppo_update_smoke
    )
    limited_quasi_real_ppo_update_smoke_readiness = (
        _limited_quasi_real_ppo_update_smoke_readiness(
            limited_quasi_real_ppo_update_smoke
        )
    )
    generated_sequential_gate_metric_accounting_readiness = (
        _generated_sequential_gate_metric_accounting_readiness(
            generated_sequential_gate_metric_accounting_audit
        )
    )
    generated_sequential_long_horizon_teacher_skill_contract_readiness = (
        _generated_sequential_long_horizon_teacher_skill_contract_readiness(
            generated_sequential_long_horizon_teacher_skill_contract
        )
    )
    limited_quasi_real_generated_sequential_override = (
        _long_horizon_contract_overrides_limited_quasi_real_generated_blocker(
            limited_quasi_real_ppo_update_smoke_readiness,
            generated_sequential_long_horizon_teacher_skill_contract_readiness,
        )
    )
    iterative_ppo_mini_loop_readiness = _iterative_ppo_mini_loop_stability_readiness(
        iterative_ppo_mini_loop
    )
    return_aligned_guarded_multistep_collector_readiness = (
        _return_aligned_guarded_multistep_collector_readiness(
            return_aligned_guarded_multistep_collector
        )
    )
    return_aligned_guarded_ppo_update_smoke_readiness = (
        _return_aligned_guarded_ppo_update_smoke_readiness(
            return_aligned_guarded_ppo_update_smoke
        )
    )
    guarded_ppo_rollout_pilot_readiness = _guarded_ppo_rollout_pilot_readiness(
        guarded_ppo_rollout_pilot
    )
    quasi_real_guarded_ppo_rollout_pilot_readiness = (
        _quasi_real_guarded_ppo_rollout_pilot_readiness(
            quasi_real_guarded_ppo_rollout_pilot
        )
    )
    quasi_real_guarded_ppo_stability_replay_readiness = (
        _quasi_real_guarded_ppo_stability_replay_readiness(
            quasi_real_guarded_ppo_stability_replay
        )
    )
    quasi_real_guarded_ppo_horizon5_batch_expansion_readiness = (
        _quasi_real_guarded_ppo_horizon5_batch_expansion_readiness(
            quasi_real_guarded_ppo_horizon5_batch_expansion
        )
    )
    quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness = (
        _quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness(
            quasi_real_guarded_ppo_scale512_multiseed_preflight
        )
    )
    quasi_real_guarded_ppo_iterative_miniloop_stability_readiness = (
        _quasi_real_guarded_ppo_iterative_miniloop_stability_readiness(
            quasi_real_guarded_ppo_iterative_miniloop_stability
        )
    )
    quasi_real_guarded_formal_ppo_preflight_readiness = (
        _quasi_real_guarded_formal_ppo_preflight_readiness(
            quasi_real_guarded_formal_ppo_preflight
        )
    )
    quasi_real_guarded_formal_ppo_rollout_canary_readiness = (
        _quasi_real_guarded_formal_ppo_rollout_canary_readiness(
            quasi_real_guarded_formal_ppo_rollout_canary
        )
    )
    quasi_real_guarded_formal_ppo_stability_holdout_readiness = (
        _quasi_real_guarded_formal_ppo_stability_holdout_readiness(
            quasi_real_guarded_formal_ppo_stability_holdout
        )
    )
    policy_training_cuda_device_support_readiness = (
        _policy_training_cuda_device_support_readiness(policy_training_cuda_device_support)
    )
    quasi_real_map_domain_gap_readiness = _quasi_real_map_domain_gap_readiness(
        quasi_real_map_domain_gap
    )
    quasi_real_shadow_policy_behavior_readiness = (
        _quasi_real_shadow_policy_behavior_readiness(quasi_real_shadow_policy_behavior)
    )
    quasi_real_shadow_alignment_readiness = _quasi_real_shadow_alignment_readiness(
        quasi_real_shadow_alignment
    )
    quasi_real_guarded_policy_pilot_readiness = _quasi_real_guarded_policy_pilot_readiness(
        quasi_real_guarded_policy_pilot
    )
    quasi_real_safe_alternative_opportunity_readiness = (
        _quasi_real_safe_alternative_opportunity_readiness(
            quasi_real_safe_alternative_opportunity
        )
    )
    quasi_real_safe_better_opportunity_expansion_readiness = (
        _quasi_real_safe_better_opportunity_expansion_readiness(
            quasi_real_safe_better_opportunity_expansion
        )
    )
    quasi_real_teacher_equivalent_validation_readiness = (
        _quasi_real_teacher_equivalent_validation_readiness(
            quasi_real_teacher_equivalent_validation
        )
    )
    quasi_real_teacher_distillation_readiness = _quasi_real_teacher_distillation_readiness(
        quasi_real_teacher_distillation
    )
    quasi_real_guarded_teacher_following_pilot_readiness = (
        _quasi_real_guarded_teacher_following_pilot_readiness(
            quasi_real_guarded_teacher_following_pilot
        )
    )
    controlled_candidate_readiness = _controlled_hybrid_training_candidate_readiness(
        candidate=controlled_candidate,
        holdout=controlled_holdout,
        allow_fresh_holdout_substitute=(
            fresh_holdout_readiness["present"]
            or scenario_rollout_readiness["present"]
            or raw_strict_rollout_readiness["present"]
        ),
    )
    training_blockers: list[str] = []
    if validation_reason_codes:
        for reason in validation_reason_codes:
            _append_reason(training_blockers, reason)
    if (
        not anchor_only_mode
        and thresholds["require_smoke_ready_for_training_review"]
        and smoke.get("recommended_next_action") != READY_SMOKE_ACTION
    ):
        _append_reason(training_blockers, "calibrated_application_smoke_not_ready_for_training_review")
    if not anchor_only_mode and applied_count < thresholds["min_applied_calibrated_candidate_count"]:
        _append_reason(training_blockers, "applied_calibrated_candidate_count_below_training_threshold")
    if not anchor_only_mode and calibrated_rate - source_rate < thresholds["min_calibrated_selection_rate_delta"]:
        _append_reason(training_blockers, "calibrated_selection_rate_delta_below_training_threshold")
    if rejected_goal_blocked_count > thresholds["max_rejected_goal_blocked_count"]:
        _append_reason(training_blockers, "goal_blocked_candidates_excluded_from_training_positive_evidence")
    if safety_regression_count > thresholds["max_safety_regression_count"]:
        _append_reason(training_blockers, "safety_regression_blocks_training_readiness")
    if fallback_or_open_grid_count > thresholds["max_fallback_or_open_grid_count"]:
        _append_reason(training_blockers, "fallback_or_open_grid_evidence_blocks_training_readiness")
    if contract_mutations:
        _append_reason(training_blockers, "contract_mutation_blocks_training_readiness")
    if (
        anchor_only_mode
        and anchor_projection_readiness["candidate_generation_nontrainable_count"] > 0
        and anchor_projection_readiness["ppo_consumable_trainable_target_count"] <= 0
    ):
        _append_reason(training_blockers, "anchor_projection_nontrainable_contexts_remain")
    for reason in anchor_projection_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in hybrid_training_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in controlled_candidate_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in fresh_holdout_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in scenario_rollout_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in raw_strict_rollout_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in raw_generalization_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in policy_canary_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in sequential_canary_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in ppo_collector_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in limited_ppo_update_smoke_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    if not limited_quasi_real_generated_sequential_override:
        for reason in limited_quasi_real_ppo_update_smoke_readiness["training_blockers"]:
            _append_reason(training_blockers, reason)
    if not generated_sequential_long_horizon_teacher_skill_contract_readiness["completed"]:
        for reason in generated_sequential_gate_metric_accounting_readiness["training_blockers"]:
            _append_reason(training_blockers, reason)
    for reason in generated_sequential_long_horizon_teacher_skill_contract_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in iterative_ppo_mini_loop_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in return_aligned_guarded_multistep_collector_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in return_aligned_guarded_ppo_update_smoke_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in guarded_ppo_rollout_pilot_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_ppo_rollout_pilot_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_ppo_stability_replay_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_ppo_horizon5_batch_expansion_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness[
        "training_blockers"
    ]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_ppo_iterative_miniloop_stability_readiness[
        "training_blockers"
    ]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_formal_ppo_preflight_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_formal_ppo_rollout_canary_readiness[
        "training_blockers"
    ]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_formal_ppo_stability_holdout_readiness[
        "training_blockers"
    ]:
        _append_reason(training_blockers, reason)
    for reason in policy_training_cuda_device_support_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_map_domain_gap_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_shadow_policy_behavior_readiness["training_blockers"]:
        if not quasi_real_shadow_alignment_readiness["present"]:
            _append_reason(training_blockers, reason)
    for reason in quasi_real_shadow_alignment_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_policy_pilot_readiness["training_blockers"]:
        if not quasi_real_safe_alternative_opportunity_readiness["present"]:
            _append_reason(training_blockers, reason)
    for reason in quasi_real_safe_alternative_opportunity_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)
    for reason in quasi_real_safe_better_opportunity_expansion_readiness["training_blockers"]:
        if not (
            quasi_real_teacher_equivalent_validation_readiness["present"]
            or quasi_real_teacher_distillation_readiness["present"]
            or quasi_real_guarded_teacher_following_pilot_readiness["present"]
        ):
            _append_reason(training_blockers, reason)
    for reason in quasi_real_teacher_equivalent_validation_readiness["training_blockers"]:
        if not quasi_real_teacher_distillation_readiness["present"]:
            _append_reason(training_blockers, reason)
    for reason in quasi_real_teacher_distillation_readiness["training_blockers"]:
        if not quasi_real_guarded_teacher_following_pilot_readiness["present"]:
            _append_reason(training_blockers, reason)
    for reason in quasi_real_guarded_teacher_following_pilot_readiness["training_blockers"]:
        _append_reason(training_blockers, reason)

    hard_validation_failed = bool(validation_reason_codes)
    if hard_validation_failed:
        training_readiness_status = "blocked_by_validation"
        recommended_next_action = "fix_validation_failures_before_training_readiness_review"
    elif training_blockers:
        training_readiness_status = "needs_training_contract_refinement"
        recommended_next_action = (
            generated_sequential_long_horizon_teacher_skill_contract_readiness.get(
                "next_required_change"
            )
            or
            generated_sequential_gate_metric_accounting_readiness.get("next_required_change")
            or "needs_training_contract_refinement"
        )
    elif (
        quasi_real_guarded_teacher_following_pilot_readiness["present"]
        and quasi_real_guarded_teacher_following_pilot_readiness["completed"]
        and not ppo_collector_readiness["present"]
    ):
        training_readiness_status = QUASI_REAL_GUARDED_TEACHER_FOLLOWING_PILOT_EVALUATED_ACTION
        recommended_next_action = QUASI_REAL_GUARDED_TEACHER_FOLLOWING_PILOT_EVALUATED_ACTION
    elif (
        quasi_real_teacher_distillation_readiness["present"]
        and quasi_real_teacher_distillation_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_TEACHER_DISTILLATION_EVALUATED_ACTION
        recommended_next_action = QUASI_REAL_TEACHER_DISTILLATION_EVALUATED_ACTION
    elif (
        quasi_real_teacher_equivalent_validation_readiness["present"]
        and quasi_real_teacher_equivalent_validation_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_TEACHER_EQUIVALENT_VALIDATED_ACTION
        recommended_next_action = QUASI_REAL_TEACHER_EQUIVALENT_VALIDATED_ACTION
    elif (
        quasi_real_safe_better_opportunity_expansion_readiness["present"]
        and quasi_real_safe_better_opportunity_expansion_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_SAFE_BETTER_OPPORTUNITY_EXPANDED_ACTION
        recommended_next_action = QUASI_REAL_SAFE_BETTER_OPPORTUNITY_EXPANDED_ACTION
    elif (
        quasi_real_safe_alternative_opportunity_readiness["present"]
        and quasi_real_safe_alternative_opportunity_readiness["completed"]
    ):
        training_readiness_status = (
            QUASI_REAL_SAFE_ALTERNATIVE_OPPORTUNITY_DIAGNOSED_ACTION
        )
        recommended_next_action = (
            QUASI_REAL_SAFE_ALTERNATIVE_OPPORTUNITY_DIAGNOSED_ACTION
        )
    elif (
        quasi_real_guarded_policy_pilot_readiness["present"]
        and quasi_real_guarded_policy_pilot_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_GUARDED_POLICY_PILOT_EVALUATED_ACTION
        recommended_next_action = QUASI_REAL_GUARDED_POLICY_PILOT_EVALUATED_ACTION
    elif (
        quasi_real_shadow_alignment_readiness["present"]
        and quasi_real_shadow_alignment_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_SHADOW_ALIGNMENT_EVALUATED_ACTION
        recommended_next_action = QUASI_REAL_SHADOW_ALIGNMENT_EVALUATED_ACTION
    elif (
        quasi_real_shadow_policy_behavior_readiness["present"]
        and quasi_real_shadow_policy_behavior_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_SHADOW_POLICY_BEHAVIOR_AUDITED_ACTION
        recommended_next_action = QUASI_REAL_SHADOW_POLICY_BEHAVIOR_AUDITED_ACTION
    elif (
        quasi_real_map_domain_gap_readiness["present"]
        and quasi_real_map_domain_gap_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_MAP_DOMAIN_GAP_EVALUATED_ACTION
        recommended_next_action = QUASI_REAL_MAP_DOMAIN_GAP_EVALUATED_ACTION
    elif (
        quasi_real_guarded_formal_ppo_stability_holdout_readiness["present"]
        and quasi_real_guarded_formal_ppo_stability_holdout_readiness["completed"]
    ):
        training_readiness_status = (
            QUASI_REAL_GUARDED_FORMAL_PPO_STABILITY_HOLDOUT_VALIDATED_ACTION
        )
        recommended_next_action = (
            QUASI_REAL_GUARDED_FORMAL_PPO_STABILITY_HOLDOUT_VALIDATED_ACTION
        )
    elif (
        quasi_real_guarded_formal_ppo_rollout_canary_readiness["present"]
        and quasi_real_guarded_formal_ppo_rollout_canary_readiness["completed"]
    ):
        training_readiness_status = (
            QUASI_REAL_GUARDED_FORMAL_PPO_ROLLOUT_CANARY_EVALUATED_ACTION
        )
        recommended_next_action = (
            QUASI_REAL_GUARDED_FORMAL_PPO_ROLLOUT_CANARY_EVALUATED_ACTION
        )
    elif (
        quasi_real_guarded_formal_ppo_preflight_readiness["present"]
        and quasi_real_guarded_formal_ppo_preflight_readiness["completed"]
    ):
        training_readiness_status = (
            QUASI_REAL_GUARDED_FORMAL_PPO_PREFLIGHT_EVALUATED_ACTION
        )
        recommended_next_action = (
            QUASI_REAL_GUARDED_FORMAL_PPO_PREFLIGHT_EVALUATED_ACTION
        )
    elif (
        quasi_real_guarded_ppo_iterative_miniloop_stability_readiness["present"]
        and quasi_real_guarded_ppo_iterative_miniloop_stability_readiness["completed"]
    ):
        training_readiness_status = (
            QUASI_REAL_GUARDED_PPO_ITERATIVE_MINILOOP_STABILITY_EVALUATED_ACTION
        )
        recommended_next_action = (
            QUASI_REAL_GUARDED_PPO_ITERATIVE_MINILOOP_STABILITY_EVALUATED_ACTION
        )
    elif (
        quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness["present"]
        and quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness["completed"]
    ):
        training_readiness_status = (
            QUASI_REAL_GUARDED_PPO_SCALE512_MULTISEED_PREFLIGHT_EVALUATED_ACTION
        )
        recommended_next_action = (
            QUASI_REAL_GUARDED_PPO_SCALE512_MULTISEED_PREFLIGHT_EVALUATED_ACTION
        )
    elif (
        policy_training_cuda_device_support_readiness["present"]
        and policy_training_cuda_device_support_readiness["completed"]
    ):
        training_readiness_status = POLICY_TRAINING_CUDA_DEVICE_SUPPORT_EVALUATED_ACTION
        recommended_next_action = POLICY_TRAINING_CUDA_DEVICE_SUPPORT_EVALUATED_ACTION
    elif (
        quasi_real_guarded_ppo_horizon5_batch_expansion_readiness["present"]
        and quasi_real_guarded_ppo_horizon5_batch_expansion_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_GUARDED_PPO_HORIZON5_BATCH_EXPANSION_EVALUATED_ACTION
        recommended_next_action = QUASI_REAL_GUARDED_PPO_HORIZON5_BATCH_EXPANSION_EVALUATED_ACTION
    elif (
        quasi_real_guarded_ppo_stability_replay_readiness["present"]
        and quasi_real_guarded_ppo_stability_replay_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_GUARDED_PPO_STABILITY_REPLAY_EVALUATED_ACTION
        recommended_next_action = QUASI_REAL_GUARDED_PPO_STABILITY_REPLAY_EVALUATED_ACTION
    elif (
        quasi_real_guarded_ppo_rollout_pilot_readiness["present"]
        and quasi_real_guarded_ppo_rollout_pilot_readiness["completed"]
    ):
        training_readiness_status = QUASI_REAL_GUARDED_PPO_ROLLOUT_PILOT_EVALUATED_ACTION
        recommended_next_action = QUASI_REAL_GUARDED_PPO_ROLLOUT_PILOT_EVALUATED_ACTION
    elif (
        return_aligned_guarded_ppo_update_smoke_readiness["present"]
        and return_aligned_guarded_ppo_update_smoke_readiness["completed"]
    ):
        training_readiness_status = RETURN_ALIGNED_GUARDED_PPO_UPDATE_SMOKE_EVALUATED_ACTION
        recommended_next_action = RETURN_ALIGNED_GUARDED_PPO_UPDATE_SMOKE_EVALUATED_ACTION
    elif (
        return_aligned_guarded_multistep_collector_readiness["present"]
        and return_aligned_guarded_multistep_collector_readiness["completed"]
    ):
        training_readiness_status = RETURN_ALIGNED_GUARDED_MULTISTEP_COLLECTOR_EVALUATED_ACTION
        recommended_next_action = RETURN_ALIGNED_GUARDED_MULTISTEP_COLLECTOR_EVALUATED_ACTION
    elif (
        guarded_ppo_rollout_pilot_readiness["present"]
        and guarded_ppo_rollout_pilot_readiness["completed"]
    ):
        training_readiness_status = GUARDED_PPO_ROLLOUT_PILOT_EVALUATED_ACTION
        recommended_next_action = GUARDED_PPO_ROLLOUT_PILOT_EVALUATED_ACTION
    elif (
        iterative_ppo_mini_loop_readiness["present"]
        and iterative_ppo_mini_loop_readiness["completed"]
    ):
        training_readiness_status = ITERATIVE_PPO_MINI_LOOP_STABILITY_EVALUATED_ACTION
        recommended_next_action = ITERATIVE_PPO_MINI_LOOP_STABILITY_EVALUATED_ACTION
    elif (
        limited_quasi_real_ppo_update_smoke_readiness["present"]
        and (
            limited_quasi_real_ppo_update_smoke_readiness["completed"]
            or limited_quasi_real_generated_sequential_override
        )
    ):
        training_readiness_status = LIMITED_QUASI_REAL_PPO_UPDATE_SMOKE_EVALUATED_ACTION
        recommended_next_action = LIMITED_QUASI_REAL_PPO_UPDATE_SMOKE_EVALUATED_ACTION
    elif (
        limited_ppo_update_smoke_readiness["present"]
        and limited_ppo_update_smoke_readiness["completed"]
    ):
        training_readiness_status = LIMITED_PPO_UPDATE_SMOKE_EVALUATED_ACTION
        recommended_next_action = LIMITED_PPO_UPDATE_SMOKE_EVALUATED_ACTION
    elif ppo_collector_readiness["present"] and ppo_collector_readiness["completed"]:
        training_readiness_status = PPO_ROLLOUT_COLLECTOR_DRY_RUN_EVALUATED_ACTION
        recommended_next_action = PPO_ROLLOUT_COLLECTOR_DRY_RUN_EVALUATED_ACTION
    elif sequential_canary_readiness["present"] and sequential_canary_readiness["completed"]:
        if sequential_canary_readiness.get("sequential_multi_step_opportunity_evaluated"):
            training_readiness_status = (
                POLICY_GATED_SEQUENTIAL_MULTI_STEP_OPPORTUNITY_EVALUATED_ACTION
            )
            recommended_next_action = (
                POLICY_GATED_SEQUENTIAL_MULTI_STEP_OPPORTUNITY_EVALUATED_ACTION
            )
        elif sequential_canary_readiness.get("sequential_safe_choice_calibrated"):
            training_readiness_status = POLICY_GATED_SEQUENTIAL_SAFE_CHOICE_CALIBRATED_ACTION
            recommended_next_action = POLICY_GATED_SEQUENTIAL_SAFE_CHOICE_CALIBRATED_ACTION
        else:
            training_readiness_status = POLICY_GATED_SEQUENTIAL_CANARY_ROLLOUT_EVALUATED_ACTION
            recommended_next_action = POLICY_GATED_SEQUENTIAL_CANARY_ROLLOUT_EVALUATED_ACTION
    elif policy_canary_readiness["present"] and policy_canary_readiness["completed"]:
        if policy_canary_readiness.get("canary_value_stability_passed"):
            training_readiness_status = POLICY_GATED_CANARY_VALUE_STABILITY_EVALUATED_ACTION
            recommended_next_action = POLICY_GATED_CANARY_VALUE_STABILITY_EVALUATED_ACTION
        elif policy_canary_readiness.get("canary_full_family_opportunity_passed"):
            training_readiness_status = (
                POLICY_GATED_CANARY_FULL_FAMILY_OPPORTUNITY_EVALUATED_ACTION
            )
            recommended_next_action = (
                POLICY_GATED_CANARY_FULL_FAMILY_OPPORTUNITY_EVALUATED_ACTION
            )
        elif policy_canary_readiness.get("canary_opportunity_quality_passed"):
            training_readiness_status = POLICY_GATED_CANARY_OPPORTUNITY_QUALITY_EVALUATED_ACTION
            recommended_next_action = POLICY_GATED_CANARY_OPPORTUNITY_QUALITY_EVALUATED_ACTION
        elif policy_canary_readiness.get("canary_diversity_passed"):
            training_readiness_status = POLICY_GATED_CANARY_DIVERSITY_EVALUATED_ACTION
            recommended_next_action = POLICY_GATED_CANARY_DIVERSITY_EVALUATED_ACTION
        else:
            training_readiness_status = POLICY_GATED_CANARY_ROLLOUT_EVALUATED_ACTION
            recommended_next_action = POLICY_GATED_CANARY_ROLLOUT_EVALUATED_ACTION
    elif raw_generalization_readiness["present"] and raw_generalization_readiness["completed"]:
        training_readiness_status = RAW_POLICY_GENERALIZATION_EVALUATED_ACTION
        recommended_next_action = RAW_POLICY_GENERALIZATION_EVALUATED_ACTION
    elif raw_strict_rollout_readiness["present"] and raw_strict_rollout_readiness["completed"]:
        training_readiness_status = RAW_POLICY_DECISION_ALIGNMENT_EVALUATED_ACTION
        recommended_next_action = RAW_POLICY_DECISION_ALIGNMENT_EVALUATED_ACTION
    elif scenario_rollout_readiness["present"] and scenario_rollout_readiness["completed"]:
        training_readiness_status = SCENARIO_DISJOINT_POLICY_ROLLOUT_EVALUATED_ACTION
        recommended_next_action = SCENARIO_DISJOINT_POLICY_ROLLOUT_EVALUATED_ACTION
    elif fresh_holdout_readiness["present"] and fresh_holdout_readiness["completed"]:
        if fresh_holdout_readiness.get("scenario_disjoint_completed"):
            training_readiness_status = SCENARIO_DISJOINT_POLICY_CANDIDATE_EVALUATED_ACTION
            recommended_next_action = SCENARIO_DISJOINT_POLICY_CANDIDATE_EVALUATED_ACTION
        else:
            training_readiness_status = FRESH_HOLDOUT_CANDIDATE_EVALUATED_ACTION
            recommended_next_action = FRESH_HOLDOUT_CANDIDATE_EVALUATED_ACTION
    elif controlled_candidate_readiness["present"] and controlled_candidate_readiness["completed"]:
        training_readiness_status = CONTROLLED_HYBRID_CANDIDATE_EVALUATED_ACTION
        recommended_next_action = CONTROLLED_HYBRID_CANDIDATE_EVALUATED_ACTION
    elif hybrid_training_readiness["present"] and hybrid_training_readiness["completed"]:
        training_readiness_status = HYBRID_DRY_RUN_COMPLETED_ACTION
        recommended_next_action = HYBRID_DRY_RUN_COMPLETED_ACTION
    else:
        training_readiness_status = READY_DRY_RUN_ACTION
        recommended_next_action = READY_DRY_RUN_ACTION

    excluded_candidate_count = rejected_goal_blocked_count + safety_regression_count + fallback_or_open_grid_count
    contract_status = "compatible_audit_only" if not contract_mutations else "contract_mutation_detected"
    return {
        "training_readiness_status": training_readiness_status,
        "source_selected_candidate_changed_rate": source_rate,
        "calibrated_selected_candidate_changed_rate": calibrated_rate,
        "calibrated_selection_rate_delta": calibrated_rate - source_rate,
        "applied_calibrated_candidate_count": applied_count,
        "changed_scenario_ids": changed_scenario_ids,
        "rejected_goal_blocked_count": rejected_goal_blocked_count,
        "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
        "platform_goal_trainable_anchor_projection_count": platform_goal_trainable_anchor_projection_count,
        "platform_goal_nontrainable_blocked_target_count": platform_goal_nontrainable_blocked_target_count,
        "platform_goal_anchor_available_count": platform_goal_anchor_available_count,
        "platform_goal_unresolved_count": platform_goal_unresolved_count,
        "platform_goal_feasibility_class_counts": platform_goal_class_counts,
        "safety_regression_count": safety_regression_count,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "training_positive_candidate_count": applied_count,
        "excluded_candidate_count": excluded_candidate_count,
        "anchor_projection_readiness": anchor_projection_readiness,
        "hybrid_training_readiness": hybrid_training_readiness,
        "controlled_hybrid_training_candidate_readiness": controlled_candidate_readiness,
        "fresh_holdout_policy_candidate_readiness": fresh_holdout_readiness,
        "scenario_disjoint_policy_rollout_readiness": scenario_rollout_readiness,
        "raw_policy_strict_rollout_readiness": raw_strict_rollout_readiness,
        "raw_policy_generalization_readiness": raw_generalization_readiness,
        "policy_gated_canary_rollout_readiness": policy_canary_readiness,
        "policy_gated_sequential_canary_rollout_readiness": sequential_canary_readiness,
        "ppo_rollout_collector_readiness": ppo_collector_readiness,
        "limited_ppo_update_smoke_readiness": limited_ppo_update_smoke_readiness,
        "limited_quasi_real_ppo_update_smoke_readiness": (
            limited_quasi_real_ppo_update_smoke_readiness
        ),
        "generated_sequential_gate_metric_accounting_readiness": (
            generated_sequential_gate_metric_accounting_readiness
        ),
        "generated_sequential_long_horizon_teacher_skill_contract_readiness": (
            generated_sequential_long_horizon_teacher_skill_contract_readiness
        ),
        "limited_quasi_real_generated_sequential_blocker_overridden_by_long_horizon_contract": (
            limited_quasi_real_generated_sequential_override
        ),
        "iterative_ppo_mini_loop_stability_readiness": iterative_ppo_mini_loop_readiness,
        "return_aligned_guarded_multistep_collector_readiness": (
            return_aligned_guarded_multistep_collector_readiness
        ),
        "return_aligned_guarded_ppo_update_smoke_readiness": (
            return_aligned_guarded_ppo_update_smoke_readiness
        ),
        "guarded_ppo_rollout_pilot_readiness": guarded_ppo_rollout_pilot_readiness,
        "quasi_real_guarded_ppo_rollout_pilot_readiness": (
            quasi_real_guarded_ppo_rollout_pilot_readiness
        ),
        "quasi_real_guarded_ppo_stability_replay_readiness": (
            quasi_real_guarded_ppo_stability_replay_readiness
        ),
        "quasi_real_guarded_ppo_horizon5_batch_expansion_readiness": (
            quasi_real_guarded_ppo_horizon5_batch_expansion_readiness
        ),
        "quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness": (
            quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness
        ),
        "quasi_real_guarded_ppo_iterative_miniloop_stability_readiness": (
            quasi_real_guarded_ppo_iterative_miniloop_stability_readiness
        ),
        "quasi_real_guarded_formal_ppo_preflight_readiness": (
            quasi_real_guarded_formal_ppo_preflight_readiness
        ),
        "quasi_real_guarded_formal_ppo_rollout_canary_readiness": (
            quasi_real_guarded_formal_ppo_rollout_canary_readiness
        ),
        "quasi_real_guarded_formal_ppo_stability_holdout_validation_readiness": (
            quasi_real_guarded_formal_ppo_stability_holdout_readiness
        ),
        "policy_training_cuda_device_support_readiness": policy_training_cuda_device_support_readiness,
        "quasi_real_map_domain_gap_readiness": quasi_real_map_domain_gap_readiness,
        "quasi_real_shadow_policy_behavior_readiness": quasi_real_shadow_policy_behavior_readiness,
        "quasi_real_shadow_alignment_readiness": quasi_real_shadow_alignment_readiness,
        "quasi_real_guarded_policy_pilot_readiness": quasi_real_guarded_policy_pilot_readiness,
        "quasi_real_safe_alternative_opportunity_readiness": (
            quasi_real_safe_alternative_opportunity_readiness
        ),
        "quasi_real_safe_better_opportunity_expansion_readiness": (
            quasi_real_safe_better_opportunity_expansion_readiness
        ),
        "quasi_real_teacher_equivalent_validation_readiness": (
            quasi_real_teacher_equivalent_validation_readiness
        ),
        "quasi_real_teacher_distillation_readiness": (
            quasi_real_teacher_distillation_readiness
        ),
        "quasi_real_guarded_teacher_following_pilot_readiness": (
            quasi_real_guarded_teacher_following_pilot_readiness
        ),
        "anchor_projection_candidate_generation_trainable_count": anchor_projection_readiness[
            "candidate_generation_trainable_count"
        ],
        "anchor_projection_contract_trainable_count": anchor_projection_readiness["contract_trainable_count"],
        "anchor_projection_readiness_trainable_count": anchor_projection_readiness["readiness_trainable_count"],
        "anchor_projection_ppo_consumable_trainable_target_count": anchor_projection_readiness[
            "ppo_consumable_trainable_target_count"
        ],
        "anchor_projection_planner_validated_trainable_target_count": anchor_projection_readiness[
            "planner_validated_trainable_target_count"
        ],
        "anchor_projection_planner_validated_distance_exception_count": anchor_projection_readiness[
            "planner_validated_distance_exception_count"
        ],
        "anchor_projection_candidate_contract_alignment_gap_count": anchor_projection_readiness[
            "candidate_contract_alignment_gap_count"
        ],
        "anchor_projection_anchor_unreachable_count": anchor_projection_readiness["anchor_unreachable_count"],
        "anchor_projection_source_candidate_not_selected_count": anchor_projection_readiness[
            "source_candidate_not_selected_count"
        ],
        "anchor_projection_reachable_substitute_anchor_found_count": anchor_projection_readiness[
            "reachable_substitute_anchor_found_count"
        ],
        "anchor_projection_anchor_unreachable_repaired_by_reachable_substitute_count": (
            anchor_projection_readiness["anchor_unreachable_repaired_by_reachable_substitute_count"]
        ),
        "anchor_projection_true_geometry_unreachable_count": anchor_projection_readiness[
            "true_geometry_unreachable_count"
        ],
        "anchor_projection_source_selected_but_distance_rejected_count": anchor_projection_readiness[
            "source_selected_but_distance_rejected_count"
        ],
        "anchor_projection_distance_contract_rejected_source_selected_count": anchor_projection_readiness[
            "distance_contract_rejected_source_selected_count"
        ],
        "anchor_projection_distance_contract_rejected_by_distance_bin": anchor_projection_readiness[
            "distance_contract_rejected_by_distance_bin"
        ],
        "anchor_projection_source_candidate_not_selected_by_best_alternative_reason": (
            anchor_projection_readiness["source_candidate_not_selected_by_best_alternative_reason"]
        ),
        "anchor_projection_source_selection_quality_tradeoff_summary": anchor_projection_readiness[
            "source_selection_quality_tradeoff_summary"
        ],
        "anchor_projection_audit_proxy_positive_count": anchor_projection_readiness[
            "audit_proxy_positive_count"
        ],
        "training_blockers": training_blockers,
        "next_required_change": (
            quasi_real_guarded_teacher_following_pilot_readiness.get("next_required_change")
            or
            quasi_real_teacher_distillation_readiness.get("next_required_change")
            or
            quasi_real_teacher_equivalent_validation_readiness.get(
                "next_required_change"
            )
            or (
                None
                if quasi_real_teacher_equivalent_validation_readiness["present"]
                else quasi_real_safe_better_opportunity_expansion_readiness.get(
                    "next_required_change"
                )
            )
            or
            quasi_real_safe_alternative_opportunity_readiness.get("next_required_change")
            or
            quasi_real_guarded_policy_pilot_readiness.get("next_required_change")
            or
            quasi_real_shadow_alignment_readiness.get("next_required_change")
            or quasi_real_shadow_policy_behavior_readiness.get("next_required_change")
            or quasi_real_map_domain_gap_readiness.get("next_required_change")
            or quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness.get(
                "next_required_change"
            )
            or quasi_real_guarded_ppo_iterative_miniloop_stability_readiness.get(
                "next_required_change"
            )
            or quasi_real_guarded_ppo_horizon5_batch_expansion_readiness.get(
                "next_required_change"
            )
            or policy_training_cuda_device_support_readiness.get("next_required_change")
            or
            guarded_ppo_rollout_pilot_readiness.get("next_required_change")
            or quasi_real_guarded_ppo_rollout_pilot_readiness.get("next_required_change")
            or quasi_real_guarded_ppo_stability_replay_readiness.get("next_required_change")
            or iterative_ppo_mini_loop_readiness.get("next_required_change")
            or generated_sequential_gate_metric_accounting_readiness.get(
                "next_required_change"
            )
            or limited_ppo_update_smoke_readiness.get("next_required_change")
            or ppo_collector_readiness.get("next_required_change")
            or sequential_canary_readiness.get("next_required_change")
            or policy_canary_readiness.get("next_required_change")
            or raw_strict_rollout_readiness.get("next_required_change")
            or raw_generalization_readiness.get("next_required_change")
            or fresh_holdout_readiness.get("next_required_change")
            or scenario_rollout_readiness.get("next_required_change")
            or controlled_candidate_readiness.get("next_required_change")
        ),
        "contract_impact": {
            "training_contract_status": contract_status,
            "contract_mutations": contract_mutations,
            "source_policy_target_selection_improvement_claimed": bool(
                smoke.get(
                    "source_policy_target_selection_improvement_claimed",
                    calibration.get("source_supports_policy_target_selection_improvement_claim", False),
                )
            ),
            "calibrated_selection_only": source_rate == 0.0 and calibrated_rate > 0.0,
            "policy_training_scope": _policy_training_scope(recommended_next_action),
        },
        "recommended_next_action": recommended_next_action,
        "readiness_source_status": {
            "smoke_recommended_next_action": str(smoke.get("recommended_next_action", "")),
            "readiness_status": str(readiness.get("readiness_status", "")),
            "calibrated_readiness_status": str(readiness.get("calibrated_readiness_status", "")),
            "coverage_recommended_next_action": str(coverage.get("recommended_next_action", "")),
        },
    }


def _policy_training_scope(recommended_next_action: str) -> str:
    if recommended_next_action == QUASI_REAL_GUARDED_PPO_STABILITY_REPLAY_EVALUATED_ACTION:
        return "quasi_real_guarded_ppo_stability_replay_only"
    if recommended_next_action == QUASI_REAL_GUARDED_TEACHER_FOLLOWING_PILOT_EVALUATED_ACTION:
        return "quasi_real_guarded_teacher_following_pilot_only"
    if recommended_next_action == QUASI_REAL_GUARDED_PPO_ROLLOUT_PILOT_EVALUATED_ACTION:
        return "quasi_real_guarded_ppo_rollout_pilot_only"
    if recommended_next_action == QUASI_REAL_TEACHER_DISTILLATION_EVALUATED_ACTION:
        return "quasi_real_teacher_distillation_robustness_only"
    if recommended_next_action == QUASI_REAL_TEACHER_EQUIVALENT_VALIDATED_ACTION:
        return "quasi_real_teacher_equivalent_validation_only"
    if recommended_next_action == QUASI_REAL_SAFE_BETTER_OPPORTUNITY_EXPANDED_ACTION:
        return "quasi_real_safe_better_opportunity_expansion_only"
    if recommended_next_action == QUASI_REAL_SHADOW_ALIGNMENT_EVALUATED_ACTION:
        return "quasi_real_shadow_alignment_evaluation_only"
    if recommended_next_action == QUASI_REAL_SHADOW_POLICY_BEHAVIOR_AUDITED_ACTION:
        return "quasi_real_shadow_policy_behavior_audit_only"
    if recommended_next_action == QUASI_REAL_MAP_DOMAIN_GAP_EVALUATED_ACTION:
        return "quasi_real_map_domain_gap_evaluation_only"
    if recommended_next_action == POLICY_TRAINING_CUDA_DEVICE_SUPPORT_EVALUATED_ACTION:
        return "policy_training_cuda_device_support_only"
    if (
        recommended_next_action
        == QUASI_REAL_GUARDED_PPO_SCALE512_MULTISEED_PREFLIGHT_EVALUATED_ACTION
    ):
        return "quasi_real_guarded_ppo_scale512_multiseed_preflight_only"
    if recommended_next_action == QUASI_REAL_GUARDED_PPO_HORIZON5_BATCH_EXPANSION_EVALUATED_ACTION:
        return "quasi_real_guarded_ppo_horizon5_batch_expansion_only"
    if recommended_next_action == QUASI_REAL_GUARDED_PPO_STABILITY_REPLAY_EVALUATED_ACTION:
        return "quasi_real_guarded_ppo_stability_replay_only"
    if recommended_next_action == QUASI_REAL_GUARDED_PPO_ROLLOUT_PILOT_EVALUATED_ACTION:
        return "quasi_real_guarded_ppo_rollout_pilot_only"
    if recommended_next_action == GUARDED_PPO_ROLLOUT_PILOT_EVALUATED_ACTION:
        return "guarded_ppo_rollout_pilot_only"
    if recommended_next_action == ITERATIVE_PPO_MINI_LOOP_STABILITY_EVALUATED_ACTION:
        return "iterative_ppo_mini_loop_stability_only"
    if recommended_next_action == LIMITED_QUASI_REAL_PPO_UPDATE_SMOKE_EVALUATED_ACTION:
        return "limited_quasi_real_ppo_update_smoke_only"
    if recommended_next_action == LIMITED_PPO_UPDATE_SMOKE_EVALUATED_ACTION:
        return "limited_ppo_update_smoke_only"
    if recommended_next_action == PPO_ROLLOUT_COLLECTOR_DRY_RUN_EVALUATED_ACTION:
        return "ppo_rollout_collector_dry_run_only"
    if recommended_next_action == POLICY_GATED_SEQUENTIAL_SAFE_CHOICE_CALIBRATED_ACTION:
        return "policy_gated_sequential_safe_choice_calibration_only"
    if recommended_next_action == POLICY_GATED_SEQUENTIAL_CANARY_ROLLOUT_EVALUATED_ACTION:
        return "policy_gated_sequential_canary_rollout_evaluation_only"
    if recommended_next_action == POLICY_GATED_CANARY_VALUE_STABILITY_EVALUATED_ACTION:
        return "policy_gated_canary_value_stability_evaluation_only"
    if recommended_next_action == POLICY_GATED_CANARY_FULL_FAMILY_OPPORTUNITY_EVALUATED_ACTION:
        return "policy_gated_canary_full_family_opportunity_evaluation_only"
    if recommended_next_action == POLICY_GATED_CANARY_OPPORTUNITY_QUALITY_EVALUATED_ACTION:
        return "policy_gated_canary_opportunity_quality_evaluation_only"
    if recommended_next_action == POLICY_GATED_CANARY_DIVERSITY_EVALUATED_ACTION:
        return "policy_gated_canary_diversity_evaluation_only"
    if recommended_next_action == POLICY_GATED_CANARY_ROLLOUT_EVALUATED_ACTION:
        return "policy_gated_canary_rollout_evaluation_only"
    if recommended_next_action == RAW_POLICY_GENERALIZATION_EVALUATED_ACTION:
        return "raw_policy_generalization_evaluation_only"
    if recommended_next_action == RAW_POLICY_DECISION_ALIGNMENT_EVALUATED_ACTION:
        return "raw_policy_decision_alignment_evaluation_only"
    if recommended_next_action == SCENARIO_DISJOINT_POLICY_ROLLOUT_EVALUATED_ACTION:
        return "scenario_disjoint_policy_rollout_evaluation_only"
    if recommended_next_action == SCENARIO_DISJOINT_POLICY_CANDIDATE_EVALUATED_ACTION:
        return "scenario_disjoint_policy_candidate_evaluation_only"
    if recommended_next_action == FRESH_HOLDOUT_CANDIDATE_EVALUATED_ACTION:
        return "fresh_holdout_policy_candidate_evaluation_only"
    if recommended_next_action == CONTROLLED_HYBRID_CANDIDATE_EVALUATED_ACTION:
        return "controlled_hybrid_training_candidate_evaluation_only"
    if recommended_next_action == HYBRID_DRY_RUN_COMPLETED_ACTION:
        return "hybrid_training_dry_run_only"
    if recommended_next_action == READY_DRY_RUN_ACTION:
        return "limited_policy_training_dry_run_only"
    return "audit_contract_refinement_only"


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    output_files = payload.get("output_files")
    if not isinstance(output_files, dict) or not _nonempty_string(
        output_files.get("policy_training_readiness_review_summary")
    ):
        raise ConfigError("output_files.policy_training_readiness_review_summary must be a non-empty string")
    thresholds = payload.get("readiness_thresholds")
    if not isinstance(thresholds, dict):
        raise ConfigError("readiness_thresholds must be an object")
    normalized_thresholds = {
        "min_applied_calibrated_candidate_count": _int_value(
            thresholds.get("min_applied_calibrated_candidate_count", 1),
            "readiness_thresholds.min_applied_calibrated_candidate_count",
        ),
        "min_calibrated_selection_rate_delta": _float_value(
            thresholds.get("min_calibrated_selection_rate_delta", 0.01),
            "readiness_thresholds.min_calibrated_selection_rate_delta",
        ),
        "max_rejected_goal_blocked_count": _int_value(
            thresholds.get("max_rejected_goal_blocked_count", 0),
            "readiness_thresholds.max_rejected_goal_blocked_count",
        ),
        "max_safety_regression_count": _int_value(
            thresholds.get("max_safety_regression_count", 0),
            "readiness_thresholds.max_safety_regression_count",
        ),
        "max_fallback_or_open_grid_count": _int_value(
            thresholds.get("max_fallback_or_open_grid_count", 0),
            "readiness_thresholds.max_fallback_or_open_grid_count",
        ),
        "max_anchor_projection_source_selection_quality_regression_count": _int_value(
            thresholds.get("max_anchor_projection_source_selection_quality_regression_count", 0),
            "readiness_thresholds.max_anchor_projection_source_selection_quality_regression_count",
        ),
        "max_anchor_projection_path_cost_regression": _optional_nonnegative_float(
            thresholds.get("max_anchor_projection_path_cost_regression"),
            "readiness_thresholds.max_anchor_projection_path_cost_regression",
        ),
        "max_anchor_projection_risk_regression": _optional_nonnegative_float(
            thresholds.get("max_anchor_projection_risk_regression"),
            "readiness_thresholds.max_anchor_projection_risk_regression",
        ),
        "require_smoke_ready_for_training_review": bool(
            thresholds.get("require_smoke_ready_for_training_review", True)
        ),
    }
    if normalized_thresholds["min_applied_calibrated_candidate_count"] < 0:
        raise ConfigError("readiness_thresholds.min_applied_calibrated_candidate_count must be >= 0")
    if normalized_thresholds["min_calibrated_selection_rate_delta"] < 0.0:
        raise ConfigError("readiness_thresholds.min_calibrated_selection_rate_delta must be >= 0")
    for key in (
        "max_rejected_goal_blocked_count",
        "max_safety_regression_count",
        "max_fallback_or_open_grid_count",
        "max_anchor_projection_source_selection_quality_regression_count",
    ):
        if normalized_thresholds[key] < 0:
            raise ConfigError(f"readiness_thresholds.{key} must be >= 0")
    config = dict(payload)
    config["readiness_thresholds"] = normalized_thresholds
    return config


def _load_source(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    repo_root: Path,
    reason_codes: list[str],
    source_summaries: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {"path": _display_path(path, repo_root), "exists": path.is_file()}
    if not path.is_file():
        _append_reason(reason_codes, f"{label}_missing")
        source_summaries[label] = record
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _append_reason(reason_codes, f"{label}_invalid_json")
        source_summaries[label] = record
        return {}
    if not isinstance(payload, dict):
        _append_reason(reason_codes, f"{label}_not_object")
        source_summaries[label] = record
        return {}
    record.update(
        {
            "schema_version": payload.get("schema_version"),
            "status": payload.get("status"),
            "reason_codes": _string_list(payload.get("reason_codes")),
        }
    )
    if payload.get("schema_version") != expected_schema:
        _append_reason(reason_codes, f"{label}_schema_mismatch")
    source_summaries[label] = record
    return payload


def _load_optional_source(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    repo_root: Path,
    reason_codes: list[str],
    source_summaries: dict[str, Any],
    required: bool = False,
) -> dict[str, Any]:
    if path.is_file() or required:
        return _load_source(
            path,
            label=label,
            expected_schema=expected_schema,
            repo_root=repo_root,
            reason_codes=reason_codes,
            source_summaries=source_summaries,
        )
    source_summaries[label] = {
        "path": _display_path(path, repo_root),
        "exists": False,
        "optional": True,
    }
    return {}


def _inspect_git(
    payload: dict[str, Any],
    *,
    label: str,
    current_git: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> bool:
    return _inspect_source_git_provenance(
        payload,
        label=label,
        current_git=current_git,
        require_current_git_match=_require_current_git_match(config),
        reason_codes=reason_codes,
        submodules=SUBMODULES,
        allow_dirty_current_git_match=_allow_dirty_current_git_match(config),
    )


def _source_rate(
    smoke: dict[str, Any],
    readiness: dict[str, Any],
    coverage: dict[str, Any],
    calibration: dict[str, Any],
) -> float:
    for payload, key in (
        (smoke, "source_selected_candidate_changed_rate"),
        (readiness, "source_selected_candidate_changed_rate"),
        (coverage, "source_selected_candidate_changed_rate"),
        (calibration, "source_selected_candidate_changed_rate"),
    ):
        if payload.get(key) is not None:
            return _float_value_or_default(payload.get(key), 0.0)
    return 0.0


def _calibrated_rate(
    smoke: dict[str, Any],
    readiness: dict[str, Any],
    coverage: dict[str, Any],
    calibration: dict[str, Any],
) -> float:
    for payload, key in (
        (smoke, "calibrated_selected_candidate_changed_rate"),
        (coverage, "calibrated_selected_candidate_changed_rate"),
        (readiness, "calibration_selected_candidate_changed_rate"),
        (calibration, "selected_candidate_changed_rate"),
    ):
        if payload.get(key) is not None:
            return _float_value_or_default(payload.get(key), 0.0)
    return 0.0


def _fallback_or_open_grid_count(payload: dict[str, Any]) -> int:
    return max(_int_value_or_default(payload.get(field), 0) for field in FALLBACK_COUNT_FIELDS)


def _contract_mutations(labeled_payloads: dict[str, dict[str, Any]]) -> list[str]:
    mutations: list[str] = []
    for label, payload in labeled_payloads.items():
        for field in CONTRACT_GUARD_FIELDS:
            if payload.get(field) is False:
                mutations.append(f"{label}.{field}")
    return sorted(set(mutations))


def _anchor_projection_readiness(
    *,
    candidate: dict[str, Any],
    contract: dict[str, Any],
    contract_aware_target: dict[str, Any],
    planner_validated_mining: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    candidate_present = bool(candidate)
    contract_present = bool(contract)
    candidate_trainable = _int_value_or_default(candidate.get("trainable_anchor_projection_count"), 0)
    candidate_nontrainable = _int_value_or_default(
        candidate.get("nontrainable_blocked_target_count"),
        _int_value_or_default(candidate.get("nontrainable_anchor_projection_count"), 0),
    )
    contract_trainable = _int_value_or_default(contract.get("trainable_anchor_projection_count"), 0)
    contract_nontrainable = _int_value_or_default(
        contract.get("nontrainable_blocked_target_count"),
        _int_value_or_default(contract.get("nontrainable_anchor_projection_count"), 0),
    )
    ppo_consumable_trainable_target_count = max(
        _int_value_or_default(candidate.get("ppo_consumable_trainable_target_count"), 0),
        _int_value_or_default(contract_aware_target.get("ppo_consumable_trainable_target_count"), 0),
        _int_value_or_default(
            planner_validated_mining.get("planner_validated_trainable_target_count"),
            0,
        ),
    )
    planner_validated_trainable_target_count = _int_value_or_default(
        planner_validated_mining.get("planner_validated_trainable_target_count"),
        0,
    )
    planner_validated_distance_exception_count = _int_value_or_default(
        planner_validated_mining.get("planner_validated_distance_exception_count"),
        0,
    )
    if candidate_present and contract_present:
        readiness_trainable = min(candidate_trainable, contract_trainable)
        alignment_gap = max(candidate_trainable - contract_trainable, 0)
    else:
        readiness_trainable = 0
        alignment_gap = candidate_trainable if candidate_present else 0
    alignment_gap = max(
        alignment_gap,
        _int_value_or_default(candidate.get("candidate_contract_alignment_gap_count"), 0),
        _int_value_or_default(contract_aware_target.get("candidate_contract_alignment_gap_count"), 0),
        _int_value_or_default(planner_validated_mining.get("candidate_contract_alignment_gap_count"), 0),
    )
    anchor_unreachable_count = _candidate_nontrainable_reason_count(candidate, "anchor_unreachable")
    source_candidate_not_selected_count = _candidate_nontrainable_reason_count(
        candidate,
        "source_candidate_not_selected",
    )
    reachable_substitute_anchor_found_count = max(
        _int_value_or_default(candidate.get("reachable_substitute_anchor_found_count"), 0),
        _int_value_or_default(contract.get("reachable_substitute_anchor_found_count"), 0),
        _coverage_diagnosis_int(candidate, "reachable_substitute_anchor_found_count"),
    )
    anchor_unreachable_repaired_count = max(
        _int_value_or_default(
            candidate.get("anchor_unreachable_repaired_by_reachable_substitute_count"),
            0,
        ),
        _int_value_or_default(
            contract.get("anchor_unreachable_repaired_by_reachable_substitute_count"),
            0,
        ),
        _coverage_diagnosis_int(
            candidate,
            "anchor_unreachable_repaired_by_reachable_substitute_count",
        ),
    )
    true_geometry_unreachable_count = max(
        _int_value_or_default(candidate.get("true_geometry_unreachable_count"), 0),
        _int_value_or_default(contract.get("true_geometry_unreachable_count"), 0),
        _coverage_diagnosis_int(candidate, "true_geometry_unreachable_count"),
    )
    source_selected_but_distance_rejected_count = max(
        _int_value_or_default(candidate.get("source_selected_but_distance_rejected_count"), 0),
        _int_value_or_default(contract.get("source_selected_but_distance_rejected_count"), 0),
        _coverage_diagnosis_int(candidate, "source_selected_but_distance_rejected_count"),
    )
    distance_contract_rejected_source_selected_count = max(
        _int_value_or_default(candidate.get("distance_contract_rejected_source_selected_count"), 0),
        _int_value_or_default(contract.get("distance_contract_rejected_source_selected_count"), 0),
        _coverage_diagnosis_int(candidate, "distance_contract_rejected_source_selected_count"),
    )
    distance_contract_rejected_by_distance_bin = _mapping_or_empty(
        candidate.get("distance_contract_rejected_by_distance_bin"),
        contract.get("distance_contract_rejected_by_distance_bin"),
        _coverage_diagnosis_mapping(candidate, "distance_contract_rejected_by_distance_bin"),
    )
    source_candidate_not_selected_by_best_alternative_reason = _mapping_or_empty(
        candidate.get("source_candidate_not_selected_by_best_alternative_reason"),
        contract.get("source_candidate_not_selected_by_best_alternative_reason"),
        _coverage_diagnosis_mapping(candidate, "source_candidate_not_selected_by_best_alternative_reason"),
    )
    source_selection_quality_tradeoff_summary = _mapping_or_empty(
        candidate.get("source_selection_quality_tradeoff_summary"),
        contract.get("source_selection_quality_tradeoff_summary"),
        _coverage_diagnosis_mapping(candidate, "source_selection_quality_tradeoff_summary"),
    )
    audit_proxy_positive_count = max(
        _int_value_or_default(candidate.get("positive_training_evidence_contains_audit_proxy_anchor_count"), 0),
        _int_value_or_default(candidate.get("audit_proxy_positive_count"), 0),
        _int_value_or_default(contract.get("positive_training_evidence_contains_audit_proxy_anchor_count"), 0),
        _int_value_or_default(contract.get("audit_proxy_positive_count"), 0),
    )
    source_quality_regression_count = _int_value_or_default(
        candidate.get("source_selection_quality_regression_count"),
        _quality_regression_count_from_contexts(candidate),
    )
    diagnostic_max_path_margin = _max_numeric(
        candidate.get("max_source_selection_path_cost_margin_vs_best_alternative"),
        _max_context_field(candidate, "source_selection_path_cost_margin_vs_best_alternative"),
    )
    diagnostic_max_risk_margin = _max_numeric(
        candidate.get("max_source_selection_risk_margin_vs_best_alternative"),
        _max_context_field(candidate, "source_selection_risk_margin_vs_best_alternative"),
    )
    max_path_margin = _max_context_field(
        candidate,
        "source_selection_path_cost_margin_vs_best_alternative",
        predicate=_context_margin_can_block_readiness,
    )
    max_risk_margin = _max_context_field(
        candidate,
        "source_selection_risk_margin_vs_best_alternative",
        predicate=_context_margin_can_block_readiness,
    )
    if source_quality_regression_count > 0:
        max_path_margin = _max_numeric(
            max_path_margin,
            candidate.get("max_source_selection_path_cost_margin_vs_best_alternative"),
        )
        max_risk_margin = _max_numeric(
            max_risk_margin,
            candidate.get("max_source_selection_risk_margin_vs_best_alternative"),
        )
    quality_blockers: list[str] = []
    training_blockers: list[str] = []
    max_quality_regressions = thresholds.get("max_anchor_projection_source_selection_quality_regression_count")
    if source_quality_regression_count > _int_value_or_default(max_quality_regressions, 0):
        _append_reason(quality_blockers, "anchor_projection_source_selection_quality_regression")
    max_allowed_path_margin = thresholds.get("max_anchor_projection_path_cost_regression")
    if (
        max_allowed_path_margin is not None
        and max_path_margin is not None
        and max_path_margin > float(max_allowed_path_margin)
    ):
        _append_reason(quality_blockers, "anchor_projection_source_selection_path_cost_regression")
    max_allowed_risk_margin = thresholds.get("max_anchor_projection_risk_regression")
    if (
        max_allowed_risk_margin is not None
        and max_risk_margin is not None
        and max_risk_margin > float(max_allowed_risk_margin)
    ):
        _append_reason(quality_blockers, "anchor_projection_source_selection_risk_regression")
    if contract_present and candidate_present and contract_trainable < candidate_trainable:
        _append_reason(training_blockers, "anchor_projection_contract_trainable_count_below_candidate_generation")
    if alignment_gap > 0:
        _append_reason(training_blockers, "anchor_projection_candidate_contract_alignment_gap")
    contract_aware_failures = _string_list(contract_aware_target.get("main_success_gate_failures"))
    contract_aware_next_required_change = contract_aware_target.get("next_required_change")
    if contract_aware_failures or contract_aware_next_required_change:
        readiness_impact = contract_aware_target.get("readiness_impact")
        readiness_impact = readiness_impact if isinstance(readiness_impact, dict) else {}
        recommended = _string_list(readiness_impact.get("recommended_training_blockers"))
        if not recommended:
            recommended = ["anchor_projection_nontrainable_contexts_remain"]
        for reason in recommended:
            _append_reason(training_blockers, reason)
    mining_failures = _string_list(planner_validated_mining.get("main_success_gate_failures"))
    mining_next_required_change = planner_validated_mining.get("next_required_change")
    if mining_failures or mining_next_required_change:
        readiness_impact = planner_validated_mining.get("readiness_impact")
        readiness_impact = readiness_impact if isinstance(readiness_impact, dict) else {}
        recommended = _string_list(readiness_impact.get("recommended_training_blockers"))
        if not recommended:
            recommended = ["anchor_projection_nontrainable_contexts_remain"]
        for reason in recommended:
            _append_reason(training_blockers, reason)
    elif planner_validated_mining:
        training_blockers = [
            reason
            for reason in training_blockers
            if reason != "anchor_projection_nontrainable_contexts_remain"
        ]
    if audit_proxy_positive_count > 0:
        _append_reason(training_blockers, "anchor_projection_positive_evidence_contains_audit_proxy_anchor")
    for reason in quality_blockers:
        _append_reason(training_blockers, reason)
    return {
        "candidate_generation_present": candidate_present,
        "contract_present": contract_present,
        "candidate_generation_trainable_count": candidate_trainable,
        "candidate_generation_nontrainable_count": candidate_nontrainable,
        "contract_trainable_count": contract_trainable,
        "contract_nontrainable_count": contract_nontrainable,
        "readiness_trainable_count": readiness_trainable,
        "ppo_consumable_trainable_target_count": ppo_consumable_trainable_target_count,
        "planner_validated_trainable_target_count": planner_validated_trainable_target_count,
        "planner_validated_distance_exception_count": planner_validated_distance_exception_count,
        "candidate_contract_alignment_gap_count": alignment_gap,
        "anchor_unreachable_count": anchor_unreachable_count,
        "source_candidate_not_selected_count": source_candidate_not_selected_count,
        "reachable_substitute_anchor_found_count": reachable_substitute_anchor_found_count,
        "anchor_unreachable_repaired_by_reachable_substitute_count": anchor_unreachable_repaired_count,
        "true_geometry_unreachable_count": true_geometry_unreachable_count,
        "source_selected_but_distance_rejected_count": source_selected_but_distance_rejected_count,
        "distance_contract_rejected_source_selected_count": (
            distance_contract_rejected_source_selected_count
        ),
        "distance_contract_rejected_by_distance_bin": distance_contract_rejected_by_distance_bin,
        "source_candidate_not_selected_by_best_alternative_reason": (
            source_candidate_not_selected_by_best_alternative_reason
        ),
        "source_selection_quality_tradeoff_summary": source_selection_quality_tradeoff_summary,
        "audit_proxy_positive_count": audit_proxy_positive_count,
        "source_selection_quality_regression_count": source_quality_regression_count,
        "max_source_selection_path_cost_margin_vs_best_alternative": max_path_margin,
        "max_source_selection_risk_margin_vs_best_alternative": max_risk_margin,
        "diagnostic_max_source_selection_path_cost_margin_vs_best_alternative": diagnostic_max_path_margin,
        "diagnostic_max_source_selection_risk_margin_vs_best_alternative": diagnostic_max_risk_margin,
        "quality_regression_blockers": quality_blockers,
        "training_blockers": training_blockers,
    }


def _hybrid_training_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {
            "present": False,
            "completed": False,
            "training_blockers": [],
            "formal_training_ready_claimed": False,
            "action_label_positive_count": 0,
            "existing_preference_pair_count": 0,
            "residual_preference_pair_count": 0,
            "pairwise_preference_signal_count": 0,
            "hybrid_train_signal_count": 0,
            "hard_positive_added_count": 0,
            "invalid_action_mask_count": 0,
            "empty_action_mask_count": 0,
            "publishes_checkpoint": False,
            "performance_claimed": False,
        }
    blockers: list[str] = []
    if summary.get("status") != "passed" or summary.get("dry_run_status") != "passed":
        _append_reason(blockers, "hybrid_training_dry_run_not_passed")
    if _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "hybrid_training_dry_run_reason_codes_present")
    if _int_value_or_default(summary.get("action_label_positive_count"), 0) != 24:
        _append_reason(blockers, "hybrid_action_label_positive_count_mismatch")
    if _int_value_or_default(summary.get("existing_preference_pair_count"), 0) != 24:
        _append_reason(blockers, "hybrid_existing_preference_pair_count_mismatch")
    if _int_value_or_default(summary.get("residual_preference_pair_count"), 0) != 30:
        _append_reason(blockers, "hybrid_residual_preference_pair_count_mismatch")
    if _int_value_or_default(summary.get("pairwise_preference_signal_count"), 0) != 54:
        _append_reason(blockers, "hybrid_pairwise_preference_signal_count_mismatch")
    if _int_value_or_default(summary.get("hybrid_train_signal_count"), 0) != 78:
        _append_reason(blockers, "hybrid_train_signal_count_mismatch")
    if _int_value_or_default(summary.get("hard_positive_added_count"), 0) != 0:
        _append_reason(blockers, "hybrid_hard_positive_added_count_nonzero")
    if _int_value_or_default(summary.get("invalid_action_mask_count"), 0) != 0:
        _append_reason(blockers, "hybrid_invalid_action_mask_count_nonzero")
    if _int_value_or_default(summary.get("empty_action_mask_count"), 0) != 0:
        _append_reason(blockers, "hybrid_empty_action_mask_count_nonzero")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "hybrid_checkpoint_publication_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "hybrid_policy_performance_claimed")
    formal_training_ready_claimed = bool(
        summary.get("formal_training_ready_claimed")
        or summary.get("policy_training_ready")
        or summary.get("performance_claimed")
    )
    if formal_training_ready_claimed:
        _append_reason(blockers, "hybrid_formal_training_ready_claimed")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "formal_training_ready_claimed": formal_training_ready_claimed,
        "action_label_positive_count": _int_value_or_default(
            summary.get("action_label_positive_count"),
            0,
        ),
        "existing_preference_pair_count": _int_value_or_default(
            summary.get("existing_preference_pair_count"),
            0,
        ),
        "residual_preference_pair_count": _int_value_or_default(
            summary.get("residual_preference_pair_count"),
            0,
        ),
        "pairwise_preference_signal_count": _int_value_or_default(
            summary.get("pairwise_preference_signal_count"),
            0,
        ),
        "hybrid_train_signal_count": _int_value_or_default(
            summary.get("hybrid_train_signal_count"),
            0,
        ),
        "hard_positive_added_count": _int_value_or_default(
            summary.get("hard_positive_added_count"),
            0,
        ),
        "invalid_action_mask_count": _int_value_or_default(
            summary.get("invalid_action_mask_count"),
            0,
        ),
        "empty_action_mask_count": _int_value_or_default(
            summary.get("empty_action_mask_count"),
            0,
        ),
        "publishes_checkpoint": bool(summary.get("publishes_checkpoint")),
        "performance_claimed": bool(summary.get("performance_claimed")),
    }


def _controlled_hybrid_training_candidate_readiness(
    *,
    candidate: dict[str, Any],
    holdout: dict[str, Any],
    allow_fresh_holdout_substitute: bool = False,
) -> dict[str, Any]:
    if not candidate and not holdout:
        return {
            "present": False,
            "completed": False,
            "training_blockers": [],
            "next_required_change": None,
            "formal_training_ready_claimed": False,
            "performance_claimed": False,
            "action_label_positive_count": 0,
            "pairwise_preference_signal_count": 0,
            "hybrid_train_signal_count": 0,
            "hard_positive_added_count": 0,
            "action_mask_invalid_count": 0,
            "empty_action_mask_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
            "preference_margin_improved_count": 0,
        }
    blockers: list[str] = []
    if not candidate:
        _append_reason(blockers, "controlled_hybrid_training_candidate_summary_missing")
    if not holdout and not allow_fresh_holdout_substitute:
        _append_reason(blockers, "controlled_hybrid_holdout_evaluation_summary_missing")
    if candidate and (
        candidate.get("status") != "passed"
        or candidate.get("candidate_training_status") != "passed"
        or _string_list(candidate.get("reason_codes"))
    ):
        _append_reason(blockers, "controlled_hybrid_training_candidate_not_passed")
    if holdout and (holdout.get("status") != "passed" or _string_list(holdout.get("reason_codes"))):
        _append_reason(blockers, "controlled_hybrid_holdout_evaluation_not_passed")
    if _int_value_or_default(candidate.get("action_label_positive_count"), 0) != 24:
        _append_reason(blockers, "controlled_hybrid_action_label_positive_count_mismatch")
    if _int_value_or_default(candidate.get("pairwise_preference_signal_count"), 0) != 54:
        _append_reason(blockers, "controlled_hybrid_pairwise_preference_signal_count_mismatch")
    if _int_value_or_default(candidate.get("hybrid_train_signal_count"), 0) != 78:
        _append_reason(blockers, "controlled_hybrid_train_signal_count_mismatch")
    if _int_value_or_default(candidate.get("hard_positive_added_count"), 0) != 0:
        _append_reason(blockers, "controlled_hybrid_hard_positive_added_count_nonzero")
    if candidate.get("experimental_checkpoint") is not True:
        _append_reason(blockers, "controlled_hybrid_checkpoint_not_experimental")
    if candidate.get("publishes_checkpoint") is True or holdout.get("publishes_checkpoint") is True:
        _append_reason(blockers, "controlled_hybrid_checkpoint_publication_claimed")
    if candidate.get("replaces_default_policy") is True or holdout.get("replaces_default_policy") is True:
        _append_reason(blockers, "controlled_hybrid_default_policy_replacement_claimed")
    if candidate.get("performance_claimed") is True or holdout.get("performance_claimed") is True:
        _append_reason(blockers, "controlled_hybrid_policy_performance_claimed")

    action_mask_invalid_count = max(
        _int_value_or_default(candidate.get("invalid_action_mask_count"), 0),
        _int_value_or_default(holdout.get("action_mask_invalid_count"), 0),
    )
    empty_action_mask_count = max(
        _int_value_or_default(candidate.get("empty_action_mask_count"), 0),
        _int_value_or_default(holdout.get("empty_action_mask_count"), 0),
    )
    fallback_or_open_grid_count = _int_value_or_default(
        holdout.get("fallback_or_open_grid_count"),
        0,
    )
    safety_regression_count = _int_value_or_default(holdout.get("safety_regression_count"), 0)
    contract_violation_count = _int_value_or_default(holdout.get("contract_violation_count"), 0)
    path_cost_regression_count = _int_value_or_default(holdout.get("path_cost_regression_count"), 0)
    risk_regression_count = _int_value_or_default(holdout.get("risk_regression_count"), 0)
    source_selection_regression_count = _int_value_or_default(
        holdout.get("source_selection_regression_count"),
        0,
    )
    if action_mask_invalid_count:
        _append_reason(blockers, "controlled_hybrid_holdout_action_mask_invalid")
    if empty_action_mask_count:
        _append_reason(blockers, "controlled_hybrid_holdout_empty_action_mask")
    if fallback_or_open_grid_count:
        _append_reason(blockers, "controlled_hybrid_holdout_fallback_or_open_grid")
    if safety_regression_count:
        _append_reason(blockers, "controlled_hybrid_holdout_safety_regression")
    if contract_violation_count:
        _append_reason(blockers, "controlled_hybrid_holdout_contract_violation")
    if path_cost_regression_count:
        _append_reason(blockers, "controlled_hybrid_holdout_path_cost_regression")
    if risk_regression_count:
        _append_reason(blockers, "controlled_hybrid_holdout_risk_regression")
    if source_selection_regression_count:
        _append_reason(blockers, "controlled_hybrid_holdout_source_selection_regression")

    next_required_change = (
        CONTROLLED_HYBRID_NEXT_REQUIRED_CHANGE
        if path_cost_regression_count or risk_regression_count or source_selection_regression_count
        else holdout.get("next_required_change")
    )
    formal_training_ready_claimed = bool(
        candidate.get("formal_training_ready_claimed")
        or holdout.get("formal_training_ready_claimed")
        or candidate.get("policy_training_ready")
        or holdout.get("policy_training_ready")
        or candidate.get("performance_claimed")
        or holdout.get("performance_claimed")
    )
    if formal_training_ready_claimed:
        _append_reason(blockers, "controlled_hybrid_formal_training_ready_claimed")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": next_required_change,
        "formal_training_ready_claimed": formal_training_ready_claimed,
        "performance_claimed": bool(candidate.get("performance_claimed") or holdout.get("performance_claimed")),
        "action_label_positive_count": _int_value_or_default(
            candidate.get("action_label_positive_count"),
            0,
        ),
        "pairwise_preference_signal_count": _int_value_or_default(
            candidate.get("pairwise_preference_signal_count"),
            0,
        ),
        "hybrid_train_signal_count": _int_value_or_default(candidate.get("hybrid_train_signal_count"), 0),
        "hard_positive_added_count": _int_value_or_default(candidate.get("hard_positive_added_count"), 0),
        "action_mask_invalid_count": action_mask_invalid_count,
        "empty_action_mask_count": empty_action_mask_count,
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "contract_violation_count": contract_violation_count,
        "path_cost_regression_count": path_cost_regression_count,
        "risk_regression_count": risk_regression_count,
        "source_selection_regression_count": source_selection_regression_count,
        "source_selection_agreement_count": _int_value_or_default(
            holdout.get("source_selection_agreement_count"),
            0,
        ),
        "preference_margin_improved_count": _int_value_or_default(
            holdout.get("preference_margin_improved_count"),
            0,
        ),
        "holdout_substituted_by_fresh_holdout": bool(
            allow_fresh_holdout_substitute and not holdout
        ),
        "publishes_checkpoint": bool(candidate.get("publishes_checkpoint") or holdout.get("publishes_checkpoint")),
        "replaces_default_policy": bool(
            candidate.get("replaces_default_policy") or holdout.get("replaces_default_policy")
        ),
    }


def _fresh_holdout_policy_candidate_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {
            "present": False,
            "completed": False,
            "training_blockers": [],
            "next_required_change": None,
            "formal_training_ready_claimed": False,
            "performance_claimed": False,
            "scenario_disjoint": False,
            "scenario_disjoint_completed": False,
            "context_id_missing_count": 0,
            "legacy_identity_fallback_count": 0,
            "context_id_coverage_rate": 0.0,
            "candidate_git_current_matches_sources": False,
            "checkpoint_metadata_git_current_matches_sources": False,
            "fresh_disjoint_context_count": 0,
            "identity_overlap_count": 0,
            "identity_key_missing_count": 0,
            "scenario_overlap_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
            "preference_margin_satisfied_count": 0,
            "preference_margin_failed_count": 0,
        }
    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "fresh_holdout_policy_candidate_evaluation_not_passed")
    if _int_value_or_default(summary.get("fresh_disjoint_context_count"), 0) <= 0:
        _append_reason(blockers, "fresh_holdout_disjoint_context_count_zero")
    if _int_value_or_default(summary.get("accepted_identity_overlap_count"), 0) != 0:
        _append_reason(blockers, "fresh_holdout_accepted_identity_overlap")
    if _int_value_or_default(summary.get("accepted_identity_key_missing_count"), 0) != 0:
        _append_reason(blockers, "fresh_holdout_accepted_identity_key_missing")
    scenario_disjoint = bool(summary.get("scenario_disjoint"))
    require_context_id = summary.get("require_context_id") is True
    require_scenario_disjoint = summary.get("require_scenario_disjoint") is True
    context_id_missing_count = _int_value_or_default(summary.get("context_id_missing_count"), 0)
    legacy_identity_fallback_count = _int_value_or_default(
        summary.get("legacy_identity_fallback_count"),
        0,
    )
    candidate_git_current_matches_sources = summary.get("candidate_git_current_matches_sources") is not False
    checkpoint_metadata_git_current_matches_sources = (
        summary.get("checkpoint_metadata_git_current_matches_sources") is not False
    )
    scenario_disjoint_completed = (
        require_context_id
        and require_scenario_disjoint
        and scenario_disjoint
        and _int_value_or_default(summary.get("scenario_overlap_count"), 0) == 0
        and _int_value_or_default(summary.get("identity_overlap_count"), 0) == 0
        and context_id_missing_count == 0
        and legacy_identity_fallback_count == 0
        and candidate_git_current_matches_sources
        and checkpoint_metadata_git_current_matches_sources
    )
    if require_scenario_disjoint and summary.get("scenario_disjoint") is False:
        _append_reason(blockers, "fresh_holdout_scenario_overlap")
    if require_context_id and context_id_missing_count:
        _append_reason(blockers, "fresh_holdout_context_id_missing")
    if require_context_id and legacy_identity_fallback_count:
        _append_reason(blockers, "fresh_holdout_legacy_identity_fallback_used")
    if require_scenario_disjoint and not candidate_git_current_matches_sources:
        _append_reason(blockers, "fresh_holdout_candidate_git_current_mismatch")
    if require_scenario_disjoint and not checkpoint_metadata_git_current_matches_sources:
        _append_reason(blockers, "fresh_holdout_checkpoint_metadata_git_current_mismatch")
    if summary.get("experimental_checkpoint") is not True:
        _append_reason(blockers, "fresh_holdout_checkpoint_not_experimental")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "fresh_holdout_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "fresh_holdout_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "fresh_holdout_policy_performance_claimed")

    fallback_or_open_grid_count = _int_value_or_default(summary.get("fallback_or_open_grid_count"), 0)
    safety_regression_count = _int_value_or_default(summary.get("safety_regression_count"), 0)
    contract_violation_count = _int_value_or_default(summary.get("contract_violation_count"), 0)
    path_cost_regression_count = _int_value_or_default(summary.get("path_cost_regression_count"), 0)
    risk_regression_count = _int_value_or_default(summary.get("risk_regression_count"), 0)
    source_selection_regression_count = _int_value_or_default(
        summary.get("source_selection_regression_count"),
        0,
    )
    if fallback_or_open_grid_count:
        _append_reason(blockers, "fresh_holdout_fallback_or_open_grid")
    if safety_regression_count:
        _append_reason(blockers, "fresh_holdout_safety_regression")
    if contract_violation_count:
        _append_reason(blockers, "fresh_holdout_contract_violation")
    if path_cost_regression_count:
        _append_reason(blockers, "fresh_holdout_path_cost_regression")
    if risk_regression_count:
        _append_reason(blockers, "fresh_holdout_risk_regression")
    if source_selection_regression_count:
        _append_reason(blockers, "fresh_holdout_source_selection_regression")

    formal_training_ready_claimed = bool(
        summary.get("formal_training_ready_claimed")
        or summary.get("policy_training_ready")
        or summary.get("performance_claimed")
    )
    if formal_training_ready_claimed:
        _append_reason(blockers, "fresh_holdout_formal_training_ready_claimed")
    next_required_change = (
        summary.get("next_required_change")
        if blockers
        else None
    )
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": next_required_change,
        "formal_training_ready_claimed": formal_training_ready_claimed,
        "performance_claimed": bool(summary.get("performance_claimed")),
        "scenario_disjoint": scenario_disjoint,
        "require_context_id": require_context_id,
        "require_scenario_disjoint": require_scenario_disjoint,
        "scenario_disjoint_completed": scenario_disjoint_completed,
        "context_id_missing_count": context_id_missing_count,
        "legacy_identity_fallback_count": legacy_identity_fallback_count,
        "context_id_coverage_rate": float(summary.get("context_id_coverage_rate") or 0.0),
        "candidate_git_current_matches_sources": candidate_git_current_matches_sources,
        "checkpoint_metadata_git_current_matches_sources": (
            checkpoint_metadata_git_current_matches_sources
        ),
        "fresh_disjoint_context_count": _int_value_or_default(
            summary.get("fresh_disjoint_context_count"),
            0,
        ),
        "identity_overlap_count": _int_value_or_default(summary.get("identity_overlap_count"), 0),
        "identity_key_missing_count": _int_value_or_default(summary.get("identity_key_missing_count"), 0),
        "scenario_overlap_count": _int_value_or_default(summary.get("scenario_overlap_count"), 0),
        "fallback_or_open_grid_count": fallback_or_open_grid_count,
        "safety_regression_count": safety_regression_count,
        "contract_violation_count": contract_violation_count,
        "path_cost_regression_count": path_cost_regression_count,
        "risk_regression_count": risk_regression_count,
        "source_selection_regression_count": source_selection_regression_count,
        "preference_margin_satisfied_count": _int_value_or_default(
            summary.get("preference_margin_satisfied_count"),
            0,
        ),
        "preference_margin_failed_count": _int_value_or_default(
            summary.get("preference_margin_failed_count"),
            0,
        ),
        "publishes_checkpoint": bool(summary.get("publishes_checkpoint")),
        "replaces_default_policy": bool(summary.get("replaces_default_policy")),
    }


def _scenario_disjoint_policy_rollout_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {
            "present": False,
            "completed": False,
            "training_blockers": [],
            "next_required_change": None,
            "formal_training_ready_claimed": False,
            "performance_claimed": False,
            "scenario_disjoint_context_count": 0,
            "policy_decision_count": 0,
            "decision_changed_count": 0,
            "aligned_decision_count": 0,
            "acceptable_alternative_count": 0,
            "regression_count": 0,
            "invalid_action_mask_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
        }
    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "scenario_disjoint_policy_rollout_evaluation_not_passed")
    if _int_value_or_default(summary.get("scenario_disjoint_context_count"), 0) <= 0:
        _append_reason(blockers, "scenario_disjoint_policy_rollout_context_count_zero")
    if _int_value_or_default(summary.get("policy_decision_count"), 0) <= 0:
        _append_reason(blockers, "scenario_disjoint_policy_rollout_decision_count_zero")
    if _int_value_or_default(summary.get("invalid_action_mask_count"), 0):
        _append_reason(blockers, "scenario_disjoint_policy_rollout_invalid_action_mask")
    if _int_value_or_default(summary.get("fallback_or_open_grid_count"), 0):
        _append_reason(blockers, "scenario_disjoint_policy_rollout_fallback_or_open_grid")
    if _int_value_or_default(summary.get("safety_regression_count"), 0):
        _append_reason(blockers, "scenario_disjoint_policy_rollout_safety_regression")
    if _int_value_or_default(summary.get("contract_violation_count"), 0):
        _append_reason(blockers, "scenario_disjoint_policy_rollout_contract_violation")
    if _int_value_or_default(summary.get("path_cost_regression_count"), 0):
        _append_reason(blockers, "scenario_disjoint_policy_rollout_path_cost_regression")
    if _int_value_or_default(summary.get("risk_regression_count"), 0):
        _append_reason(blockers, "scenario_disjoint_policy_rollout_risk_regression")
    if _int_value_or_default(summary.get("source_selection_regression_count"), 0):
        _append_reason(blockers, "scenario_disjoint_policy_rollout_source_selection_regression")
    if _int_value_or_default(summary.get("regression_count"), 0):
        _append_reason(blockers, "scenario_disjoint_policy_rollout_regression")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "scenario_disjoint_policy_rollout_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "scenario_disjoint_policy_rollout_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "scenario_disjoint_policy_rollout_policy_performance_claimed")
    if summary.get("candidate_git_current_matches_sources") is False:
        _append_reason(blockers, "scenario_disjoint_policy_rollout_candidate_git_current_mismatch")
    if summary.get("checkpoint_metadata_git_current_matches_sources") is False:
        _append_reason(blockers, "scenario_disjoint_policy_rollout_checkpoint_metadata_git_current_mismatch")

    formal_training_ready_claimed = bool(
        summary.get("formal_training_ready_claimed")
        or summary.get("policy_training_ready")
        or summary.get("performance_claimed")
    )
    if formal_training_ready_claimed:
        _append_reason(blockers, "scenario_disjoint_policy_rollout_formal_training_ready_claimed")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "formal_training_ready_claimed": formal_training_ready_claimed,
        "performance_claimed": bool(summary.get("performance_claimed")),
        "scenario_disjoint_context_count": _int_value_or_default(summary.get("scenario_disjoint_context_count"), 0),
        "policy_decision_count": _int_value_or_default(summary.get("policy_decision_count"), 0),
        "decision_changed_count": _int_value_or_default(summary.get("decision_changed_count"), 0),
        "aligned_decision_count": _int_value_or_default(summary.get("aligned_decision_count"), 0),
        "acceptable_alternative_count": _int_value_or_default(summary.get("acceptable_alternative_count"), 0),
        "regression_count": _int_value_or_default(summary.get("regression_count"), 0),
        "raw_policy_regression_count": _int_value_or_default(summary.get("raw_policy_regression_count"), 0),
        "invalid_action_mask_count": _int_value_or_default(summary.get("invalid_action_mask_count"), 0),
        "fallback_or_open_grid_count": _int_value_or_default(summary.get("fallback_or_open_grid_count"), 0),
        "safety_regression_count": _int_value_or_default(summary.get("safety_regression_count"), 0),
        "contract_violation_count": _int_value_or_default(summary.get("contract_violation_count"), 0),
        "path_cost_regression_count": _int_value_or_default(summary.get("path_cost_regression_count"), 0),
        "risk_regression_count": _int_value_or_default(summary.get("risk_regression_count"), 0),
        "source_selection_regression_count": _int_value_or_default(
            summary.get("source_selection_regression_count"),
            0,
        ),
    }


def _raw_policy_strict_rollout_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "formal_training_ready_claimed": False,
        "performance_claimed": False,
        "scenario_disjoint_context_count": 0,
        "policy_decision_count": 0,
        "regression_count": 0,
        "raw_policy_regression_count": 0,
        "invalid_action_mask_count": 0,
        "fallback_or_open_grid_count": 0,
        "safety_regression_count": 0,
        "contract_violation_count": 0,
        "path_cost_regression_count": 0,
        "risk_regression_count": 0,
        "source_selection_regression_count": 0,
    }
    if not summary:
        return empty
    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "raw_policy_strict_rollout_evaluation_not_passed")
    if _int_value_or_default(summary.get("scenario_disjoint_context_count"), 0) <= 0:
        _append_reason(blockers, "raw_policy_strict_rollout_context_count_zero")
    if _int_value_or_default(summary.get("policy_decision_count"), 0) <= 0:
        _append_reason(blockers, "raw_policy_strict_rollout_decision_count_zero")
    if summary.get("raw_policy_alignment_improved") is False:
        _append_reason(blockers, "raw_policy_alignment_not_improved")
    if _int_value_or_default(summary.get("regression_count"), 0):
        _append_reason(blockers, "raw_policy_controlled_rollout_regression")
    if _int_value_or_default(summary.get("invalid_action_mask_count"), 0):
        _append_reason(blockers, "raw_policy_strict_rollout_invalid_action_mask")
    if _int_value_or_default(summary.get("fallback_or_open_grid_count"), 0):
        _append_reason(blockers, "raw_policy_strict_rollout_fallback_or_open_grid")
    if _int_value_or_default(summary.get("safety_regression_count"), 0):
        _append_reason(blockers, "raw_policy_strict_rollout_safety_regression")
    if _int_value_or_default(summary.get("contract_violation_count"), 0):
        _append_reason(blockers, "raw_policy_strict_rollout_contract_violation")
    if _int_value_or_default(summary.get("path_cost_regression_count"), 0):
        _append_reason(blockers, "raw_policy_strict_rollout_path_cost_regression")
    if _int_value_or_default(summary.get("risk_regression_count"), 0):
        _append_reason(blockers, "raw_policy_strict_rollout_risk_regression")
    if _int_value_or_default(summary.get("source_selection_regression_count"), 0):
        _append_reason(blockers, "raw_policy_strict_rollout_source_selection_regression")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "raw_policy_strict_rollout_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "raw_policy_strict_rollout_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "raw_policy_strict_rollout_policy_performance_claimed")
    if summary.get("candidate_git_current_matches_sources") is False:
        _append_reason(blockers, "raw_policy_strict_rollout_candidate_git_current_mismatch")
    if summary.get("checkpoint_metadata_git_current_matches_sources") is False:
        _append_reason(blockers, "raw_policy_strict_rollout_checkpoint_metadata_git_current_mismatch")

    formal_training_ready_claimed = bool(
        summary.get("formal_training_ready_claimed")
        or summary.get("policy_training_ready")
        or summary.get("performance_claimed")
    )
    if formal_training_ready_claimed:
        _append_reason(blockers, "raw_policy_strict_rollout_formal_training_ready_claimed")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": (
            summary.get("next_required_change")
            if blockers
            else None
        )
        or (RAW_POLICY_ALIGNMENT_NEXT_REQUIRED_CHANGE if blockers else None),
        "formal_training_ready_claimed": formal_training_ready_claimed,
        "performance_claimed": bool(summary.get("performance_claimed")),
        "scenario_disjoint_context_count": _int_value_or_default(
            summary.get("scenario_disjoint_context_count"),
            0,
        ),
        "policy_decision_count": _int_value_or_default(summary.get("policy_decision_count"), 0),
        "regression_count": _int_value_or_default(summary.get("regression_count"), 0),
        "raw_policy_regression_count": _int_value_or_default(
            summary.get("raw_policy_regression_count"),
            0,
        ),
        "invalid_action_mask_count": _int_value_or_default(summary.get("invalid_action_mask_count"), 0),
        "fallback_or_open_grid_count": _int_value_or_default(
            summary.get("fallback_or_open_grid_count"),
            0,
        ),
        "safety_regression_count": _int_value_or_default(summary.get("safety_regression_count"), 0),
        "contract_violation_count": _int_value_or_default(summary.get("contract_violation_count"), 0),
        "path_cost_regression_count": _int_value_or_default(summary.get("path_cost_regression_count"), 0),
        "risk_regression_count": _int_value_or_default(summary.get("risk_regression_count"), 0),
        "source_selection_regression_count": _int_value_or_default(
            summary.get("source_selection_regression_count"),
            0,
        ),
    }


def _raw_policy_generalization_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "formal_training_ready_claimed": False,
        "performance_claimed": False,
        "test_generalization_passed": False,
        "test_raw_policy_regression_reduction_rate": 0.0,
        "overfit_gap": 0.0,
        "test_regression_count": 0,
        "test_invalid_action_mask_count": 0,
        "test_fallback_or_open_grid_count": 0,
        "test_safety_regression_count": 0,
        "test_contract_violation_count": 0,
        "test_path_cost_regression_count": 0,
        "test_risk_regression_count": 0,
        "test_source_selection_regression_count": 0,
    }
    if not summary:
        return empty
    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "raw_policy_generalization_evaluation_not_passed")
    if summary.get("test_generalization_passed") is not True:
        _append_reason(blockers, "raw_policy_generalization_test_not_passed")
    if _float_value_or_default(summary.get("test_raw_policy_regression_reduction_rate"), 0.0) < 0.5:
        _append_reason(blockers, "raw_policy_generalization_reduction_below_threshold")
    if _float_value_or_default(summary.get("overfit_gap"), 0.0) > 0.15:
        _append_reason(blockers, "raw_policy_generalization_overfit_gap")
    for field, reason in (
        ("test_regression_count", "raw_policy_generalization_controlled_regression"),
        ("test_invalid_action_mask_count", "raw_policy_generalization_invalid_action_mask"),
        ("test_fallback_or_open_grid_count", "raw_policy_generalization_fallback_or_open_grid"),
        ("test_safety_regression_count", "raw_policy_generalization_safety_regression"),
        ("test_contract_violation_count", "raw_policy_generalization_contract_violation"),
        ("test_path_cost_regression_count", "raw_policy_generalization_path_cost_regression"),
        ("test_risk_regression_count", "raw_policy_generalization_risk_regression"),
        ("test_source_selection_regression_count", "raw_policy_generalization_source_selection_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "raw_policy_generalization_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "raw_policy_generalization_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "raw_policy_generalization_policy_performance_claimed")

    formal_training_ready_claimed = bool(
        summary.get("formal_training_ready_claimed")
        or summary.get("policy_training_ready")
        or summary.get("performance_claimed")
    )
    if formal_training_ready_claimed:
        _append_reason(blockers, "raw_policy_generalization_formal_training_ready_claimed")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": (
            summary.get("next_required_change")
            if blockers
            else None
        )
        or (RAW_POLICY_GENERALIZATION_NEXT_REQUIRED_CHANGE if blockers else None),
        "formal_training_ready_claimed": formal_training_ready_claimed,
        "performance_claimed": bool(summary.get("performance_claimed")),
        "test_generalization_passed": summary.get("test_generalization_passed") is True,
        "test_raw_policy_regression_reduction_rate": _float_value_or_default(
            summary.get("test_raw_policy_regression_reduction_rate"),
            0.0,
        ),
        "overfit_gap": _float_value_or_default(summary.get("overfit_gap"), 0.0),
        "test_regression_count": _int_value_or_default(summary.get("test_regression_count"), 0),
        "test_invalid_action_mask_count": _int_value_or_default(
            summary.get("test_invalid_action_mask_count"),
            0,
        ),
        "test_fallback_or_open_grid_count": _int_value_or_default(
            summary.get("test_fallback_or_open_grid_count"),
            0,
        ),
        "test_safety_regression_count": _int_value_or_default(
            summary.get("test_safety_regression_count"),
            0,
        ),
        "test_contract_violation_count": _int_value_or_default(
            summary.get("test_contract_violation_count"),
            0,
        ),
        "test_path_cost_regression_count": _int_value_or_default(
            summary.get("test_path_cost_regression_count"),
            0,
        ),
        "test_risk_regression_count": _int_value_or_default(
            summary.get("test_risk_regression_count"),
            0,
        ),
        "test_source_selection_regression_count": _int_value_or_default(
            summary.get("test_source_selection_regression_count"),
            0,
        ),
    }


def _policy_gated_sequential_canary_rollout_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "episode_count": 0,
        "step_count": 0,
        "accepted_takeover_step_count": 0,
        "accepted_better_step_count": 0,
        "multi_step_accepted_episode_count": 0,
        "family_with_multi_step_accepted_episode_count": 0,
        "accepted_takeover_family_count": 0,
        "sequential_safe_choice_calibrated": False,
        "sequential_multi_step_opportunity_evaluated": False,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "policy_gated_sequential_canary_rollout_not_passed")
    for field, reason in (
        ("episode_count", "policy_gated_sequential_canary_episode_count_zero"),
        ("step_count", "policy_gated_sequential_canary_step_count_zero"),
        ("policy_takeover_step_count", "policy_gated_sequential_canary_takeover_step_count_zero"),
        ("accepted_takeover_step_count", "policy_gated_sequential_canary_accepted_step_count_zero"),
        ("multi_step_accepted_episode_count", "policy_gated_sequential_canary_multi_step_episode_count_zero"),
        (
            "family_with_multi_step_accepted_episode_count",
            "policy_gated_sequential_canary_family_coverage_zero",
        ),
    ):
        if _int_value_or_default(summary.get(field), 0) <= 0:
            _append_reason(blockers, reason)
    for field, reason in (
        ("state_continuity_violation_count", "policy_gated_sequential_canary_state_continuity_violation"),
        ("episode_fallback_count", "policy_gated_sequential_canary_episode_fallback"),
        ("canary_rejected_policy_choice_count", "policy_gated_sequential_canary_rejected_policy_choice"),
        ("invalid_action_mask_count", "policy_gated_sequential_canary_invalid_action_mask"),
        ("fallback_or_open_grid_count", "policy_gated_sequential_canary_fallback_or_open_grid"),
        ("cumulative_safety_regression_count", "policy_gated_sequential_canary_safety_regression"),
        ("cumulative_contract_violation_count", "policy_gated_sequential_canary_contract_violation"),
        ("cumulative_path_cost_regression_count", "policy_gated_sequential_canary_path_cost_regression"),
        ("cumulative_risk_regression_count", "policy_gated_sequential_canary_risk_regression"),
        (
            "cumulative_source_selection_regression_count",
            "policy_gated_sequential_canary_source_selection_regression",
        ),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "policy_gated_sequential_canary_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "policy_gated_sequential_canary_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "policy_gated_sequential_canary_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "policy_gated_sequential_canary_formal_training_ready_claimed")
    if summary.get("candidate_git_current_matches_sources") is False:
        _append_reason(blockers, "policy_gated_sequential_canary_candidate_git_current_mismatch")
    if summary.get("checkpoint_metadata_git_current_matches_sources") is False:
        _append_reason(blockers, "policy_gated_sequential_canary_checkpoint_metadata_git_current_mismatch")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "episode_count": _int_value_or_default(summary.get("episode_count"), 0),
        "step_count": _int_value_or_default(summary.get("step_count"), 0),
        "accepted_takeover_step_count": _int_value_or_default(
            summary.get("accepted_takeover_step_count"),
            0,
        ),
        "accepted_better_step_count": _int_value_or_default(
            summary.get("accepted_better_step_count"),
            0,
        ),
        "multi_step_accepted_episode_count": _int_value_or_default(
            summary.get("multi_step_accepted_episode_count"),
            0,
        ),
        "family_with_multi_step_accepted_episode_count": _int_value_or_default(
            summary.get("family_with_multi_step_accepted_episode_count"),
            0,
        ),
        "accepted_takeover_family_count": _int_value_or_default(
            summary.get("accepted_takeover_family_count"),
            0,
        ),
        "sequential_safe_choice_calibrated": _sequential_safe_choice_calibrated(summary),
        "sequential_multi_step_opportunity_evaluated": _sequential_multi_step_opportunity_evaluated(summary),
    }


def _sequential_safe_choice_calibrated(summary: dict[str, Any]) -> bool:
    stage = str(summary.get("calibration_stage") or summary.get("evaluation_stage") or "")
    batch_root = str(summary.get("batch_root") or "")
    source_root = str(summary.get("source_root") or "")
    return (
        stage == "sequential_safe_choice_calibration"
        or "sequential_safe_choice" in batch_root
        or "sequential_safe_choice" in source_root
    )


def _sequential_multi_step_opportunity_evaluated(summary: dict[str, Any]) -> bool:
    stage = str(summary.get("calibration_stage") or summary.get("evaluation_stage") or "")
    batch_root = str(summary.get("batch_root") or "")
    source_root = str(summary.get("source_root") or "")
    return (
        stage == "sequential_multi_step_opportunity"
        or "sequential_multi_step_opportunity" in batch_root
        or "sequential_multi_step_opportunity" in source_root
    )


def _ppo_rollout_collector_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "ppo_trainable_transition_count": 0,
        "episode_count": 0,
        "step_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "ppo_rollout_collector_not_passed")
    for field, reason in (
        ("episode_count", "ppo_rollout_collector_episode_count_zero"),
        ("step_count", "ppo_rollout_collector_step_count_zero"),
        ("ppo_trainable_transition_count", "ppo_trainable_transition_count_insufficient"),
    ):
        if _int_value_or_default(summary.get(field), 0) <= 0:
            _append_reason(blockers, reason)
    for field, reason in (
        ("source_fallback_trainable_count", "ppo_rollout_collector_source_fallback_trainable"),
        ("invalid_action_mask_count", "ppo_rollout_collector_invalid_action_mask"),
        ("empty_action_mask_count", "ppo_rollout_collector_empty_action_mask"),
        ("missing_log_prob_count", "ppo_logprob_value_missing"),
        ("missing_value_count", "ppo_logprob_value_missing"),
        ("non_finite_reward_count", "ppo_reward_contract_invalid"),
        ("state_continuity_violation_count", "ppo_rollout_collector_state_continuity_violation"),
        ("fallback_or_open_grid_count", "ppo_rollout_collector_fallback_or_open_grid"),
        ("safety_regression_count", "ppo_rollout_collector_safety_regression"),
        ("contract_violation_count", "ppo_rollout_collector_contract_violation"),
        ("path_cost_regression_count", "ppo_rollout_collector_path_cost_regression"),
        ("risk_regression_count", "ppo_rollout_collector_risk_regression"),
        ("source_selection_regression_count", "ppo_rollout_collector_source_selection_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "ppo_rollout_collector_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "ppo_rollout_collector_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "ppo_rollout_collector_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "ppo_rollout_collector_formal_training_ready_claimed")
    if summary.get("candidate_git_current_matches_sources") is False:
        _append_reason(blockers, "ppo_rollout_collector_candidate_git_current_mismatch")
    if summary.get("checkpoint_metadata_git_current_matches_sources") is False:
        _append_reason(blockers, "ppo_rollout_collector_checkpoint_metadata_git_current_mismatch")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "ppo_trainable_transition_count": _int_value_or_default(
            summary.get("ppo_trainable_transition_count"),
            0,
        ),
        "episode_count": _int_value_or_default(summary.get("episode_count"), 0),
        "step_count": _int_value_or_default(summary.get("step_count"), 0),
    }


def _filter_stale_source_git_for_quasi_real_collector(
    reason_codes: list[str],
    *,
    ppo_collector: dict[str, Any],
    quasi_real_guarded_teacher_following_pilot: dict[str, Any],
) -> list[str]:
    if not _quasi_real_collector_summary_passed(ppo_collector):
        return reason_codes
    if not _quasi_real_guarded_teacher_following_summary_passed(
        quasi_real_guarded_teacher_following_pilot
    ):
        return reason_codes
    protected_prefixes = ("ppo_rollout_collector_summary_",)
    protected_mismatch_present = any(
        reason.startswith(protected_prefixes)
        and reason.endswith("_current_git_provenance_mismatch")
        for reason in reason_codes
    )
    filtered: list[str] = []
    for reason in reason_codes:
        if reason == "current_git_provenance_mismatch" and not protected_mismatch_present:
            continue
        if (
            reason.endswith("_current_git_provenance_mismatch")
            and not reason.startswith(protected_prefixes)
        ):
            continue
        filtered.append(reason)
    return filtered


def _filter_stale_source_git_for_iterative_only(
    reason_codes: list[str],
    *,
    iterative_ppo_mini_loop: dict[str, Any],
    iterative_only_mode: bool,
) -> list[str]:
    if not iterative_only_mode:
        return reason_codes
    if not _iterative_ppo_mini_loop_stability_readiness(iterative_ppo_mini_loop)[
        "completed"
    ]:
        return reason_codes
    protected_prefix = "iterative_ppo_mini_loop_stability_summary_"
    protected_mismatch_present = any(
        reason.startswith(protected_prefix)
        and (
            reason.endswith("_current_git_provenance_mismatch")
            or reason.endswith("_git_provenance_mismatch")
        )
        for reason in reason_codes
    )
    filtered: list[str] = []
    for reason in reason_codes:
        if reason in {"current_git_provenance_mismatch", "git_provenance_mismatch"}:
            if not protected_mismatch_present:
                continue
        if (
            reason.endswith("_current_git_provenance_mismatch")
            or reason.endswith("_git_provenance_mismatch")
        ) and not reason.startswith(protected_prefix):
            continue
        filtered.append(reason)
    return filtered


def _quasi_real_collector_summary_passed(summary: dict[str, Any]) -> bool:
    return (
        summary.get("schema_version") == PPO_ROLLOUT_COLLECTOR_SCHEMA_VERSION
        and summary.get("status") == "passed"
        and not _string_list(summary.get("reason_codes"))
        and _int_value_or_default(summary.get("ppo_trainable_transition_count"), 0) > 0
    )


def _quasi_real_guarded_teacher_following_summary_passed(summary: dict[str, Any]) -> bool:
    return (
        summary.get("schema_version")
        == QUASI_REAL_GUARDED_TEACHER_FOLLOWING_PILOT_SCHEMA_VERSION
        and summary.get("status") == "passed"
        and not _string_list(summary.get("reason_codes"))
        and summary.get("teacher_following_pilot_verdict")
        == "teacher_following_pilot_validated"
    )


def _limited_ppo_update_smoke_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "input_ppo_trainable_transition_count": 0,
        "optimizer_train_transition_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "limited_ppo_update_smoke_not_passed")
    for field, reason in (
        ("input_ppo_trainable_transition_count", "limited_ppo_update_input_transition_count_zero"),
        ("optimizer_train_transition_count", "ppo_trainable_transition_count_insufficient"),
    ):
        if _int_value_or_default(summary.get(field), 0) <= 0:
            _append_reason(blockers, reason)
    for field, reason in (
        ("source_fallback_trainable_count", "limited_ppo_update_source_fallback_trainable"),
        ("loss_non_finite_count", "ppo_update_loss_non_finite"),
        ("non_finite_gradient_count", "ppo_update_loss_non_finite"),
        ("non_finite_reward_count", "ppo_reward_contract_invalid"),
        ("non_finite_return_count", "ppo_update_loss_non_finite"),
        ("non_finite_advantage_count", "ppo_update_loss_non_finite"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if _float_value_or_default(summary.get("old_log_prob_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "ppo_update_not_on_collector_policy")
    if _float_value_or_default(summary.get("old_value_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "ppo_update_not_on_collector_policy")
    if _float_value_or_default(summary.get("parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(blockers, "limited_ppo_update_input_contract_invalid")
    if _float_value_or_default(summary.get("approx_kl"), float("inf")) > 0.25:
        _append_reason(blockers, "ppo_update_too_large")
    if _float_value_or_default(summary.get("max_grad_norm_after_clip"), float("inf")) > 1.0 + 1.0e-8:
        _append_reason(blockers, "ppo_update_too_large")
    if summary.get("experimental_checkpoint") is not True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_not_experimental")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "limited_ppo_update_smoke_git_current_mismatch")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "input_ppo_trainable_transition_count": _int_value_or_default(
            summary.get("input_ppo_trainable_transition_count"),
            0,
        ),
        "optimizer_train_transition_count": _int_value_or_default(
            summary.get("optimizer_train_transition_count"),
            0,
        ),
    }


def _limited_quasi_real_ppo_update_smoke_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "input_ppo_trainable_transition_count": 0,
        "optimizer_train_transition_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "limited_quasi_real_ppo_update_smoke_not_passed")
    if _int_value_or_default(summary.get("input_ppo_trainable_transition_count"), 0) < 24:
        _append_reason(blockers, "ppo_trainable_transition_count_insufficient")
    if _int_value_or_default(summary.get("optimizer_train_transition_count"), 0) < 24:
        _append_reason(blockers, "ppo_trainable_transition_count_insufficient")
    for field, reason in (
        ("validation_test_optimizer_transition_count", "limited_quasi_real_ppo_update_split_leakage"),
        ("non_empty_gate_reason_optimizer_transition_count", "limited_quasi_real_ppo_update_gate_regression_trainable"),
        ("disallowed_source_optimizer_transition_count", "limited_quasi_real_ppo_update_disallowed_source_trainable"),
        ("source_fallback_trainable_count", "limited_ppo_update_source_fallback_trainable"),
        ("loss_non_finite_count", "ppo_update_loss_non_finite"),
        ("non_finite_gradient_count", "ppo_update_loss_non_finite"),
        ("non_finite_reward_count", "ppo_reward_contract_invalid"),
        ("non_finite_return_count", "ppo_update_loss_non_finite"),
        ("non_finite_advantage_count", "ppo_update_loss_non_finite"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if _float_value_or_default(summary.get("old_log_prob_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "ppo_update_not_on_collector_policy")
    if _float_value_or_default(summary.get("old_value_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "ppo_update_not_on_collector_policy")
    if _float_value_or_default(summary.get("parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(blockers, "limited_ppo_update_input_contract_invalid")
    if abs(_float_value_or_default(summary.get("approx_kl"), float("inf"))) > 0.25:
        _append_reason(blockers, "ppo_update_too_large")
    if _float_value_or_default(summary.get("max_grad_norm_after_clip"), float("inf")) > 1.0 + 1.0e-8:
        _append_reason(blockers, "ppo_update_too_large")
    if summary.get("experimental_checkpoint") is not True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_not_experimental")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _post_update_quasi_real_gate_regressed(summary):
        _append_reason(blockers, "limited_quasi_real_ppo_update_post_update_gate_regression")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "limited_quasi_real_ppo_update_smoke_git_current_mismatch")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "input_ppo_trainable_transition_count": _int_value_or_default(
            summary.get("input_ppo_trainable_transition_count"),
            0,
        ),
        "optimizer_train_transition_count": _int_value_or_default(
            summary.get("optimizer_train_transition_count"),
            0,
        ),
    }


def _generated_sequential_gate_metric_accounting_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "legacy_mismatch_count": 0,
        "raw_policy_path_cost_regression_count": 0,
        "raw_policy_risk_regression_count": 0,
        "controlled_path_cost_regression_count": 0,
        "controlled_risk_regression_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "generated_sequential_gate_metric_accounting_audit_not_passed")
    if _int_value_or_default(summary.get("legacy_mismatch_count"), 0) <= 0:
        _append_reason(blockers, "generated_sequential_gate_metric_accounting_no_legacy_mismatch")
    if summary.get("diagnosis_verdict_after_origin_split") != "pre_existing_generated_sequential_contract_mismatch":
        _append_reason(blockers, "generated_sequential_gate_metric_accounting_verdict_unresolved")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "generated_sequential_gate_metric_accounting_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "generated_sequential_gate_metric_accounting_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "generated_sequential_gate_metric_accounting_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "generated_sequential_gate_metric_accounting_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "generated_sequential_gate_metric_accounting_git_current_mismatch")

    if not blockers:
        _append_reason(blockers, "generated_sequential_contract_alignment_required")
    return {
        "present": True,
        "completed": False,
        "training_blockers": blockers,
        "next_required_change": (
            "generated_sequential_contract_alignment_required"
            if "generated_sequential_contract_alignment_required" in blockers
            else summary.get("recommended_next_action")
        ),
        "legacy_mismatch_count": _int_value_or_default(summary.get("legacy_mismatch_count"), 0),
        "raw_policy_path_cost_regression_count": _int_value_or_default(
            summary.get("raw_policy_path_cost_regression_count"),
            0,
        ),
        "raw_policy_risk_regression_count": _int_value_or_default(
            summary.get("raw_policy_risk_regression_count"),
            0,
        ),
        "controlled_path_cost_regression_count": _int_value_or_default(
            summary.get("controlled_path_cost_regression_count"),
            0,
        ),
        "controlled_risk_regression_count": _int_value_or_default(
            summary.get("controlled_risk_regression_count"),
            0,
        ),
    }


def _generated_sequential_long_horizon_teacher_skill_contract_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "teacher_equivalent_episode_count": 0,
        "beyond_teacher_episode_count": 0,
        "controlled_regression_episode_count": 0,
        "dominated_raw_choice_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "generated_sequential_long_horizon_contract_not_passed")
    verdict = str(summary.get("verdict") or "")
    if verdict != "long_horizon_teacher_skill_contract_aligned":
        if verdict == "long_horizon_contract_still_blocked":
            _append_reason(blockers, "generated_sequential_long_horizon_contract_still_blocked")
        elif verdict == "missing_or_stale_input_evidence":
            _append_reason(blockers, "generated_sequential_long_horizon_input_missing_or_stale")
        elif verdict == "return_accounting_inconclusive":
            _append_reason(blockers, "generated_sequential_long_horizon_return_accounting_inconclusive")
        else:
            _append_reason(blockers, "generated_sequential_long_horizon_contract_unresolved")
    if _int_value_or_default(summary.get("teacher_equivalent_episode_count"), 0) <= 0:
        _append_reason(blockers, "generated_sequential_long_horizon_teacher_equivalent_missing")
    if _int_value_or_default(summary.get("controlled_regression_episode_count"), 0) > 0:
        _append_reason(blockers, "generated_sequential_long_horizon_controlled_regression")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "generated_sequential_long_horizon_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "generated_sequential_long_horizon_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "generated_sequential_long_horizon_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "generated_sequential_long_horizon_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "generated_sequential_long_horizon_git_current_mismatch")

    completed = not blockers
    return {
        "present": True,
        "completed": completed,
        "training_blockers": blockers,
        "next_required_change": (
            None if completed else "generated_sequential_contract_alignment_required"
        ),
        "teacher_equivalent_episode_count": _int_value_or_default(
            summary.get("teacher_equivalent_episode_count"),
            0,
        ),
        "beyond_teacher_episode_count": _int_value_or_default(
            summary.get("beyond_teacher_episode_count"),
            0,
        ),
        "controlled_regression_episode_count": _int_value_or_default(
            summary.get("controlled_regression_episode_count"),
            0,
        ),
        "dominated_raw_choice_count": _int_value_or_default(
            summary.get("dominated_raw_choice_count"),
            0,
        ),
    }


def _long_horizon_contract_overrides_limited_quasi_real_generated_blocker(
    limited_readiness: dict[str, Any],
    long_horizon_readiness: dict[str, Any],
) -> bool:
    if not limited_readiness.get("present") or not long_horizon_readiness.get("completed"):
        return False
    blockers = set(_string_list(limited_readiness.get("training_blockers")))
    generated_only_blockers = {
        "limited_quasi_real_ppo_update_smoke_not_passed",
        "limited_quasi_real_ppo_update_post_update_gate_regression",
    }
    return bool(blockers) and blockers.issubset(generated_only_blockers)


def _post_update_quasi_real_gate_regressed(summary: dict[str, Any]) -> bool:
    status_fields = (
        "post_update_raw_generalization_status",
        "post_update_sequential_canary_status",
        "post_update_generated_collector_status",
        "post_update_quasi_real_teacher_following_status",
        "post_update_quasi_real_collector_status",
    )
    if any(summary.get(field) != "passed" for field in status_fields):
        return True
    if _int_value_or_default(summary.get("post_update_raw_test_regression_count"), 0):
        return True
    if _int_value_or_default(summary.get("post_update_sequential_rejected_choice_count"), 0):
        return True
    if _int_value_or_default(summary.get("post_update_generated_collector_trainable_transition_count"), 0) < 24:
        return True
    if _int_value_or_default(summary.get("post_update_quasi_real_collector_trainable_transition_count"), 0) < 24:
        return True
    if _float_value_or_default(summary.get("post_update_quasi_real_teacher_agreement_rate"), 0.0) < 0.9:
        return True
    if _int_value_or_default(summary.get("post_update_quasi_real_unsafe_disagreement_count"), 0):
        return True
    return False


def _iterative_ppo_mini_loop_stability_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "round_count": 0,
        "failed_round_count": 0,
        "stability_passed": False,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "iterative_ppo_mini_loop_stability_not_passed")
    if summary.get("stability_passed") is not True:
        _append_reason(blockers, "iterative_ppo_policy_drift_detected")
    if _int_value_or_default(summary.get("round_count"), 0) < 3:
        _append_reason(blockers, "iterative_ppo_round_count_insufficient")
    if _int_value_or_default(summary.get("failed_round_count"), 0):
        _append_reason(blockers, "iterative_ppo_post_update_gate_regression")
    if (
        "min_optimizer_train_transition_count" in summary
        and _int_value_or_default(summary.get("min_optimizer_train_transition_count"), 0) < 24
    ):
        _append_reason(blockers, "iterative_ppo_trainable_transition_count_insufficient")
    if (
        "min_ppo_trainable_transition_count" in summary
        and _int_value_or_default(summary.get("min_ppo_trainable_transition_count"), 0) < 24
    ):
        _append_reason(blockers, "iterative_ppo_trainable_transition_count_insufficient")
    if _float_value_or_default(summary.get("max_abs_approx_kl"), float("inf")) > 0.25:
        _append_reason(blockers, "iterative_ppo_policy_drift_detected")
    cumulative_delta = _float_value_or_default(summary.get("cumulative_parameter_l2_delta"), 0.0)
    if cumulative_delta <= 0.0 or cumulative_delta > 0.05:
        _append_reason(blockers, "iterative_ppo_policy_drift_detected")
    if _int_value_or_default(summary.get("raw_test_regression_count"), 0):
        _append_reason(blockers, "iterative_ppo_post_update_gate_regression")
    if _int_value_or_default(summary.get("sequential_rejected_count"), 0):
        _append_reason(blockers, "iterative_ppo_post_update_gate_regression")
    if _int_value_or_default(summary.get("collector_regression_count"), 0):
        _append_reason(blockers, "iterative_ppo_post_update_gate_regression")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "iterative_ppo_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "iterative_ppo_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "iterative_ppo_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "iterative_ppo_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "round_count": _int_value_or_default(summary.get("round_count"), 0),
        "failed_round_count": _int_value_or_default(summary.get("failed_round_count"), 0),
        "stability_passed": summary.get("stability_passed") is True,
    }


def _return_aligned_guarded_multistep_collector_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "trainable_episode_count": 0,
        "trainable_transition_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "return_aligned_guarded_multistep_collector_not_passed")
    if _int_value_or_default(summary.get("trainable_episode_count"), 0) < 24:
        _append_reason(blockers, "return_aligned_trainable_episode_count_insufficient")
    if _int_value_or_default(summary.get("trainable_transition_count"), 0) < 24:
        _append_reason(blockers, "return_aligned_trainable_transition_count_insufficient")
    for key, reason in (
        ("validation_trainable_count", "return_aligned_validation_trainable_leakage"),
        ("test_trainable_count", "return_aligned_test_trainable_leakage"),
        ("source_fallback_trainable_count", "return_aligned_source_fallback_trainable_detected"),
        ("invalid_action_mask_count", "return_aligned_input_collector_invalid_mask_detected"),
        ("empty_action_mask_count", "return_aligned_input_collector_empty_mask_detected"),
        ("missing_log_prob_count", "return_aligned_input_collector_missing_log_prob_detected"),
        ("missing_value_count", "return_aligned_input_collector_missing_value_detected"),
        ("non_finite_reward_count", "return_aligned_non_finite_reward_detected"),
        ("non_finite_return_count", "return_aligned_non_finite_return_detected"),
        ("non_finite_advantage_count", "return_aligned_non_finite_advantage_detected"),
        ("controlled_regression_count", "return_aligned_controlled_regression_detected"),
        ("controlled_safety_regression_count", "return_aligned_controlled_regression_detected"),
        ("controlled_contract_regression_count", "return_aligned_controlled_regression_detected"),
        ("controlled_path_risk_regression_count", "return_aligned_controlled_regression_detected"),
        ("controlled_source_selection_regression_count", "return_aligned_controlled_regression_detected"),
    ):
        if _int_value_or_default(summary.get(key), 0):
            _append_reason(blockers, reason)
    if summary.get("uses_multistep_discounted_return") is not True:
        _append_reason(blockers, "return_aligned_multistep_discounted_return_missing")
    if summary.get("not_single_step_best_action") is not True:
        _append_reason(blockers, "return_aligned_single_step_best_action_claim_detected")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "return_aligned_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "return_aligned_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "return_aligned_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "return_aligned_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "trainable_episode_count": _int_value_or_default(summary.get("trainable_episode_count"), 0),
        "trainable_transition_count": _int_value_or_default(summary.get("trainable_transition_count"), 0),
    }


def _return_aligned_guarded_ppo_update_smoke_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "input_return_aligned_trainable_transition_count": 0,
        "optimizer_train_transition_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "return_aligned_ppo_update_smoke_not_passed")
    if _int_value_or_default(summary.get("input_return_aligned_trainable_transition_count"), 0) < 24:
        _append_reason(blockers, "ppo_trainable_transition_count_insufficient")
    if _int_value_or_default(summary.get("optimizer_train_transition_count"), 0) < 24:
        _append_reason(blockers, "ppo_trainable_transition_count_insufficient")
    for key, reason in (
        ("validation_test_optimizer_transition_count", "return_aligned_ppo_update_split_leakage"),
        ("source_fallback_optimizer_transition_count", "return_aligned_ppo_update_source_fallback_optimizer"),
        ("non_empty_gate_reason_optimizer_transition_count", "return_aligned_ppo_update_gate_regression_trainable"),
        ("source_fallback_trainable_count", "return_aligned_source_fallback_trainable_detected"),
        ("materialization_error_count", "return_aligned_ppo_update_materialization_failed"),
        ("loss_non_finite_count", "ppo_update_loss_non_finite"),
        ("non_finite_gradient_count", "ppo_update_loss_non_finite"),
        ("non_finite_reward_count", "ppo_reward_contract_invalid"),
        ("non_finite_return_count", "ppo_update_loss_non_finite"),
        ("non_finite_advantage_count", "ppo_update_loss_non_finite"),
        ("post_update_controlled_regression_count", "return_aligned_ppo_update_post_update_gate_regression"),
    ):
        if _int_value_or_default(summary.get(key), 0):
            _append_reason(blockers, reason)
    if _float_value_or_default(summary.get("old_log_prob_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "ppo_update_not_on_collector_policy")
    if _float_value_or_default(summary.get("old_value_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "ppo_update_not_on_collector_policy")
    if _float_value_or_default(summary.get("parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(blockers, "limited_ppo_update_input_contract_invalid")
    if abs(_float_value_or_default(summary.get("approx_kl"), float("inf"))) > 0.25:
        _append_reason(blockers, "ppo_update_too_large")
    if _float_value_or_default(summary.get("max_grad_norm_after_clip"), float("inf")) > 1.0 + 1.0e-8:
        _append_reason(blockers, "ppo_update_too_large")
    if summary.get("uses_multistep_discounted_return") is not True:
        _append_reason(blockers, "return_aligned_multistep_discounted_return_missing")
    if summary.get("not_single_step_best_action") is not True:
        _append_reason(blockers, "return_aligned_single_step_best_action_claim_detected")
    if summary.get("post_update_gates_evaluated") is not True:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("post_update_raw_generalization_status") not in {"passed"}:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("post_update_generated_sequential_status") not in {"passed", "failed"}:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("post_update_generated_collector_status") not in {"passed"}:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("post_update_quasi_real_teacher_following_status") not in {"passed"}:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("post_update_quasi_real_collector_status") not in {"passed"}:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if _int_value_or_default(summary.get("post_update_raw_test_regression_count"), 0):
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if _int_value_or_default(summary.get("post_update_quasi_real_unsafe_disagreement_count"), 0):
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    generated_trainable = summary.get("post_update_generated_collector_trainable_transition_count")
    if generated_trainable is not None and _int_value_or_default(generated_trainable, 0) < 24:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    quasi_trainable = summary.get("post_update_quasi_real_collector_trainable_transition_count")
    if quasi_trainable is not None and _int_value_or_default(quasi_trainable, 0) < 24:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    return_aligned_replay_trainable = summary.get("post_update_return_aligned_replay_trainable_transition_count")
    if return_aligned_replay_trainable is not None and _int_value_or_default(return_aligned_replay_trainable, 0) < 24:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("post_update_teacher_agreement_rate") is not None and _float_value_or_default(
        summary.get("post_update_teacher_agreement_rate"),
        0.0,
    ) < 0.9:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("post_update_return_aligned_replay_status") not in {"passed"}:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("post_update_long_horizon_status") not in {None, "passed"}:
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("post_update_long_horizon_verdict") != "long_horizon_teacher_skill_contract_aligned":
        _append_reason(blockers, "return_aligned_ppo_update_post_update_gate_regression")
    if summary.get("experimental_checkpoint") is not True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_not_experimental")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "input_return_aligned_trainable_transition_count": _int_value_or_default(
            summary.get("input_return_aligned_trainable_transition_count"),
            0,
        ),
        "optimizer_train_transition_count": _int_value_or_default(
            summary.get("optimizer_train_transition_count"),
            0,
        ),
    }


def _policy_training_cuda_device_support_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "resolved_device": None,
        "optimizer_train_transition_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "cuda_training_smoke_failed")
    requested_device = str(summary.get("requested_device", "cpu"))
    resolved_device = str(summary.get("resolved_device", ""))
    if requested_device not in {"cpu", "cuda", "auto"} or resolved_device not in {"cpu", "cuda"}:
        _append_reason(blockers, "training_device_contract_invalid")
    if "cuda_requested_but_unavailable" in _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "cuda_requested_but_unavailable")
    if "device_tensor_mismatch" in _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "device_tensor_mismatch")
    if _int_value_or_default(summary.get("optimizer_train_transition_count"), 0) < 24:
        _append_reason(blockers, "ppo_trainable_transition_count_insufficient")
    for field, reason in (
        ("source_fallback_trainable_count", "limited_ppo_update_input_contract_invalid"),
        ("loss_non_finite_count", "cuda_training_smoke_failed"),
        ("non_finite_gradient_count", "cuda_training_smoke_failed"),
        ("non_finite_reward_count", "ppo_reward_contract_invalid"),
        ("non_finite_return_count", "cuda_training_smoke_failed"),
        ("non_finite_advantage_count", "cuda_training_smoke_failed"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if _float_value_or_default(summary.get("old_log_prob_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "ppo_update_not_on_collector_policy")
    if _float_value_or_default(summary.get("old_value_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "ppo_update_not_on_collector_policy")
    if _float_value_or_default(summary.get("parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(blockers, "limited_ppo_update_input_contract_invalid")
    if abs(_float_value_or_default(summary.get("approx_kl"), float("inf"))) > 0.25:
        _append_reason(blockers, "ppo_update_too_large")
    if _float_value_or_default(summary.get("max_grad_norm_after_clip"), float("inf")) > 1.0 + 1.0e-8:
        _append_reason(blockers, "ppo_update_too_large")
    if summary.get("checkpoint_cpu_loadable") is not True:
        _append_reason(blockers, "training_device_contract_invalid")
    if summary.get("experimental_checkpoint") is not True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_not_experimental")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "resolved_device": resolved_device,
        "optimizer_train_transition_count": _int_value_or_default(
            summary.get("optimizer_train_transition_count"),
            0,
        ),
    }


def _quasi_real_map_domain_gap_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "domain_gap_verdict": None,
        "slice_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "real_map_domain_gap_evaluation_failed")
    verdict = str(summary.get("domain_gap_verdict", ""))
    if verdict not in {"acceptable_for_next_pilot", "scenario_expansion_required", "planner_contract_gap"}:
        _append_reason(blockers, "real_map_domain_gap_verdict_missing")
    if verdict != "acceptable_for_next_pilot":
        if verdict == "scenario_expansion_required":
            _append_reason(blockers, "real_map_distribution_gap_requires_scenario_expansion")
        elif verdict == "planner_contract_gap":
            _append_reason(blockers, "real_map_path_feedback_regression")
    if _int_value_or_default(summary.get("slice_count"), 0) < 12:
        _append_reason(blockers, "real_map_slice_count_below_threshold")
    if _int_value_or_default(summary.get("roi_group_count"), 0) < 4:
        _append_reason(blockers, "real_map_roi_group_count_below_threshold")
    if _int_value_or_default(summary.get("context_id_missing_count"), 0):
        _append_reason(blockers, "real_map_context_id_missing")
    if _int_value_or_default(summary.get("legacy_identity_fallback_count"), 0):
        _append_reason(blockers, "real_map_context_id_missing")
    for field, reason in (
        ("invalid_action_mask_count", "real_map_action_mask_contract_gap"),
        ("fallback_or_open_grid_count", "real_map_path_feedback_regression"),
        ("safety_regression_count", "real_map_path_feedback_regression"),
        ("contract_violation_count", "real_map_bridge_contract_invalid"),
        ("path_cost_regression_count", "real_map_path_feedback_regression"),
        ("risk_regression_count", "real_map_path_feedback_regression"),
        ("source_selection_regression_count", "real_map_path_feedback_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if summary.get("runs_ppo_update") is True:
        _append_reason(blockers, "quasi_real_domain_gap_unexpected_ppo_update")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "quasi_real_domain_gap_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "quasi_real_domain_gap_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "quasi_real_domain_gap_policy_performance_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "domain_gap_verdict": verdict,
        "slice_count": _int_value_or_default(summary.get("slice_count"), 0),
    }


def _quasi_real_shadow_policy_behavior_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "behavior_verdict": None,
        "shadow_context_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_shadow_policy_scoring_failed")
    verdict = str(summary.get("behavior_verdict", ""))
    allowed_verdicts = {
        "acceptable_for_quasi_real_guarded_pilot",
        "policy_real_map_alignment_refinement_required",
        "real_map_action_mask_contract_gap",
        "real_map_bridge_or_feedback_gap",
        "scenario_expansion_required",
    }
    if verdict not in allowed_verdicts:
        _append_reason(blockers, "quasi_real_shadow_policy_scoring_failed")
    elif verdict != "acceptable_for_quasi_real_guarded_pilot":
        if verdict == "policy_real_map_alignment_refinement_required":
            _append_reason(blockers, "quasi_real_shadow_policy_alignment_refinement_required")
        elif verdict == "real_map_action_mask_contract_gap":
            _append_reason(blockers, "quasi_real_shadow_action_mask_contract_gap")
        elif verdict == "real_map_bridge_or_feedback_gap":
            _append_reason(blockers, "quasi_real_shadow_policy_scoring_failed")
        elif verdict == "scenario_expansion_required":
            _append_reason(blockers, "quasi_real_shadow_context_count_below_threshold")

    shadow_context_count = _int_value_or_default(summary.get("shadow_context_count"), 0)
    policy_decision_count = _int_value_or_default(summary.get("policy_decision_count"), 0)
    if shadow_context_count < 12:
        _append_reason(blockers, "quasi_real_shadow_context_count_below_threshold")
    if policy_decision_count != shadow_context_count:
        _append_reason(blockers, "quasi_real_shadow_policy_scoring_failed")
    if _int_value_or_default(summary.get("roi_group_count"), 0) < 4:
        _append_reason(blockers, "quasi_real_shadow_roi_group_count_below_threshold")
    if _int_value_or_default(summary.get("context_id_missing_count"), 0):
        _append_reason(blockers, "quasi_real_shadow_context_id_missing")
    for field, reason in (
        ("invalid_action_mask_count", "quasi_real_shadow_action_mask_contract_gap"),
        ("fallback_or_open_grid_count", "quasi_real_shadow_gate_regression"),
        ("open_grid_fallback_count", "quasi_real_shadow_gate_regression"),
        ("safety_regression_count", "quasi_real_shadow_gate_regression"),
        ("contract_violation_count", "quasi_real_shadow_gate_regression"),
        ("contract_regression_count", "quasi_real_shadow_gate_regression"),
        ("path_cost_regression_count", "quasi_real_shadow_gate_regression"),
        ("risk_regression_count", "quasi_real_shadow_gate_regression"),
        ("source_selection_regression_count", "quasi_real_shadow_gate_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    rejected_count = _int_value_or_default(summary.get("policy_changed_gate_rejected_count"), 0)
    if rejected_count:
        _append_reason(blockers, "quasi_real_shadow_gate_regression")
    if summary.get("runs_ppo_update") is True:
        _append_reason(blockers, "quasi_real_shadow_unexpected_ppo_update")
    if summary.get("policy_takes_control") is True:
        _append_reason(blockers, "quasi_real_shadow_policy_takeover_detected")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "quasi_real_shadow_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "quasi_real_shadow_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "quasi_real_shadow_policy_performance_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "behavior_verdict": verdict,
        "shadow_context_count": shadow_context_count,
        "policy_decision_count": policy_decision_count,
    }


def _quasi_real_shadow_alignment_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "alignment_verdict": None,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_shadow_objective_weight_refinement_required")
    verdict = str(summary.get("alignment_verdict", ""))
    if verdict != "acceptable_for_quasi_real_shadow_audit":
        if verdict == "holdout_regression":
            _append_reason(blockers, "quasi_real_shadow_holdout_regression")
        elif verdict == "over_conservative_policy":
            _append_reason(blockers, "quasi_real_shadow_over_conservative_policy_detected")
        elif verdict:
            _append_reason(blockers, "quasi_real_shadow_objective_weight_refinement_required")
        else:
            _append_reason(blockers, "quasi_real_shadow_objective_weight_refinement_required")
    if _int_value_or_default(summary.get("taxonomy_failure_count"), 0) <= 0:
        _append_reason(blockers, "quasi_real_shadow_failure_taxonomy_required")
    if _int_value_or_default(summary.get("quasi_real_hard_negative_preference_count"), 0) <= 0:
        _append_reason(blockers, "quasi_real_shadow_hard_negative_signal_insufficient")
    for field in ("context_id_overlap_count", "scenario_id_overlap_count", "slice_id_overlap_count"):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, "quasi_real_shadow_alignment_context_leakage_detected")
    if _int_value_or_default(summary.get("hard_positive_added_count"), 0):
        _append_reason(blockers, "quasi_real_shadow_alignment_contract_invalid")
    if _int_value_or_default(summary.get("ppo_transition_added_count"), 0):
        _append_reason(blockers, "quasi_real_shadow_alignment_contract_invalid")
    for field in (
        "holdout_policy_changed_gate_rejected_count",
        "holdout_path_cost_regression_count",
        "holdout_risk_regression_count",
        "holdout_source_selection_regression_count",
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, "quasi_real_shadow_holdout_regression")
    if _int_value_or_default(summary.get("original_roi_regression_count"), 0):
        _append_reason(blockers, "quasi_real_shadow_holdout_regression")
    if summary.get("over_conservative_policy_detected") is True:
        _append_reason(blockers, "quasi_real_shadow_over_conservative_policy_detected")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "quasi_real_shadow_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "quasi_real_shadow_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "quasi_real_shadow_policy_performance_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "alignment_verdict": verdict,
    }


def _quasi_real_safe_alternative_opportunity_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "opportunity_verdict": None,
        "safe_better_opportunity_context_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_safe_alternative_opportunity_diagnosis_failed")
    verdict = str(summary.get("opportunity_verdict", ""))
    allowed_verdicts = {
        "quasi_real_safe_alternative_opportunity_gap",
        "acceptable_for_quasi_real_safe_choice_calibration",
        "real_map_bridge_or_feedback_gap",
        "real_map_action_mask_contract_gap",
    }
    if verdict not in allowed_verdicts:
        _append_reason(blockers, "quasi_real_safe_alternative_opportunity_diagnosis_failed")
    elif verdict == "real_map_bridge_or_feedback_gap":
        _append_reason(blockers, "real_map_bridge_or_feedback_gap")
    elif verdict == "real_map_action_mask_contract_gap":
        _append_reason(blockers, "real_map_action_mask_contract_gap")

    if _int_value_or_default(summary.get("quasi_real_context_count"), 0) < 12:
        _append_reason(blockers, "quasi_real_safe_alternative_context_count_below_threshold")
    if _int_value_or_default(summary.get("policy_decision_count"), 0) != _int_value_or_default(
        summary.get("quasi_real_context_count"),
        0,
    ):
        _append_reason(blockers, "quasi_real_safe_alternative_opportunity_diagnosis_failed")
    if _int_value_or_default(summary.get("roi_group_count"), 0) < 4:
        _append_reason(blockers, "quasi_real_safe_alternative_roi_group_count_below_threshold")
    if _int_value_or_default(summary.get("context_id_missing_count"), 0):
        _append_reason(blockers, "quasi_real_shadow_context_id_missing")
    if _int_value_or_default(summary.get("opportunity_exclusion_count"), 0):
        _append_reason(blockers, "real_map_bridge_or_feedback_gap")
    for field, reason in (
        ("invalid_action_mask_count", "real_map_action_mask_contract_gap"),
        ("fallback_or_open_grid_count", "quasi_real_safe_alternative_gate_regression"),
        ("safety_regression_count", "quasi_real_safe_alternative_gate_regression"),
        ("contract_violation_count", "real_map_action_mask_contract_gap"),
        ("path_cost_regression_count", "quasi_real_safe_alternative_gate_regression"),
        ("risk_regression_count", "quasi_real_safe_alternative_gate_regression"),
        ("source_selection_regression_count", "quasi_real_safe_alternative_gate_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if summary.get("runs_ppo_update") is True:
        _append_reason(blockers, "quasi_real_safe_alternative_unexpected_ppo_update")
    if _int_value_or_default(summary.get("ppo_transition_added_count"), 0):
        _append_reason(blockers, "quasi_real_safe_alternative_unexpected_ppo_transition")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "quasi_real_safe_alternative_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "quasi_real_safe_alternative_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "quasi_real_safe_alternative_policy_performance_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change"),
        "opportunity_verdict": verdict,
        "safe_better_opportunity_context_count": _int_value_or_default(
            summary.get("safe_better_opportunity_context_count"),
            0,
        ),
    }


def _quasi_real_safe_better_opportunity_expansion_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "safe_better_opportunity_context_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_safe_better_opportunity_expansion_failed")
    if _int_value_or_default(summary.get("candidate_context_count"), 0) < 48:
        _append_reason(blockers, "quasi_real_safe_better_candidate_context_count_below_threshold")
    if _int_value_or_default(summary.get("roi_group_count"), 0) < 4:
        _append_reason(blockers, "quasi_real_safe_better_roi_group_count_below_threshold")
    for field, reason in (
        ("start_cell_missing_count", "quasi_real_safe_better_start_cell_missing"),
        ("context_id_missing_count", "quasi_real_safe_better_context_id_missing"),
        ("context_id_overlap_count", "quasi_real_safe_better_context_id_overlap"),
        ("scenario_id_overlap_count", "quasi_real_safe_better_scenario_id_overlap"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if _int_value_or_default(summary.get("safe_alternative_context_count"), 0) < 8:
        _append_reason(blockers, "quasi_real_safe_better_safe_alternative_count_below_threshold")
    if _int_value_or_default(summary.get("safe_better_opportunity_context_count"), 0) < 4:
        _append_reason(blockers, "quasi_real_safe_better_opportunity_count_below_threshold")
    if _int_value_or_default(summary.get("roi_group_with_safe_better_opportunity_count"), 0) < 2:
        _append_reason(blockers, "quasi_real_safe_better_roi_group_opportunity_count_below_threshold")
    if summary.get("runs_ppo_update") is True:
        _append_reason(blockers, "quasi_real_safe_better_unexpected_ppo_update")
    if summary.get("policy_takes_control") is True:
        _append_reason(blockers, "quasi_real_safe_better_unexpected_policy_takeover")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "quasi_real_safe_better_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "quasi_real_safe_better_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "quasi_real_safe_better_policy_performance_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": None if not blockers else "quasi_real_roi_start_target_expansion_required",
        "safe_better_opportunity_context_count": _int_value_or_default(
            summary.get("safe_better_opportunity_context_count"),
            0,
        ),
    }


def _quasi_real_teacher_equivalent_validation_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "teacher_equivalent_context_count": 0,
        "teacher_agreement_rate": 0.0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_teacher_equivalent_scoring_failed")
    verdict = str(summary.get("teacher_equivalent_verdict", ""))
    if verdict != "teacher_equivalent_validated":
        if verdict == "quasi_real_teacher_equivalent_unsafe_disagreement":
            _append_reason(blockers, "quasi_real_teacher_equivalent_unsafe_disagreement")
        elif verdict == "quasi_real_teacher_equivalent_gate_regression":
            _append_reason(blockers, "quasi_real_teacher_equivalent_gate_regression")
        elif verdict == "quasi_real_teacher_equivalent_context_id_missing":
            _append_reason(blockers, "quasi_real_teacher_equivalent_context_id_missing")
        elif verdict == "quasi_real_teacher_equivalent_context_coverage_insufficient":
            _append_reason(blockers, "quasi_real_teacher_equivalent_context_coverage_insufficient")
        else:
            _append_reason(blockers, "quasi_real_teacher_equivalent_scoring_failed")

    context_count = _int_value_or_default(summary.get("teacher_equivalent_context_count"), 0)
    policy_decision_count = _int_value_or_default(summary.get("policy_decision_count"), 0)
    if context_count < 48:
        _append_reason(blockers, "quasi_real_teacher_equivalent_context_coverage_insufficient")
    if policy_decision_count != context_count:
        _append_reason(blockers, "quasi_real_teacher_equivalent_scoring_failed")
    if _int_value_or_default(summary.get("roi_group_count"), 0) < 4:
        _append_reason(blockers, "quasi_real_teacher_equivalent_context_coverage_insufficient")
    if _int_value_or_default(summary.get("context_id_missing_count"), 0):
        _append_reason(blockers, "quasi_real_teacher_equivalent_context_id_missing")
    teacher_agreement_rate = _float_value_or_default(summary.get("teacher_agreement_rate"), 0.0)
    if teacher_agreement_rate < 0.9:
        _append_reason(blockers, "quasi_real_teacher_equivalent_context_coverage_insufficient")
    if _int_value_or_default(summary.get("unsafe_disagreement_count"), 0):
        _append_reason(blockers, "quasi_real_teacher_equivalent_unsafe_disagreement")
    if _int_value_or_default(summary.get("policy_changed_gate_rejected_count"), 0):
        _append_reason(blockers, "quasi_real_teacher_equivalent_unsafe_disagreement")
    for field in (
        "invalid_action_mask_count",
        "fallback_or_open_grid_count",
        "open_grid_fallback_count",
        "safety_regression_count",
        "contract_violation_count",
        "contract_regression_count",
        "path_cost_regression_count",
        "risk_regression_count",
        "source_selection_regression_count",
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, "quasi_real_teacher_equivalent_gate_regression")
    if summary.get("runs_ppo_update") is True:
        _append_reason(blockers, "quasi_real_teacher_equivalent_unexpected_ppo_update")
    if summary.get("policy_takes_control") is True:
        _append_reason(blockers, "quasi_real_teacher_equivalent_unexpected_policy_takeover")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "quasi_real_teacher_equivalent_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "quasi_real_teacher_equivalent_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "quasi_real_teacher_equivalent_policy_performance_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "teacher_equivalent_context_count": context_count,
        "teacher_agreement_rate": teacher_agreement_rate,
    }


def _quasi_real_teacher_distillation_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "teacher_distillation_preference_count": 0,
        "teacher_agreement_rate": 0.0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_teacher_distillation_objective_weight_refinement_required")
    verdict = str(summary.get("teacher_distillation_verdict", ""))
    if verdict != "teacher_distillation_robustness_validated":
        _append_reason(blockers, "quasi_real_teacher_distillation_objective_weight_refinement_required")
    if _int_value_or_default(summary.get("taxonomy_unsafe_disagreement_count"), 0) < 13:
        _append_reason(blockers, "quasi_real_teacher_distillation_signal_insufficient")
    classified = _int_value_or_default(summary.get("classified_disagreement_count"), 0)
    if classified != _int_value_or_default(summary.get("taxonomy_unsafe_disagreement_count"), classified):
        _append_reason(blockers, "quasi_real_teacher_distillation_signal_insufficient")
    preference_count = _int_value_or_default(summary.get("teacher_distillation_preference_count"), 0)
    if preference_count < 8:
        _append_reason(blockers, "quasi_real_teacher_distillation_signal_insufficient")
    for field in (
        "hard_positive_added_count",
        "ppo_transition_added_count",
        "holdout_leakage_count",
        "context_id_overlap_count",
        "scenario_id_overlap_count",
        "slice_id_overlap_count",
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, "quasi_real_teacher_distillation_context_leakage_detected")

    context_count = _int_value_or_default(summary.get("teacher_equivalent_context_count"), 0)
    policy_decision_count = _int_value_or_default(summary.get("policy_decision_count"), 0)
    if context_count < 108:
        _append_reason(blockers, "quasi_real_teacher_equivalent_holdout_regression")
    if policy_decision_count != context_count:
        _append_reason(blockers, "quasi_real_teacher_equivalent_holdout_regression")
    if _int_value_or_default(summary.get("roi_group_count"), 0) < 4:
        _append_reason(blockers, "quasi_real_teacher_equivalent_holdout_regression")
    teacher_agreement_rate = _float_value_or_default(summary.get("teacher_agreement_rate"), 0.0)
    if teacher_agreement_rate < 0.9:
        _append_reason(blockers, "quasi_real_teacher_equivalent_holdout_regression")
    if _int_value_or_default(summary.get("unsafe_disagreement_count"), 0):
        _append_reason(blockers, "quasi_real_teacher_equivalent_holdout_regression")
    if _int_value_or_default(summary.get("policy_changed_gate_rejected_count"), 0):
        _append_reason(blockers, "quasi_real_teacher_equivalent_holdout_regression")
    for field in (
        "invalid_action_mask_count",
        "fallback_or_open_grid_count",
        "open_grid_fallback_count",
        "safety_regression_count",
        "contract_violation_count",
        "contract_regression_count",
        "path_cost_regression_count",
        "risk_regression_count",
        "source_selection_regression_count",
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, "quasi_real_teacher_equivalent_holdout_regression")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "quasi_real_teacher_distillation_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "quasi_real_teacher_distillation_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "quasi_real_teacher_distillation_policy_performance_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "teacher_distillation_preference_count": preference_count,
        "teacher_agreement_rate": teacher_agreement_rate,
    }


def _quasi_real_guarded_teacher_following_pilot_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "teacher_following_pilot_verdict": None,
        "teacher_agreement_rate": 0.0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_teacher_following_pilot_contract_invalid")
    verdict = str(summary.get("teacher_following_pilot_verdict", ""))
    if verdict != "teacher_following_pilot_validated":
        if verdict == "real_map_action_mask_contract_gap":
            _append_reason(blockers, "quasi_real_guarded_teacher_following_action_mask_contract_gap")
        elif verdict == "real_map_bridge_or_feedback_gap":
            _append_reason(blockers, "quasi_real_guarded_teacher_following_scoring_failed")
        elif verdict == "scenario_expansion_required":
            _append_reason(blockers, "quasi_real_guarded_teacher_following_context_count_below_threshold")
        elif verdict == "quasi_real_guarded_teacher_following_gate_regression":
            _append_reason(blockers, "quasi_real_guarded_teacher_following_gate_regression")
        else:
            _append_reason(blockers, "quasi_real_guarded_teacher_following_alignment_insufficient")

    context_count = _int_value_or_default(summary.get("quasi_real_context_count"), 0)
    policy_decision_count = _int_value_or_default(summary.get("policy_decision_count"), 0)
    teacher_agreement_rate = _float_value_or_default(summary.get("teacher_agreement_rate"), 0.0)
    if context_count < 108:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_context_count_below_threshold")
    if policy_decision_count != context_count:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_scoring_failed")
    if _int_value_or_default(summary.get("roi_group_count"), 0) < 4:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_roi_group_count_below_threshold")
    if _int_value_or_default(summary.get("context_id_missing_count"), 0):
        _append_reason(blockers, "quasi_real_guarded_teacher_following_context_id_missing")
    if teacher_agreement_rate < 0.9:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_alignment_insufficient")
    if _int_value_or_default(summary.get("teacher_following_step_count"), 0) < 90:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_alignment_insufficient")
    if _int_value_or_default(summary.get("unsafe_disagreement_count"), 0):
        _append_reason(blockers, "quasi_real_guarded_teacher_following_gate_regression")
    if _int_value_or_default(summary.get("policy_changed_gate_rejected_count"), 0):
        _append_reason(blockers, "quasi_real_guarded_teacher_following_gate_regression")
    for field, reason in (
        ("invalid_action_mask_count", "quasi_real_guarded_teacher_following_action_mask_contract_gap"),
        ("fallback_or_open_grid_count", "quasi_real_guarded_teacher_following_gate_regression"),
        ("open_grid_fallback_count", "quasi_real_guarded_teacher_following_gate_regression"),
        ("safety_regression_count", "quasi_real_guarded_teacher_following_gate_regression"),
        ("contract_violation_count", "quasi_real_guarded_teacher_following_gate_regression"),
        ("contract_regression_count", "quasi_real_guarded_teacher_following_gate_regression"),
        ("path_cost_regression_count", "quasi_real_guarded_teacher_following_gate_regression"),
        ("risk_regression_count", "quasi_real_guarded_teacher_following_gate_regression"),
        ("source_selection_regression_count", "quasi_real_guarded_teacher_following_gate_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if summary.get("runs_ppo_update") is True:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_unexpected_ppo_update")
    if summary.get("writes_ppo_transition") is True:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_unexpected_ppo_transition")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "quasi_real_guarded_teacher_following_policy_performance_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "teacher_following_pilot_verdict": verdict,
        "quasi_real_context_count": context_count,
        "policy_decision_count": policy_decision_count,
        "teacher_agreement_rate": teacher_agreement_rate,
    }


def _quasi_real_guarded_policy_pilot_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "guarded_pilot_verdict": None,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_pilot_contract_invalid")
    verdict = str(summary.get("guarded_pilot_verdict", ""))
    allowed_verdicts = {
        "acceptable_for_quasi_real_collector_dry_run",
        "policy_over_conservative_on_quasi_real",
        "policy_real_map_alignment_refinement_required",
        "real_map_action_mask_contract_gap",
        "real_map_bridge_or_feedback_gap",
        "scenario_expansion_required",
    }
    if verdict not in allowed_verdicts:
        _append_reason(blockers, "quasi_real_guarded_policy_scoring_failed")
    elif verdict != "acceptable_for_quasi_real_collector_dry_run":
        if verdict == "policy_over_conservative_on_quasi_real":
            _append_reason(blockers, "quasi_real_guarded_policy_over_conservative")
        elif verdict == "policy_real_map_alignment_refinement_required":
            _append_reason(blockers, "quasi_real_guarded_gate_regression")
        elif verdict == "real_map_action_mask_contract_gap":
            _append_reason(blockers, "quasi_real_guarded_action_mask_contract_gap")
        elif verdict == "real_map_bridge_or_feedback_gap":
            _append_reason(blockers, "quasi_real_guarded_policy_scoring_failed")
        elif verdict == "scenario_expansion_required":
            _append_reason(blockers, "quasi_real_guarded_context_count_below_threshold")

    context_count = _int_value_or_default(summary.get("quasi_real_context_count"), 0)
    policy_decision_count = _int_value_or_default(summary.get("policy_decision_count"), 0)
    if context_count < 12:
        _append_reason(blockers, "quasi_real_guarded_context_count_below_threshold")
    if policy_decision_count != context_count:
        _append_reason(blockers, "quasi_real_guarded_policy_scoring_failed")
    if _int_value_or_default(summary.get("roi_group_count"), 0) < 4:
        _append_reason(blockers, "quasi_real_guarded_roi_group_count_below_threshold")
    if _int_value_or_default(summary.get("context_id_missing_count"), 0):
        _append_reason(blockers, "quasi_real_guarded_context_id_missing")
    for field, reason in (
        ("invalid_action_mask_count", "quasi_real_guarded_action_mask_contract_gap"),
        ("fallback_or_open_grid_count", "quasi_real_guarded_gate_regression"),
        ("open_grid_fallback_count", "quasi_real_guarded_gate_regression"),
        ("safety_regression_count", "quasi_real_guarded_gate_regression"),
        ("contract_violation_count", "quasi_real_guarded_gate_regression"),
        ("contract_regression_count", "quasi_real_guarded_gate_regression"),
        ("path_cost_regression_count", "quasi_real_guarded_gate_regression"),
        ("risk_regression_count", "quasi_real_guarded_gate_regression"),
        ("source_selection_regression_count", "quasi_real_guarded_gate_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if _int_value_or_default(summary.get("policy_changed_gate_rejected_count"), 0):
        _append_reason(blockers, "quasi_real_guarded_gate_regression")
    if _int_value_or_default(summary.get("policy_changed_gate_passed_count"), 0) <= 0:
        _append_reason(blockers, "quasi_real_guarded_policy_over_conservative")
    if summary.get("runs_ppo_update") is True:
        _append_reason(blockers, "quasi_real_guarded_unexpected_ppo_update")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "quasi_real_guarded_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "quasi_real_guarded_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "quasi_real_guarded_policy_performance_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "guarded_pilot_verdict": verdict,
        "quasi_real_context_count": context_count,
        "policy_decision_count": policy_decision_count,
    }


def _guarded_ppo_rollout_pilot_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "ppo_trainable_transition_count": 0,
        "optimizer_train_transition_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "guarded_ppo_rollout_pilot_not_passed")
    if summary.get("guarded_rollout_pilot_passed") is not True:
        _append_reason(blockers, "guarded_ppo_rollout_contract_invalid")
    for field, reason in (
        ("episode_count", "guarded_ppo_rollout_contract_invalid"),
        ("step_count", "guarded_ppo_rollout_contract_invalid"),
        ("ppo_trainable_transition_count", "guarded_ppo_trainable_transition_count_insufficient"),
        ("optimizer_train_transition_count", "guarded_ppo_trainable_transition_count_insufficient"),
    ):
        if _int_value_or_default(summary.get(field), 0) <= 0:
            _append_reason(blockers, reason)
    if _int_value_or_default(summary.get("ppo_trainable_transition_count"), 0) < 24:
        _append_reason(blockers, "guarded_ppo_trainable_transition_count_insufficient")
    if _int_value_or_default(summary.get("optimizer_train_transition_count"), 0) < 24:
        _append_reason(blockers, "guarded_ppo_trainable_transition_count_insufficient")
    for field, reason in (
        ("source_fallback_trainable_count", "guarded_ppo_rollout_contract_invalid"),
        ("state_continuity_violation_count", "guarded_ppo_rollout_contract_invalid"),
        ("invalid_action_mask_count", "guarded_ppo_rollout_contract_invalid"),
        ("empty_action_mask_count", "guarded_ppo_rollout_contract_invalid"),
        ("missing_log_prob_count", "guarded_ppo_on_policy_contract_invalid"),
        ("missing_value_count", "guarded_ppo_on_policy_contract_invalid"),
        ("non_finite_reward_count", "guarded_ppo_update_loss_non_finite"),
        ("post_update_raw_test_regression_count", "guarded_ppo_post_update_regression"),
        ("post_update_controlled_sequential_regression_count", "guarded_ppo_post_update_regression"),
        ("post_update_collector_regression_count", "guarded_ppo_post_update_regression"),
        ("post_update_quasi_real_unsafe_disagreement_count", "guarded_ppo_post_update_regression"),
        ("post_update_quasi_real_teacher_following_regression_count", "guarded_ppo_post_update_regression"),
        ("post_update_quasi_real_collector_regression_count", "guarded_ppo_post_update_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if summary.get("post_update_quasi_real_teacher_following_status") not in {None, "passed"}:
        _append_reason(blockers, "guarded_ppo_post_update_regression")
    if summary.get("post_update_quasi_real_collector_status") not in {None, "passed"}:
        _append_reason(blockers, "guarded_ppo_post_update_regression")
    if _float_value_or_default(summary.get("post_update_quasi_real_teacher_agreement_rate"), 1.0) < 0.9:
        _append_reason(blockers, "guarded_ppo_post_update_regression")
    quasi_collector_trainable = summary.get("post_update_quasi_real_collector_trainable_transition_count")
    if quasi_collector_trainable is not None and _int_value_or_default(quasi_collector_trainable, 0) < 24:
        _append_reason(blockers, "guarded_ppo_trainable_transition_count_insufficient")
    if _float_value_or_default(summary.get("old_log_prob_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "guarded_ppo_on_policy_contract_invalid")
    if _float_value_or_default(summary.get("old_value_max_abs_error"), float("inf")) > 1.0e-4:
        _append_reason(blockers, "guarded_ppo_on_policy_contract_invalid")
    if _float_value_or_default(summary.get("parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(blockers, "guarded_ppo_policy_drift_detected")
    if abs(_float_value_or_default(summary.get("approx_kl"), float("inf"))) > 0.25:
        _append_reason(blockers, "guarded_ppo_policy_drift_detected")
    if _float_value_or_default(summary.get("max_grad_norm_after_clip"), float("inf")) > 1.0 + 1.0e-8:
        _append_reason(blockers, "guarded_ppo_policy_drift_detected")
    if summary.get("experimental_checkpoint") is not True:
        _append_reason(blockers, "guarded_ppo_checkpoint_not_experimental")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "guarded_ppo_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "guarded_ppo_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "guarded_ppo_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "guarded_ppo_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "ppo_trainable_transition_count": _int_value_or_default(
            summary.get("ppo_trainable_transition_count"),
            0,
        ),
        "optimizer_train_transition_count": _int_value_or_default(
            summary.get("optimizer_train_transition_count"),
            0,
        ),
    }


def _quasi_real_guarded_ppo_rollout_pilot_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "episode_count": 0,
        "step_count": 0,
        "trainable_transition_count": 0,
        "teacher_agreement_rate": 0.0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_ppo_rollout_pilot_not_passed")
    episode_count = _int_value_or_default(summary.get("episode_count"), 0)
    step_count = _int_value_or_default(summary.get("step_count"), 0)
    trainable_transition_count = _int_value_or_default(
        summary.get("trainable_transition_count"),
        _int_value_or_default(summary.get("ppo_trainable_transition_count"), 0),
    )
    if episode_count < 36:
        _append_reason(blockers, "quasi_real_guarded_ppo_rollout_episode_count_below_threshold")
    if step_count < 108:
        _append_reason(blockers, "quasi_real_guarded_ppo_rollout_step_count_below_threshold")
    if trainable_transition_count < 24:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_rollout_trainable_transition_count_below_threshold",
        )
    for field, reason in (
        ("validation_trainable_count", "quasi_real_guarded_ppo_rollout_split_leakage"),
        ("test_trainable_count", "quasi_real_guarded_ppo_rollout_split_leakage"),
        ("source_fallback_trainable_count", "quasi_real_guarded_ppo_rollout_fallback_trainable"),
        ("teacher_fallback_trainable_count", "quasi_real_guarded_ppo_rollout_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "quasi_real_guarded_ppo_rollout_gate_reason_trainable"),
        ("missing_observation_count", "quasi_real_guarded_ppo_rollout_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_guarded_ppo_rollout_contract_invalid"),
        ("missing_value_count", "quasi_real_guarded_ppo_rollout_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_guarded_ppo_rollout_non_finite_return"),
        ("non_finite_return_count", "quasi_real_guarded_ppo_rollout_non_finite_return"),
        ("non_finite_advantage_count", "quasi_real_guarded_ppo_rollout_non_finite_return"),
        ("controlled_regression_count", "quasi_real_guarded_ppo_rollout_controlled_regression"),
        ("controlled_safety_regression_count", "quasi_real_guarded_ppo_rollout_controlled_regression"),
        ("controlled_contract_regression_count", "quasi_real_guarded_ppo_rollout_controlled_regression"),
        ("controlled_path_risk_regression_count", "quasi_real_guarded_ppo_rollout_controlled_regression"),
        ("controlled_source_selection_regression_count", "quasi_real_guarded_ppo_rollout_controlled_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)

    teacher_agreement_rate = _float_value_or_default(summary.get("teacher_agreement_rate"), 0.0)
    if teacher_agreement_rate < 0.9:
        _append_reason(blockers, "quasi_real_guarded_ppo_rollout_teacher_alignment_insufficient")
    if summary.get("quasi_real_collector_replay_status") not in {"passed"}:
        _append_reason(blockers, "quasi_real_guarded_ppo_rollout_collector_replay_not_passed")
    replay_trainable = summary.get("quasi_real_collector_replay_trainable_transition_count")
    if replay_trainable is not None and _int_value_or_default(replay_trainable, 0) < 24:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_rollout_trainable_transition_count_below_threshold",
        )
    if summary.get("post_pilot_long_horizon_status") not in {None, "passed"}:
        _append_reason(blockers, "quasi_real_guarded_ppo_rollout_long_horizon_not_aligned")
    if summary.get("post_pilot_long_horizon_verdict") != "long_horizon_teacher_skill_contract_aligned":
        _append_reason(blockers, "quasi_real_guarded_ppo_rollout_long_horizon_not_aligned")
    if summary.get("uses_multistep_discounted_return") is not True:
        _append_reason(blockers, "return_aligned_multistep_discounted_return_missing")
    if summary.get("not_single_step_best_action") is not True:
        _append_reason(blockers, "return_aligned_single_step_best_action_claim_detected")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "episode_count": episode_count,
        "step_count": step_count,
        "trainable_transition_count": trainable_transition_count,
        "teacher_agreement_rate": teacher_agreement_rate,
    }


def _quasi_real_guarded_ppo_stability_replay_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "replay_count": 0,
        "passed_replay_count": 0,
        "episode_count": 0,
        "step_count": 0,
        "trainable_transition_count": 0,
        "teacher_agreement_rate": 0.0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_ppo_stability_replay_not_passed")
    replay_count = _int_value_or_default(summary.get("replay_count"), 0)
    passed_replay_count = _int_value_or_default(summary.get("passed_replay_count"), 0)
    episode_count = _int_value_or_default(summary.get("episode_count"), 0)
    step_count = _int_value_or_default(summary.get("step_count"), 0)
    trainable_transition_count = _int_value_or_default(
        summary.get("ppo_trainable_transition_count"),
        _int_value_or_default(summary.get("trainable_transition_count"), 0),
    )
    if replay_count < 3:
        _append_reason(blockers, "quasi_real_guarded_ppo_stability_replay_count_below_threshold")
    if passed_replay_count != replay_count:
        _append_reason(blockers, "quasi_real_guarded_ppo_stability_replay_not_all_passed")
    if episode_count < 36:
        _append_reason(blockers, "quasi_real_guarded_ppo_stability_episode_count_below_threshold")
    if step_count < 108:
        _append_reason(blockers, "quasi_real_guarded_ppo_stability_step_count_below_threshold")
    if trainable_transition_count < 24:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_stability_trainable_transition_count_below_threshold",
        )
    for field, reason in (
        ("validation_trainable_count", "quasi_real_guarded_ppo_stability_split_leakage"),
        ("test_trainable_count", "quasi_real_guarded_ppo_stability_split_leakage"),
        ("source_fallback_trainable_count", "quasi_real_guarded_ppo_stability_fallback_trainable"),
        ("teacher_fallback_trainable_count", "quasi_real_guarded_ppo_stability_fallback_trainable"),
        ("missing_observation_count", "quasi_real_guarded_ppo_stability_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_guarded_ppo_stability_contract_invalid"),
        ("missing_value_count", "quasi_real_guarded_ppo_stability_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_guarded_ppo_stability_non_finite_return"),
        ("non_finite_return_count", "quasi_real_guarded_ppo_stability_non_finite_return"),
        ("non_finite_advantage_count", "quasi_real_guarded_ppo_stability_non_finite_return"),
        ("controlled_regression_count", "quasi_real_guarded_ppo_stability_controlled_regression"),
        ("controlled_safety_regression_count", "quasi_real_guarded_ppo_stability_controlled_regression"),
        ("controlled_contract_regression_count", "quasi_real_guarded_ppo_stability_controlled_regression"),
        ("controlled_path_risk_regression_count", "quasi_real_guarded_ppo_stability_controlled_regression"),
        ("controlled_source_selection_regression_count", "quasi_real_guarded_ppo_stability_controlled_regression"),
        ("baseline_replay_behavior_drift_count", "quasi_real_guarded_ppo_stability_behavior_drift"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    teacher_agreement_rate = _float_value_or_default(summary.get("teacher_agreement_rate"), 0.0)
    if teacher_agreement_rate < 0.9:
        _append_reason(blockers, "quasi_real_guarded_ppo_stability_teacher_alignment_insufficient")
    if summary.get("quasi_real_collector_replay_status") != "passed":
        _append_reason(blockers, "quasi_real_guarded_ppo_stability_collector_replay_not_passed")
    if summary.get("long_horizon_verdict") != "long_horizon_teacher_skill_contract_aligned":
        _append_reason(blockers, "quasi_real_guarded_ppo_stability_long_horizon_not_aligned")
    if summary.get("acceptance_contract_refined") is not True:
        _append_reason(blockers, "quasi_real_guarded_ppo_stability_acceptance_contract_missing")
    if summary.get("runs_ppo_update") is True:
        _append_reason(blockers, "limited_ppo_update_unexpected")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "replay_count": replay_count,
        "passed_replay_count": passed_replay_count,
        "episode_count": episode_count,
        "step_count": step_count,
        "trainable_transition_count": trainable_transition_count,
        "teacher_agreement_rate": teacher_agreement_rate,
    }


def _quasi_real_guarded_ppo_horizon5_batch_expansion_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "horizon": 0,
        "replay_count": 0,
        "passed_replay_count": 0,
        "episode_count": 0,
        "step_count": 0,
        "trainable_transition_count": 0,
        "teacher_agreement_rate": 0.0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_ppo_horizon5_batch_expansion_not_passed")
    horizon = _int_value_or_default(summary.get("horizon"), 0)
    replay_count = _int_value_or_default(summary.get("replay_count"), 0)
    passed_replay_count = _int_value_or_default(summary.get("passed_replay_count"), 0)
    episode_count = _int_value_or_default(summary.get("episode_count"), 0)
    step_count = _int_value_or_default(summary.get("step_count"), 0)
    trainable_transition_count = _int_value_or_default(
        summary.get("ppo_trainable_transition_count"),
        _int_value_or_default(summary.get("trainable_transition_count"), 0),
    )
    if horizon != 5:
        _append_reason(blockers, "quasi_real_guarded_ppo_horizon5_horizon_invalid")
    if episode_count < 96:
        _append_reason(blockers, "quasi_real_guarded_ppo_horizon5_episode_count_below_threshold")
    if step_count < 480:
        _append_reason(blockers, "quasi_real_guarded_ppo_horizon5_step_count_below_threshold")
    if trainable_transition_count < 96:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_horizon5_trainable_transition_count_below_threshold",
        )
    if replay_count < 3:
        _append_reason(blockers, "quasi_real_guarded_ppo_horizon5_replay_count_below_threshold")
    if passed_replay_count != replay_count:
        _append_reason(blockers, "quasi_real_guarded_ppo_horizon5_replay_not_all_passed")
    for field, reason in (
        ("validation_trainable_count", "quasi_real_guarded_ppo_horizon5_split_leakage"),
        ("test_trainable_count", "quasi_real_guarded_ppo_horizon5_split_leakage"),
        ("source_fallback_trainable_count", "quasi_real_guarded_ppo_horizon5_fallback_trainable"),
        ("teacher_fallback_trainable_count", "quasi_real_guarded_ppo_horizon5_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "quasi_real_guarded_ppo_horizon5_gate_reason_trainable"),
        ("missing_observation_count", "quasi_real_guarded_ppo_horizon5_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_guarded_ppo_horizon5_contract_invalid"),
        ("missing_value_count", "quasi_real_guarded_ppo_horizon5_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_guarded_ppo_horizon5_non_finite_return"),
        ("non_finite_return_count", "quasi_real_guarded_ppo_horizon5_non_finite_return"),
        ("non_finite_advantage_count", "quasi_real_guarded_ppo_horizon5_non_finite_return"),
        ("controlled_regression_count", "quasi_real_guarded_ppo_horizon5_controlled_regression"),
        ("controlled_safety_regression_count", "quasi_real_guarded_ppo_horizon5_controlled_regression"),
        ("controlled_contract_regression_count", "quasi_real_guarded_ppo_horizon5_controlled_regression"),
        ("controlled_path_risk_regression_count", "quasi_real_guarded_ppo_horizon5_controlled_regression"),
        ("controlled_source_selection_regression_count", "quasi_real_guarded_ppo_horizon5_controlled_regression"),
        ("baseline_replay_behavior_drift_count", "quasi_real_guarded_ppo_horizon5_behavior_drift"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    teacher_agreement_rate = _float_value_or_default(summary.get("teacher_agreement_rate"), 0.0)
    if teacher_agreement_rate < 0.95:
        _append_reason(blockers, "quasi_real_guarded_ppo_horizon5_teacher_alignment_insufficient")
    if summary.get("quasi_real_collector_replay_status") != "passed":
        _append_reason(blockers, "quasi_real_guarded_ppo_horizon5_collector_replay_not_passed")
    replay_trainable = summary.get("quasi_real_collector_replay_trainable_transition_count")
    if replay_trainable is not None and _int_value_or_default(replay_trainable, 0) < 96:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_horizon5_trainable_transition_count_below_threshold",
        )
    if summary.get("long_horizon_verdict") != "long_horizon_teacher_skill_contract_aligned":
        _append_reason(blockers, "quasi_real_guarded_ppo_horizon5_long_horizon_not_aligned")
    if summary.get("uses_multistep_discounted_return") is not True:
        _append_reason(blockers, "return_aligned_multistep_discounted_return_missing")
    if summary.get("not_single_step_best_action") is not True:
        _append_reason(blockers, "return_aligned_single_step_best_action_claim_detected")
    if summary.get("runs_ppo_update") is True:
        _append_reason(blockers, "limited_ppo_update_unexpected")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")
    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "horizon": horizon,
        "replay_count": replay_count,
        "passed_replay_count": passed_replay_count,
        "episode_count": episode_count,
        "step_count": step_count,
        "trainable_transition_count": trainable_transition_count,
        "teacher_agreement_rate": teacher_agreement_rate,
    }


def _quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "horizon": 0,
        "trainable_transition_count": 0,
        "unique_trainable_context_count": 0,
        "seed_count": 0,
        "passed_seed_count": 0,
        "teacher_agreement_rate": 0.0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_ppo_scale512_preflight_not_passed")

    horizon = _int_value_or_default(summary.get("horizon"), 0)
    trainable_transition_count = _int_value_or_default(
        summary.get("ppo_trainable_transition_count"),
        _int_value_or_default(summary.get("trainable_transition_count"), 0),
    )
    unique_trainable_context_count = _int_value_or_default(
        summary.get("unique_trainable_context_count"), 0
    )
    seed_count = _int_value_or_default(summary.get("seed_count"), 0)
    passed_seed_count = _int_value_or_default(summary.get("passed_seed_count"), 0)
    seed_failure_count = _int_value_or_default(summary.get("seed_failure_count"), 0)
    teacher_agreement_rate = _float_value_or_default(
        summary.get("teacher_agreement_rate"), 0.0
    )

    if horizon < 5:
        _append_reason(blockers, "quasi_real_guarded_ppo_scale512_horizon_invalid")
    if trainable_transition_count < 512:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_scale512_trainable_transition_count_below_threshold",
        )
    if unique_trainable_context_count < 512:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_scale512_unique_context_count_below_threshold",
        )
    for field, reason in (
        ("validation_trainable_count", "quasi_real_guarded_ppo_scale512_split_leakage"),
        ("test_trainable_count", "quasi_real_guarded_ppo_scale512_split_leakage"),
        (
            "source_fallback_trainable_count",
            "quasi_real_guarded_ppo_scale512_fallback_trainable",
        ),
        (
            "teacher_fallback_trainable_count",
            "quasi_real_guarded_ppo_scale512_fallback_trainable",
        ),
        (
            "non_empty_gate_reason_trainable_count",
            "quasi_real_guarded_ppo_scale512_gate_reason_trainable",
        ),
        ("missing_observation_count", "quasi_real_guarded_ppo_scale512_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_guarded_ppo_scale512_contract_invalid"),
        ("missing_value_count", "quasi_real_guarded_ppo_scale512_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_guarded_ppo_scale512_non_finite_return"),
        ("non_finite_return_count", "quasi_real_guarded_ppo_scale512_non_finite_return"),
        ("non_finite_advantage_count", "quasi_real_guarded_ppo_scale512_non_finite_return"),
        ("controlled_regression_count", "quasi_real_guarded_ppo_scale512_controlled_regression"),
        (
            "controlled_safety_regression_count",
            "quasi_real_guarded_ppo_scale512_controlled_regression",
        ),
        (
            "controlled_contract_regression_count",
            "quasi_real_guarded_ppo_scale512_controlled_regression",
        ),
        (
            "controlled_path_risk_regression_count",
            "quasi_real_guarded_ppo_scale512_controlled_regression",
        ),
        (
            "controlled_source_selection_regression_count",
            "quasi_real_guarded_ppo_scale512_controlled_regression",
        ),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)

    if teacher_agreement_rate < 0.95:
        _append_reason(blockers, "quasi_real_guarded_ppo_scale512_teacher_alignment_insufficient")
    if seed_count < 3:
        _append_reason(blockers, "quasi_real_guarded_ppo_scale512_seed_count_below_threshold")
    if passed_seed_count != seed_count or seed_failure_count:
        _append_reason(blockers, "quasi_real_guarded_ppo_scale512_seed_smoke_not_all_passed")
    if _float_value_or_default(summary.get("seed_max_old_log_prob_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_ppo_scale512_old_policy_reconstruction_error")
    if _float_value_or_default(summary.get("seed_max_old_value_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_ppo_scale512_old_policy_reconstruction_error")
    for field in (
        "seed_loss_non_finite_count",
        "seed_non_finite_gradient_count",
        "seed_non_finite_reward_count",
        "seed_non_finite_return_count",
        "seed_non_finite_advantage_count",
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, "quasi_real_guarded_ppo_scale512_seed_non_finite")
    if abs(_float_value_or_default(summary.get("seed_max_abs_approx_kl"), 0.0)) > 0.25:
        _append_reason(blockers, "quasi_real_guarded_ppo_scale512_seed_kl_too_large")
    if _float_value_or_default(summary.get("seed_max_grad_norm_after_clip"), 0.0) > 1.0:
        _append_reason(blockers, "quasi_real_guarded_ppo_scale512_seed_grad_norm_too_large")
    min_post_update_trainable = _int_value_or_default(
        summary.get("min_post_update_guarded_collector_trainable_transition_count"), 0
    )
    if min_post_update_trainable < 512:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_scale512_post_update_collector_below_threshold",
        )
    if summary.get("runs_formal_ppo_rollout") is True:
        _append_reason(blockers, "formal_ppo_rollout_unexpected")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "horizon": horizon,
        "trainable_transition_count": trainable_transition_count,
        "unique_trainable_context_count": unique_trainable_context_count,
        "seed_count": seed_count,
        "passed_seed_count": passed_seed_count,
        "teacher_agreement_rate": teacher_agreement_rate,
    }


def _quasi_real_guarded_ppo_iterative_miniloop_stability_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "trainable_transition_count": 0,
        "unique_trainable_context_count": 0,
        "seed_count": 0,
        "iteration_count": 0,
        "passed_iteration_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_ppo_iterative_miniloop_not_passed")

    trainable_transition_count = _int_value_or_default(
        summary.get("input_trainable_transition_count"),
        _int_value_or_default(summary.get("ppo_trainable_transition_count"), 0),
    )
    unique_trainable_context_count = _int_value_or_default(
        summary.get("unique_trainable_context_count"), 0
    )
    seed_count = _int_value_or_default(summary.get("seed_count"), 0)
    iteration_count = _int_value_or_default(summary.get("iteration_count"), 0)
    passed_iteration_count = _int_value_or_default(summary.get("passed_iteration_count"), 0)
    failed_iteration_count = _int_value_or_default(summary.get("failed_iteration_count"), 0)
    expected_iterations = seed_count * iteration_count

    if trainable_transition_count != 684:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_iterative_trainable_transition_count_mismatch",
        )
    if unique_trainable_context_count != 684:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_iterative_unique_context_count_mismatch",
        )
    if seed_count < 3:
        _append_reason(blockers, "quasi_real_guarded_ppo_iterative_seed_count_below_threshold")
    if iteration_count < 3:
        _append_reason(
            blockers,
            "quasi_real_guarded_ppo_iterative_iteration_count_below_threshold",
        )
    if passed_iteration_count != expected_iterations or failed_iteration_count:
        _append_reason(blockers, "quasi_real_guarded_ppo_iterative_iterations_not_all_passed")
    if _int_value_or_default(summary.get("min_optimizer_train_transition_count"), 0) != 684:
        _append_reason(blockers, "quasi_real_guarded_ppo_iterative_optimizer_count_mismatch")

    for field, reason in (
        ("validation_trainable_count", "quasi_real_guarded_ppo_iterative_split_leakage"),
        ("test_trainable_count", "quasi_real_guarded_ppo_iterative_split_leakage"),
        ("source_fallback_trainable_count", "quasi_real_guarded_ppo_iterative_fallback_trainable"),
        ("teacher_fallback_trainable_count", "quasi_real_guarded_ppo_iterative_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "quasi_real_guarded_ppo_iterative_gate_reason_trainable"),
        ("missing_observation_count", "quasi_real_guarded_ppo_iterative_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_guarded_ppo_iterative_contract_invalid"),
        ("missing_value_count", "quasi_real_guarded_ppo_iterative_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_guarded_ppo_iterative_non_finite"),
        ("non_finite_return_count", "quasi_real_guarded_ppo_iterative_non_finite"),
        ("non_finite_advantage_count", "quasi_real_guarded_ppo_iterative_non_finite"),
        ("loss_non_finite_count", "quasi_real_guarded_ppo_iterative_non_finite"),
        ("non_finite_gradient_count", "quasi_real_guarded_ppo_iterative_non_finite"),
        ("controlled_regression_count", "quasi_real_guarded_ppo_iterative_controlled_regression"),
        ("controlled_safety_regression_count", "quasi_real_guarded_ppo_iterative_controlled_regression"),
        ("controlled_contract_regression_count", "quasi_real_guarded_ppo_iterative_controlled_regression"),
        ("controlled_path_risk_regression_count", "quasi_real_guarded_ppo_iterative_controlled_regression"),
        ("controlled_source_selection_regression_count", "quasi_real_guarded_ppo_iterative_controlled_regression"),
        ("behavior_drift_count", "quasi_real_guarded_ppo_iterative_behavior_drift"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)

    if _float_value_or_default(summary.get("max_old_log_prob_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_ppo_iterative_old_policy_reconstruction_error")
    if _float_value_or_default(summary.get("max_old_value_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_ppo_iterative_old_policy_reconstruction_error")
    if abs(_float_value_or_default(summary.get("max_abs_approx_kl"), 0.0)) > 0.25:
        _append_reason(blockers, "quasi_real_guarded_ppo_iterative_kl_too_large")
    if _float_value_or_default(summary.get("max_grad_norm_after_clip"), 0.0) > 1.0:
        _append_reason(blockers, "quasi_real_guarded_ppo_iterative_grad_norm_too_large")
    if _float_value_or_default(summary.get("min_teacher_agreement_rate"), 0.0) < 0.95:
        _append_reason(blockers, "quasi_real_guarded_ppo_iterative_teacher_alignment_insufficient")
    if summary.get("runs_formal_ppo_rollout") is True:
        _append_reason(blockers, "formal_ppo_rollout_unexpected")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "trainable_transition_count": trainable_transition_count,
        "unique_trainable_context_count": unique_trainable_context_count,
        "seed_count": seed_count,
        "iteration_count": iteration_count,
        "passed_iteration_count": passed_iteration_count,
    }


def _quasi_real_guarded_formal_ppo_preflight_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "trainable_transition_count": 0,
        "optimizer_train_transition_count": 0,
        "unique_trainable_context_count": 0,
        "seed_count": 0,
        "passed_seed_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_not_passed")

    trainable_transition_count = _int_value_or_default(
        summary.get("input_trainable_transition_count"), 0
    )
    optimizer_train_transition_count = _int_value_or_default(
        summary.get("optimizer_train_transition_count"), 0
    )
    unique_trainable_context_count = _int_value_or_default(
        summary.get("unique_trainable_context_count"), 0
    )
    seed_count = _int_value_or_default(summary.get("seed_count"), 0)
    passed_seed_count = _int_value_or_default(summary.get("passed_seed_count"), 0)
    if trainable_transition_count != 684 or optimizer_train_transition_count != 684:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_trainable_count_mismatch")
    if unique_trainable_context_count != 684:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_unique_context_count_mismatch")
    if seed_count < 3 or passed_seed_count != seed_count:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_seed_not_all_passed")
    for field, reason in (
        ("validation_trainable_count", "quasi_real_guarded_formal_ppo_preflight_split_leakage"),
        ("test_trainable_count", "quasi_real_guarded_formal_ppo_preflight_split_leakage"),
        ("fallback_trainable_count", "quasi_real_guarded_formal_ppo_preflight_fallback_trainable"),
        ("source_fallback_trainable_count", "quasi_real_guarded_formal_ppo_preflight_fallback_trainable"),
        ("teacher_fallback_trainable_count", "quasi_real_guarded_formal_ppo_preflight_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "quasi_real_guarded_formal_ppo_preflight_gate_reason_trainable"),
        ("missing_observation_count", "quasi_real_guarded_formal_ppo_preflight_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_guarded_formal_ppo_preflight_contract_invalid"),
        ("missing_value_count", "quasi_real_guarded_formal_ppo_preflight_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_guarded_formal_ppo_preflight_non_finite"),
        ("non_finite_return_count", "quasi_real_guarded_formal_ppo_preflight_non_finite"),
        ("non_finite_advantage_count", "quasi_real_guarded_formal_ppo_preflight_non_finite"),
        ("loss_non_finite_count", "quasi_real_guarded_formal_ppo_preflight_non_finite"),
        ("non_finite_gradient_count", "quasi_real_guarded_formal_ppo_preflight_non_finite"),
        ("controlled_regression_count", "quasi_real_guarded_formal_ppo_preflight_controlled_regression"),
        ("controlled_safety_regression_count", "quasi_real_guarded_formal_ppo_preflight_controlled_regression"),
        ("controlled_contract_regression_count", "quasi_real_guarded_formal_ppo_preflight_controlled_regression"),
        ("controlled_path_risk_regression_count", "quasi_real_guarded_formal_ppo_preflight_controlled_regression"),
        ("controlled_source_selection_regression_count", "quasi_real_guarded_formal_ppo_preflight_controlled_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if _float_value_or_default(summary.get("max_old_log_prob_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_old_policy_reconstruction_error")
    if _float_value_or_default(summary.get("max_old_value_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_old_policy_reconstruction_error")
    if abs(_float_value_or_default(summary.get("max_abs_approx_kl"), 0.0)) > 0.25:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_kl_too_large")
    if _float_value_or_default(summary.get("max_grad_norm_after_clip"), 0.0) > 1.0:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_grad_norm_too_large")
    if _float_value_or_default(summary.get("min_parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_parameter_delta_missing")
    if _float_value_or_default(summary.get("teacher_agreement_rate"), 0.0) < 0.95:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_teacher_alignment_insufficient")
    if not summary.get("rollback_manifest"):
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_preflight_rollback_manifest_missing")
    if summary.get("runs_formal_ppo_rollout") is True:
        _append_reason(blockers, "formal_ppo_rollout_unexpected")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "trainable_transition_count": trainable_transition_count,
        "optimizer_train_transition_count": optimizer_train_transition_count,
        "unique_trainable_context_count": unique_trainable_context_count,
        "seed_count": seed_count,
        "passed_seed_count": passed_seed_count,
    }


def _quasi_real_guarded_formal_ppo_rollout_canary_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "trainable_transition_count": 0,
        "optimizer_train_transition_count": 0,
        "unique_trainable_context_count": 0,
        "seed_count": 0,
        "passed_seed_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_not_passed")

    trainable_transition_count = _int_value_or_default(
        summary.get("input_trainable_transition_count"), 0
    )
    optimizer_train_transition_count = _int_value_or_default(
        summary.get("optimizer_train_transition_count"), 0
    )
    unique_trainable_context_count = _int_value_or_default(
        summary.get("unique_trainable_context_count"), 0
    )
    seed_count = _int_value_or_default(summary.get("seed_count"), 0)
    passed_seed_count = _int_value_or_default(summary.get("passed_seed_count"), 0)
    if trainable_transition_count != 684 or optimizer_train_transition_count != 684:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_trainable_count_mismatch")
    if unique_trainable_context_count != 684:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_unique_context_count_mismatch")
    if seed_count < 3 or passed_seed_count != seed_count:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_seed_not_all_passed")
    for field, reason in (
        ("validation_trainable_count", "quasi_real_guarded_formal_ppo_rollout_canary_split_leakage"),
        ("test_trainable_count", "quasi_real_guarded_formal_ppo_rollout_canary_split_leakage"),
        ("fallback_trainable_count", "quasi_real_guarded_formal_ppo_rollout_canary_fallback_trainable"),
        ("source_fallback_trainable_count", "quasi_real_guarded_formal_ppo_rollout_canary_fallback_trainable"),
        ("teacher_fallback_trainable_count", "quasi_real_guarded_formal_ppo_rollout_canary_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "quasi_real_guarded_formal_ppo_rollout_canary_gate_reason_trainable"),
        ("missing_observation_count", "quasi_real_guarded_formal_ppo_rollout_canary_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_guarded_formal_ppo_rollout_canary_contract_invalid"),
        ("missing_value_count", "quasi_real_guarded_formal_ppo_rollout_canary_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_guarded_formal_ppo_rollout_canary_non_finite"),
        ("non_finite_return_count", "quasi_real_guarded_formal_ppo_rollout_canary_non_finite"),
        ("non_finite_advantage_count", "quasi_real_guarded_formal_ppo_rollout_canary_non_finite"),
        ("loss_non_finite_count", "quasi_real_guarded_formal_ppo_rollout_canary_non_finite"),
        ("non_finite_gradient_count", "quasi_real_guarded_formal_ppo_rollout_canary_non_finite"),
        ("controlled_regression_count", "quasi_real_guarded_formal_ppo_rollout_canary_controlled_regression"),
        ("controlled_safety_regression_count", "quasi_real_guarded_formal_ppo_rollout_canary_controlled_regression"),
        ("controlled_contract_regression_count", "quasi_real_guarded_formal_ppo_rollout_canary_controlled_regression"),
        ("controlled_path_risk_regression_count", "quasi_real_guarded_formal_ppo_rollout_canary_controlled_regression"),
        ("controlled_source_selection_regression_count", "quasi_real_guarded_formal_ppo_rollout_canary_controlled_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if _float_value_or_default(summary.get("max_old_log_prob_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_old_policy_reconstruction_error")
    if _float_value_or_default(summary.get("max_old_value_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_old_policy_reconstruction_error")
    if abs(_float_value_or_default(summary.get("max_abs_approx_kl"), 0.0)) > 0.25:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_kl_too_large")
    if _float_value_or_default(summary.get("max_grad_norm_after_clip"), 0.0) > 1.0:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_grad_norm_too_large")
    if _float_value_or_default(summary.get("min_parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_parameter_delta_missing")
    if _float_value_or_default(summary.get("teacher_agreement_rate"), 0.0) < 0.95:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_teacher_alignment_insufficient")
    if not summary.get("rollback_manifest"):
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_rollback_manifest_missing")
    if summary.get("runs_formal_ppo_rollout_canary") is not True:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_rollout_canary_not_run")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "trainable_transition_count": trainable_transition_count,
        "optimizer_train_transition_count": optimizer_train_transition_count,
        "unique_trainable_context_count": unique_trainable_context_count,
        "seed_count": seed_count,
        "passed_seed_count": passed_seed_count,
    }


def _quasi_real_guarded_formal_ppo_stability_holdout_readiness(
    summary: dict[str, Any],
) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "trainable_transition_count": 0,
        "optimizer_train_transition_count": 0,
        "unique_trainable_context_count": 0,
        "seed_count": 0,
        "budget_count": 0,
        "run_count": 0,
        "passed_run_count": 0,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_not_passed")

    trainable_transition_count = _int_value_or_default(
        summary.get("input_trainable_transition_count"), 0
    )
    optimizer_train_transition_count = _int_value_or_default(
        summary.get("optimizer_train_transition_count"), 0
    )
    unique_trainable_context_count = _int_value_or_default(
        summary.get("unique_trainable_context_count"), 0
    )
    seed_count = _int_value_or_default(summary.get("seed_count"), 0)
    budget_count = _int_value_or_default(summary.get("budget_count"), 0)
    run_count = _int_value_or_default(summary.get("run_count"), 0)
    passed_run_count = _int_value_or_default(summary.get("passed_run_count"), 0)
    run_failure_count = _int_value_or_default(summary.get("run_failure_count"), 0)
    if trainable_transition_count != 684 or optimizer_train_transition_count != 684:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_trainable_count_mismatch")
    if unique_trainable_context_count != 684:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_unique_context_count_mismatch")
    if seed_count < 5:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_seed_count_below_threshold")
    if budget_count < 2:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_budget_count_below_threshold")
    if run_count < seed_count * budget_count or passed_run_count != run_count or run_failure_count:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_run_not_all_passed")
    for field, reason in (
        ("validation_trainable_count", "quasi_real_guarded_formal_ppo_stability_holdout_split_leakage"),
        ("test_trainable_count", "quasi_real_guarded_formal_ppo_stability_holdout_split_leakage"),
        ("fallback_trainable_count", "quasi_real_guarded_formal_ppo_stability_holdout_fallback_trainable"),
        ("source_fallback_trainable_count", "quasi_real_guarded_formal_ppo_stability_holdout_fallback_trainable"),
        ("teacher_fallback_trainable_count", "quasi_real_guarded_formal_ppo_stability_holdout_fallback_trainable"),
        ("non_empty_gate_reason_trainable_count", "quasi_real_guarded_formal_ppo_stability_holdout_gate_reason_trainable"),
        ("missing_observation_count", "quasi_real_guarded_formal_ppo_stability_holdout_contract_invalid"),
        ("missing_log_prob_count", "quasi_real_guarded_formal_ppo_stability_holdout_contract_invalid"),
        ("missing_value_count", "quasi_real_guarded_formal_ppo_stability_holdout_contract_invalid"),
        ("non_finite_reward_count", "quasi_real_guarded_formal_ppo_stability_holdout_non_finite"),
        ("non_finite_return_count", "quasi_real_guarded_formal_ppo_stability_holdout_non_finite"),
        ("non_finite_advantage_count", "quasi_real_guarded_formal_ppo_stability_holdout_non_finite"),
        ("loss_non_finite_count", "quasi_real_guarded_formal_ppo_stability_holdout_non_finite"),
        ("non_finite_gradient_count", "quasi_real_guarded_formal_ppo_stability_holdout_non_finite"),
        ("controlled_regression_count", "quasi_real_guarded_formal_ppo_stability_holdout_controlled_regression"),
        ("train_controlled_regression_count", "quasi_real_guarded_formal_ppo_stability_holdout_controlled_regression"),
        ("validation_controlled_regression_count", "quasi_real_guarded_formal_ppo_stability_holdout_controlled_regression"),
        ("test_controlled_regression_count", "quasi_real_guarded_formal_ppo_stability_holdout_controlled_regression"),
        ("controlled_safety_regression_count", "quasi_real_guarded_formal_ppo_stability_holdout_controlled_regression"),
        ("controlled_contract_regression_count", "quasi_real_guarded_formal_ppo_stability_holdout_controlled_regression"),
        ("controlled_path_risk_regression_count", "quasi_real_guarded_formal_ppo_stability_holdout_controlled_regression"),
        ("controlled_source_selection_regression_count", "quasi_real_guarded_formal_ppo_stability_holdout_controlled_regression"),
        ("family_regression_count", "quasi_real_guarded_formal_ppo_stability_holdout_family_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)
    if _float_value_or_default(summary.get("max_old_log_prob_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_old_policy_reconstruction_error")
    if _float_value_or_default(summary.get("max_old_value_abs_error"), 0.0) > 1.0e-4:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_old_policy_reconstruction_error")
    if abs(_float_value_or_default(summary.get("max_abs_approx_kl"), 0.0)) > 0.25:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_kl_too_large")
    if _float_value_or_default(summary.get("max_grad_norm_after_clip"), 0.0) > 1.0:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_grad_norm_too_large")
    if _float_value_or_default(summary.get("min_parameter_l2_delta"), 0.0) <= 0.0:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_parameter_delta_missing")
    if _float_value_or_default(summary.get("teacher_agreement_rate"), 0.0) < 0.95:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_teacher_alignment_insufficient")
    if not summary.get("rollback_manifest") or not summary.get("baseline_manifest"):
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_rollback_manifest_missing")
    if summary.get("runs_formal_ppo_stability_holdout_validation") is not True:
        _append_reason(blockers, "quasi_real_guarded_formal_ppo_stability_holdout_not_run")
    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "limited_ppo_update_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "limited_ppo_update_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_policy_performance_claimed")
    if summary.get("formal_training_ready_claimed") is True:
        _append_reason(blockers, "limited_ppo_update_formal_training_ready_claimed")
    if _git_current_matches(summary) is False:
        _append_reason(blockers, "clean_head_evidence_refresh_required")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "trainable_transition_count": trainable_transition_count,
        "optimizer_train_transition_count": optimizer_train_transition_count,
        "unique_trainable_context_count": unique_trainable_context_count,
        "seed_count": seed_count,
        "budget_count": budget_count,
        "run_count": run_count,
        "passed_run_count": passed_run_count,
    }


def _policy_gated_canary_rollout_readiness(summary: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "present": False,
        "completed": False,
        "training_blockers": [],
        "next_required_change": None,
        "formal_training_ready_claimed": False,
        "performance_claimed": False,
        "policy_decision_count": 0,
        "canary_opportunity_context_count": 0,
        "policy_changed_decision_count": 0,
        "canary_accepted_policy_choice_count": 0,
        "canary_rejected_policy_choice_count": 0,
        "controlled_regression_count": 0,
        "raw_policy_regression_count": 0,
        "invalid_action_mask_count": 0,
        "fallback_or_open_grid_count": 0,
        "safety_regression_count": 0,
        "contract_violation_count": 0,
        "path_cost_regression_count": 0,
        "risk_regression_count": 0,
        "source_selection_regression_count": 0,
        "scenario_family_count": 0,
        "accepted_scenario_family_count": 0,
        "canary_diversity_passed": False,
        "family_with_acceptable_alternative_count": 0,
        "source_aligned_with_acceptable_alternative_count": 0,
        "canary_missed_opportunity_preference_pair_count": 0,
        "missed_safe_choice_family_count": 0,
        "hard_positive_added_count": 0,
        "dense_choke_acceptable_alternative_count": 0,
        "dense_choke_accepted_policy_choice_count": 0,
        "canary_full_family_opportunity_passed": False,
        "canary_opportunity_quality_passed": False,
        "accepted_equal_choice_count": 0,
        "accepted_better_choice_count": 0,
        "accepted_better_family_count": 0,
        "policy_change_rate": 0.0,
        "accepted_choice_rate": 0.0,
        "canary_value_stability_passed": False,
    }
    if not summary:
        return empty

    blockers: list[str] = []
    if summary.get("status") != "passed" or _string_list(summary.get("reason_codes")):
        _append_reason(blockers, "policy_gated_canary_rollout_not_passed")
    if _int_value_or_default(summary.get("policy_decision_count"), 0) <= 0:
        _append_reason(blockers, "policy_gated_canary_decision_count_zero")
    if _int_value_or_default(summary.get("canary_opportunity_context_count"), 0) <= 0:
        _append_reason(blockers, "policy_gated_canary_opportunity_count_zero")
    if _int_value_or_default(summary.get("policy_changed_decision_count"), 0) <= 0:
        _append_reason(blockers, "policy_gated_canary_changed_decision_count_zero")
    if _int_value_or_default(summary.get("canary_accepted_policy_choice_count"), 0) <= 0:
        _append_reason(blockers, "policy_gated_canary_accepted_choice_count_zero")

    for field, reason in (
        ("controlled_regression_count", "policy_gated_canary_controlled_regression"),
        ("invalid_action_mask_count", "policy_gated_canary_invalid_action_mask"),
        ("fallback_or_open_grid_count", "policy_gated_canary_fallback_or_open_grid"),
        ("safety_regression_count", "policy_gated_canary_safety_regression"),
        ("contract_violation_count", "policy_gated_canary_contract_violation"),
        ("path_cost_regression_count", "policy_gated_canary_path_cost_regression"),
        ("risk_regression_count", "policy_gated_canary_risk_regression"),
        ("source_selection_regression_count", "policy_gated_canary_source_selection_regression"),
    ):
        if _int_value_or_default(summary.get(field), 0):
            _append_reason(blockers, reason)

    if summary.get("publishes_checkpoint") is True:
        _append_reason(blockers, "policy_gated_canary_checkpoint_publication_claimed")
    if summary.get("replaces_default_policy") is True:
        _append_reason(blockers, "policy_gated_canary_default_policy_replacement_claimed")
    if summary.get("performance_claimed") is True:
        _append_reason(blockers, "policy_gated_canary_policy_performance_claimed")
    if summary.get("candidate_git_current_matches_sources") is False:
        _append_reason(blockers, "policy_gated_canary_candidate_git_current_mismatch")
    if summary.get("checkpoint_metadata_git_current_matches_sources") is False:
        _append_reason(blockers, "policy_gated_canary_checkpoint_metadata_git_current_mismatch")

    formal_training_ready_claimed = bool(
        summary.get("formal_training_ready_claimed")
        or summary.get("policy_training_ready")
        or summary.get("performance_claimed")
    )
    if formal_training_ready_claimed:
        _append_reason(blockers, "policy_gated_canary_formal_training_ready_claimed")

    return {
        "present": True,
        "completed": not blockers,
        "training_blockers": blockers,
        "next_required_change": summary.get("next_required_change") if blockers else None,
        "formal_training_ready_claimed": formal_training_ready_claimed,
        "performance_claimed": bool(summary.get("performance_claimed")),
        "policy_decision_count": _int_value_or_default(summary.get("policy_decision_count"), 0),
        "canary_opportunity_context_count": _int_value_or_default(
            summary.get("canary_opportunity_context_count"),
            0,
        ),
        "policy_changed_decision_count": _int_value_or_default(
            summary.get("policy_changed_decision_count"),
            0,
        ),
        "canary_accepted_policy_choice_count": _int_value_or_default(
            summary.get("canary_accepted_policy_choice_count"),
            0,
        ),
        "canary_rejected_policy_choice_count": _int_value_or_default(
            summary.get("canary_rejected_policy_choice_count"),
            0,
        ),
        "accepted_equal_choice_count": _int_value_or_default(
            summary.get("accepted_equal_choice_count"),
            0,
        ),
        "accepted_better_choice_count": _int_value_or_default(
            summary.get("accepted_better_choice_count"),
            0,
        ),
        "accepted_better_family_count": _int_value_or_default(
            summary.get("accepted_better_family_count"),
            0,
        ),
        "policy_change_rate": _float_value_or_default(
            summary.get("policy_change_rate"),
            0.0,
        ),
        "accepted_choice_rate": _float_value_or_default(
            summary.get("accepted_choice_rate"),
            0.0,
        ),
        "controlled_regression_count": _int_value_or_default(
            summary.get("controlled_regression_count"),
            0,
        ),
        "raw_policy_regression_count": _int_value_or_default(
            summary.get("raw_policy_regression_count"),
            0,
        ),
        "invalid_action_mask_count": _int_value_or_default(
            summary.get("invalid_action_mask_count"),
            0,
        ),
        "fallback_or_open_grid_count": _int_value_or_default(
            summary.get("fallback_or_open_grid_count"),
            0,
        ),
        "safety_regression_count": _int_value_or_default(
            summary.get("safety_regression_count"),
            0,
        ),
        "contract_violation_count": _int_value_or_default(
            summary.get("contract_violation_count"),
            0,
        ),
        "path_cost_regression_count": _int_value_or_default(
            summary.get("path_cost_regression_count"),
            0,
        ),
        "risk_regression_count": _int_value_or_default(
            summary.get("risk_regression_count"),
            0,
        ),
        "source_selection_regression_count": _int_value_or_default(
            summary.get("source_selection_regression_count"),
            0,
        ),
        "scenario_family_count": _int_value_or_default(
            summary.get("scenario_family_count"),
            0,
        ),
        "accepted_scenario_family_count": _int_value_or_default(
            summary.get("accepted_scenario_family_count"),
            0,
        ),
        "canary_diversity_passed": bool(summary.get("canary_diversity_passed")),
        "family_with_acceptable_alternative_count": _int_value_or_default(
            summary.get("family_with_acceptable_alternative_count"),
            0,
        ),
        "source_aligned_with_acceptable_alternative_count": _int_value_or_default(
            summary.get("source_aligned_with_acceptable_alternative_count"),
            0,
        ),
        "canary_missed_opportunity_preference_pair_count": _int_value_or_default(
            summary.get("canary_missed_opportunity_preference_pair_count"),
            0,
        ),
        "missed_safe_choice_family_count": _int_value_or_default(
            summary.get("missed_safe_choice_family_count"),
            0,
        ),
        "hard_positive_added_count": _int_value_or_default(
            summary.get("hard_positive_added_count"),
            0,
        ),
        "dense_choke_acceptable_alternative_count": _int_value_or_default(
            summary.get("dense_choke_acceptable_alternative_count"),
            0,
        ),
        "dense_choke_accepted_policy_choice_count": _int_value_or_default(
            summary.get("dense_choke_accepted_policy_choice_count"),
            0,
        ),
        "canary_full_family_opportunity_passed": bool(
            summary.get("canary_full_family_opportunity_passed")
        ),
        "canary_opportunity_quality_passed": bool(
            summary.get("canary_opportunity_quality_passed")
        ),
        "canary_value_stability_passed": bool(
            summary.get("canary_value_stability_passed")
        ),
    }


def _candidate_nontrainable_reason_count(payload: dict[str, Any], reason: str) -> int:
    explicit_fields = {
        "anchor_unreachable": (
            "nontrainable_anchor_unreachable_count",
            "anchor_unreachable_count",
        ),
        "source_candidate_not_selected": (
            "nontrainable_source_candidate_not_selected_count",
            "source_candidate_not_selected_count",
        ),
    }
    for field in explicit_fields.get(reason, ()):
        if payload.get(field) is not None:
            return _int_value_or_default(payload.get(field), 0)
    diagnosis = payload.get("anchor_projection_coverage_diagnosis")
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    reason_counts = diagnosis.get("nontrainable_primary_reason_counts")
    if isinstance(reason_counts, dict) and reason_counts.get(reason) is not None:
        return _int_value_or_default(reason_counts.get(reason), 0)
    fallback_fields = {
        "anchor_unreachable": "anchor_unreachable_not_generated_count",
        "source_candidate_not_selected": "projected_candidate_not_source_selected_count",
    }
    field = fallback_fields.get(reason)
    if field:
        return _int_value_or_default(diagnosis.get(field), 0)
    return 0


def _coverage_diagnosis_int(payload: dict[str, Any], field: str) -> int:
    diagnosis = payload.get("anchor_projection_coverage_diagnosis")
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    return _int_value_or_default(diagnosis.get(field), 0)


def _coverage_diagnosis_mapping(payload: dict[str, Any], field: str) -> dict[str, Any]:
    diagnosis = payload.get("anchor_projection_coverage_diagnosis")
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    value = diagnosis.get(field)
    return value if isinstance(value, dict) else {}


def _mapping_or_empty(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _quality_regression_count_from_contexts(payload: dict[str, Any]) -> int:
    contexts = payload.get("context_records")
    if not isinstance(contexts, list):
        return 0
    return sum(
        1
        for context in contexts
        if isinstance(context, dict)
        and (
            context.get("source_selection_quality_regression") is True
            or "source_selection_quality_regression" in _string_list(context.get("reject_reasons"))
        )
    )


def _max_context_field(
    payload: dict[str, Any],
    field: str,
    *,
    predicate: Any | None = None,
) -> float | None:
    contexts = payload.get("context_records")
    if not isinstance(contexts, list):
        return None
    return _max_numeric(
        *(
            context.get(field)
            for context in contexts
            if isinstance(context, dict) and (predicate is None or predicate(context))
        )
    )


def _context_margin_can_block_readiness(context: dict[str, Any]) -> bool:
    if (
        context.get("source_selection_quality_regression") is True
        or "source_selection_quality_regression" in _string_list(context.get("reject_reasons"))
    ):
        return True
    return (
        context.get("training_use") == "trainable_anchor_projection_contrast"
        and context.get("source_selection_status") in {"source_selected", "source_selected_quality_regression", None}
    )


def _max_numeric(*values: Any) -> float | None:
    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric_values:
        return None
    return max(numeric_values)


def _output_file(batch_root: Path, config: dict[str, Any]) -> Path:
    return batch_root / config["output_files"]["policy_training_readiness_review_summary"]


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": config.get("schema_version"),
        "validation": dict(config.get("validation", {})) if isinstance(config.get("validation"), dict) else {},
        "readiness_thresholds": dict(config.get("readiness_thresholds", {})),
        "output_files": dict(config.get("output_files", {})),
    }


def _require_current_git_match(config: dict[str, Any]) -> bool:
    validation = config.get("validation")
    if not isinstance(validation, dict):
        return True
    return bool(validation.get("require_current_git_match", True))


def _allow_dirty_current_git_match(config: dict[str, Any]) -> bool:
    validation = config.get("validation")
    if not isinstance(validation, dict):
        return False
    return bool(validation.get("allow_dirty_current_git_match", False))


def _fail_on_input_failure(config: dict[str, Any]) -> bool:
    validation = config.get("validation")
    if not isinstance(validation, dict):
        return True
    return bool(validation.get("fail_on_input_failure", True))


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_reason(reason_codes: list[str], code: str) -> None:
    if code not in reason_codes:
        reason_codes.append(code)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def _max_counter_dict(*values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, count in value.items():
            parsed = _int_value_or_default(count, 0)
            key_text = str(key)
            result[key_text] = max(result.get(key_text, 0), parsed)
    return dict(sorted(result.items()))


def _int_value(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be an integer") from exc


def _float_value(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a number") from exc


def _optional_nonnegative_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    parsed = _float_value(value, label)
    if parsed < 0.0:
        raise ConfigError(f"{label} must be >= 0")
    return parsed


def _int_value_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _git_current_matches(summary: dict[str, Any]) -> bool | None:
    provenance = summary.get("git_provenance")
    if isinstance(provenance, dict) and provenance.get("current_matches_sources") is False:
        return False
    return None


if __name__ == "__main__":
    raise SystemExit(main())
