from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now
from global_99_governance_common import global_99_boundary_defaults
from platform_command import display_command, python_script_command


CONFIG_SCHEMA_VERSION = "xunce-stage18-research-evidence-pipeline-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage18-research-evidence-pipeline-summary/v1"
STAGE18_5_ATTRIBUTION_SCHEMA_VERSION = "xunce-stage18-5-evidence-attribution-summary/v1"
STAGE18_5_GUARD_SCHEMA_VERSION = "xunce-stage18-5-guard-evaluation/v1"
STAGE18_5_ROUTING_SCHEMA_VERSION = "xunce-stage18-5-next-stage-routing/v1"
STAGE18_6_GUARD_REFINEMENT_SCHEMA_VERSION = "xunce-stage18-6-guard-refinement-summary/v1"
STAGE18_6_ROUTING_SCHEMA_VERSION = "xunce-stage18-6-next-stage-routing/v1"
STAGE18_7_CANDIDATE_COUNT_SCALING_SCHEMA_VERSION = "xunce-stage18-7-candidate-count-scaling-summary/v1"
STAGE18_7_ROUTING_SCHEMA_VERSION = "xunce-stage18-7-next-stage-routing/v1"
STAGE18_9_TRAJECTORY_RISK_REWARD_SCHEMA_VERSION = "xunce-stage18-9-trajectory-risk-reward-summary/v1"
STAGE18_9_ROUTING_SCHEMA_VERSION = "xunce-stage18-9-next-stage-routing/v1"
STAGE18_9_STAGE19_READINESS_SCHEMA_VERSION = "xunce-stage18-9-stage19-readiness/v1"
STAGE18_9_EXPECTED_PROFILE_ID = "xunce-coverage-cost-risk-boundary-v3"
STAGE18_9_EXPECTED_PROFILE_VERSION = "v3"
STAGE18_11_PATH_COST_WEIGHT_SCHEMA_VERSION = "xunce-stage18-11-path-cost-weight-calibration-summary/v1"
STAGE18_11_ROUTING_SCHEMA_VERSION = "xunce-stage18-11-next-stage-routing/v1"
STAGE19_EVALUATOR_CRITIC_SCHEMA_VERSION = "xunce-stage19-evaluator-critic-preflight-summary/v1"
STAGE19_ROUTING_SCHEMA_VERSION = "xunce-stage19-next-stage-routing/v1"

ROI_EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
STAGE18I_SUMMARY_FILES = (
    "xunce-true-frontier-nbv-candidate-source-summary.json",
    "xunce-risk-aware-frontier-nbv-candidate-repair-summary.json",
    "xunce-risk-constrained-frontier-nbv-candidate-generation-summary.json",
)
MODEL_COMPARISON_SUMMARY_FILE = "xunce-high-fidelity-real-map-comparison-summary.json"
TRUE_BINDING_SUMMARY_FILE = "xunce-true-incumbent-selection-binding-summary.json"
QUANTIZATION_SUMMARY_FILE = "xunce-risk-coverage-cost-quantization-summary.json"
ORACLE_SUMMARY_FILE = "xunce-oracle-separability-summary.json"
COVERAGE_COMPARISON_SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
COVERAGE_COMPARISON_AGGREGATE_FILE = "xunce-exploration-coverage-comparison-aggregate.json"

REVIEW_METRICS_NEXT_REQUIRED_CHANGE = "review_xunce_incumbent_comparison_metrics"
STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE = "prepare_stage19_evaluator_critic_preflight"
REFRESH_STAGE18_NEXT_REQUIRED_CHANGE = "refresh_stage18_research_evidence_pipeline"
ROOT_REPAIR_NEXT_REQUIRED_CHANGE = "rerun_stage18_downstream_evidence_for_candidate_root"
BOUNDARY_REPAIR_NEXT_REQUIRED_CHANGE = "resolve_stage18_research_evidence_boundary_rejections"
DYNAMIC_ROLLOUT_NEXT_REQUIRED_CHANGE = "run_dynamic_frontier_nbv_rollout_comparison"
STAGE18_5_ATTRIBUTION_SUMMARY_FILE = "xunce-stage18-5-evidence-attribution-summary.json"
STAGE18_6_GUARD_REFINEMENT_SUMMARY_FILE = "xunce-stage18-6-guard-refinement-summary.json"
STAGE18_7_CANDIDATE_COUNT_SCALING_SUMMARY_FILE = "xunce-stage18-7-candidate-count-scaling-summary.json"
STAGE18_9_TRAJECTORY_RISK_REWARD_SUMMARY_FILE = "xunce-stage18-9-trajectory-risk-reward-summary.json"
STAGE18_11_PATH_COST_WEIGHT_SUMMARY_FILE = "xunce-stage18-11-path-cost-weight-calibration-summary.json"
STAGE19_EVALUATOR_CRITIC_SUMMARY_FILE = "xunce-stage19-evaluator-critic-preflight-summary.json"
STAGE18_5_ALLOWED_ROUTES = {
    "rerun_xunce_stage18_4e_coverage_comparison_with_required_artifacts",
    "resolve_stage18_5_evidence_review_boundary_rejections",
    "refine_coverage_reward_and_cost_guard",
    "establish_same_candidate_set_policy_selection_advantage",
    STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE,
}
STAGE18_6_ALLOWED_ROUTES = {
    "rerun_stage18_4e_with_candidate_metric_audit",
    "rerun_xunce_stage18_5_evidence_attribution_review",
    "rerun_xunce_stage18_4e_coverage_comparison_with_required_artifacts",
    "resolve_stage18_6_guard_refinement_boundary_rejections",
    "expand_candidate_generation_roi_complexity",
    "refine_coverage_reward_and_cost_guard",
    STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE,
}
STAGE18_7_ALLOWED_ROUTES = {
    "run_missing_candidate_count_sweeps_with_metric_audit",
    "repair_stage18_7_lineage_or_config_drift",
    "continue_candidate_count_scaling_with_bounded_budget",
    "expand_candidate_generation_roi_complexity",
    "refine_coverage_reward_and_cost_guard",
    "resolve_stage18_7_candidate_count_scaling_boundary_rejections",
    STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE,
}
STAGE18_9_ALLOWED_ROUTES = {
    "resolve_stage18_9_boundary_rejections",
    "rerun_required_stage18_9_inputs",
    "repair_path_risk_boundary_filtering",
    "refine_coverage_cost_reward_weights",
    "calibrate_soft_risk_exposure_weight",
    STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE,
}
STAGE18_11_ALLOWED_ROUTES = {
    "resolve_stage18_11_boundary_rejections",
    "rerun_stage18_11_required_inputs",
    "repair_path_risk_boundary_filtering",
    "run_stage18_11_reward_rerank_diagnostic_rollouts",
    "stage18_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage",
    "continue_path_cost_weight_calibration_at_99pct_coverage",
    STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE,
}
STAGE19_ALLOWED_ROUTES = {
    "resolve_stage19_evaluator_critic_preflight_boundary_rejections",
    "rerun_stage18_11_reward_rerank_diagnostic_rollouts",
    "repair_path_risk_boundary_filtering",
    "stage18_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage",
    "continue_path_cost_weight_calibration_at_99pct_coverage",
    "collect_more_reward_rerank_preference_evidence",
    "stage20_reward_rerank_oracle_preference_dataset_preparation",
}

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "default_policy_replacement_approved",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "real_world_release_approved",
    "real_world_performance_claimed",
)


@dataclass(frozen=True)
class Stage18ModuleSpec:
    module_id: str
    stage_label: str
    description: str
    root_key: str
    summary_files: tuple[str, ...]
    stage_ids: tuple[str, ...]
    support_status: str


@dataclass(frozen=True)
class Stage18RootSet:
    roi_expansion_root: Path
    candidate_root: Path
    model_inference_root: Path
    true_binding_root: Path
    quantization_root: Path
    oracle_root: Path
    coverage_comparison_root: Path
    attribution_root: Path | None = None
    guard_refinement_root: Path | None = None
    candidate_count_scaling_root: Path | None = None
    trajectory_risk_reward_root: Path | None = None
    path_cost_weight_calibration_root: Path | None = None
    evaluator_critic_preflight_root: Path | None = None


MODULE_SPECS: tuple[Stage18ModuleSpec, ...] = (
    Stage18ModuleSpec(
        module_id="stage18_1_scenario_evidence_preparation",
        stage_label="Stage 18.1",
        description="scenario and high-fidelity quasi-real map evidence preparation",
        root_key="roi_expansion_root",
        summary_files=(ROI_EXPANSION_SUMMARY_FILE,),
        stage_ids=("xunce-high-fidelity-real-map-roi-expansion",),
        support_status="active_mainline",
    ),
    Stage18ModuleSpec(
        module_id="stage18_2_candidate_generation_and_validation",
        stage_label="Stage 18.2",
        description="candidate source generation plus planner/path-feedback validation",
        root_key="candidate_root",
        summary_files=STAGE18I_SUMMARY_FILES,
        stage_ids=("xunce-true-frontier-nbv-candidate-source-replacement",),
        support_status="active_mainline",
    ),
    Stage18ModuleSpec(
        module_id="stage18_3_true_inference_and_binding",
        stage_label="Stage 18.3",
        description="true checkpoint inference and true incumbent binding",
        root_key="true_binding_root",
        summary_files=(TRUE_BINDING_SUMMARY_FILE,),
        stage_ids=("xunce-high-fidelity-real-map-comparison", "xunce-true-incumbent-selection-binding"),
        support_status="active_mainline",
    ),
    Stage18ModuleSpec(
        module_id="stage18_4_quantization_oracle_and_rollout",
        stage_label="Stage 18.4",
        description="risk/coverage/cost quantization, oracle baselines, and offline coverage rollout",
        root_key="coverage_comparison_root",
        summary_files=(COVERAGE_COMPARISON_SUMMARY_FILE,),
        stage_ids=(
            "xunce-risk-coverage-cost-quantization-audit",
            "xunce-oracle-separability-benchmark",
            "xunce-high-fidelity-exploration-coverage-comparison",
        ),
        support_status="active_mainline",
    ),
    Stage18ModuleSpec(
        module_id="stage18_5_evidence_closure_and_attribution",
        stage_label="Stage 18.5",
        description="pipeline closure, conclusion attribution, and next-stage routing",
        root_key="coverage_comparison_root",
        summary_files=(COVERAGE_COMPARISON_SUMMARY_FILE,),
        stage_ids=("xunce-stage18-research-evidence-pipeline",),
        support_status="active_mainline",
    ),
)


def load_stage18_config(
    *,
    config_path: Path,
    repo_root: Path,
    overrides: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"expected schema_version {CONFIG_SCHEMA_VERSION}")

    merged = dict(payload)
    for key, value in (overrides or {}).items():
        if value is not None:
            merged[key] = value
    if float(merged.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        raise ConfigError("canary_traffic_fraction must be 0.0 for offline Stage 18 research evidence")

    required = (
        "stage18_1_roi_expansion_root",
        "stage18_2_candidate_root",
        "stage18_3_model_inference_root",
        "stage18_3_true_incumbent_binding_root",
        "stage18_4_quantization_root",
        "stage18_4_oracle_root",
        "stage18_4_coverage_comparison_root",
    )
    config: dict[str, Any] = {
        "schema_version": merged["schema_version"],
        "canary_traffic_fraction": 0.0,
        "execution_mode": str(merged.get("execution_mode", "plan_only")),
    }
    for key in required:
        value = merged.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{key} must be a non-empty path string")
        config[key] = str(resolve_path(Path(value), repo_root).resolve())
    attribution_root = merged.get("stage18_5_attribution_root")
    if attribution_root is not None:
        if not isinstance(attribution_root, str) or not attribution_root:
            raise ConfigError("stage18_5_attribution_root must be a non-empty path string")
        config["stage18_5_attribution_root"] = str(resolve_path(Path(attribution_root), repo_root).resolve())
    guard_refinement_root = merged.get("stage18_6_guard_refinement_root")
    if guard_refinement_root is not None:
        if not isinstance(guard_refinement_root, str) or not guard_refinement_root:
            raise ConfigError("stage18_6_guard_refinement_root must be a non-empty path string")
        config["stage18_6_guard_refinement_root"] = str(resolve_path(Path(guard_refinement_root), repo_root).resolve())
    candidate_count_scaling_root = merged.get("stage18_7_candidate_count_scaling_root")
    if candidate_count_scaling_root is not None:
        if not isinstance(candidate_count_scaling_root, str) or not candidate_count_scaling_root:
            raise ConfigError("stage18_7_candidate_count_scaling_root must be a non-empty path string")
        config["stage18_7_candidate_count_scaling_root"] = str(resolve_path(Path(candidate_count_scaling_root), repo_root).resolve())
    trajectory_risk_reward_root = merged.get("stage18_9_trajectory_risk_reward_root")
    if trajectory_risk_reward_root is not None:
        if not isinstance(trajectory_risk_reward_root, str) or not trajectory_risk_reward_root:
            raise ConfigError("stage18_9_trajectory_risk_reward_root must be a non-empty path string")
        config["stage18_9_trajectory_risk_reward_root"] = str(resolve_path(Path(trajectory_risk_reward_root), repo_root).resolve())
    path_cost_weight_calibration_root = merged.get("stage18_11_path_cost_weight_calibration_root")
    if path_cost_weight_calibration_root is not None:
        if not isinstance(path_cost_weight_calibration_root, str) or not path_cost_weight_calibration_root:
            raise ConfigError("stage18_11_path_cost_weight_calibration_root must be a non-empty path string")
        config["stage18_11_path_cost_weight_calibration_root"] = str(resolve_path(Path(path_cost_weight_calibration_root), repo_root).resolve())
    evaluator_critic_preflight_root = merged.get("stage19_evaluator_critic_preflight_root")
    if evaluator_critic_preflight_root is not None:
        if not isinstance(evaluator_critic_preflight_root, str) or not evaluator_critic_preflight_root:
            raise ConfigError("stage19_evaluator_critic_preflight_root must be a non-empty path string")
        config["stage19_evaluator_critic_preflight_root"] = str(resolve_path(Path(evaluator_critic_preflight_root), repo_root).resolve())
    return config


def root_set_from_config(config: dict[str, Any]) -> Stage18RootSet:
    return Stage18RootSet(
        roi_expansion_root=Path(config["stage18_1_roi_expansion_root"]),
        candidate_root=Path(config["stage18_2_candidate_root"]),
        model_inference_root=Path(config["stage18_3_model_inference_root"]),
        true_binding_root=Path(config["stage18_3_true_incumbent_binding_root"]),
        quantization_root=Path(config["stage18_4_quantization_root"]),
        oracle_root=Path(config["stage18_4_oracle_root"]),
        coverage_comparison_root=Path(config["stage18_4_coverage_comparison_root"]),
        attribution_root=Path(config["stage18_5_attribution_root"]) if config.get("stage18_5_attribution_root") else None,
        guard_refinement_root=Path(config["stage18_6_guard_refinement_root"]) if config.get("stage18_6_guard_refinement_root") else None,
        candidate_count_scaling_root=Path(config["stage18_7_candidate_count_scaling_root"]) if config.get("stage18_7_candidate_count_scaling_root") else None,
        trajectory_risk_reward_root=Path(config["stage18_9_trajectory_risk_reward_root"]) if config.get("stage18_9_trajectory_risk_reward_root") else None,
        path_cost_weight_calibration_root=Path(config["stage18_11_path_cost_weight_calibration_root"]) if config.get("stage18_11_path_cost_weight_calibration_root") else None,
        evaluator_critic_preflight_root=Path(config["stage19_evaluator_critic_preflight_root"]) if config.get("stage19_evaluator_critic_preflight_root") else None,
    )


def build_stage18_pipeline_summary(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    overrides: dict[str, str | None] | None = None,
    plan_only: bool = True,
    dry_run: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    config = load_stage18_config(config_path=config_path, repo_root=repo_root, overrides=overrides)
    roots = root_set_from_config(config)
    evidence = load_stage18_evidence(roots)
    commands = build_stage18_command_plan(repo_root=repo_root, roots=roots)
    diagnostics = evaluate_stage18_evidence(evidence=evidence, roots=roots)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": diagnostics["status"],
        "execution_mode": "execute" if execute else "dry_run" if dry_run else "plan_only" if plan_only else "plan_only",
        "plan_only": bool(plan_only),
        "dry_run": bool(dry_run),
        "execute_requested": bool(execute),
        "execute_supported": False,
        "evidence_status": diagnostics["evidence_status"],
        "candidate_validity_status": diagnostics["candidate_validity_status"],
        "comparison_verdict": diagnostics["comparison_verdict"],
        "release_readiness": "not_authorized",
        "training_readiness": "not_authorized",
        "overall_conclusion": diagnostics["overall_conclusion"],
        "next_required_change": diagnostics["next_required_change"],
        "reason_codes": diagnostics["reason_codes"],
        "missing_reason_codes": diagnostics["missing_reason_codes"],
        "blocking_reason_codes": diagnostics["blocking_reason_codes"],
        "diagnostic_reason_codes": diagnostics["diagnostic_reason_codes"],
        "diagnostic_recommended_changes": diagnostics["diagnostic_recommended_changes"],
        "module_results": diagnostics["module_results"],
        "resolved_roots": resolved_roots_payload(roots),
        "stage_command_plan": commands,
        "single_step_comparison_summary": diagnostics["single_step_comparison_summary"],
        "coverage_rollout_comparison_summary": diagnostics["coverage_rollout_comparison_summary"],
        "stage18_5_attribution_summary": diagnostics["stage18_5_attribution_summary"],
        "stage18_5_guard_verdict": diagnostics["stage18_5_guard_verdict"],
        "stage18_5_primary_next_required_change": diagnostics["stage18_5_primary_next_required_change"],
        "stage18_6_guard_refinement_summary": diagnostics["stage18_6_guard_refinement_summary"],
        "stage18_6_guard_refinement_verdict": diagnostics["stage18_6_guard_refinement_verdict"],
        "stage18_6_primary_next_required_change": diagnostics["stage18_6_primary_next_required_change"],
        "stage18_6_candidate_metric_readiness": diagnostics["stage18_6_candidate_metric_readiness"],
        "stage18_7_candidate_count_scaling_summary": diagnostics["stage18_7_candidate_count_scaling_summary"],
        "stage18_7_candidate_count_scaling_verdict": diagnostics["stage18_7_candidate_count_scaling_verdict"],
        "stage18_7_primary_next_required_change": diagnostics["stage18_7_primary_next_required_change"],
        "stage18_9_trajectory_risk_reward_summary": diagnostics["stage18_9_trajectory_risk_reward_summary"],
        "stage18_9_trajectory_risk_reward_verdict": diagnostics["stage18_9_trajectory_risk_reward_verdict"],
        "stage18_9_primary_next_required_change": diagnostics["stage18_9_primary_next_required_change"],
        "stage18_11_path_cost_weight_calibration_summary": diagnostics["stage18_11_path_cost_weight_calibration_summary"],
        "stage18_11_path_cost_weight_calibration_verdict": diagnostics["stage18_11_path_cost_weight_calibration_verdict"],
        "stage18_11_primary_next_required_change": diagnostics["stage18_11_primary_next_required_change"],
        "stage19_evaluator_critic_preflight_summary": diagnostics["stage19_evaluator_critic_preflight_summary"],
        "stage19_evaluator_critic_preflight_verdict": diagnostics["stage19_evaluator_critic_preflight_verdict"],
        "stage19_primary_next_required_change": diagnostics["stage19_primary_next_required_change"],
        "stage19_primary_target_candidate_count": diagnostics["stage19_primary_target_candidate_count"],
        "stage19_primary_target_path_cost_weight": diagnostics["stage19_primary_target_path_cost_weight"],
        "stage19_oracle_target_feasible": diagnostics["stage19_oracle_target_feasible"],
        "stage19_xunce_checkpoint_advantage_established": diagnostics["stage19_xunce_checkpoint_advantage_established"],
        "stage19_training_authorized": diagnostics["stage19_training_authorized"],
        "comparison_metric_summary": diagnostics["comparison_metric_summary"],
        "coverage_delta_distribution": diagnostics["coverage_delta_distribution"],
        "cost_delta_distribution": diagnostics["cost_delta_distribution"],
        "risk_delta_distribution": diagnostics["risk_delta_distribution"],
        "scenario_win_loss_summary": diagnostics["scenario_win_loss_summary"],
        "utility_profile_summary": diagnostics["utility_profile_summary"],
        "legacy_label_summary": diagnostics["legacy_label_summary"],
        "oracle_summary": diagnostics["oracle_summary"],
        "candidate_generation_summary": diagnostics["candidate_generation_summary"],
        "governance_boundary": boundary_payload(),
        "config": str(config_path),
        "output_root": str(output_root),
        **boundary_payload(),
    }
    return summary


def load_stage18_evidence(roots: Stage18RootSet) -> dict[str, Any]:
    missing: list[str] = []
    stage18_1 = _read_json(roots.roi_expansion_root / ROI_EXPANSION_SUMMARY_FILE, missing, "missing_stage18_1_roi_expansion")
    stage18_2 = _read_first_json(roots.candidate_root, STAGE18I_SUMMARY_FILES, missing, "missing_stage18_2_candidate_generation")
    model = _read_json(roots.model_inference_root / MODEL_COMPARISON_SUMMARY_FILE, missing, "missing_stage18_3_model_inference")
    binding = _read_json(roots.true_binding_root / TRUE_BINDING_SUMMARY_FILE, missing, "missing_stage18_3_true_incumbent_binding")
    quant = _read_json(roots.quantization_root / QUANTIZATION_SUMMARY_FILE, missing, "missing_stage18_4_quantization")
    oracle = _read_json(roots.oracle_root / ORACLE_SUMMARY_FILE, missing, "missing_stage18_4_oracle")
    coverage = _read_json(
        roots.coverage_comparison_root / COVERAGE_COMPARISON_SUMMARY_FILE,
        missing,
        "missing_stage18_4_coverage_comparison",
    )
    coverage_aggregate = _read_optional_json(roots.coverage_comparison_root / COVERAGE_COMPARISON_AGGREGATE_FILE)
    attribution, stage18_5_blocking = _read_stage18_5_attribution(roots)
    guard_refinement, stage18_6_blocking = _read_stage18_6_guard_refinement(roots)
    candidate_count_scaling, stage18_7_blocking = _read_stage18_7_candidate_count_scaling(roots)
    trajectory_risk_reward, stage18_9_blocking = _read_stage18_9_trajectory_risk_reward(roots)
    path_cost_weight_calibration, stage18_11_blocking = _read_stage18_11_path_cost_weight_calibration(roots)
    evaluator_critic_preflight, stage19_blocking = _read_stage19_evaluator_critic_preflight(roots)
    return {
        "stage18_1": stage18_1,
        "stage18_2": stage18_2,
        "model_inference": model,
        "true_binding": binding,
        "quantization": quant,
        "oracle": oracle,
        "coverage_comparison": coverage,
        "coverage_comparison_aggregate": coverage_aggregate,
        "stage18_5_attribution": attribution,
        "stage18_5_blocking_reason_codes": stage18_5_blocking,
        "stage18_6_guard_refinement": guard_refinement,
        "stage18_6_blocking_reason_codes": stage18_6_blocking,
        "stage18_7_candidate_count_scaling": candidate_count_scaling,
        "stage18_7_blocking_reason_codes": stage18_7_blocking,
        "stage18_9_trajectory_risk_reward": trajectory_risk_reward,
        "stage18_9_blocking_reason_codes": stage18_9_blocking,
        "stage18_11_path_cost_weight_calibration": path_cost_weight_calibration,
        "stage18_11_blocking_reason_codes": stage18_11_blocking,
        "stage19_evaluator_critic_preflight": evaluator_critic_preflight,
        "stage19_blocking_reason_codes": stage19_blocking,
        "missing_reason_codes": unique_sorted(missing),
    }


def evaluate_stage18_evidence(*, evidence: dict[str, Any], roots: Stage18RootSet) -> dict[str, Any]:
    missing = list(evidence["missing_reason_codes"])
    blocking: list[str] = []
    diagnostic: list[str] = []
    module_results = _module_results(evidence)

    stage18_2 = evidence["stage18_2"]
    model = evidence["model_inference"]
    binding = evidence["true_binding"]
    quant = evidence["quantization"]
    oracle = evidence["oracle"]
    coverage = evidence["coverage_comparison"]
    coverage_aggregate = evidence["coverage_comparison_aggregate"]
    attribution = evidence["stage18_5_attribution"]
    guard_refinement = evidence["stage18_6_guard_refinement"]
    candidate_count_scaling = evidence["stage18_7_candidate_count_scaling"]
    trajectory_risk_reward = evidence["stage18_9_trajectory_risk_reward"]
    path_cost_weight_calibration = evidence["stage18_11_path_cost_weight_calibration"]
    evaluator_critic_preflight = evidence["stage19_evaluator_critic_preflight"]

    if _boundary_violation(*(payload for payload in evidence.values() if isinstance(payload, dict))):
        blocking.append("boundary_violation")
    blocking.extend(evidence.get("stage18_5_blocking_reason_codes", []))
    blocking.extend(evidence.get("stage18_6_blocking_reason_codes", []))
    blocking.extend(evidence.get("stage18_7_blocking_reason_codes", []))
    blocking.extend(evidence.get("stage18_9_blocking_reason_codes", []))
    blocking.extend(evidence.get("stage18_11_blocking_reason_codes", []))
    blocking.extend(evidence.get("stage19_blocking_reason_codes", []))
    if model and model.get("true_model_inference_executed") is not True:
        blocking.append("true_model_inference_not_executed")
    if model and model.get("proxy_selection_used") is True:
        blocking.append("proxy_selection_used")
    if model and (model.get("xunce_checkpoint_loaded") is False or model.get("incumbent_checkpoint_loaded") is False):
        blocking.append("checkpoint_not_loaded")
    if binding and binding.get("true_incumbent_selection_bound") is not True:
        blocking.append("true_incumbent_selection_not_bound")
    if binding and int(binding.get("fallback_action_index_0_count", 0) or 0) > 0:
        blocking.append("fallback_action_index_0_present")
    if binding and int(binding.get("candidate_cell_mismatch_count", 0) or 0) > 0:
        blocking.append("candidate_cell_mismatch")
    if stage18_2 and stage18_2.get("status") not in (None, "passed"):
        blocking.append("stage18_2_candidate_generation_not_passed")
    if quant and quant.get("candidate_validity_gate_passed") is False:
        blocking.append("candidate_validity_gate_failed")

    blocking.extend(_root_consistency_reasons(roots=roots, binding=binding, quant=quant, oracle=oracle, coverage=coverage))

    if quant and int(quant.get("safe_efficient_candidate_count", 0) or 0) <= 0:
        diagnostic.append("safe_efficient_candidate_missing")
    if oracle and oracle.get("oracle_separable") is not True:
        diagnostic.append("oracle_not_separable")
    if coverage and coverage.get("xunce_coverage_advantage_established") is not True:
        diagnostic.append("xunce_coverage_advantage_not_established")
    if coverage and coverage.get("candidate_refresh_mode") != "dynamic_frontier_nbv_in_process":
        missing.append("missing_dynamic_frontier_nbv_rollout_comparison")
        diagnostic.append("dynamic_frontier_nbv_rollout_not_executed")
    if coverage and coverage.get("candidate_refresh_mode") == "dynamic_frontier_nbv_in_process":
        if coverage.get("dynamic_candidate_generation_executed") is not True:
            blocking.append("dynamic_candidate_generation_not_executed")
        if int(coverage.get("dynamic_contract_sidecar_missing_count", 0) or 0) > 0:
            blocking.append("dynamic_contract_sidecar_missing")
        if int(coverage.get("dynamic_candidate_generation_missing_count", 0) or 0) > 0:
            blocking.append("dynamic_candidate_generation_missing")
        if int(coverage.get("model_inference_failure_count", 0) or 0) > 0:
            blocking.append("model_inference_failure")
        if int(coverage.get("candidate_generation_exhausted_count", 0) or 0) > 0:
            diagnostic.append("candidate_generation_exhausted")
        if (
            int(coverage.get("coverage_frontier_candidate_count", 0) or 0)
            + int(coverage.get("undercovered_component_candidate_count", 0) or 0)
            <= 0
        ):
            diagnostic.append("dynamic_candidate_source_not_frontier_dominant")
        if coverage.get("dynamic_validation_full_adapter_evidence_passed") is not True:
            if int(coverage.get("sidecar_grid_astar_screening_count", 0) or 0) > 0 or int(coverage.get("dynamic_sidecar_grid_astar_fallback_count", 0) or 0) > 0:
                diagnostic.append("sidecar_screening_not_full_adapter_evidence")
            elif int(coverage.get("in_process_batch_astar_validation_count", 0) or 0) > 0:
                diagnostic.append("dynamic_batch_astar_screening_not_full_adapter_evidence")
            else:
                diagnostic.append("dynamic_validation_not_full_adapter_evidence")
    if model and model.get("xunce_candidate_advantage_established") is not True:
        diagnostic.append("single_step_xunce_advantage_not_established")
    if coverage and int(coverage.get("xunce_efficiency_regression_count", 0) or 0) > 0:
        diagnostic.append("coverage_efficiency_regression_present")

    evidence_status = _evidence_status(missing=missing, blocking=blocking)
    candidate_validity_status = _candidate_validity_status(missing=missing, blocking=blocking, stage18_2=stage18_2, quant=quant)
    stage18_5_summary = _stage18_5_attribution_summary(attribution)
    stage18_6_summary = _stage18_6_guard_refinement_summary(guard_refinement)
    stage18_7_summary = _stage18_7_candidate_count_scaling_summary(candidate_count_scaling)
    stage18_9_summary = _stage18_9_trajectory_risk_reward_summary(trajectory_risk_reward)
    stage18_11_summary = _stage18_11_path_cost_weight_calibration_summary(path_cost_weight_calibration)
    stage19_summary = _stage19_evaluator_critic_preflight_summary(evaluator_critic_preflight)
    comparison_verdict = _comparison_verdict(
        evidence_status=evidence_status,
        model=model,
        coverage=coverage,
        stage18_5_summary=stage18_5_summary,
        stage18_9_summary=stage18_9_summary,
    )
    overall_conclusion = _overall_conclusion(
        evidence_status=evidence_status,
        candidate_validity_status=candidate_validity_status,
        comparison_verdict=comparison_verdict,
    )
    stage18_5_route = stage18_5_summary["primary_next_required_change"]
    stage18_6_route = stage18_6_summary["primary_next_required_change"]
    stage18_7_route = stage18_7_summary["primary_next_required_change"]
    stage18_9_route = stage18_9_summary["primary_next_required_change"]
    stage18_11_route = stage18_11_summary["primary_next_required_change"]
    stage19_route = stage19_summary["primary_next_required_change"]
    return {
        "status": "failed" if blocking else "partial" if missing else "passed",
        "evidence_status": evidence_status,
        "candidate_validity_status": candidate_validity_status,
        "comparison_verdict": comparison_verdict,
        "overall_conclusion": overall_conclusion,
        "next_required_change": _next_required_change(
            missing=missing,
            blocking=blocking,
            comparison_verdict=comparison_verdict,
            stage18_5_route=stage18_5_route,
            stage18_6_route=stage18_6_route,
            stage18_7_route=stage18_7_route,
            stage18_9_route=stage18_9_route,
            stage18_11_route=stage18_11_route,
            stage19_route=stage19_route,
        ),
        "reason_codes": unique_sorted([*missing, *blocking]),
        "missing_reason_codes": unique_sorted(missing),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
        "diagnostic_recommended_changes": _diagnostic_recommendations(diagnostic),
        "module_results": module_results,
        "single_step_comparison_summary": _single_step_summary(model),
        "coverage_rollout_comparison_summary": _coverage_summary(coverage, coverage_aggregate),
        "stage18_5_attribution_summary": stage18_5_summary["summary"],
        "stage18_5_guard_verdict": stage18_5_summary["guard_verdict"],
        "stage18_5_primary_next_required_change": stage18_5_route,
        "stage18_6_guard_refinement_summary": stage18_6_summary["summary"],
        "stage18_6_guard_refinement_verdict": stage18_6_summary["guard_verdict"],
        "stage18_6_primary_next_required_change": stage18_6_route,
        "stage18_6_candidate_metric_readiness": stage18_6_summary["candidate_metric_readiness"],
        "stage18_7_candidate_count_scaling_summary": stage18_7_summary["summary"],
        "stage18_7_candidate_count_scaling_verdict": stage18_7_summary["guard_verdict"],
        "stage18_7_primary_next_required_change": stage18_7_route,
        "stage18_9_trajectory_risk_reward_summary": stage18_9_summary["summary"],
        "stage18_9_trajectory_risk_reward_verdict": stage18_9_summary["guard_verdict"],
        "stage18_9_primary_next_required_change": stage18_9_route,
        "stage18_11_path_cost_weight_calibration_summary": stage18_11_summary["summary"],
        "stage18_11_path_cost_weight_calibration_verdict": stage18_11_summary["guard_verdict"],
        "stage18_11_primary_next_required_change": stage18_11_route,
        "stage19_evaluator_critic_preflight_summary": stage19_summary["summary"],
        "stage19_evaluator_critic_preflight_verdict": stage19_summary["guard_verdict"],
        "stage19_primary_next_required_change": stage19_route,
        "stage19_primary_target_candidate_count": stage19_summary["primary_target_candidate_count"],
        "stage19_primary_target_path_cost_weight": stage19_summary["primary_target_path_cost_weight"],
        "stage19_oracle_target_feasible": stage19_summary["oracle_target_feasible"],
        "stage19_xunce_checkpoint_advantage_established": stage19_summary["xunce_checkpoint_advantage_established"],
        "stage19_training_authorized": stage19_summary["training_authorized"],
        "comparison_metric_summary": _comparison_metric_summary(coverage, coverage_aggregate),
        "coverage_delta_distribution": _distribution_summary(coverage_aggregate, "coverage_delta_cells"),
        "cost_delta_distribution": _distribution_summary(coverage_aggregate, "path_cost_delta_m"),
        "risk_delta_distribution": _distribution_summary(coverage_aggregate, "risk_delta"),
        "scenario_win_loss_summary": _scenario_win_loss_summary(coverage_aggregate),
        "utility_profile_summary": (coverage_aggregate or {}).get("utility_profile_summary", {}),
        "legacy_label_summary": _legacy_label_summary(model, coverage, oracle),
        "oracle_summary": _oracle_summary(oracle),
        "candidate_generation_summary": _candidate_summary(stage18_2, quant),
    }


def build_stage18_command_plan(*, repo_root: Path, roots: Stage18RootSet) -> list[dict[str, Any]]:
    run_stage = repo_root / "scripts" / "run_stage.py"
    commands = [
        ("18.1", "xunce-high-fidelity-real-map-roi-expansion", ["--output-root", str(roots.roi_expansion_root)]),
        ("18.2", "xunce-true-frontier-nbv-candidate-source-replacement", ["--output-root", str(roots.candidate_root)]),
        (
            "18.3",
            "xunce-high-fidelity-real-map-comparison",
            [
                "--output-root",
                str(roots.model_inference_root),
                "--extra-arg",
                "--source-roi-expansion-root",
                "--extra-arg",
                str(roots.candidate_root),
            ],
        ),
        (
            "18.3",
            "xunce-true-incumbent-selection-binding",
            [
                "--output-root",
                str(roots.true_binding_root),
                "--extra-arg",
                "--source-materialized-coverage-root",
                "--extra-arg",
                str(roots.candidate_root),
                "--extra-arg",
                "--source-model-inference-root",
                "--extra-arg",
                str(roots.model_inference_root),
            ],
        ),
        (
            "18.4",
            "xunce-risk-coverage-cost-quantization-audit",
            [
                "--output-root",
                str(roots.quantization_root),
                "--extra-arg",
                "--source-bound-coverage-root",
                "--extra-arg",
                str(roots.true_binding_root),
            ],
        ),
        (
            "18.4",
            "xunce-oracle-separability-benchmark",
            [
                "--output-root",
                str(roots.oracle_root),
                "--extra-arg",
                "--source-materialized-coverage-root",
                "--extra-arg",
                str(roots.quantization_root),
            ],
        ),
        (
            "18.4",
            "xunce-high-fidelity-exploration-coverage-comparison",
            [
                "--output-root",
                str(roots.coverage_comparison_root),
                "--extra-arg",
                "--source-roi-expansion-root",
                "--extra-arg",
                str(roots.quantization_root),
                "--extra-arg",
                "--candidate-refresh-mode",
                "--extra-arg",
                "dynamic_frontier_nbv_in_process",
                "--extra-arg",
                "--dynamic-candidate-validation-mode",
                "--extra-arg",
                "in_process_path_planner_astar_batch",
                "--extra-arg",
                "--dynamic-validation-work-root",
                "--extra-arg",
                "outputs/_xunce_dynamic_validation_work",
                "--extra-arg",
                "--coverage-metric-mode",
                "--extra-arg",
                "path_line_plus_endpoint",
                "--extra-arg",
                "--include-oracle-baselines",
                "--extra-arg",
                "--include-roi-weighted-coverage",
            ],
        ),
    ]
    if roots.attribution_root is not None:
        commands.append(
            (
                "18.5",
                "xunce-stage18-5-evidence-attribution-review",
                [
                    "--output-root",
                    str(roots.attribution_root),
                    "--extra-arg",
                    "--coverage-comparison-root",
                    "--extra-arg",
                    str(roots.coverage_comparison_root),
                ],
            )
        )
    if roots.guard_refinement_root is not None:
        stage18_6_args = [
            "--output-root",
            str(roots.guard_refinement_root),
        ]
        if roots.attribution_root is not None:
            stage18_6_args.extend(
                [
                    "--extra-arg",
                    "--stage18-5-attribution-root",
                    "--extra-arg",
                    str(roots.attribution_root),
                ]
            )
        stage18_6_args.extend(
            [
                "--extra-arg",
                "--coverage-comparison-root",
                "--extra-arg",
                str(roots.coverage_comparison_root),
            ]
        )
        commands.append(
            (
                "18.6",
                "xunce-stage18-6-coverage-reward-cost-risk-guard-refinement",
                stage18_6_args,
            )
        )
    if roots.candidate_count_scaling_root is not None:
        commands.append(
            (
                "18.7",
                "xunce-stage18-7-candidate-count-scaling-audit",
                [
                    "--output-root",
                    str(roots.candidate_count_scaling_root),
                ],
            )
        )
    if roots.trajectory_risk_reward_root is not None:
        stage18_9_args = [
            "--output-root",
            str(roots.trajectory_risk_reward_root),
            "--extra-arg",
            "--coverage-comparison-root",
            "--extra-arg",
            str(roots.coverage_comparison_root),
        ]
        if roots.candidate_count_scaling_root is not None:
            stage18_9_args.extend(
                [
                    "--extra-arg",
                    "--stage18-7-candidate-count-scaling-root",
                    "--extra-arg",
                    str(roots.candidate_count_scaling_root),
                ]
            )
        commands.append(
            (
                "18.9",
                "xunce-stage18-9-trajectory-risk-boundary-reward-audit",
                stage18_9_args,
            )
        )
    if roots.path_cost_weight_calibration_root is not None:
        commands.append(
            (
                "18.11",
                "xunce-stage18-11-path-cost-weight-calibration",
                [
                    "--output-root",
                    str(roots.path_cost_weight_calibration_root),
                ],
            )
        )
    if roots.evaluator_critic_preflight_root is not None:
        stage19_args = [
            "--output-root",
            str(roots.evaluator_critic_preflight_root),
        ]
        if roots.path_cost_weight_calibration_root is not None:
            stage19_args.extend(
                [
                    "--extra-arg",
                    "--stage18-11-path-cost-weight-calibration-root",
                    "--extra-arg",
                    str(roots.path_cost_weight_calibration_root),
                ]
            )
        commands.append(
            (
                "19",
                "xunce-stage19-evaluator-critic-preflight",
                stage19_args,
            )
        )
    plan: list[dict[str, Any]] = []
    for module, stage_id, args in commands:
        argv = python_script_command(run_stage, "--stage", stage_id, *args)
        plan.append(
            {
                "module": module,
                "stage": stage_id,
                "argv": argv,
                "display": display_command(argv),
            }
        )
    return plan


def resolved_roots_payload(roots: Stage18RootSet) -> dict[str, str]:
    return {
        "stage18_1_roi_expansion_root": str(roots.roi_expansion_root),
        "stage18_2_candidate_root": str(roots.candidate_root),
        "stage18_3_model_inference_root": str(roots.model_inference_root),
        "stage18_3_true_incumbent_binding_root": str(roots.true_binding_root),
        "stage18_4_quantization_root": str(roots.quantization_root),
        "stage18_4_oracle_root": str(roots.oracle_root),
        "stage18_4_coverage_comparison_root": str(roots.coverage_comparison_root),
        "stage18_5_attribution_root": str(roots.attribution_root) if roots.attribution_root is not None else None,
        "stage18_6_guard_refinement_root": str(roots.guard_refinement_root) if roots.guard_refinement_root is not None else None,
        "stage18_7_candidate_count_scaling_root": str(roots.candidate_count_scaling_root) if roots.candidate_count_scaling_root is not None else None,
        "stage18_9_trajectory_risk_reward_root": str(roots.trajectory_risk_reward_root) if roots.trajectory_risk_reward_root is not None else None,
        "stage18_11_path_cost_weight_calibration_root": str(roots.path_cost_weight_calibration_root) if roots.path_cost_weight_calibration_root is not None else None,
        "stage19_evaluator_critic_preflight_root": str(roots.evaluator_critic_preflight_root) if roots.evaluator_critic_preflight_root is not None else None,
    }


def write_stage18_pipeline_artifacts(output_root: Path, summary: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "xunce-stage18-pipeline-summary.json", summary)
    _write_json(output_root / "xunce-stage18-resolved-roots.json", summary["resolved_roots"])
    module_lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in summary["module_results"]]
    (output_root / "xunce-stage18-module-results.jsonl").write_text("\n".join(module_lines) + ("\n" if module_lines else ""), encoding="utf-8")
    (output_root / "xunce-stage18-pipeline-report.md").write_text(render_stage18_report(summary), encoding="utf-8")


def render_stage18_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 18 Research Evidence Pipeline",
            "",
            f"- status: `{summary['status']}`",
            f"- evidence_status: `{summary['evidence_status']}`",
            f"- candidate_validity_status: `{summary['candidate_validity_status']}`",
            f"- comparison_verdict: `{summary['comparison_verdict']}`",
            f"- overall_conclusion: `{summary['overall_conclusion']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- release_readiness: `{summary['release_readiness']}`",
            f"- training_readiness: `{summary['training_readiness']}`",
            f"- stage18_5_guard_verdict: `{summary['stage18_5_guard_verdict']}`",
            f"- stage18_5_primary_next_required_change: `{summary['stage18_5_primary_next_required_change']}`",
            f"- stage18_6_guard_refinement_verdict: `{summary['stage18_6_guard_refinement_verdict']}`",
            f"- stage18_6_primary_next_required_change: `{summary['stage18_6_primary_next_required_change']}`",
            f"- stage18_7_candidate_count_scaling_verdict: `{summary['stage18_7_candidate_count_scaling_verdict']}`",
            f"- stage18_7_primary_next_required_change: `{summary['stage18_7_primary_next_required_change']}`",
            f"- stage18_9_trajectory_risk_reward_verdict: `{summary['stage18_9_trajectory_risk_reward_verdict']}`",
            f"- stage18_9_primary_next_required_change: `{summary['stage18_9_primary_next_required_change']}`",
            f"- stage18_11_path_cost_weight_calibration_verdict: `{summary['stage18_11_path_cost_weight_calibration_verdict']}`",
            f"- stage18_11_primary_next_required_change: `{summary['stage18_11_primary_next_required_change']}`",
            f"- stage19_evaluator_critic_preflight_verdict: `{summary['stage19_evaluator_critic_preflight_verdict']}`",
            f"- stage19_primary_next_required_change: `{summary['stage19_primary_next_required_change']}`",
            f"- stage19_primary_target_candidate_count: `{summary['stage19_primary_target_candidate_count']}`",
            f"- stage19_primary_target_path_cost_weight: `{summary['stage19_primary_target_path_cost_weight']}`",
            f"- stage19_oracle_target_feasible: `{summary['stage19_oracle_target_feasible']}`",
            f"- stage19_xunce_checkpoint_advantage_established: `{summary['stage19_xunce_checkpoint_advantage_established']}`",
            f"- stage19_training_authorized: `{summary['stage19_training_authorized']}`",
            "",
            "## Quantitative Comparison",
            "",
            f"- mean coverage delta cells: `{summary['comparison_metric_summary'].get('coverage_delta_cells_mean')}`",
            f"- mean path cost delta m: `{summary['comparison_metric_summary'].get('path_cost_delta_m_mean')}`",
            f"- mean risk delta: `{summary['comparison_metric_summary'].get('risk_delta_mean')}`",
            f"- mean coverage per 100m delta: `{summary['comparison_metric_summary'].get('coverage_per_100m_delta_mean')}`",
            f"- scenario win/tie/loss: `{summary['scenario_win_loss_summary']}`",
            "",
            "This pipeline is an offline research evidence closure. It does not approve checkpoint publication, default policy replacement, executor connection, PPO updates, or online canary traffic.",
            "",
        ]
    )


def _read_json(path: Path, missing: list[str], reason_code: str) -> dict[str, Any]:
    if not path.is_file():
        missing.append(reason_code)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        missing.append(reason_code)
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_stage18_5_attribution(roots: Stage18RootSet) -> tuple[dict[str, Any], list[str]]:
    if roots.attribution_root is None:
        return {}, []
    path = roots.attribution_root / STAGE18_5_ATTRIBUTION_SUMMARY_FILE
    if not path.is_file():
        return {}, ["missing_stage18_5_attribution_summary"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ["invalid_stage18_5_attribution_summary"]
    if not isinstance(payload, dict):
        return {}, ["invalid_stage18_5_attribution_summary"]
    if payload.get("schema_version") != STAGE18_5_ATTRIBUTION_SCHEMA_VERSION:
        return {}, ["invalid_stage18_5_attribution_summary_schema"]
    coverage_root = payload.get("coverage_comparison_root")
    if not isinstance(coverage_root, str) or not coverage_root:
        return {}, ["missing_stage18_5_coverage_comparison_root"]
    if not _same_path(coverage_root, roots.coverage_comparison_root):
        return {}, ["stale_stage18_5_attribution_root"]
    routing = payload.get("next_stage_routing")
    guard = payload.get("guard_evaluation")
    if not isinstance(routing, dict):
        return {}, ["invalid_stage18_5_attribution_summary"]
    if routing.get("schema_version") != STAGE18_5_ROUTING_SCHEMA_VERSION:
        return {}, ["invalid_stage18_5_routing_summary_schema"]
    route = routing.get("primary_route")
    if not isinstance(route, str) or not route or route not in STAGE18_5_ALLOWED_ROUTES:
        return {}, ["invalid_stage18_5_attribution_summary"]
    if routing.get("stage19_authorized") is not False:
        return {}, ["invalid_stage18_5_attribution_summary"]
    if not isinstance(guard, dict):
        return {}, ["invalid_stage18_5_attribution_summary"]
    if guard.get("schema_version") != STAGE18_5_GUARD_SCHEMA_VERSION:
        return {}, ["invalid_stage18_5_guard_summary_schema"]
    if not isinstance(guard.get("passed"), bool):
        return {}, ["invalid_stage18_5_attribution_summary"]
    failed_guards = guard.get("failed_guards")
    if failed_guards is not None and not isinstance(failed_guards, list):
        return {}, ["invalid_stage18_5_attribution_summary"]
    if _boundary_violation(payload):
        return {}, ["boundary_violation"]
    return payload, []


def _read_stage18_6_guard_refinement(roots: Stage18RootSet) -> tuple[dict[str, Any], list[str]]:
    if roots.guard_refinement_root is None:
        return {}, []
    path = roots.guard_refinement_root / STAGE18_6_GUARD_REFINEMENT_SUMMARY_FILE
    if not path.is_file():
        return {}, ["missing_stage18_6_guard_refinement_summary"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ["invalid_stage18_6_guard_refinement_summary"]
    if not isinstance(payload, dict):
        return {}, ["invalid_stage18_6_guard_refinement_summary"]
    if payload.get("schema_version") != STAGE18_6_GUARD_REFINEMENT_SCHEMA_VERSION:
        return {}, ["invalid_stage18_6_guard_refinement_summary_schema"]
    if not _same_path(payload.get("coverage_comparison_root"), roots.coverage_comparison_root):
        return {}, ["stale_stage18_6_guard_refinement_root"]
    if roots.attribution_root is not None and not _same_path(payload.get("stage18_5_attribution_root"), roots.attribution_root):
        return {}, ["stale_stage18_6_guard_refinement_root"]
    if not _valid_profile_identity(payload):
        return {}, ["invalid_stage18_6_guard_refinement_profile_lineage"]
    if roots.attribution_root is not None:
        stage18_5_payload = _read_json_if_present(roots.attribution_root / STAGE18_5_ATTRIBUTION_SUMMARY_FILE)
        if not _valid_profile_identity(stage18_5_payload) or not _profile_identity_matches(payload, stage18_5_payload):
            return {}, ["invalid_stage18_6_guard_refinement_profile_lineage"]
    routing = payload.get("next_stage_routing")
    if not isinstance(routing, dict):
        return {}, ["invalid_stage18_6_guard_refinement_summary"]
    if routing.get("schema_version") != STAGE18_6_ROUTING_SCHEMA_VERSION:
        return {}, ["invalid_stage18_6_guard_refinement_routing_schema"]
    route = routing.get("primary_route")
    if not isinstance(route, str) or route not in STAGE18_6_ALLOWED_ROUTES:
        return {}, ["invalid_stage18_6_guard_refinement_summary"]
    if routing.get("stage19_authorized") is not False:
        return {}, ["invalid_stage18_6_guard_refinement_summary"]
    readiness = payload.get("stage19_readiness")
    if not isinstance(readiness, dict) or readiness.get("authorized") is not False:
        return {}, ["invalid_stage18_6_guard_refinement_summary"]
    if not isinstance(payload.get("guard_refinement_passed"), bool):
        return {}, ["invalid_stage18_6_guard_refinement_summary"]
    candidate_readiness = payload.get("candidate_metric_readiness")
    if not isinstance(candidate_readiness, dict):
        return {}, ["invalid_stage18_6_guard_refinement_summary"]
    if route == STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE and not _stage18_6_preflight_semantics_are_clean(payload):
        return {}, ["invalid_stage18_6_guard_refinement_preflight_semantics"]
    if _boundary_violation(payload):
        return {}, ["boundary_violation"]
    return payload, []


def _read_stage18_7_candidate_count_scaling(roots: Stage18RootSet) -> tuple[dict[str, Any], list[str]]:
    if roots.candidate_count_scaling_root is None:
        return {}, []
    if roots.guard_refinement_root is None:
        return {}, ["missing_stage18_6_guard_refinement_for_stage18_7"]
    path = roots.candidate_count_scaling_root / STAGE18_7_CANDIDATE_COUNT_SCALING_SUMMARY_FILE
    if not path.is_file():
        return {}, ["missing_stage18_7_candidate_count_scaling_summary"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ["invalid_stage18_7_candidate_count_scaling_summary"]
    if not isinstance(payload, dict):
        return {}, ["invalid_stage18_7_candidate_count_scaling_summary"]
    if payload.get("schema_version") != STAGE18_7_CANDIDATE_COUNT_SCALING_SCHEMA_VERSION:
        return {}, ["invalid_stage18_7_candidate_count_scaling_summary_schema"]
    if not _valid_profile_identity(payload):
        return {}, ["invalid_stage18_7_candidate_count_scaling_profile_lineage"]
    stage18_6_payload = _read_json_if_present(roots.guard_refinement_root / STAGE18_6_GUARD_REFINEMENT_SUMMARY_FILE)
    if not _valid_profile_identity(stage18_6_payload) or not _profile_identity_matches(payload, stage18_6_payload):
        return {}, ["invalid_stage18_7_candidate_count_scaling_profile_lineage"]
    routing = payload.get("next_stage_routing")
    if not isinstance(routing, dict):
        return {}, ["invalid_stage18_7_candidate_count_scaling_summary"]
    if routing.get("schema_version") != STAGE18_7_ROUTING_SCHEMA_VERSION:
        return {}, ["invalid_stage18_7_candidate_count_scaling_routing_schema"]
    route = routing.get("primary_route")
    if not isinstance(route, str) or route not in STAGE18_7_ALLOWED_ROUTES:
        return {}, ["invalid_stage18_7_candidate_count_scaling_summary"]
    if routing.get("stage19_authorized") is not False:
        return {}, ["invalid_stage18_7_candidate_count_scaling_summary"]
    readiness = payload.get("stage19_readiness")
    if not isinstance(readiness, dict) or readiness.get("authorized") is not False:
        return {}, ["invalid_stage18_7_candidate_count_scaling_summary"]
    if _boundary_violation(payload):
        return {}, ["boundary_violation"]
    if not _stage18_7_results_are_valid(payload):
        return {}, ["invalid_stage18_7_candidate_count_scaling_results"]
    if route == STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE and not _stage18_7_preflight_semantics_are_clean(payload):
        return {}, ["invalid_stage18_7_candidate_count_scaling_preflight_semantics"]
    if route == STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE and not _stage18_6_preflight_semantics_are_clean(stage18_6_payload):
        return {}, ["invalid_stage18_7_candidate_count_scaling_preflight_semantics"]
    return payload, []


def _read_stage18_9_trajectory_risk_reward(roots: Stage18RootSet) -> tuple[dict[str, Any], list[str]]:
    if roots.trajectory_risk_reward_root is None:
        return {}, []
    path = roots.trajectory_risk_reward_root / STAGE18_9_TRAJECTORY_RISK_REWARD_SUMMARY_FILE
    if not path.is_file():
        return {}, ["missing_stage18_9_trajectory_risk_reward_summary"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ["invalid_stage18_9_trajectory_risk_reward_summary"]
    if not isinstance(payload, dict):
        return {}, ["invalid_stage18_9_trajectory_risk_reward_summary"]
    if payload.get("schema_version") != STAGE18_9_TRAJECTORY_RISK_REWARD_SCHEMA_VERSION:
        return {}, ["invalid_stage18_9_trajectory_risk_reward_summary_schema"]
    if not _same_path(payload.get("coverage_comparison_root"), roots.coverage_comparison_root):
        return {}, ["stale_stage18_9_trajectory_risk_reward_root"]
    if roots.candidate_count_scaling_root is not None and payload.get("stage18_7_candidate_count_scaling_root") is not None:
        if not _same_path(payload.get("stage18_7_candidate_count_scaling_root"), roots.candidate_count_scaling_root):
            return {}, ["stale_stage18_9_trajectory_risk_reward_root"]
    if not _valid_stage18_9_profile_identity(payload):
        return {}, ["invalid_stage18_9_trajectory_risk_reward_profile_lineage"]
    routing = payload.get("next_stage_routing")
    if not isinstance(routing, dict):
        return {}, ["invalid_stage18_9_trajectory_risk_reward_summary"]
    if routing.get("schema_version") != STAGE18_9_ROUTING_SCHEMA_VERSION:
        return {}, ["invalid_stage18_9_trajectory_risk_reward_routing_schema"]
    route = routing.get("primary_route")
    if not isinstance(route, str) or route not in STAGE18_9_ALLOWED_ROUTES:
        return {}, ["invalid_stage18_9_trajectory_risk_reward_summary"]
    if routing.get("stage19_authorized") is not False:
        return {}, ["invalid_stage18_9_trajectory_risk_reward_summary"]
    readiness = payload.get("stage19_readiness")
    if (
        not isinstance(readiness, dict)
        or readiness.get("schema_version") != STAGE18_9_STAGE19_READINESS_SCHEMA_VERSION
        or readiness.get("authorized") is not False
    ):
        return {}, ["invalid_stage18_9_trajectory_risk_reward_summary"]
    if not isinstance(payload.get("trajectory_guard_passed"), bool):
        return {}, ["invalid_stage18_9_trajectory_risk_reward_summary"]
    if route == STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE and not _stage18_9_preflight_semantics_are_clean(payload):
        return {}, ["invalid_stage18_9_trajectory_risk_reward_preflight_semantics"]
    if _boundary_violation(payload):
        return {}, ["boundary_violation"]
    return payload, []


def _read_stage18_11_path_cost_weight_calibration(roots: Stage18RootSet) -> tuple[dict[str, Any], list[str]]:
    if roots.path_cost_weight_calibration_root is None:
        return {}, []
    path = roots.path_cost_weight_calibration_root / STAGE18_11_PATH_COST_WEIGHT_SUMMARY_FILE
    if not path.is_file():
        return {}, ["missing_stage18_11_path_cost_weight_calibration_summary"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ["invalid_stage18_11_path_cost_weight_calibration_summary"]
    if not isinstance(payload, dict):
        return {}, ["invalid_stage18_11_path_cost_weight_calibration_summary"]
    if payload.get("schema_version") != STAGE18_11_PATH_COST_WEIGHT_SCHEMA_VERSION:
        return {}, ["invalid_stage18_11_path_cost_weight_calibration_summary_schema"]
    if payload.get("profile_id") != STAGE18_9_EXPECTED_PROFILE_ID or payload.get("profile_version") != STAGE18_9_EXPECTED_PROFILE_VERSION:
        return {}, ["invalid_stage18_11_path_cost_weight_calibration_profile_lineage"]
    routing = payload.get("next_stage_routing")
    if not isinstance(routing, dict) or routing.get("schema_version") != STAGE18_11_ROUTING_SCHEMA_VERSION:
        return {}, ["invalid_stage18_11_path_cost_weight_calibration_routing_schema"]
    route = routing.get("primary_route")
    if not isinstance(route, str) or route not in STAGE18_11_ALLOWED_ROUTES:
        return {}, ["invalid_stage18_11_path_cost_weight_calibration_route"]
    if routing.get("stage19_authorized") is not False or payload.get("stage19_authorized") is not False:
        return {}, ["stage18_11_stage19_authorized_not_false"]
    if _boundary_violation(payload):
        return {}, ["boundary_violation"]
    if route == STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE and not _stage18_11_preflight_semantics_are_clean(payload):
        return {}, ["invalid_stage18_11_path_cost_weight_calibration_preflight_semantics"]
    return payload, []


def _read_stage19_evaluator_critic_preflight(roots: Stage18RootSet) -> tuple[dict[str, Any], list[str]]:
    if roots.evaluator_critic_preflight_root is None:
        return {}, []
    path = roots.evaluator_critic_preflight_root / STAGE19_EVALUATOR_CRITIC_SUMMARY_FILE
    if not path.is_file():
        return {}, ["missing_stage19_evaluator_critic_preflight_summary"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ["invalid_stage19_evaluator_critic_preflight_summary"]
    if not isinstance(payload, dict):
        return {}, ["invalid_stage19_evaluator_critic_preflight_summary"]
    if payload.get("schema_version") != STAGE19_EVALUATOR_CRITIC_SCHEMA_VERSION:
        return {}, ["invalid_stage19_evaluator_critic_preflight_summary_schema"]
    if payload.get("profile_id") != STAGE18_9_EXPECTED_PROFILE_ID or payload.get("profile_version") != STAGE18_9_EXPECTED_PROFILE_VERSION:
        return {}, ["invalid_stage19_evaluator_critic_preflight_profile_lineage"]
    if roots.path_cost_weight_calibration_root is not None and payload.get("stage18_11_path_cost_weight_calibration_root") is not None:
        if not _same_path(payload.get("stage18_11_path_cost_weight_calibration_root"), roots.path_cost_weight_calibration_root):
            return {}, ["stale_stage19_evaluator_critic_preflight_root"]
    routing = payload.get("next_stage_routing")
    if not isinstance(routing, dict) or routing.get("schema_version") != STAGE19_ROUTING_SCHEMA_VERSION:
        return {}, ["invalid_stage19_evaluator_critic_preflight_routing_schema"]
    route = routing.get("primary_route")
    if not isinstance(route, str) or route not in STAGE19_ALLOWED_ROUTES:
        return {}, ["invalid_stage19_evaluator_critic_preflight_route"]
    if routing.get("stage20_authorized") is not False or payload.get("stage20_authorized") is not False:
        return {}, ["stage19_stage20_authorized_not_false"]
    if payload.get("training_or_release_authorized") is not False:
        return {}, ["stage19_training_authorized_not_false"]
    if _boundary_violation(payload):
        return {}, ["boundary_violation"]
    if route == "stage20_reward_rerank_oracle_preference_dataset_preparation" and not _stage19_preflight_semantics_are_clean(payload):
        return {}, ["invalid_stage19_evaluator_critic_preflight_semantics"]
    return payload, []


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _valid_profile_identity(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    return all(isinstance(payload.get(field), str) and bool(payload.get(field)) for field in ("profile_id", "profile_version", "profile_hash"))


def _valid_stage18_9_profile_identity(payload: dict[str, Any]) -> bool:
    return (
        _valid_profile_identity(payload)
        and payload.get("profile_id") == STAGE18_9_EXPECTED_PROFILE_ID
        and payload.get("profile_version") == STAGE18_9_EXPECTED_PROFILE_VERSION
    )


def _profile_identity_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in ("profile_id", "profile_version", "profile_hash"))


def _stage18_6_preflight_semantics_are_clean(payload: dict[str, Any]) -> bool:
    candidate_readiness = payload.get("candidate_metric_readiness")
    paired_summary = payload.get("paired_decision_summary")
    readiness = payload.get("stage19_readiness")
    return (
        payload.get("guard_refinement_passed") is True
        and isinstance(candidate_readiness, dict)
        and candidate_readiness.get("full_candidate_metric_replay_available") is True
        and isinstance(paired_summary, dict)
        and paired_summary.get("same_candidate_set_guard_clean_advantage_established") is True
        and isinstance(readiness, dict)
        and readiness.get("authorized") is False
    )


def _stage18_7_preflight_semantics_are_clean(payload: dict[str, Any]) -> bool:
    readiness = payload.get("stage19_readiness")
    rows = payload.get("candidate_count_results")
    if not isinstance(rows, list) or len(rows) != 4:
        return False
    clean_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("guard_refinement_passed") is True
        and row.get("same_candidate_set_guard_clean_advantage_established") is True
        and row.get("candidate_metric_replay_available") is True
        and row.get("stage19_authorized") is False
        and float(row.get("guard_clean_candidate_available_rate", 0.0) or 0.0) > 0.0
        and float(row.get("xunce_selected_guard_clean_rate", 0.0) or 0.0)
        >= float(row.get("incumbent_selected_guard_clean_rate", 0.0) or 0.0)
    ]
    return (
        isinstance(readiness, dict)
        and readiness.get("authorized") is False
        and int(payload.get("stage18_6_guard_refinement_passed_count", 0) or 0) > 0
        and int(payload.get("same_candidate_set_guard_clean_advantage_established_count", 0) or 0) > 0
        and float(payload.get("best_guard_clean_candidate_available_rate", 0.0) or 0.0) > 0.0
        and bool(clean_rows)
    )


def _stage18_9_preflight_semantics_are_clean(payload: dict[str, Any]) -> bool:
    readiness = payload.get("stage19_readiness")
    boundary = payload.get("path_risk_boundary_summary")
    trajectory = payload.get("trajectory_guard_summary")
    return (
        payload.get("status") == "passed"
        and payload.get("trajectory_guard_passed") is True
        and isinstance(readiness, dict)
        and readiness.get("schema_version") == STAGE18_9_STAGE19_READINESS_SCHEMA_VERSION
        and readiness.get("readiness") == "ready_for_stage19_preflight_human_review_only"
        and readiness.get("authorized") is False
        and readiness.get("trajectory_guard_passed") is True
        and isinstance(boundary, dict)
        and int(boundary.get("hard_risk_violation_count", 0) or 0) == 0
        and boundary.get("path_risk_boundary_passed") is True
        and isinstance(trajectory, dict)
        and trajectory.get("coverage_advantage_established") is True
        and trajectory.get("path_cost_budget_passed") is True
        and trajectory.get("coverage_efficiency_passed") is True
        and trajectory.get("soft_risk_exposure_passed") is True
    )


def _stage18_11_preflight_semantics_are_clean(payload: dict[str, Any]) -> bool:
    readiness = payload.get("stage19_readiness")
    diagnostic = payload.get("diagnostic_rollout_summary")
    return (
        payload.get("status") == "passed"
        and payload.get("next_required_change") == STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE
        and payload.get("stage19_authorized") is False
        and isinstance(readiness, dict)
        and readiness.get("authorized") is False
        and isinstance(diagnostic, dict)
        and int(diagnostic.get("complete_diagnostic_rollout_count", 0) or 0) >= 4
        and float(diagnostic.get("best_final_coverage_rate_mean", 0.0) or 0.0) >= 0.99
        and float(diagnostic.get("best_hard_risk_violation_count", 0.0) or 0.0) <= 0.0
    )


def _stage19_preflight_semantics_are_clean(payload: dict[str, Any]) -> bool:
    readiness = payload.get("stage20_readiness")
    target = payload.get("practical_target_selection")
    critic = payload.get("critic_target_readiness")
    return (
        payload.get("status") == "passed"
        and payload.get("oracle_target_feasible") is True
        and payload.get("primary_target_selected") is True
        and payload.get("xunce_checkpoint_advantage_established") is False
        and payload.get("stage20_authorized") is False
        and payload.get("training_or_release_authorized") is False
        and isinstance(readiness, dict)
        and readiness.get("authorized") is False
        and isinstance(target, dict)
        and target.get("primary_target_feasible") is True
        and target.get("primary_budget_passed") is True
        and isinstance(critic, dict)
        and critic.get("critic_target_ready") is True
    )


def _stage18_7_results_are_valid(payload: dict[str, Any]) -> bool:
    rows = payload.get("candidate_count_results")
    if not isinstance(rows, list):
        return False
    expected = {(6, 48), (12, 96), (24, 192), (36, 288)}
    observed: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        if row.get("schema_version") != "xunce-stage18-7-candidate-count-scaling-result/v1":
            return False
        try:
            count = int(row.get("candidate_count"))
            pool = int(row.get("proposal_pool_limit"))
            expected_pool = int(row.get("expected_proposal_pool_limit", pool))
        except (TypeError, ValueError):
            return False
        if (count, pool) not in expected or expected_pool != pool:
            return False
        observed.add((count, pool))
        if row.get("stage19_authorized") is not False:
            return False
        if row.get("boundary_flags_all_false") is False:
            return False
        route = row.get("stage18_6_next_required_change")
        if route is not None and route not in STAGE18_6_ALLOWED_ROUTES:
            return False
        profile_hash = row.get("profile_hash")
        if profile_hash is not None and profile_hash != payload.get("profile_hash"):
            return False
    return observed == expected


def _read_first_json(root: Path, filenames: tuple[str, ...], missing: list[str], reason_code: str) -> dict[str, Any]:
    for filename in filenames:
        path = root / filename
        if path.is_file():
            return _read_json(path, missing, reason_code)
    missing.append(reason_code)
    return {}


def _module_results(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    payloads = {
        "stage18_1_scenario_evidence_preparation": evidence["stage18_1"],
        "stage18_2_candidate_generation_and_validation": evidence["stage18_2"],
        "stage18_3_model_inference": evidence["model_inference"],
        "stage18_3_true_incumbent_binding": evidence["true_binding"],
        "stage18_4_quantization": evidence["quantization"],
        "stage18_4_oracle": evidence["oracle"],
        "stage18_4_coverage_rollout": evidence["coverage_comparison"],
    }
    return [
        {
            "module_id": module_id,
            "present": bool(payload),
            "status": payload.get("status") if payload else "missing",
            "schema_version": payload.get("schema_version") if payload else None,
            "next_required_change": payload.get("next_required_change") if payload else None,
        }
        for module_id, payload in payloads.items()
    ]


def _root_consistency_reasons(
    *,
    roots: Stage18RootSet,
    binding: dict[str, Any],
    quant: dict[str, Any],
    oracle: dict[str, Any],
    coverage: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if binding:
        if not _same_path(binding.get("source_materialized_coverage_root"), roots.candidate_root):
            reasons.append("stale_or_mixed_stage18_roots")
        if not _same_path(binding.get("source_model_inference_root"), roots.model_inference_root):
            reasons.append("stale_or_mixed_stage18_roots")
    if quant and not _same_path(quant.get("source_bound_coverage_root"), roots.true_binding_root):
        reasons.append("stale_or_mixed_stage18_roots")
    if oracle and not _same_path(oracle.get("source_materialized_coverage_root"), roots.quantization_root):
        reasons.append("stale_or_mixed_stage18_roots")
    if coverage and not _same_path(coverage.get("source_roi_expansion_root"), roots.quantization_root):
        reasons.append("stale_or_mixed_stage18_roots")
    if coverage and coverage.get("dynamic_validation_source_root") and not _same_path(coverage.get("dynamic_validation_source_root"), roots.quantization_root):
        reasons.append("stale_or_mixed_stage18_roots")
    return unique_sorted(reasons)


def _same_path(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).resolve() == expected.resolve()


def _evidence_status(*, missing: list[str], blocking: list[str]) -> str:
    if blocking:
        return "blocked"
    if missing:
        return "partial"
    return "passed"


def _candidate_validity_status(
    *,
    missing: list[str],
    blocking: list[str],
    stage18_2: dict[str, Any],
    quant: dict[str, Any],
) -> str:
    if "candidate_validity_gate_failed" in blocking or "stage18_2_candidate_generation_not_passed" in blocking:
        return "blocked"
    if {"missing_stage18_2_candidate_generation", "missing_stage18_4_quantization"} & set(missing):
        return "partial"
    if stage18_2 and quant:
        return "passed"
    return "partial"


def _comparison_verdict(
    *,
    evidence_status: str,
    model: dict[str, Any],
    coverage: dict[str, Any],
    stage18_5_summary: dict[str, Any],
    stage18_9_summary: dict[str, Any],
) -> str:
    if evidence_status != "passed":
        return "inconclusive"
    stage18_9 = stage18_9_summary.get("summary") if isinstance(stage18_9_summary, dict) else None
    if isinstance(stage18_9, dict) and stage18_9.get("trajectory_guard_passed") is True:
        return "xunce_advantage_established"
    stage18_5 = stage18_5_summary.get("summary") if isinstance(stage18_5_summary, dict) else None
    if not isinstance(stage18_5, dict):
        return "xunce_advantage_not_established"
    if stage18_5.get("stage19_authorized") is not True:
        return "xunce_advantage_not_established"
    if model.get("xunce_candidate_advantage_established") is True or coverage.get("xunce_coverage_advantage_established") is True:
        return "xunce_advantage_established"
    return "xunce_advantage_not_established"


def _overall_conclusion(*, evidence_status: str, candidate_validity_status: str, comparison_verdict: str) -> str:
    if evidence_status == "blocked" or candidate_validity_status == "blocked":
        return "stage18_evidence_blocked"
    if evidence_status == "partial" or candidate_validity_status == "partial":
        return "stage18_evidence_incomplete"
    if comparison_verdict == "xunce_advantage_established":
        return "evidence_valid_and_xunce_advantage_established_for_human_review_only"
    if comparison_verdict == "xunce_advantage_not_established":
        return "evidence_valid_but_xunce_advantage_not_established"
    return "stage18_comparison_inconclusive"


def _next_required_change(
    *,
    missing: list[str],
    blocking: list[str],
    comparison_verdict: str,
    stage18_5_route: str | None = None,
    stage18_6_route: str | None = None,
    stage18_7_route: str | None = None,
    stage18_9_route: str | None = None,
    stage18_11_route: str | None = None,
    stage19_route: str | None = None,
) -> str:
    if "boundary_violation" in blocking:
        return BOUNDARY_REPAIR_NEXT_REQUIRED_CHANGE
    if "stale_or_mixed_stage18_roots" in blocking:
        return ROOT_REPAIR_NEXT_REQUIRED_CHANGE
    if stage19_route:
        return stage19_route
    if stage18_11_route:
        return stage18_11_route
    if stage18_9_route:
        return stage18_9_route
    if stage18_7_route:
        return stage18_7_route
    if stage18_6_route:
        return stage18_6_route
    if stage18_5_route:
        return stage18_5_route
    if "missing_dynamic_frontier_nbv_rollout_comparison" in missing:
        return DYNAMIC_ROLLOUT_NEXT_REQUIRED_CHANGE
    if blocking:
        return REFRESH_STAGE18_NEXT_REQUIRED_CHANGE
    if missing:
        return REFRESH_STAGE18_NEXT_REQUIRED_CHANGE
    if comparison_verdict == "xunce_advantage_not_established":
        return REVIEW_METRICS_NEXT_REQUIRED_CHANGE
    return REVIEW_METRICS_NEXT_REQUIRED_CHANGE


def _stage18_5_attribution_summary(attribution: dict[str, Any]) -> dict[str, Any]:
    if not attribution:
        return {
            "summary": None,
            "guard_verdict": None,
            "primary_next_required_change": None,
        }
    routing = attribution.get("next_stage_routing", {})
    guard = attribution.get("guard_evaluation", {})
    stage19 = attribution.get("stage19_readiness", {})
    return {
        "summary": {
            "status": attribution.get("status"),
            "evidence_authenticity_gate_passed": attribution.get("evidence_authenticity_gate_passed"),
            "candidate_validity_gate_passed": attribution.get("candidate_validity_gate_passed"),
            "guard_passed": guard.get("passed"),
            "failed_guards": guard.get("failed_guards", []),
            "primary_next_required_change": routing.get("primary_route", attribution.get("next_required_change")),
            "stage19_authorized": bool(routing.get("stage19_authorized") is True or stage19.get("authorized") is True),
            "stage19_readiness": stage19.get("readiness"),
        },
        "guard_verdict": "passed" if guard.get("passed") is True else "failed",
        "primary_next_required_change": routing.get("primary_route", attribution.get("next_required_change")),
    }


def _stage18_6_guard_refinement_summary(guard_refinement: dict[str, Any]) -> dict[str, Any]:
    if not guard_refinement:
        return {
            "summary": None,
            "guard_verdict": None,
            "primary_next_required_change": None,
            "candidate_metric_readiness": None,
        }
    routing = guard_refinement.get("next_stage_routing", {})
    readiness = guard_refinement.get("stage19_readiness", {})
    candidate_readiness = guard_refinement.get("candidate_metric_readiness", {})
    return {
        "summary": {
            "status": guard_refinement.get("status"),
            "guard_refinement_passed": guard_refinement.get("guard_refinement_passed"),
            "counterfactual_reselection_claimed": guard_refinement.get("counterfactual_reselection_claimed"),
            "candidate_metric_readiness": candidate_readiness,
            "primary_next_required_change": routing.get("primary_route", guard_refinement.get("next_required_change")),
            "stage19_authorized": bool(routing.get("stage19_authorized") is True or readiness.get("authorized") is True),
            "stage19_readiness": readiness.get("readiness"),
        },
        "guard_verdict": "passed" if guard_refinement.get("guard_refinement_passed") is True else "failed",
        "primary_next_required_change": routing.get("primary_route", guard_refinement.get("next_required_change")),
        "candidate_metric_readiness": candidate_readiness,
    }


def _stage18_7_candidate_count_scaling_summary(candidate_count_scaling: dict[str, Any]) -> dict[str, Any]:
    if not candidate_count_scaling:
        return {
            "summary": None,
            "guard_verdict": None,
            "primary_next_required_change": None,
        }
    routing = candidate_count_scaling.get("next_stage_routing", {})
    readiness = candidate_count_scaling.get("stage19_readiness", {})
    return {
        "summary": {
            "status": candidate_count_scaling.get("status"),
            "sweep_complete_count": candidate_count_scaling.get("sweep_complete_count"),
            "stage18_6_guard_refinement_passed_count": candidate_count_scaling.get("stage18_6_guard_refinement_passed_count"),
            "same_candidate_set_guard_clean_advantage_established_count": candidate_count_scaling.get(
                "same_candidate_set_guard_clean_advantage_established_count"
            ),
            "best_guard_clean_candidate_available_rate": candidate_count_scaling.get("best_guard_clean_candidate_available_rate"),
            "best_xunce_selected_guard_clean_rate": candidate_count_scaling.get("best_xunce_selected_guard_clean_rate"),
            "best_incumbent_selected_guard_clean_rate": candidate_count_scaling.get("best_incumbent_selected_guard_clean_rate"),
            "primary_next_required_change": routing.get("primary_route", candidate_count_scaling.get("next_required_change")),
            "stage19_authorized": bool(routing.get("stage19_authorized") is True or readiness.get("authorized") is True),
            "stage19_readiness": readiness.get("readiness"),
        },
        "guard_verdict": "passed"
        if candidate_count_scaling.get("stage18_6_guard_refinement_passed_count", 0)
        and candidate_count_scaling.get("same_candidate_set_guard_clean_advantage_established_count", 0)
        else "failed",
        "primary_next_required_change": routing.get("primary_route", candidate_count_scaling.get("next_required_change")),
    }


def _stage18_9_trajectory_risk_reward_summary(trajectory_risk_reward: dict[str, Any]) -> dict[str, Any]:
    if not trajectory_risk_reward:
        return {
            "summary": None,
            "guard_verdict": None,
            "primary_next_required_change": None,
        }
    routing = trajectory_risk_reward.get("next_stage_routing", {})
    readiness = trajectory_risk_reward.get("stage19_readiness", {})
    trajectory_summary = trajectory_risk_reward.get("trajectory_guard_summary", {})
    boundary_summary = trajectory_risk_reward.get("path_risk_boundary_summary", {})
    return {
        "summary": {
            "status": trajectory_risk_reward.get("status"),
            "trajectory_guard_passed": trajectory_risk_reward.get("trajectory_guard_passed"),
            "path_risk_boundary_passed": boundary_summary.get("path_risk_boundary_passed"),
            "hard_risk_violation_count": boundary_summary.get("hard_risk_violation_count"),
            "coverage_advantage_established": trajectory_summary.get("coverage_advantage_established"),
            "path_cost_budget_passed": trajectory_summary.get("path_cost_budget_passed"),
            "coverage_efficiency_passed": trajectory_summary.get("coverage_efficiency_passed"),
            "soft_risk_exposure_passed": trajectory_summary.get("soft_risk_exposure_passed"),
            "candidate_diagnostics": trajectory_risk_reward.get("candidate_diagnostics"),
            "primary_next_required_change": routing.get("primary_route", trajectory_risk_reward.get("next_required_change")),
            "stage19_authorized": bool(routing.get("stage19_authorized") is True or readiness.get("authorized") is True),
            "stage19_readiness": readiness.get("readiness"),
        },
        "guard_verdict": "passed" if trajectory_risk_reward.get("trajectory_guard_passed") is True else "failed",
        "primary_next_required_change": routing.get("primary_route", trajectory_risk_reward.get("next_required_change")),
    }


def _stage18_11_path_cost_weight_calibration_summary(path_cost_weight_calibration: dict[str, Any]) -> dict[str, Any]:
    if not path_cost_weight_calibration:
        return {
            "summary": None,
            "guard_verdict": None,
            "primary_next_required_change": None,
        }
    routing = path_cost_weight_calibration.get("next_stage_routing", {})
    readiness = path_cost_weight_calibration.get("stage19_readiness", {})
    diagnostic = path_cost_weight_calibration.get("diagnostic_rollout_summary", {})
    return {
        "summary": {
            "status": path_cost_weight_calibration.get("status"),
            "target_final_coverage_rate": path_cost_weight_calibration.get("target_final_coverage_rate"),
            "best_diagnostic_rollout_candidate_count": path_cost_weight_calibration.get("best_diagnostic_rollout_candidate_count"),
            "best_diagnostic_rollout_path_cost_weight": path_cost_weight_calibration.get("best_diagnostic_rollout_path_cost_weight"),
            "best_diagnostic_final_coverage_rate_mean": path_cost_weight_calibration.get("best_diagnostic_final_coverage_rate_mean"),
            "best_diagnostic_final_coverage_rate_max": path_cost_weight_calibration.get("best_diagnostic_final_coverage_rate_max"),
            "complete_diagnostic_rollout_count": diagnostic.get("complete_diagnostic_rollout_count") if isinstance(diagnostic, dict) else None,
            "primary_next_required_change": routing.get("primary_route", path_cost_weight_calibration.get("next_required_change")),
            "stage19_authorized": bool(routing.get("stage19_authorized") is True or readiness.get("authorized") is True),
            "stage19_readiness": readiness.get("readiness") if isinstance(readiness, dict) else None,
        },
        "guard_verdict": "passed" if path_cost_weight_calibration.get("status") == "passed" else "failed",
        "primary_next_required_change": routing.get("primary_route", path_cost_weight_calibration.get("next_required_change")),
    }


def _stage19_evaluator_critic_preflight_summary(evaluator_critic_preflight: dict[str, Any]) -> dict[str, Any]:
    if not evaluator_critic_preflight:
        return {
            "summary": None,
            "guard_verdict": None,
            "primary_next_required_change": None,
            "primary_target_candidate_count": None,
            "primary_target_path_cost_weight": None,
            "oracle_target_feasible": None,
            "xunce_checkpoint_advantage_established": None,
            "training_authorized": None,
        }
    routing = evaluator_critic_preflight.get("next_stage_routing", {})
    readiness = evaluator_critic_preflight.get("stage20_readiness", {})
    target = evaluator_critic_preflight.get("practical_target_selection", {})
    critic = evaluator_critic_preflight.get("critic_target_readiness", {})
    return {
        "summary": {
            "status": evaluator_critic_preflight.get("status"),
            "oracle_target_feasible": evaluator_critic_preflight.get("oracle_target_feasible"),
            "primary_target_selected": evaluator_critic_preflight.get("primary_target_selected"),
            "selected_candidate_count": evaluator_critic_preflight.get("selected_candidate_count"),
            "selected_path_cost_weight": evaluator_critic_preflight.get("selected_path_cost_weight"),
            "xunce_checkpoint_advantage_established": evaluator_critic_preflight.get("xunce_checkpoint_advantage_established"),
            "critic_target_ready": critic.get("critic_target_ready") if isinstance(critic, dict) else None,
            "preference_pair_count": critic.get("preference_pair_count") if isinstance(critic, dict) else None,
            "primary_next_required_change": routing.get("primary_route", evaluator_critic_preflight.get("next_required_change")),
            "stage20_authorized": bool(routing.get("stage20_authorized") is True or readiness.get("authorized") is True),
            "stage20_readiness": readiness.get("readiness") if isinstance(readiness, dict) else None,
        },
        "guard_verdict": "passed" if evaluator_critic_preflight.get("status") == "passed" else "failed",
        "primary_next_required_change": routing.get("primary_route", evaluator_critic_preflight.get("next_required_change")),
        "primary_target_candidate_count": target.get("selected_candidate_count") if isinstance(target, dict) else None,
        "primary_target_path_cost_weight": target.get("selected_path_cost_weight") if isinstance(target, dict) else None,
        "oracle_target_feasible": evaluator_critic_preflight.get("oracle_target_feasible"),
        "xunce_checkpoint_advantage_established": evaluator_critic_preflight.get("xunce_checkpoint_advantage_established"),
        "training_authorized": bool(evaluator_critic_preflight.get("training_or_release_authorized") is True),
    }


def _diagnostic_recommendations(diagnostic: list[str]) -> list[str]:
    recommendations: list[str] = []
    reason_set = set(diagnostic)
    if {"safe_efficient_candidate_missing", "oracle_not_separable"} & reason_set:
        recommendations.append("candidate_generation_or_roi_complexity_review")
    if {"single_step_xunce_advantage_not_established", "xunce_coverage_advantage_not_established"} & reason_set:
        recommendations.append(STAGE19_PREFLIGHT_NEXT_REQUIRED_CHANGE)
    if "coverage_efficiency_regression_present" in reason_set:
        recommendations.append("review_coverage_cost_tradeoff_metrics")
    if "dynamic_frontier_nbv_rollout_not_executed" in reason_set:
        recommendations.append(DYNAMIC_ROLLOUT_NEXT_REQUIRED_CHANGE)
    if {"sidecar_screening_not_full_adapter_evidence", "dynamic_validation_not_full_adapter_evidence"} & reason_set:
        recommendations.append("repair_dynamic_path_planner_adapter_evidence")
    return unique_sorted(recommendations)


def _single_step_summary(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": model.get("status"),
        "true_model_inference_executed": model.get("true_model_inference_executed"),
        "proxy_selection_used": model.get("proxy_selection_used"),
        "xunce_candidate_advantage_established": model.get("xunce_candidate_advantage_established"),
        "xunce_better_than_incumbent_count": model.get("xunce_better_than_incumbent_count"),
        "xunce_worse_than_incumbent_count": model.get("xunce_worse_than_incumbent_count"),
        "latency_ratio_vs_incumbent": model.get("latency_ratio_vs_incumbent"),
    }


def _coverage_summary(coverage: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": coverage.get("status"),
        "candidate_refresh_mode": coverage.get("candidate_refresh_mode"),
        "dynamic_candidate_generation_executed": coverage.get("dynamic_candidate_generation_executed"),
        "dynamic_validation_work_root": coverage.get("dynamic_validation_work_root"),
        "dynamic_validation_work_root_path_length": coverage.get("dynamic_validation_work_root_path_length"),
        "dynamic_validation_max_path_length": coverage.get("dynamic_validation_max_path_length"),
        "dynamic_candidate_validation_mode": coverage.get("dynamic_candidate_validation_mode"),
        "dynamic_validation_attempt_count": coverage.get("dynamic_validation_attempt_count"),
        "dynamic_validation_success_count": coverage.get("dynamic_validation_success_count"),
        "dynamic_path_length_preflight_failure_count": coverage.get("dynamic_path_length_preflight_failure_count"),
        "in_process_batch_astar_validation_count": coverage.get("in_process_batch_astar_validation_count"),
        "path_planner_route_adapter_success_count": coverage.get("path_planner_route_adapter_success_count"),
        "path_planner_route_adapter_failure_count": coverage.get("path_planner_route_adapter_failure_count"),
        "path_planner_route_adapter_audit_sample_count": coverage.get("path_planner_route_adapter_audit_sample_count"),
        "path_planner_route_adapter_audit_failure_count": coverage.get("path_planner_route_adapter_audit_failure_count"),
        "adapter_batch_astar_mismatch_count": coverage.get("adapter_batch_astar_mismatch_count"),
        "adapter_audit_passed": coverage.get("adapter_audit_passed"),
        "sidecar_grid_astar_screening_count": coverage.get("sidecar_grid_astar_screening_count"),
        "sidecar_grid_astar_diagnostic_count": coverage.get("sidecar_grid_astar_diagnostic_count"),
        "adapter_error_type_counts": coverage.get("adapter_error_type_counts"),
        "adapter_error_message_samples": coverage.get("adapter_error_message_samples"),
        "planner_validation_backend_counts": coverage.get("planner_validation_backend_counts", coverage.get("dynamic_planner_validation_backend_counts")),
        "validation_evidence_kind_counts": coverage.get("validation_evidence_kind_counts"),
        "coverage_frontier_candidate_count": coverage.get("coverage_frontier_candidate_count", 0),
        "undercovered_component_candidate_count": coverage.get("undercovered_component_candidate_count", 0),
        "candidate_generation_algorithm_source_counts": coverage.get("candidate_generation_algorithm_source_counts", {}),
        "low_cost_bridge_candidate_count": coverage.get("low_cost_bridge_candidate_count", 0),
        "conservative_local_candidate_count": coverage.get("conservative_local_candidate_count", 0),
        "validated_pareto_frontier_count": coverage.get("validated_pareto_frontier_count", 0),
        "validated_low_cost_candidate_count": coverage.get("validated_low_cost_candidate_count", 0),
        "validated_efficiency_candidate_count": coverage.get("validated_efficiency_candidate_count", 0),
        "risk_source_counts": coverage.get("risk_source_counts", {}),
        "formal_risk_source_counts": coverage.get("formal_risk_source_counts", {}),
        "selected_risk_source_counts": coverage.get("selected_risk_source_counts", {}),
        "route_derived_risk_count": coverage.get("route_derived_risk_count", 0),
        "roi_weight_source_counts": coverage.get("roi_weight_source_counts", {}),
        "candidate_selection_mode": coverage.get("candidate_selection_mode"),
        "dynamic_validation_full_adapter_evidence_passed": coverage.get("dynamic_validation_full_adapter_evidence_passed"),
        "dynamic_candidate_generation_missing_count": coverage.get("dynamic_candidate_generation_missing_count", 0),
        "candidate_generation_exhausted_count": coverage.get("candidate_generation_exhausted_count", 0),
        "model_inference_failure_count": coverage.get("model_inference_failure_count", 0),
        "coverage_rate_saturation_episode_count": coverage.get("coverage_rate_saturation_episode_count", aggregate.get("coverage_rate_saturation_episode_count", 0)),
        "max_final_coverage_rate_raw": coverage.get("max_final_coverage_rate_raw", aggregate.get("max_final_coverage_rate_raw", 0)),
        "max_coverage_rate_saturation_excess": coverage.get("max_coverage_rate_saturation_excess", aggregate.get("max_coverage_rate_saturation_excess", 0)),
        "paired_decision_audit_row_count": coverage.get("paired_decision_audit_row_count"),
        "candidate_generation_effect_scope": coverage.get("candidate_generation_effect_scope"),
        "model_selection_evidence_scope": coverage.get("model_selection_evidence_scope"),
        "closed_loop_dynamic_rollout_summary": coverage.get("closed_loop_dynamic_rollout_summary"),
        "same_candidate_set_policy_selection_summary": coverage.get("same_candidate_set_policy_selection_summary"),
        "coverage_delta_cells_mean": aggregate.get("coverage_delta_cells_mean", coverage.get("xunce_new_covered_cell_delta_vs_incumbent")),
        "coverage_delta_cells_median": aggregate.get("coverage_delta_cells_median"),
        "path_cost_delta_m_mean": aggregate.get("path_cost_delta_m_mean", coverage.get("xunce_path_cost_delta_vs_incumbent")),
        "risk_delta_mean": aggregate.get("risk_delta_mean", coverage.get("xunce_risk_delta_vs_incumbent")),
        "coverage_per_100m_delta_mean": aggregate.get("coverage_per_100m_delta_mean", coverage.get("coverage_per_100m_delta_vs_incumbent")),
        "policy_disagreement_count": coverage.get("policy_disagreement_count"),
        "useful_disagreement_count": coverage.get("useful_disagreement_count"),
    }


def _comparison_metric_summary(coverage: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": coverage.get("status"),
        "coverage_delta_cells_mean": aggregate.get("coverage_delta_cells_mean", coverage.get("xunce_new_covered_cell_delta_vs_incumbent")),
        "path_cost_delta_m_mean": aggregate.get("path_cost_delta_m_mean", coverage.get("xunce_path_cost_delta_vs_incumbent")),
        "risk_delta_mean": aggregate.get("risk_delta_mean", coverage.get("xunce_risk_delta_vs_incumbent")),
        "coverage_per_100m_delta_mean": aggregate.get("coverage_per_100m_delta_mean", coverage.get("coverage_per_100m_delta_vs_incumbent")),
        "greedy_oracle_coverage_regret_delta_mean": aggregate.get("greedy_oracle_coverage_regret_delta_mean"),
        "cost_aware_oracle_utility_regret_delta_mean": aggregate.get("cost_aware_oracle_utility_regret_delta_mean"),
    }


def _distribution_summary(aggregate: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "mean": aggregate.get(f"{prefix}_mean"),
        "median": aggregate.get(f"{prefix}_median"),
        "iqr": aggregate.get(f"{prefix}_iqr"),
        "min": aggregate.get(f"{prefix}_min"),
        "max": aggregate.get(f"{prefix}_max"),
    }


def _scenario_win_loss_summary(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "xunce_coverage_win_count": aggregate.get("xunce_coverage_win_count"),
        "xunce_coverage_tie_count": aggregate.get("xunce_coverage_tie_count"),
        "xunce_coverage_loss_count": aggregate.get("xunce_coverage_loss_count"),
        "xunce_coverage_win_rate": aggregate.get("xunce_coverage_win_rate"),
    }


def _legacy_label_summary(model: dict[str, Any], coverage: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    return {
        "xunce_candidate_advantage_established": model.get("xunce_candidate_advantage_established"),
        "xunce_coverage_advantage_established": coverage.get("xunce_coverage_advantage_established"),
        "xunce_efficiency_regression_count": coverage.get("xunce_efficiency_regression_count"),
        "oracle_separable": oracle.get("oracle_separable"),
    }


def _oracle_summary(oracle: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": oracle.get("status"),
        "oracle_separable": oracle.get("oracle_separable"),
        "greedy_oracle_coverage_return_delta_vs_incumbent": oracle.get("greedy_oracle_coverage_return_delta_vs_incumbent"),
        "cost_aware_oracle_coverage_return_delta_vs_incumbent": oracle.get("cost_aware_oracle_coverage_return_delta_vs_incumbent"),
        "oracle_better_scenario_fraction": oracle.get("oracle_better_scenario_fraction"),
        "safe_efficient_opportunity_count": oracle.get("safe_efficient_opportunity_count"),
    }


def _candidate_summary(stage18_2: dict[str, Any], quant: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage18_2_status": stage18_2.get("status"),
        "candidate_count": stage18_2.get("candidate_count", quant.get("candidate_count")),
        "valid_candidate_count": stage18_2.get("valid_candidate_count", quant.get("valid_candidate_count")),
        "safe_efficient_candidate_count": quant.get("safe_efficient_candidate_count", stage18_2.get("safe_efficient_candidate_count")),
        "candidate_coverage_spread_range": stage18_2.get("candidate_coverage_spread_range"),
        "candidate_validation_mode": stage18_2.get("candidate_validation_mode"),
        "coverage_frontier_candidate_count": stage18_2.get("coverage_frontier_candidate_count"),
        "undercovered_component_candidate_count": stage18_2.get("undercovered_component_candidate_count"),
        "low_cost_bridge_candidate_count": stage18_2.get("low_cost_bridge_candidate_count"),
        "candidate_selection_mode": stage18_2.get("candidate_selection_mode"),
    }


def _boundary_violation(*payloads: dict[str, Any]) -> bool:
    for payload in payloads:
        if not payload:
            continue
        if float(payload.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
            return True
        for field in BOUNDARY_FIELDS:
            if payload.get(field) is True:
                return True
    return False


def boundary_payload() -> dict[str, Any]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields["canary_traffic_fraction"] = 0.0
    return fields


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
