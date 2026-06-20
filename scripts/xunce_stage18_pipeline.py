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
    return {
        "stage18_1": stage18_1,
        "stage18_2": stage18_2,
        "model_inference": model,
        "true_binding": binding,
        "quantization": quant,
        "oracle": oracle,
        "coverage_comparison": coverage,
        "coverage_comparison_aggregate": coverage_aggregate,
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

    if _boundary_violation(*(payload for payload in evidence.values() if isinstance(payload, dict))):
        blocking.append("boundary_violation")
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
    comparison_verdict = _comparison_verdict(evidence_status=evidence_status, model=model, coverage=coverage)
    overall_conclusion = _overall_conclusion(
        evidence_status=evidence_status,
        candidate_validity_status=candidate_validity_status,
        comparison_verdict=comparison_verdict,
    )

    return {
        "status": "failed" if blocking else "partial" if missing else "passed",
        "evidence_status": evidence_status,
        "candidate_validity_status": candidate_validity_status,
        "comparison_verdict": comparison_verdict,
        "overall_conclusion": overall_conclusion,
        "next_required_change": _next_required_change(missing=missing, blocking=blocking, comparison_verdict=comparison_verdict),
        "reason_codes": unique_sorted([*missing, *blocking]),
        "missing_reason_codes": unique_sorted(missing),
        "blocking_reason_codes": unique_sorted(blocking),
        "diagnostic_reason_codes": unique_sorted(diagnostic),
        "diagnostic_recommended_changes": _diagnostic_recommendations(diagnostic),
        "module_results": module_results,
        "single_step_comparison_summary": _single_step_summary(model),
        "coverage_rollout_comparison_summary": _coverage_summary(coverage, coverage_aggregate),
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
        return True
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


def _comparison_verdict(*, evidence_status: str, model: dict[str, Any], coverage: dict[str, Any]) -> str:
    if evidence_status != "passed":
        return "inconclusive"
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


def _next_required_change(*, missing: list[str], blocking: list[str], comparison_verdict: str) -> str:
    if "boundary_violation" in blocking:
        return BOUNDARY_REPAIR_NEXT_REQUIRED_CHANGE
    if "stale_or_mixed_stage18_roots" in blocking:
        return ROOT_REPAIR_NEXT_REQUIRED_CHANGE
    if "missing_dynamic_frontier_nbv_rollout_comparison" in missing:
        return DYNAMIC_ROLLOUT_NEXT_REQUIRED_CHANGE
    if blocking:
        return REFRESH_STAGE18_NEXT_REQUIRED_CHANGE
    if missing:
        return REFRESH_STAGE18_NEXT_REQUIRED_CHANGE
    if comparison_verdict == "xunce_advantage_not_established":
        return "review_xunce_incumbent_comparison_metrics_or_prepare_stage19_evaluator_preflight"
    return REVIEW_METRICS_NEXT_REQUIRED_CHANGE


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
