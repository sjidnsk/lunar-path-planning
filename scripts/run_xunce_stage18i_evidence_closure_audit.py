from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from global_99_governance_common import global_99_boundary_defaults
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from scripts.global_99_governance_common import global_99_boundary_defaults


CONFIG_SCHEMA_VERSION = "xunce-stage18i-evidence-closure-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage18i-evidence-closure-summary/v1"
DEFAULT_CONFIG = "configs/xunce_stage18i_evidence_closure_audit_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_stage18i_evidence_closure_v1"

STAGE18I_SUMMARY_FILE = "xunce-risk-constrained-frontier-nbv-candidate-generation-summary.json"
STAGE18I2_SUMMARY_FILE = "xunce-risk-aware-frontier-nbv-candidate-repair-summary.json"
COMPARISON_SUMMARY_FILE = "xunce-high-fidelity-real-map-comparison-summary.json"
BINDING_SUMMARY_FILE = "xunce-true-incumbent-selection-binding-summary.json"
QUANTIZATION_SUMMARY_FILE = "xunce-risk-coverage-cost-quantization-summary.json"
ORACLE_SUMMARY_FILE = "xunce-oracle-separability-summary.json"
COVERAGE_SUMMARY_FILE = "xunce-exploration-coverage-comparison-summary.json"
SUMMARY_FILE = "xunce-stage18i-evidence-closure-summary.json"
REPORT_FILE = "xunce-stage18i-evidence-closure-report.md"

RUN_STAGE18I_NEXT_REQUIRED_CHANGE = "run_xunce_risk_constrained_frontier_nbv_candidate_generation"
RUN_STAGE18B_NEXT_REQUIRED_CHANGE = "run_xunce_high_fidelity_real_map_comparison_after_stage18i"
RUN_STAGE18G0_NEXT_REQUIRED_CHANGE = "run_true_incumbent_selection_binding_after_stage18i"
RUN_STAGE18H0_NEXT_REQUIRED_CHANGE = "run_risk_coverage_cost_quantization_after_stage18i"
RUN_STAGE18F_NEXT_REQUIRED_CHANGE = "run_oracle_separability_after_stage18i"
RUN_STAGE18C_NEXT_REQUIRED_CHANGE = "run_stage18c_v2_after_stage18i"
REPAIR_CANDIDATE_NEXT_REQUIRED_CHANGE = "refine_frontier_nbv_candidate_generation_or_expand_roi_map_complexity"
STAGE18J_NEXT_REQUIRED_CHANGE = "stage18j_coverage_cost_evaluator_critic_preflight"
AUTHORIZATION_NEXT_REQUIRED_CHANGE = "xunce_default_policy_candidate_authorization_preflight"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_stage18i_evidence_closure_boundary_rejections"
REVIEW_COMPARISON_NEXT_REQUIRED_CHANGE = "review_xunce_incumbent_comparison_metrics"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
    "runs_new_ppo_update",
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 18I after-stage evidence closure.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--stage18i-candidate-generation-root")
    parser.add_argument("--after-stage18i-model-inference-root")
    parser.add_argument("--after-stage18i-true-incumbent-binding-root")
    parser.add_argument("--after-stage18i-quantization-root")
    parser.add_argument("--after-stage18i-oracle-separability-root")
    parser.add_argument("--after-stage18i-coverage-comparison-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "stage18i_candidate_generation_root": args.stage18i_candidate_generation_root,
            "after_stage18i_model_inference_root": args.after_stage18i_model_inference_root,
            "after_stage18i_true_incumbent_binding_root": args.after_stage18i_true_incumbent_binding_root,
            "after_stage18i_quantization_root": args.after_stage18i_quantization_root,
            "after_stage18i_oracle_separability_root": args.after_stage18i_oracle_separability_root,
            "after_stage18i_coverage_comparison_root": args.after_stage18i_coverage_comparison_root,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_stage18i_evidence_closure_audit(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "oracle_separable": summary["oracle_separable"],
                "xunce_coverage_advantage_established": summary["xunce_coverage_advantage_established"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage18i_evidence_closure_audit(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    evidence = _load_evidence(config)
    metrics = _metrics(evidence)
    decision = _decision(metrics)
    paths = {"summary": output_root / SUMMARY_FILE, "report": output_root / REPORT_FILE}
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "blocking_reason_codes": decision["blocking_reason_codes"],
        "diagnostic_reason_codes": decision["diagnostic_reason_codes"],
        "diagnostic_recommended_change": decision["diagnostic_recommended_change"],
        "evidence_authenticity_gate_passed": decision["evidence_authenticity_gate_passed"],
        "candidate_validity_gate_passed": decision["candidate_validity_gate_passed"],
        "comparison_allowed": decision["comparison_allowed"],
        "next_required_change": decision["next_required_change"],
        **metrics,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "report": str(paths["report"]),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


def _load_config(config_path: Path, repo_root: Path, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"expected schema_version {CONFIG_SCHEMA_VERSION}")
    payload = {**payload, **(config_overrides or {})}
    required_roots = (
        "stage18i_candidate_generation_root",
        "after_stage18i_model_inference_root",
        "after_stage18i_true_incumbent_binding_root",
        "after_stage18i_quantization_root",
        "after_stage18i_oracle_separability_root",
        "after_stage18i_coverage_comparison_root",
    )
    config = {"schema_version": payload["schema_version"]}
    for key in required_roots:
        config[key] = str(resolve_path(Path(str(payload[key])), repo_root))
    config["canary_traffic_fraction"] = float(payload.get("canary_traffic_fraction", 0.0))
    return config


def _load_evidence(config: dict[str, Any]) -> dict[str, Any]:
    specs = {
        "comparison": (
            Path(config["after_stage18i_model_inference_root"]) / COMPARISON_SUMMARY_FILE,
            "missing_after_stage18i_model_inference",
        ),
        "binding": (
            Path(config["after_stage18i_true_incumbent_binding_root"]) / BINDING_SUMMARY_FILE,
            "missing_after_stage18i_true_incumbent_binding",
        ),
        "quantization": (
            Path(config["after_stage18i_quantization_root"]) / QUANTIZATION_SUMMARY_FILE,
            "missing_after_stage18i_quantization",
        ),
        "oracle": (
            Path(config["after_stage18i_oracle_separability_root"]) / ORACLE_SUMMARY_FILE,
            "missing_after_stage18i_oracle_separability",
        ),
        "coverage": (
            Path(config["after_stage18i_coverage_comparison_root"]) / COVERAGE_SUMMARY_FILE,
            "missing_after_stage18i_coverage_comparison",
        ),
    }
    evidence: dict[str, Any] = {"reason_codes": []}
    stage18i_root = Path(config["stage18i_candidate_generation_root"])
    evidence["stage18i"] = _read_first_json(
        (
            stage18i_root / STAGE18I_SUMMARY_FILE,
            stage18i_root / STAGE18I2_SUMMARY_FILE,
        ),
        evidence["reason_codes"],
        "missing_stage18i_candidate_generation",
    )
    for key, (path, reason) in specs.items():
        evidence[key] = _read_json(path, evidence["reason_codes"], reason)
    evidence["reason_codes"] = unique_sorted(evidence["reason_codes"])
    return evidence


def _metrics(evidence: dict[str, Any]) -> dict[str, Any]:
    stage18i = evidence["stage18i"]
    comparison = evidence["comparison"]
    binding = evidence["binding"]
    quantization = evidence["quantization"]
    oracle = evidence["oracle"]
    coverage = evidence["coverage"]
    reason_codes = list(evidence["reason_codes"])

    stage18i_passed = stage18i.get("status") == "passed"
    true_inference = comparison.get("true_model_inference_executed") is True and comparison.get("proxy_selection_used") is not True
    true_bound = (
        binding.get("true_incumbent_selection_bound") is True
        and int(binding.get("fallback_action_index_0_count", 0) or 0) == 0
        and int(binding.get("candidate_cell_mismatch_count", 0) or 0) == 0
    )
    safe_count = int(quantization.get("safe_efficient_candidate_count", stage18i.get("safe_efficient_candidate_count", 0)) or 0)
    safe_roi_count = int(quantization.get("roi_group_with_safe_efficient_candidate_count", stage18i.get("roi_group_with_safe_efficient_candidate_count", 0)) or 0)
    oracle_separable = oracle.get("oracle_separable") is True
    coverage_advantage = coverage.get("xunce_coverage_advantage_established") is True
    boundary_violation = _boundary_violation(stage18i, comparison, binding, quantization, oracle, coverage)

    if stage18i and not stage18i_passed:
        reason_codes.append("stage18i_candidate_generation_not_passed")
    if comparison and not true_inference:
        reason_codes.append("after_stage18i_true_inference_not_executed")
    if binding and not true_bound:
        reason_codes.append("after_stage18i_true_incumbent_binding_not_bound")
    if boundary_violation:
        reason_codes.append("boundary_violation")

    return {
        "reason_codes": unique_sorted(reason_codes),
        "stage18i_candidate_generation_passed": stage18i_passed,
        "true_model_inference_executed": true_inference,
        "true_incumbent_selection_bound": true_bound,
        "safe_efficient_candidate_count": safe_count,
        "roi_group_with_safe_efficient_candidate_count": safe_roi_count,
        "oracle_separable": oracle_separable,
        "cost_aware_oracle_efficiency_regression_count": int(oracle.get("cost_aware_oracle_efficiency_regression_count", 0) or 0),
        "xunce_coverage_advantage_established": coverage_advantage,
        "canary_traffic_fraction": 0.0,
        "stage18i_summary_status": stage18i.get("status"),
        "comparison_summary_status": comparison.get("status"),
        "binding_summary_status": binding.get("status"),
        "quantization_summary_status": quantization.get("status"),
        "oracle_summary_status": oracle.get("status"),
        "coverage_summary_status": coverage.get("status"),
    }


def _decision(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons = list(metrics["reason_codes"])
    diagnostic_reasons: list[str] = []
    if metrics["safe_efficient_candidate_count"] <= 0:
        diagnostic_reasons.append("safe_efficient_candidate_missing")
    if metrics["roi_group_with_safe_efficient_candidate_count"] <= 0:
        diagnostic_reasons.append("safe_efficient_roi_spread_insufficient")
    if not metrics["oracle_separable"]:
        diagnostic_reasons.append("oracle_not_separable")
    if not metrics["xunce_coverage_advantage_established"]:
        diagnostic_reasons.append("xunce_coverage_advantage_not_established")
    if reasons:
        return _decision_payload(
            status="failed",
            blocking_reasons=reasons,
            diagnostic_reasons=diagnostic_reasons,
            next_required_change=_route_for_reasons(reasons),
            evidence_gate=False,
            candidate_gate=False,
        )
    return _decision_payload(
        status="passed",
        blocking_reasons=[],
        diagnostic_reasons=diagnostic_reasons,
        next_required_change=REVIEW_COMPARISON_NEXT_REQUIRED_CHANGE,
        evidence_gate=True,
        candidate_gate=True,
    )


def _decision_payload(
    *,
    status: str,
    blocking_reasons: list[str],
    diagnostic_reasons: list[str],
    next_required_change: str,
    evidence_gate: bool,
    candidate_gate: bool,
) -> dict[str, Any]:
    blocking_reason_codes = unique_sorted(blocking_reasons)
    diagnostic_reason_codes = unique_sorted(diagnostic_reasons)
    return {
        "status": status,
        "reason_codes": blocking_reason_codes,
        "blocking_reason_codes": blocking_reason_codes,
        "diagnostic_reason_codes": diagnostic_reason_codes,
        "diagnostic_recommended_change": _diagnostic_recommended_change(diagnostic_reason_codes),
        "evidence_authenticity_gate_passed": evidence_gate,
        "candidate_validity_gate_passed": candidate_gate,
        "comparison_allowed": status == "passed" and evidence_gate and candidate_gate,
        "next_required_change": next_required_change,
    }


def _diagnostic_recommended_change(diagnostic_reasons: list[str]) -> str:
    reason_set = set(diagnostic_reasons)
    if "oracle_not_separable" in reason_set:
        return REPAIR_CANDIDATE_NEXT_REQUIRED_CHANGE
    if "xunce_coverage_advantage_not_established" in reason_set:
        return STAGE18J_NEXT_REQUIRED_CHANGE
    if {"safe_efficient_candidate_missing", "safe_efficient_roi_spread_insufficient"} & reason_set:
        return REPAIR_CANDIDATE_NEXT_REQUIRED_CHANGE
    return ""


def _route_for_reasons(reasons: list[str]) -> str:
    priorities = (
        ("missing_stage18i_candidate_generation", RUN_STAGE18I_NEXT_REQUIRED_CHANGE),
        ("stage18i_candidate_generation_not_passed", RUN_STAGE18I_NEXT_REQUIRED_CHANGE),
        ("missing_after_stage18i_model_inference", RUN_STAGE18B_NEXT_REQUIRED_CHANGE),
        ("after_stage18i_true_inference_not_executed", RUN_STAGE18B_NEXT_REQUIRED_CHANGE),
        ("missing_after_stage18i_true_incumbent_binding", RUN_STAGE18G0_NEXT_REQUIRED_CHANGE),
        ("after_stage18i_true_incumbent_binding_not_bound", RUN_STAGE18G0_NEXT_REQUIRED_CHANGE),
        ("missing_after_stage18i_quantization", RUN_STAGE18H0_NEXT_REQUIRED_CHANGE),
        ("missing_after_stage18i_oracle_separability", RUN_STAGE18F_NEXT_REQUIRED_CHANGE),
        ("missing_after_stage18i_coverage_comparison", RUN_STAGE18C_NEXT_REQUIRED_CHANGE),
        ("boundary_violation", BOUNDARY_NEXT_REQUIRED_CHANGE),
    )
    reason_set = set(reasons)
    for reason, route in priorities:
        if reason in reason_set:
            return route
    return REPAIR_CANDIDATE_NEXT_REQUIRED_CHANGE


def _read_json(path: Path, reasons: list[str], reason_code: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(reason_code)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(reason_code)
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_first_json(paths: tuple[Path, ...], reasons: list[str], reason_code: str) -> dict[str, Any]:
    for path in paths:
        if path.is_file():
            return _read_json(path, reasons, reason_code)
    reasons.append(reason_code)
    return {}


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


def _boundary_fields() -> dict[str, Any]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields["canary_traffic_fraction"] = 0.0
    return fields


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Stage 18I Evidence Closure Audit",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- stage18i_candidate_generation_passed: `{summary['stage18i_candidate_generation_passed']}`",
            f"- true_model_inference_executed: `{summary['true_model_inference_executed']}`",
            f"- true_incumbent_selection_bound: `{summary['true_incumbent_selection_bound']}`",
            f"- safe_efficient_candidate_count: `{summary['safe_efficient_candidate_count']}`",
            f"- oracle_separable: `{summary['oracle_separable']}`",
            f"- xunce_coverage_advantage_established: `{summary['xunce_coverage_advantage_established']}`",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
