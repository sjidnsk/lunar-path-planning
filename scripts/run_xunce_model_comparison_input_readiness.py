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
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json


SUMMARY_SCHEMA_VERSION = "xunce-model-comparison-input-readiness-audit/v1"
DEFAULT_OUTPUT_ROOT = "outputs/xunce_model_comparison_input_readiness_v1"
AUDIT_FILE = "input-readiness-audit.json"

ROI_EXPANSION_CONFIG = "configs/xunce_high_fidelity_real_map_roi_expansion_v1.json"
COMPARISON_CONFIG = "configs/xunce_high_fidelity_real_map_comparison_v1.json"

READY_FOR_STAGE_18A = "ready_for_stage_18a"
READY_FOR_STAGE_18B = "ready_for_stage_18b"
ROUTE_XUNCE = "restore_or_generate_xunce_training_candidate"
ROUTE_INCUMBENT = "restore_or_generate_incumbent_value_stability_checkpoint"
ROUTE_DOMAIN_GAP = "restore_or_generate_quasi_real_domain_gap"
ROUTE_CONFIG = "fix_xunce_model_comparison_input_configuration"

XUNCE_STAGE_ARTIFACTS = (
    ("xunce-design-freeze-audit", "outputs/path_feedback_batch_xunce_design_freeze_v1", "xunce-design-freeze-summary.json", "stage_0"),
    ("xunce-current-head-evidence-refresh", "outputs/path_feedback_batch_xunce_current_head_evidence_refresh_v1", "xunce-current-head-evidence-refresh-summary.json", "stage_1"),
    ("xunce-network-literature-bottleneck-review", "outputs/path_feedback_batch_xunce_network_literature_bottleneck_review_v1", "xunce-network-literature-bottleneck-review-summary.json", "stage_2"),
    ("xunce-topology-observation-contract", "outputs/path_feedback_batch_xunce_topology_observation_contract_v1", "xunce-topology-observation-contract-summary.json", "stage_3"),
    ("xunce-topology-feature-extraction-audit", "outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1", "xunce-topology-feature-extraction-audit-summary.json", "stage_4"),
    ("xunce-topology-graph-proto", "outputs/path_feedback_batch_xunce_topology_graph_proto_v1", "xunce-topology-graph-proto-summary.json", "stage_5"),
    ("xunce-proto-mechanism-validation", "outputs/path_feedback_batch_xunce_proto_mechanism_validation_v1", "xunce-proto-mechanism-validation-summary.json", "stage_6"),
    ("xunce-architecture-contrast-evaluation", "outputs/path_feedback_batch_xunce_architecture_contrast_evaluation_v1", "xunce-architecture-contrast-evaluation-summary.json", "stage_7"),
    ("xunce-full-network-v1", "outputs/path_feedback_batch_xunce_full_network_v1", "xunce-full-network-v1-summary.json", "stage_8"),
    ("xunce-full-network-static-contract-validation", "outputs/path_feedback_batch_xunce_full_network_static_contract_validation_v1", "xunce-full-network-static-contract-validation-summary.json", "stage_9"),
    ("xunce-full-network-ablation-experiments", "outputs/path_feedback_batch_xunce_full_network_ablation_experiments_v1", "xunce-full-network-ablation-experiments-summary.json", "stage_10"),
    ("xunce-full-network-stress-evaluation", "outputs/path_feedback_batch_xunce_full_network_stress_evaluation_v1", "xunce-full-network-stress-evaluation-summary.json", "stage_11"),
    ("xunce-guarded-training-candidate-preflight", "outputs/path_feedback_batch_xunce_guarded_training_candidate_preflight_v1", "xunce-guarded-training-candidate-preflight-summary.json", "stage_12"),
    ("xunce-controlled-training-candidate", "outputs/path_feedback_batch_xunce_controlled_training_candidate_v1", "xunce-controlled-training-candidate-summary.json", "stage_13"),
    ("xunce-post-training-offline-evaluation", "outputs/path_feedback_batch_xunce_post_training_offline_evaluation_v1", "xunce-post-training-offline-evaluation-summary.json", "stage_14"),
    ("xunce-shadow-replay-validation", "outputs/path_feedback_batch_xunce_shadow_replay_validation_v1", "xunce-shadow-replay-validation-summary.json", "stage_15"),
    ("xunce-sandbox-candidate-preflight", "outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1", "xunce-sandbox-candidate-preflight-summary.json", "stage_16"),
    ("xunce-release-governance-gate", "outputs/path_feedback_batch_xunce_release_governance_gate_v1", "xunce-release-governance-gate-summary.json", "stage_17"),
)

DOMAIN_GAP_REQUIRED_FILES = (
    "quasi-real-map-domain-gap-summary.json",
    "quasi-real-map-slices.jsonl",
    "quasi-real-map-path-feedback-summary.json",
)

ROI_EXPANSION_REQUIRED_FILES = (
    "xunce-high-fidelity-real-map-roi-expansion-summary.json",
    "xunce-high-fidelity-real-map-slices.jsonl",
    "xunce-high-fidelity-path-feedback-audit.json",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit inputs required for Xunce true model comparison.")
    parser.add_argument("--repo-root")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--roi-expansion-config", default=ROI_EXPANSION_CONFIG)
    parser.add_argument("--comparison-config", default=COMPARISON_CONFIG)
    parser.add_argument("--source-release-root")
    parser.add_argument("--source-domain-gap-root")
    parser.add_argument("--source-roi-expansion-root")
    parser.add_argument("--xunce-candidate-checkpoint")
    parser.add_argument("--incumbent-policy-checkpoint")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    output_root = resolve_path(Path(args.output_root), repo_root)
    try:
        audit = run_xunce_model_comparison_input_readiness(
            repo_root=repo_root,
            output_root=output_root,
            roi_expansion_config=resolve_path(Path(args.roi_expansion_config), repo_root),
            comparison_config=resolve_path(Path(args.comparison_config), repo_root),
            source_release_root=args.source_release_root,
            source_domain_gap_root=args.source_domain_gap_root,
            source_roi_expansion_root=args.source_roi_expansion_root,
            xunce_candidate_checkpoint=args.xunce_candidate_checkpoint,
            incumbent_policy_checkpoint=args.incumbent_policy_checkpoint,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": audit["status"],
                "next_required_change": audit["next_required_change"],
                "ready_for_stage_18a": audit["ready_for_stage_18a"],
                "ready_for_stage_18b": audit["ready_for_stage_18b"],
                "missing_required_input_count": len(audit["missing_required_inputs"]),
                "audit": audit["audit"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_xunce_model_comparison_input_readiness(
    *,
    repo_root: Path,
    output_root: Path,
    roi_expansion_config: Path,
    comparison_config: Path,
    source_release_root: str | None = None,
    source_domain_gap_root: str | None = None,
    source_roi_expansion_root: str | None = None,
    xunce_candidate_checkpoint: str | None = None,
    incumbent_policy_checkpoint: str | None = None,
) -> dict[str, Any]:
    roi_config = _load_optional_config(roi_expansion_config)
    comparison = _load_optional_config(comparison_config)

    release_root = _resolve_config_path(
        source_release_root,
        roi_config,
        "source_xunce_release_governance_root",
        repo_root,
    )
    domain_gap_root = _resolve_config_path(
        source_domain_gap_root,
        roi_config,
        "source_quasi_real_domain_gap_root",
        repo_root,
    )
    roi_root = _resolve_config_path(
        source_roi_expansion_root,
        comparison,
        "source_roi_expansion_root",
        repo_root,
    )
    xunce_checkpoint = _resolve_config_path(
        xunce_candidate_checkpoint,
        comparison,
        "xunce_candidate_checkpoint",
        repo_root,
    )
    incumbent_checkpoint = _resolve_config_path(
        incumbent_policy_checkpoint,
        comparison,
        "incumbent_policy_checkpoint",
        repo_root,
    )
    matrix_manifest = _resolve_config_path(
        None,
        roi_config,
        "source_matrix_manifest",
        repo_root,
        fallback=Path("model-explorer/data/manifests/lunar_south_pole_lro_lola_selection_matrix_v1.json"),
    )

    config_checks = [
        _file_check("roi_expansion_config", roi_expansion_config, "configuration"),
        _file_check("comparison_config", comparison_config, "configuration"),
        _file_check("lola_selection_matrix_manifest", matrix_manifest, "stage_18a"),
    ]
    xunce_stage_checks = [
        _file_check(stage_id, resolve_path(Path(root), repo_root) / filename, required_for)
        for stage_id, root, filename, required_for in XUNCE_STAGE_ARTIFACTS
    ]
    release_checks = [_file_check("xunce_release_governance_summary", release_root / "xunce-release-governance-gate-summary.json", "stage_18a")]
    domain_gap_checks = [
        _file_check(f"quasi_real_domain_gap_{Path(filename).stem}", domain_gap_root / filename, "stage_18a")
        for filename in DOMAIN_GAP_REQUIRED_FILES
    ]
    roi_checks = [
        _file_check(f"roi_expansion_{Path(filename).stem}", roi_root / filename, "stage_18b")
        for filename in ROI_EXPANSION_REQUIRED_FILES
    ]
    checkpoint_checks = [
        _file_check("xunce_candidate_checkpoint", xunce_checkpoint, "stage_18b"),
        _file_check("incumbent_policy_checkpoint", incumbent_checkpoint, "stage_18b"),
    ]

    missing_configs = _missing(config_checks)
    missing_xunce_checkpoint = [item for item in checkpoint_checks if item["id"] == "xunce_candidate_checkpoint" and not item["exists"]]
    missing_incumbent = [item for item in checkpoint_checks if item["id"] == "incumbent_policy_checkpoint" and not item["exists"]]
    missing_release = _missing(release_checks)
    missing_domain_gap = _missing(domain_gap_checks)
    missing_roi = _missing(roi_checks)

    ready_for_stage_18a = not missing_configs and not missing_release and not missing_domain_gap
    ready_for_stage_18b = ready_for_stage_18a and not missing_roi and not missing_xunce_checkpoint and not missing_incumbent
    if missing_configs:
        next_required_change = ROUTE_CONFIG
    elif missing_xunce_checkpoint or missing_release:
        next_required_change = ROUTE_XUNCE
    elif missing_incumbent:
        next_required_change = ROUTE_INCUMBENT
    elif missing_domain_gap:
        next_required_change = ROUTE_DOMAIN_GAP
    elif missing_roi:
        next_required_change = READY_FOR_STAGE_18A
    else:
        next_required_change = READY_FOR_STAGE_18B

    missing_required_inputs = unique_sorted(
        item["id"]
        for item in (
            missing_configs
            + missing_xunce_checkpoint
            + missing_incumbent
            + missing_release
            + missing_domain_gap
            + missing_roi
        )
    )
    generated_at = utc_now()
    audit_path = output_root / AUDIT_FILE
    audit = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready" if ready_for_stage_18b else "blocked",
        "next_required_change": next_required_change,
        "ready_for_stage_18a": ready_for_stage_18a,
        "ready_for_stage_18b": ready_for_stage_18b,
        "missing_required_inputs": missing_required_inputs,
        "audit": str(audit_path),
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "input_source_policy": "read_only_existing_outputs_or_external_bundle",
        "large_artifact_policy": "do_not_copy_large_artifacts_into_tracked_config",
        "config_checks": config_checks,
        "xunce_upstream_stage_artifacts": xunce_stage_checks,
        "release_governance_inputs": release_checks,
        "quasi_real_domain_gap_inputs": domain_gap_checks,
        "roi_expansion_inputs": roi_checks,
        "checkpoint_inputs": checkpoint_checks,
        "regeneration_hint": _regeneration_hint(next_required_change),
        "git_provenance": git_snapshot(repo_root),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    write_json(audit_path, audit)
    return audit


def _load_optional_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"config root must be an object: {path}")
    return payload


def _resolve_config_path(
    override: str | None,
    config: dict[str, Any],
    key: str,
    repo_root: Path,
    *,
    fallback: Path | None = None,
) -> Path:
    value: Any = override if override is not None else config.get(key)
    if value is None:
        value = fallback
    if value is None or not str(value).strip():
        return repo_root / f"missing-{key}"
    return resolve_path(Path(str(value)), repo_root).resolve()


def _file_check(check_id: str, path: Path, required_for: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "path": str(path),
        "exists": path.is_file(),
        "required_for": required_for,
        "reason_code": None if path.is_file() else f"missing_{check_id}",
    }


def _missing(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in checks if not item["exists"]]


def _regeneration_hint(next_required_change: str) -> str:
    if next_required_change == ROUTE_XUNCE:
        return "restore evidence bundle or run Xunce stages through sandbox candidate preflight and release governance"
    if next_required_change == ROUTE_INCUMBENT:
        return "restore value-stability candidate bundle or generate model-explorer incumbent checkpoint"
    if next_required_change == ROUTE_DOMAIN_GAP:
        return "restore or regenerate quasi-real bridge/path-feedback/domain-gap evidence"
    if next_required_change == READY_FOR_STAGE_18A:
        return "run xunce-high-fidelity-real-map-roi-expansion"
    if next_required_change == READY_FOR_STAGE_18B:
        return "run xunce-high-fidelity-real-map-comparison"
    return "fix missing config or manifest input"


if __name__ == "__main__":
    raise SystemExit(main())
