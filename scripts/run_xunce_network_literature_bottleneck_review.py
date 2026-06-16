from __future__ import annotations

import argparse
import json
import re
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


CONFIG_SCHEMA_VERSION = "xunce-network-literature-bottleneck-review-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-network-literature-bottleneck-review-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-network-literature-bottleneck-review-manifest/v1"
LITERATURE_AUDIT_SCHEMA_VERSION = "xunce-network-literature-map-audit/v1"
BOTTLENECK_AUDIT_SCHEMA_VERSION = "xunce-project-bottleneck-audit/v1"
SCOPE_AUDIT_SCHEMA_VERSION = "xunce-network-research-scope-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-network-literature-bottleneck-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_network_literature_bottleneck_review_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_network_literature_bottleneck_review_v1"

SUMMARY_FILE = "xunce-network-literature-bottleneck-review-summary.json"
MANIFEST_FILE = "xunce-network-literature-bottleneck-review-manifest.json"
LITERATURE_AUDIT_FILE = "xunce-network-literature-map-audit.json"
BOTTLENECK_AUDIT_FILE = "xunce-project-bottleneck-audit.json"
SCOPE_AUDIT_FILE = "xunce-network-research-scope-audit.json"
REJECTION_REPORT_FILE = "xunce-network-literature-bottleneck-rejection-report.json"
REPORT_FILE = "xunce-network-literature-bottleneck-review-report.md"

PASS_NEXT_REQUIRED_CHANGE = "topology_observation_contract"
FIX_STAGE1_NEXT_REQUIRED_CHANGE = "fix_xunce_current_head_evidence_refresh"
FIX_REVIEW_NEXT_REQUIRED_CHANGE = "fix_xunce_network_literature_bottleneck_review"

SOURCE_FILES = {
    "current_head_evidence_refresh": (
        "source_current_head_evidence_refresh_root",
        "xunce-current-head-evidence-refresh-summary.json",
    ),
    "network_readiness": (
        "source_network_readiness_root",
        "network-architecture-upgrade-readiness-summary.json",
    ),
    "multi_map": (
        "source_multi_map_root",
        "global-99-multi-map-generalization-summary.json",
    ),
    "real_map_multi_roi": (
        "source_real_map_multi_roi_root",
        "global-99-real-map-multi-roi-generalization-summary.json",
    ),
}

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "starts_online_canary",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Network Literature Bottleneck Review v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_network_literature_bottleneck_review(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "primary_bottleneck_decision": summary["primary_bottleneck_decision"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_network_literature_bottleneck_review(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(Path(config_path), repo_root)
    paths = _artifact_paths(output_root)
    source_paths = _source_paths(config, repo_root)
    sources = _load_sources(source_paths)
    literature_audit = _literature_audit(config)
    bottleneck_audit = _bottleneck_audit(config, sources)
    scope_audit = _scope_audit(config, sources)
    decision = _decision(sources, literature_audit, bottleneck_audit, scope_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=Path(config_path),
        output_root=output_root,
        paths=paths,
        source_paths=source_paths,
        literature_audit=literature_audit,
        bottleneck_audit=bottleneck_audit,
        scope_audit=scope_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, Path(config_path), output_root, paths, summary)
    rejection_report = _rejection_report(generated_at, decision, literature_audit, bottleneck_audit, scope_audit)

    write_json(paths["literature_audit"], literature_audit)
    write_json(paths["bottleneck_audit"], bottleneck_audit)
    write_json(paths["scope_audit"], scope_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, literature_audit, bottleneck_audit, rejection_report), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
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
    normalized = dict(payload)
    for root_key, _filename in SOURCE_FILES.values():
        if not isinstance(payload.get(root_key), str) or not payload[root_key].strip():
            raise ConfigError(f"{root_key} must be a non-empty string")
        normalized[root_key] = str(resolve_path(Path(payload[root_key]), repo_root))
    for key in ("policy_architectures_path", "policy_features_path"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    for key in ("require_current_head_refresh_passed", "require_closed_boundaries"):
        if not isinstance(payload.get(key), bool):
            raise ConfigError(f"{key} must be a boolean")
    normalized["min_literature_reference_count"] = _positive_int(payload.get("min_literature_reference_count"), "min_literature_reference_count")
    normalized["max_policy_guard_fallback_rate"] = _unit_float(payload.get("max_policy_guard_fallback_rate"), "max_policy_guard_fallback_rate")
    refs = payload.get("literature_references")
    if not isinstance(refs, list):
        raise ConfigError("literature_references must be a list")
    normalized["literature_references"] = [dict(item) for item in refs if isinstance(item, dict)]
    return normalized


def _source_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    return {
        label: resolve_path(Path(config[root_key]), repo_root) / filename
        for label, (root_key, filename) in SOURCE_FILES.items()
    }


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "literature_audit": output_root / LITERATURE_AUDIT_FILE,
        "bottleneck_audit": output_root / BOTTLENECK_AUDIT_FILE,
        "scope_audit": output_root / SCOPE_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_sources(source_paths: dict[str, Path]) -> dict[str, dict[str, Any] | None]:
    return {label: _load_json(path) for label, path in source_paths.items()}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _literature_audit(config: dict[str, Any]) -> dict[str, Any]:
    refs = config["literature_references"]
    rows = []
    invalid_ids = []
    primary_url_count = 0
    family_counts: dict[str, int] = {}
    for ref in refs:
        ref_id = str(ref.get("id", "")).strip()
        title = str(ref.get("title", "")).strip()
        url = str(ref.get("url", "")).strip()
        family = str(ref.get("family", "")).strip()
        relevance = str(ref.get("project_relevance", "")).strip()
        valid = bool(ref_id and title and family and relevance and _is_primary_url(url))
        if not valid:
            invalid_ids.append(ref_id or "<missing-id>")
        if _is_primary_url(url):
            primary_url_count += 1
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
        rows.append(
            {
                "id": ref_id,
                "title": title,
                "url": url,
                "family": family,
                "project_relevance": relevance,
                "valid_primary_source": valid,
            }
        )
    reasons: list[str] = []
    if len(rows) < config["min_literature_reference_count"]:
        reasons.append("insufficient_literature_references")
    if invalid_ids:
        reasons.append("invalid_literature_references")
    return {
        "schema_version": LITERATURE_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not reasons else "failed",
        "reason_codes": unique_sorted(reasons),
        "reference_count": len(rows),
        "primary_source_reference_count": primary_url_count,
        "min_literature_reference_count": config["min_literature_reference_count"],
        "family_counts": family_counts,
        "invalid_reference_ids": invalid_ids,
        "references": rows,
    }


def _bottleneck_audit(config: dict[str, Any], sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    network = sources.get("network_readiness") or {}
    multi_map = sources.get("multi_map") or {}
    real_map = sources.get("real_map_multi_roi") or {}
    architecture_inventory = _architecture_inventory(Path(config["policy_architectures_path"]), Path(config["policy_features_path"]))

    failed_required = _int_value(multi_map.get("failed_required_scenario_count"), 0) + _int_value(
        real_map.get("failed_required_scenario_count"),
        0,
    )
    aggregate = _float_value(network.get("aggregate_achieved_coverage_rate"), _float_value(multi_map.get("aggregate_achieved_coverage_rate"), 0.0))
    min_coverage = _float_value(network.get("min_scenario_achieved_coverage_rate"), _float_value(multi_map.get("min_scenario_achieved_coverage_rate"), 0.0))
    policy_worse = _int_value(network.get("policy_worse_than_baseline_count"), _int_value(multi_map.get("policy_worse_than_baseline_count"), 0))
    controlled_regression = _int_value(network.get("controlled_regression_count"), _int_value(multi_map.get("controlled_regression_count"), 0))
    fallback_rate = _fallback_rate(network, multi_map)
    coverage_or_generalization = bool(failed_required > 0 or aggregate < 0.99 or min_coverage < 0.99)
    fallback_or_policy = bool(
        policy_worse > 0
        or controlled_regression > 0
        or fallback_rate > config["max_policy_guard_fallback_rate"]
    )
    topology_gap = not architecture_inventory["has_topology_graph_or_memory_architecture"]
    latency_params_gap = not architecture_inventory["has_param_latency_reporting"]
    no_release_blocker = bool(
        not coverage_or_generalization
        and not fallback_or_policy
        and network.get("network_upgrade_recommended") is False
    )
    decision = _primary_bottleneck_decision(
        coverage_or_generalization=coverage_or_generalization,
        fallback_or_policy=fallback_or_policy,
        latency_params_gap=latency_params_gap,
        topology_gap=topology_gap,
        no_release_blocker=no_release_blocker,
    )
    return {
        "schema_version": BOTTLENECK_AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "reason_codes": [],
        "coverage_or_generalization_bottleneck": coverage_or_generalization,
        "fallback_or_policy_regression_bottleneck": fallback_or_policy,
        "latency_or_parameter_evidence_gap": latency_params_gap,
        "topology_representation_gap": topology_gap,
        "no_current_release_blocking_network_bottleneck": no_release_blocker,
        "primary_bottleneck_decision": decision,
        "aggregate_achieved_coverage_rate": aggregate,
        "min_scenario_achieved_coverage_rate": min_coverage,
        "failed_required_scenario_count": failed_required,
        "policy_worse_than_baseline_count": policy_worse,
        "controlled_regression_count": controlled_regression,
        "policy_guard_fallback_rate": fallback_rate,
        "network_upgrade_recommended": network.get("network_upgrade_recommended"),
        "architecture_inventory": architecture_inventory,
        "next_research_gate_rationale": "Proceed to additive topology observation contract for contrast evidence; do not train or publish.",
    }


def _scope_audit(config: dict[str, Any], sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    violations = []
    if config["require_closed_boundaries"]:
        for label, payload in sources.items():
            if not isinstance(payload, dict):
                continue
            for field in BOUNDARY_FIELDS:
                if payload.get(field) is True:
                    violations.append({"source": label, "field": field, "value": True})
    return {
        "schema_version": SCOPE_AUDIT_SCHEMA_VERSION,
        "status": "passed" if not violations else "failed",
        "reason_codes": [] if not violations else ["research_scope_boundary_violation"],
        "violations": violations,
        **global_99_boundary_defaults(),
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
    }


def _decision(
    sources: dict[str, dict[str, Any] | None],
    literature_audit: dict[str, Any],
    bottleneck_audit: dict[str, Any],
    scope_audit: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    for label, payload in sources.items():
        if payload is None:
            reasons.append(f"missing_{label}_summary")
        elif payload.get("status") != "passed":
            if label == "current_head_evidence_refresh":
                reasons.append("current_head_evidence_refresh_not_passed")
            else:
                reasons.append(f"{label}_not_passed")
    stage1 = sources.get("current_head_evidence_refresh") or {}
    if stage1 and stage1.get("next_required_change") != "network_literature_bottleneck_review":
        reasons.append("current_head_evidence_refresh_next_required_change_invalid")
    reasons.extend(literature_audit.get("reason_codes", []))
    reasons.extend(bottleneck_audit.get("reason_codes", []))
    reasons.extend(scope_audit.get("reason_codes", []))
    reasons = unique_sorted(reasons)
    status = "passed" if not reasons else "failed"
    if status == "passed":
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif "current_head_evidence_refresh_not_passed" in reasons or "current_head_evidence_refresh_next_required_change_invalid" in reasons or "missing_current_head_evidence_refresh_summary" in reasons:
        next_required_change = FIX_STAGE1_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FIX_REVIEW_NEXT_REQUIRED_CHANGE
    return {"status": status, "reason_codes": reasons, "next_required_change": next_required_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_paths: dict[str, Path],
    literature_audit: dict[str, Any],
    bottleneck_audit: dict[str, Any],
    scope_audit: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary": str(paths["summary"]),
        "config_path": str(config_path),
        "output_root": str(output_root),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "literature_bottleneck_review_passed": decision["status"] == "passed",
        "literature_map_audit_passed": literature_audit["status"] == "passed",
        "project_bottleneck_audit_passed": bottleneck_audit["status"] == "passed",
        "research_scope_audit_passed": scope_audit["status"] == "passed",
        "literature_reference_count": literature_audit["reference_count"],
        "primary_source_reference_count": literature_audit["primary_source_reference_count"],
        "coverage_or_generalization_bottleneck": bottleneck_audit["coverage_or_generalization_bottleneck"],
        "fallback_or_policy_regression_bottleneck": bottleneck_audit["fallback_or_policy_regression_bottleneck"],
        "latency_or_parameter_evidence_gap": bottleneck_audit["latency_or_parameter_evidence_gap"],
        "topology_representation_gap": bottleneck_audit["topology_representation_gap"],
        "no_current_release_blocking_network_bottleneck": bottleneck_audit["no_current_release_blocking_network_bottleneck"],
        "primary_bottleneck_decision": bottleneck_audit["primary_bottleneck_decision"],
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "next_required_change": decision["next_required_change"],
        **global_99_boundary_defaults(),
        "real_world_release_approved": False,
        "real_world_performance_claimed": False,
        "default_policy_replacement_approved": False,
        "real_executor_connection_approved": False,
        "git_provenance": {"current": git_snapshot(repo_root)},
    }


def _manifest(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "status": summary["status"],
        "next_required_change": summary["next_required_change"],
        "artifacts": {key: str(value) for key, value in paths.items()},
    }


def _rejection_report(
    generated_at: str,
    decision: dict[str, Any],
    literature_audit: dict[str, Any],
    bottleneck_audit: dict[str, Any],
    scope_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if decision["status"] == "passed" else "failed",
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "literature_rejections": literature_audit.get("reason_codes", []),
        "bottleneck_rejections": bottleneck_audit.get("reason_codes", []),
        "scope_rejections": scope_audit.get("violations", []),
    }


def _render_report(
    summary: dict[str, Any],
    literature_audit: dict[str, Any],
    bottleneck_audit: dict[str, Any],
    rejection_report: dict[str, Any],
) -> str:
    refs = "\n".join(
        f"- `{row['id']}`: {row['title']} ({row['url']})"
        for row in literature_audit.get("references", [])
    )
    return "\n".join(
        [
            "# Xunce Network Literature Bottleneck Review v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- primary_bottleneck_decision: `{summary['primary_bottleneck_decision']}`",
            f"- topology_representation_gap: `{summary['topology_representation_gap']}`",
            f"- no_current_release_blocking_network_bottleneck: `{summary['no_current_release_blocking_network_bottleneck']}`",
            "",
            "## Literature",
            "",
            refs,
            "",
            "## Bottleneck Audit",
            "",
            json.dumps(bottleneck_audit, ensure_ascii=False, indent=2),
            "",
            "## Rejections",
            "",
            json.dumps(rejection_report, ensure_ascii=False, indent=2),
            "",
        ]
    )


def _architecture_inventory(architectures_path: Path, features_path: Path) -> dict[str, Any]:
    architectures_text = architectures_path.read_text(encoding="utf-8") if architectures_path.is_file() else ""
    features_text = features_path.read_text(encoding="utf-8") if features_path.is_file() else ""
    supported = sorted(set(re.findall(r'"([a-z0-9_]+_v1)"', architectures_text)))
    text = f"{architectures_text}\n{features_text}".lower()
    return {
        "policy_architectures_path": str(architectures_path),
        "policy_features_path": str(features_path),
        "supported_architectures": supported,
        "has_candidate_attention": "candidate_attention_v1" in supported,
        "has_topology_graph_or_memory_architecture": any(
            marker in text
            for marker in ("topology_aware", "candidate_graph", "coverage_memory_token", "graph_encoder")
        ),
        "has_param_latency_reporting": any(marker in text for marker in ("latency", "parameter_count", "param_count")),
        "observation_schema_version": "policy-observation/v1.1" if "policy-observation/v1.1" in features_text else None,
    }


def _primary_bottleneck_decision(
    *,
    coverage_or_generalization: bool,
    fallback_or_policy: bool,
    latency_params_gap: bool,
    topology_gap: bool,
    no_release_blocker: bool,
) -> str:
    if coverage_or_generalization:
        return "coverage_or_generalization_bottleneck"
    if fallback_or_policy:
        return "fallback_or_policy_regression_bottleneck"
    if no_release_blocker:
        return "no_current_release_blocking_network_bottleneck"
    if latency_params_gap:
        return "latency_or_parameter_evidence_gap"
    if topology_gap:
        return "topology_representation_gap"
    return "network_bottleneck_inconclusive"


def _fallback_rate(network: dict[str, Any], multi_map: dict[str, Any]) -> float:
    if network.get("policy_guard_fallback_rate") is not None:
        return _float_value(network.get("policy_guard_fallback_rate"), 0.0)
    fallback_count = _int_value(multi_map.get("policy_guard_fallback_count"), 0)
    decision_count = _int_value(multi_map.get("policy_guided_decision_count"), 0)
    return (fallback_count / decision_count) if decision_count else 0.0


def _is_primary_url(url: str) -> bool:
    return url.startswith("https://arxiv.org/abs/") or url.startswith("https://proceedings.neurips.cc/")


def _positive_int(value: Any, field_name: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a positive integer") from exc
    if numeric <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return numeric


def _unit_float(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a float") from exc
    if numeric < 0.0 or numeric > 1.0:
        raise ConfigError(f"{field_name} must be in [0, 1]")
    return numeric


def _int_value(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float_value(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
