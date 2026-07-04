from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import xunce_artifact_io as artifact_io
from xunce_artifact_paths import (
    STAGE21_4_CHECKPOINT,
    STAGE21_4_GRADIENT,
    STAGE21_4_LOSS,
    STAGE21_4_MANIFEST,
    STAGE21_4_ROUTING,
    STAGE21_4_SUMMARY,
    STAGE21_5_DELTA,
    STAGE21_5_MANIFEST,
    STAGE21_5_RESULTS,
    STAGE21_5_ROUTING,
    STAGE21_5_SCENARIO_DELTA,
    STAGE21_5_SUMMARY,
    STAGE26_2_BATCH_AUDIT,
    STAGE26_2_CHECKPOINT_BOUNDARY_AUDIT,
    STAGE26_2_LOSS_GRADIENT_AUDIT,
    STAGE26_2_MANIFEST,
    STAGE26_2_ROUTING,
    STAGE26_2_STAGE21_4_CONFIG,
    STAGE26_2_STAGE21_4_SUMMARY,
    STAGE26_2_SUMMARY,
    STAGE26_3_ACTION_AUDIT,
    STAGE26_3_DELTA_AUDIT,
    STAGE26_3_HIGH_FIDELITY_CONFIG,
    STAGE26_3_MANIFEST,
    STAGE26_3_ROUTING,
    STAGE26_3_STAGE21_5_CONFIG,
    STAGE26_3_STAGE21_5_SUMMARY,
    STAGE26_3_SUMMARY,
    artifact_path,
    read_json_artifact,
    read_jsonl_artifact,
    write_json_artifact,
    write_jsonl_artifact,
)


STAGE_ID = "xunce-stage26-io2-remaining-runner-long-path-migration"
CONFIG_SCHEMA_VERSION = "xunce-stage26-io2-remaining-runner-long-path-migration-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-io2-summary/v1"
PATH_AUDIT_SCHEMA_VERSION = "xunce-stage26-io2-path-audit/v1"
ALIAS_AUDIT_SCHEMA_VERSION = "xunce-stage26-io2-alias-audit/v1"
STATIC_AUDIT_SCHEMA_VERSION = "xunce-stage26-io2-runner-static-io-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-io2-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-io2-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_io2_remaining_runner_long_path_migration_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_io2"

SUMMARY_FILE = "summary.json"
PATH_AUDIT_FILE = "path_audit.json"
ALIAS_AUDIT_FILE = "alias_audit.json"
STATIC_AUDIT_FILE = "runner_static_io_audit.json"
ROUTING_FILE = "routing.json"
REPORT_FILE = "report.md"
MANIFEST_FILE = "manifest.json"

ROUTE_INPUTS = "rerun_stage26_io2_required_inputs"
ROUTE_ALIAS = "repair_stage26_io2_alias_contract"
ROUTE_STATIC_IO = "repair_stage26_io2_runner_artifact_io_migration"
ROUTE_SHORT_ROOT = "repair_stage26_io2_short_root_contract"
ROUTE_STAGE26_8N = "rerun_stage26_8n_aggressive_update_sweep_with_aligned_planning_proxy"
ROUTE_BOUNDARY = "resolve_stage26_io2_boundary_rejections"

BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

DEFAULT_SHORT_ROOT_STAGE_IDS = (
    "xunce-stage26-8m-generalized-resumable-training-pipeline",
    "xunce-stage26-8n-aggressive-sample-update-sweep",
    "xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget",
    "xunce-stage26-8p-hybrid-astar-primitive-resolution-sweep",
    "xunce-stage26-8q-derived-high-res-planning-proxy-alignment",
    "xunce-stage26-io1-artifact-path-contract-and-long-path-resilience",
    STAGE_ID,
)

DEFAULT_MIGRATED_RUNNERS = (
    "scripts/run_xunce_stage21_4_tiny_ppo_update_smoke.py",
    "scripts/run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py",
    "scripts/run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py",
    "scripts/run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py",
    "scripts/run_xunce_stage26_8d_resumable_seed_horizon_execution.py",
    "scripts/run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit.py",
    "scripts/run_xunce_stage26_8g_repair_synthetic_scenario_diversity.py",
    "scripts/run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval.py",
    "scripts/run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair.py",
    "scripts/run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget.py",
    "scripts/run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep.py",
)

FORBIDDEN_DIRECT_IO_MARKERS = (".read_text(", ".write_text(", ".is_file(", ".exists(", ".open(", ".mkdir(")
SAFE_HELPER_MARKERS = ("artifact_io.", "read_json_artifact(", "read_jsonl_artifact(", "write_json_artifact(", "write_jsonl_artifact(")


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.IO2 remaining runner long-path migration audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage26_io2_remaining_runner_long_path_migration(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            repo_root=Path(args.repo_root),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_io2_remaining_runner_long_path_migration(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)

    boundary_rejections = _boundary_rejections(config)
    alias_audit = _alias_audit(output_root)
    path_audit = _path_audit(config, output_root, repo_root)
    static_audit = _runner_static_io_audit(config, repo_root)

    route = _route(
        boundary_rejections=boundary_rejections,
        alias_audit=alias_audit,
        path_audit=path_audit,
        static_audit=static_audit,
    )
    status = "passed" if route == ROUTE_STAGE26_8N else "failed"

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "alias_contract_passed": bool(alias_audit.get("alias_contract_passed")),
        "runner_static_io_audit_passed": bool(static_audit.get("runner_static_io_audit_passed")),
        "short_output_root_contract_passed": bool(path_audit.get("short_output_root_contract_passed")),
        "direct_artifact_io_violation_count": int(static_audit.get("direct_artifact_io_violation_count") or 0),
        "path_length_failure_count": int(path_audit.get("path_length_failure_count") or 0),
        "max_path_length": path_audit.get("max_path_length"),
        "boundary_rejections": boundary_rejections,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "primary_route": route,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary": str(output_root / SUMMARY_FILE),
        "path_audit": str(output_root / PATH_AUDIT_FILE),
        "alias_audit": str(output_root / ALIAS_AUDIT_FILE),
        "runner_static_io_audit": str(output_root / STATIC_AUDIT_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
        "manifest": str(output_root / MANIFEST_FILE),
    }

    artifact_io.write_json(output_root / PATH_AUDIT_FILE, path_audit)
    artifact_io.write_json(output_root / ALIAS_AUDIT_FILE, alias_audit)
    artifact_io.write_json(output_root / STATIC_AUDIT_FILE, static_audit)
    artifact_io.write_json(output_root / ROUTING_FILE, routing)
    artifact_io.write_json(output_root / SUMMARY_FILE, summary)
    artifact_io.write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not artifact_io.path_is_file(path):
        raise ConfigError(f"config missing: {path}")
    payload = artifact_io.read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ConfigError(f"stage_id must be {STAGE_ID}")
    config = dict(payload)
    config.setdefault("stage_registry_path", "configs/stage_registry.json")
    config.setdefault("short_root_stage_ids", list(DEFAULT_SHORT_ROOT_STAGE_IDS))
    config.setdefault("migrated_runner_files", list(DEFAULT_MIGRATED_RUNNERS))
    config.setdefault("warn_path_length", 180)
    config.setdefault("fail_path_length", 240)
    config["stage_registry_path"] = str(_resolve_path(Path(str(config["stage_registry_path"])), repo_root))
    config["migrated_runner_files"] = [str(_resolve_path(Path(str(path)), repo_root)) for path in config["migrated_runner_files"]]
    config["short_root_stage_ids"] = [str(stage_id) for stage_id in config["short_root_stage_ids"]]
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    return config


def _alias_audit(output_root: Path) -> dict[str, Any]:
    root = output_root / "alias_probe"
    json_artifacts = (
        ("s21_4", STAGE21_4_SUMMARY),
        ("s21_4", STAGE21_4_GRADIENT),
        ("s21_4", STAGE21_4_CHECKPOINT),
        ("s21_4", STAGE21_4_ROUTING),
        ("s21_4", STAGE21_4_MANIFEST),
        ("s21_5", STAGE21_5_SUMMARY),
        ("s21_5", STAGE21_5_RESULTS),
        ("s21_5", STAGE21_5_DELTA),
        ("s21_5", STAGE21_5_ROUTING),
        ("s21_5", STAGE21_5_MANIFEST),
        ("s26_2", STAGE26_2_SUMMARY),
        ("s26_2", STAGE26_2_STAGE21_4_CONFIG),
        ("s26_2", STAGE26_2_STAGE21_4_SUMMARY),
        ("s26_2", STAGE26_2_BATCH_AUDIT),
        ("s26_2", STAGE26_2_LOSS_GRADIENT_AUDIT),
        ("s26_2", STAGE26_2_CHECKPOINT_BOUNDARY_AUDIT),
        ("s26_2", STAGE26_2_ROUTING),
        ("s26_2", STAGE26_2_MANIFEST),
        ("s26_3", STAGE26_3_SUMMARY),
        ("s26_3", STAGE26_3_STAGE21_5_CONFIG),
        ("s26_3", STAGE26_3_HIGH_FIDELITY_CONFIG),
        ("s26_3", STAGE26_3_STAGE21_5_SUMMARY),
        ("s26_3", STAGE26_3_ACTION_AUDIT),
        ("s26_3", STAGE26_3_DELTA_AUDIT),
        ("s26_3", STAGE26_3_ROUTING),
        ("s26_3", STAGE26_3_MANIFEST),
    )
    jsonl_artifacts = (
        ("s21_4", STAGE21_4_LOSS),
        ("s21_5", STAGE21_5_SCENARIO_DELTA),
    )

    failures: list[dict[str, Any]] = []
    for stage_dir, artifact in json_artifacts:
        stage_root = root / stage_dir
        payload = {"artifact_key": artifact.key, "status": "canonical"}
        write_json_artifact(stage_root, artifact, payload)
        read_payload, source = read_json_artifact(stage_root, artifact)
        if source != "canonical" or read_payload.get("artifact_key") != artifact.key:
            failures.append({"artifact": artifact.key, "reason": "canonical_read_failed"})

        legacy_root = root / f"{stage_dir}_legacy_{artifact.key}"
        artifact_io.write_json(artifact_path(legacy_root, artifact, canonical=False), {"artifact_key": artifact.key, "status": "legacy"})
        legacy_payload, legacy_source = read_json_artifact(legacy_root, artifact)
        if legacy_source != "legacy" or legacy_payload.get("artifact_key") != artifact.key:
            failures.append({"artifact": artifact.key, "reason": "legacy_read_failed"})

    for stage_dir, artifact in jsonl_artifacts:
        stage_root = root / stage_dir
        rows = [{"artifact_key": artifact.key, "row": 1}]
        write_jsonl_artifact(stage_root, artifact, rows)
        read_rows, source = read_jsonl_artifact(stage_root, artifact)
        if source != "canonical" or len(read_rows) != 1:
            failures.append({"artifact": artifact.key, "reason": "canonical_jsonl_read_failed"})

        legacy_root = root / f"{stage_dir}_legacy_{artifact.key}"
        artifact_io.write_jsonl(artifact_path(legacy_root, artifact, canonical=False), rows)
        legacy_rows, legacy_source = read_jsonl_artifact(legacy_root, artifact)
        if legacy_source != "legacy" or len(legacy_rows) != 1:
            failures.append({"artifact": artifact.key, "reason": "legacy_jsonl_read_failed"})

    return {
        "schema_version": ALIAS_AUDIT_SCHEMA_VERSION,
        "alias_contract_passed": not failures,
        "checked_json_artifact_count": len(json_artifacts),
        "checked_jsonl_artifact_count": len(jsonl_artifacts),
        "failure_count": len(failures),
        "failures": failures,
    }


def _path_audit(config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    registry_path = Path(str(config["stage_registry_path"]))
    if not artifact_io.path_is_file(registry_path):
        return {
            "schema_version": PATH_AUDIT_SCHEMA_VERSION,
            "short_output_root_contract_passed": False,
            "path_length_failure_count": 1,
            "max_path_length": 0,
            "failures": [{"stage_id": "stage_registry", "reason": "registry_missing", "path": str(registry_path)}],
        }
    registry_payload = artifact_io.read_json(registry_path)
    registry = registry_payload.get("stages") if isinstance(registry_payload.get("stages"), dict) else registry_payload
    roots: list[tuple[str, Path]] = [("output_root", output_root)]
    failures: list[dict[str, Any]] = []
    for stage_id in config["short_root_stage_ids"]:
        entry = registry.get(stage_id)
        if not isinstance(entry, dict):
            failures.append({"stage_id": stage_id, "reason": "registry_entry_missing"})
            continue
        root = Path(str(entry.get("default_output_root", "")))
        roots.append((stage_id, root))
    lengths = [
        {
            "stage_id": stage_id,
            "path": str(root),
            "resolved_path": str(_resolve_path(root, repo_root)),
            "path_length": artifact_io.path_length(_resolve_path(root, repo_root)),
        }
        for stage_id, root in roots
    ]
    fail_at = int(config["fail_path_length"])
    short_failures = [row for row in lengths if row["path_length"] >= fail_at]
    d_root_failures = [
        row
        for row in lengths
        if str(row["path"]).replace("\\", "/").startswith("D:/xunce/out") and row["path_length"] >= 180
    ]
    return {
        "schema_version": PATH_AUDIT_SCHEMA_VERSION,
        "short_output_root_contract_passed": not failures and not short_failures and not d_root_failures,
        "checked_stage_count": len(config["short_root_stage_ids"]),
        "max_path_length": max((row["path_length"] for row in lengths), default=0),
        "path_length_failure_count": len(failures) + len(short_failures) + len(d_root_failures),
        "entries": lengths,
        "failures": failures + short_failures + d_root_failures,
    }


def _runner_static_io_audit(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    missing_files: list[str] = []
    for raw_path in config["migrated_runner_files"]:
        path = Path(str(raw_path))
        if not artifact_io.path_is_file(path):
            missing_files.append(str(path))
            continue
        for lineno, line in enumerate(artifact_io.read_text(path).splitlines(), start=1):
            if not any(marker in line for marker in FORBIDDEN_DIRECT_IO_MARKERS):
                continue
            if any(marker in line for marker in SAFE_HELPER_MARKERS):
                continue
            violations.append({"file": str(path.relative_to(repo_root) if path.is_absolute() and _is_relative_to(path, repo_root) else path), "line": lineno, "text": line.strip()})
    return {
        "schema_version": STATIC_AUDIT_SCHEMA_VERSION,
        "runner_static_io_audit_passed": not violations and not missing_files,
        "checked_file_count": len(config["migrated_runner_files"]),
        "direct_artifact_io_violation_count": len(violations),
        "missing_file_count": len(missing_files),
        "violations": violations,
        "missing_files": missing_files,
    }


def _route(
    *,
    boundary_rejections: list[str],
    alias_audit: dict[str, Any],
    path_audit: dict[str, Any],
    static_audit: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if not alias_audit.get("alias_contract_passed"):
        return ROUTE_ALIAS
    if not path_audit.get("short_output_root_contract_passed"):
        return ROUTE_SHORT_ROOT
    if not static_audit.get("runner_static_io_audit_passed"):
        return ROUTE_STATIC_IO
    return ROUTE_STAGE26_8N


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.IO2 Remaining Runner Long-Path Migration",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- alias_contract_passed: `{summary['alias_contract_passed']}`",
            f"- runner_static_io_audit_passed: `{summary['runner_static_io_audit_passed']}`",
            f"- short_output_root_contract_passed: `{summary['short_output_root_contract_passed']}`",
            f"- direct_artifact_io_violation_count: `{summary['direct_artifact_io_violation_count']}`",
            "",
            "This stage only audits artifact IO/path behavior. It does not run PPO, change rewards, publish checkpoints, replace policies, connect executors, or start canaries.",
        ]
    ) + "\n"


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
