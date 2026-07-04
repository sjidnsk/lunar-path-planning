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
    STAGE21_1_SUMMARY,
    STAGE21_1_TRAINABLE,
    STAGE21_2_REWARDS,
    STAGE21_2_SUMMARY,
    STAGE21_3_BATCH,
    STAGE21_3_SPLITS,
    STAGE21_3_SUMMARY,
    artifact_path,
    read_json_artifact,
    read_jsonl_artifact,
    write_json_artifact,
    write_jsonl_artifact,
)


STAGE_ID = "xunce-stage26-io1-artifact-path-contract-and-long-path-resilience"
CONFIG_SCHEMA_VERSION = "xunce-stage26-io1-artifact-path-contract-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-io1-summary/v1"
PATH_AUDIT_SCHEMA_VERSION = "xunce-stage26-io1-path-audit/v1"
ALIAS_AUDIT_SCHEMA_VERSION = "xunce-stage26-io1-alias-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-io1-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-io1-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_io1_artifact_path_contract_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_io1"

SUMMARY_FILE = "summary.json"
PATH_AUDIT_FILE = "path_audit.json"
ALIAS_AUDIT_FILE = "alias_audit.json"
ROUTING_FILE = "routing.json"
REPORT_FILE = "report.md"
MANIFEST_FILE = "manifest.json"

ROUTE_LONG_PATH_IO = "repair_xunce_artifact_io_long_path"
ROUTE_ALIAS = "repair_xunce_artifact_alias_contract"
ROUTE_STAGE21_COMPAT = "repair_stage21_artifact_compatibility"
ROUTE_SHORT_ROOT = "repair_stage26_short_output_root_contract"
ROUTE_STAGE26_8N = "rerun_stage26_8n_aggressive_update_sweep_with_aligned_planning_proxy"
ROUTE_BOUNDARY = "resolve_stage26_io1_boundary_rejections"

BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.IO1 artifact path contract audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    try:
        summary = run_xunce_stage26_io1_artifact_path_contract(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            repo_root=Path(args.repo_root),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_io1_artifact_path_contract(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)

    boundary_rejections = _boundary_rejections(config)
    long_path_io_passed = _long_path_io_probe(output_root)
    alias_audit = _alias_audit(output_root)
    path_audit = _path_audit(config, output_root, repo_root)
    stage21_compat = _stage21_compatibility_audit(output_root)
    stage26_8q_audit = _stage26_8q_current_root_audit(config)

    route = _route(
        boundary_rejections=boundary_rejections,
        long_path_io_passed=long_path_io_passed,
        alias_audit=alias_audit,
        path_audit=path_audit,
        stage21_compat=stage21_compat,
    )
    status = "passed" if route == ROUTE_STAGE26_8N else "failed"

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "long_path_io_passed": long_path_io_passed,
        "alias_contract_passed": bool(alias_audit.get("alias_contract_passed")),
        "stage21_artifact_compatibility_passed": bool(stage21_compat.get("stage21_artifact_compatibility_passed")),
        "short_output_root_contract_passed": bool(path_audit.get("short_output_root_contract_passed")),
        "stage26_8q_legacy_root_readable": bool(stage26_8q_audit.get("stage26_8q_legacy_root_readable")),
        "stage26_8q_next_required_change": stage26_8q_audit.get("next_required_change"),
        "max_path_length": path_audit.get("max_path_length"),
        "path_length_failure_count": path_audit.get("path_length_failure_count"),
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
        "routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
        "manifest": str(output_root / MANIFEST_FILE),
        "stage26_8q_current_root_audit": stage26_8q_audit,
    }

    artifact_io.write_json(output_root / PATH_AUDIT_FILE, path_audit)
    artifact_io.write_json(output_root / ALIAS_AUDIT_FILE, alias_audit | {"stage21_compatibility": stage21_compat})
    artifact_io.write_json(output_root / ROUTING_FILE, routing)
    artifact_io.write_json(output_root / SUMMARY_FILE, summary)
    artifact_io.write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = artifact_io.read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ConfigError(f"stage_id must be {STAGE_ID}")
    config = dict(payload)
    config.setdefault("short_output_roots", ["D:/xunce/out/s26_io1", "D:/xunce/out/s26_8m", "D:/xunce/out/s26_8n", "D:/xunce/out/s26_8q"])
    config.setdefault("warn_path_length", 180)
    config.setdefault("fail_path_length", 240)
    config.setdefault("stage26_8q_legacy_root", "")
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    config["_config_path"] = str(path)
    config["_repo_root"] = str(repo_root)
    return config


def _long_path_io_probe(output_root: Path) -> bool:
    long_dir = output_root / "long_path_probe" / ("segment_" + "x" * 40) / ("segment_" + "y" * 40) / ("segment_" + "z" * 40)
    probe = long_dir / ("artifact_" + "q" * 90 + ".json")
    rows = long_dir / ("rows_" + "r" * 90 + ".jsonl")
    payload = {"schema_version": "xunce-long-path-io-probe/v1", "path_length": artifact_io.path_length(probe)}
    artifact_io.write_json(probe, payload)
    artifact_io.write_jsonl(rows, [{"row": 1}, {"row": 2}])
    return artifact_io.read_json(probe)["schema_version"] == payload["schema_version"] and artifact_io.count_jsonl_rows(rows) == 2


def _alias_audit(output_root: Path) -> dict[str, Any]:
    root = output_root / "alias_probe"
    write_json_artifact(root / "stage21_1", STAGE21_1_SUMMARY, {"status": "canonical"})
    write_jsonl_artifact(root / "stage21_1", STAGE21_1_TRAINABLE, [{"transition_id": "t1"}])
    preferred, preferred_source = read_json_artifact(root / "stage21_1", STAGE21_1_SUMMARY)

    legacy_root = root / "legacy_only_stage21_2"
    artifact_io.write_json(artifact_path(legacy_root, STAGE21_2_SUMMARY, canonical=False), {"status": "legacy-only"})
    artifact_io.write_jsonl(artifact_path(legacy_root, STAGE21_2_REWARDS, canonical=False), [{"transition_id": "t1"}])
    fallback, fallback_source = read_json_artifact(legacy_root, STAGE21_2_SUMMARY)
    fallback_rows, fallback_rows_source = read_jsonl_artifact(legacy_root, STAGE21_2_REWARDS)

    alias_contract_passed = (
        preferred.get("status") == "canonical"
        and preferred_source == "canonical"
        and fallback.get("status") == "legacy-only"
        and fallback_source == "legacy"
        and fallback_rows_source == "legacy"
        and len(fallback_rows) == 1
    )
    return {
        "schema_version": ALIAS_AUDIT_SCHEMA_VERSION,
        "alias_contract_passed": alias_contract_passed,
        "canonical_preferred_source": preferred_source,
        "legacy_fallback_source": fallback_source,
        "legacy_fallback_row_source": fallback_rows_source,
    }


def _stage21_compatibility_audit(output_root: Path) -> dict[str, Any]:
    root = output_root / "stage21_compat_probe"
    write_json_artifact(root / "s21_1", STAGE21_1_SUMMARY, {"status": "passed"})
    write_jsonl_artifact(root / "s21_1", STAGE21_1_TRAINABLE, [{"transition_id": "t1"}])
    write_json_artifact(root / "s21_2", STAGE21_2_SUMMARY, {"status": "passed"})
    write_jsonl_artifact(root / "s21_2", STAGE21_2_REWARDS, [{"transition_id": "t1", "reward": 1.0}])
    write_json_artifact(root / "s21_3", STAGE21_3_SUMMARY, {"status": "passed"})
    write_jsonl_artifact(root / "s21_3", STAGE21_3_BATCH, [{"transition_id": "t1"}])
    write_json_artifact(root / "s21_3", STAGE21_3_SPLITS, {"train": ["t1"], "validation": []})
    s21_1_rows, _ = read_jsonl_artifact(root / "s21_1", STAGE21_1_TRAINABLE)
    s21_2_rows, _ = read_jsonl_artifact(root / "s21_2", STAGE21_2_REWARDS)
    s21_3_rows, _ = read_jsonl_artifact(root / "s21_3", STAGE21_3_BATCH)
    return {
        "schema_version": "xunce-stage26-io1-stage21-compatibility-audit/v1",
        "stage21_artifact_compatibility_passed": len(s21_1_rows) == len(s21_2_rows) == len(s21_3_rows) == 1,
        "stage21_1_trainable_rows": len(s21_1_rows),
        "stage21_2_reward_rows": len(s21_2_rows),
        "stage21_3_batch_rows": len(s21_3_rows),
    }


def _path_audit(config: dict[str, Any], output_root: Path, repo_root: Path) -> dict[str, Any]:
    roots = [output_root, *[Path(str(root)) for root in config.get("short_output_roots", [])]]
    audit = artifact_io.path_length_audit(
        roots,
        warn_at=int(config["warn_path_length"]),
        fail_at=int(config["fail_path_length"]),
    )
    audit["schema_version"] = PATH_AUDIT_SCHEMA_VERSION
    audit["short_output_root_contract_passed"] = audit["path_length_failure_count"] == 0 and all(
        len(str(_resolve_path(root, repo_root))) < 180 for root in roots if str(root).startswith("D:/xunce/out")
    )
    return audit


def _stage26_8q_current_root_audit(config: dict[str, Any]) -> dict[str, Any]:
    raw = str(config.get("stage26_8q_legacy_root") or "").strip()
    if not raw:
        return {"stage26_8q_legacy_root_readable": False, "reason": "not_configured"}
    summary_path = Path(raw) / "xunce-stage26-8q-summary.json"
    if not artifact_io.path_is_file(summary_path):
        return {"stage26_8q_legacy_root_readable": False, "reason": "summary_missing", "summary_path": str(summary_path)}
    payload = artifact_io.read_json(summary_path)
    return {
        "stage26_8q_legacy_root_readable": True,
        "status": payload.get("status"),
        "next_required_change": payload.get("next_required_change"),
        "summary_path": str(summary_path),
    }


def _route(
    *,
    boundary_rejections: list[str],
    long_path_io_passed: bool,
    alias_audit: dict[str, Any],
    path_audit: dict[str, Any],
    stage21_compat: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if not long_path_io_passed:
        return ROUTE_LONG_PATH_IO
    if not alias_audit.get("alias_contract_passed"):
        return ROUTE_ALIAS
    if not stage21_compat.get("stage21_artifact_compatibility_passed"):
        return ROUTE_STAGE21_COMPAT
    if not path_audit.get("short_output_root_contract_passed"):
        return ROUTE_SHORT_ROOT
    return ROUTE_STAGE26_8N


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is not False]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.IO1 Artifact Path Contract",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- long_path_io_passed: `{summary['long_path_io_passed']}`",
            f"- alias_contract_passed: `{summary['alias_contract_passed']}`",
            f"- stage21_artifact_compatibility_passed: `{summary['stage21_artifact_compatibility_passed']}`",
            f"- short_output_root_contract_passed: `{summary['short_output_root_contract_passed']}`",
            "",
            "This stage only audits artifact IO/path behavior. It does not run PPO, change rewards, publish checkpoints, replace policies, connect executors, or start canaries.",
        ]
    ) + "\n"


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
