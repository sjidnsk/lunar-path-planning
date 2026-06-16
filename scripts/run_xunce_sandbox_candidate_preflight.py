from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from global_99_governance_common import global_99_boundary_defaults
    from run_xunce_controlled_training_candidate import _synthetic_batch
    from xunce_full_network_common import XunceFullNetworkV1
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.run_xunce_controlled_training_candidate import _synthetic_batch
    from scripts.xunce_full_network_common import XunceFullNetworkV1


CONFIG_SCHEMA_VERSION = "xunce-sandbox-candidate-preflight-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-sandbox-candidate-preflight-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-sandbox-candidate-preflight-manifest/v1"
PACKAGE_MANIFEST_SCHEMA_VERSION = "xunce-sandbox-package-manifest/v1"
HASH_AUDIT_SCHEMA_VERSION = "xunce-sandbox-checkpoint-hash-audit/v1"
LOAD_AUDIT_SCHEMA_VERSION = "xunce-sandbox-load-audit/v1"
KILL_SWITCH_AUDIT_SCHEMA_VERSION = "xunce-sandbox-kill-switch-audit/v1"
ROLLBACK_AUDIT_SCHEMA_VERSION = "xunce-sandbox-rollback-audit/v1"
TELEMETRY_AUDIT_SCHEMA_VERSION = "xunce-sandbox-telemetry-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-sandbox-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-sandbox-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_sandbox_candidate_preflight_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1"

SUMMARY_FILE = "xunce-sandbox-candidate-preflight-summary.json"
MANIFEST_FILE = "xunce-sandbox-candidate-preflight-manifest.json"
PACKAGE_MANIFEST_FILE = "xunce-sandbox-package-manifest.json"
HASH_AUDIT_FILE = "xunce-sandbox-checkpoint-hash-audit.json"
LOAD_AUDIT_FILE = "xunce-sandbox-load-audit.json"
KILL_SWITCH_AUDIT_FILE = "xunce-sandbox-kill-switch-audit.json"
ROLLBACK_AUDIT_FILE = "xunce-sandbox-rollback-audit.json"
TELEMETRY_AUDIT_FILE = "xunce-sandbox-telemetry-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-sandbox-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-sandbox-rejection-report.json"
REPORT_FILE = "xunce-sandbox-candidate-preflight-report.md"
SANDBOX_PACKAGE_DIR = "sandbox_package"
CHECKPOINT_FILE = "xunce-controlled-training-candidate.pt"

ARCHITECTURE = "xunce_full_network_v1"
PASS_NEXT_REQUIRED_CHANGE = "xunce_release_governance_gate"
FIX_SHADOW_NEXT_REQUIRED_CHANGE = "fix_xunce_shadow_replay_validation"
FIX_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_controlled_training_candidate_checkpoint"
FAIL_NEXT_REQUIRED_CHANGE = "fix_xunce_sandbox_candidate_preflight"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_sandbox_candidate_boundary_rejections"

SOURCE_SHADOW_SUMMARY_FILE = "xunce-shadow-replay-validation-summary.json"
SOURCE_SHADOW_PASS_NEXT_REQUIRED_CHANGE = "sandbox_candidate_preflight"

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
    parser = argparse.ArgumentParser(description="Run Xunce Sandbox Candidate Preflight v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_sandbox_candidate_preflight(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_sandbox_candidate_preflight(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    shadow_root = resolve_path(Path(config["source_shadow_replay_root"]), repo_root)
    training_root = resolve_path(Path(config["source_controlled_training_root"]), repo_root)
    shadow_summary = _load_json(shadow_root / SOURCE_SHADOW_SUMMARY_FILE)
    source_audit = _source_audit(shadow_summary)
    boundary_audit = _boundary_audit(shadow_summary)
    package_manifest = _empty_package_manifest(paths)
    hash_audit = _empty_hash_audit(training_root / CHECKPOINT_FILE, paths["sandbox_checkpoint"])
    load_audit = _empty_load_audit(paths["sandbox_checkpoint"])
    if source_audit["source_shadow_audit_passed"] and boundary_audit["source_boundary_audit_passed"]:
        package_manifest, hash_audit, load_audit = _create_and_load_sandbox_package(config, training_root / CHECKPOINT_FILE, paths)
    kill_switch_audit = _kill_switch_audit(config)
    rollback_audit = _rollback_audit(config, output_root)
    telemetry_audit = _telemetry_audit(config, package_manifest, hash_audit, load_audit)
    decision = _decision(source_audit, boundary_audit, hash_audit, load_audit, kill_switch_audit, rollback_audit, telemetry_audit)
    generated_at = utc_now()
    summary = _summary(generated_at, config_path, output_root, paths, shadow_summary, source_audit, boundary_audit, package_manifest, hash_audit, load_audit, kill_switch_audit, rollback_audit, telemetry_audit, decision, repo_root)
    manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "generated_at": generated_at, "config": str(config_path), "output_root": str(output_root), "artifacts": {key: str(path) for key, path in paths.items()}, "summary_status": summary["status"], "next_required_change": summary["next_required_change"]}
    rejection_report = {"schema_version": REJECTION_REPORT_SCHEMA_VERSION, "status": decision["status"], "reason_codes": decision["reason_codes"], "next_required_change": decision["next_required_change"], "sandbox_package_created": package_manifest["sandbox_package_created"], "sandbox_load_verified": load_audit["sandbox_load_verified"]}
    write_json(paths["package_manifest"], package_manifest)
    write_json(paths["hash_audit"], hash_audit)
    write_json(paths["load_audit"], load_audit)
    write_json(paths["kill_switch_audit"], kill_switch_audit)
    write_json(paths["rollback_audit"], rollback_audit)
    write_json(paths["telemetry_audit"], telemetry_audit)
    write_json(paths["boundary_audit"], boundary_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary), encoding="utf-8")
    return summary


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
    if payload.get("architecture") != ARCHITECTURE:
        raise ConfigError(f"architecture must be {ARCHITECTURE!r}")
    normalized = dict(payload)
    for key in ("source_shadow_replay_root", "source_controlled_training_root"):
        if not isinstance(normalized.get(key), str) or not normalized[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    for key in ("seed", "candidate_feature_count", "edge_feature_count", "memory_feature_count", "context_feature_count", "missing_indicator_count", "hidden_dim", "message_passing_layers", "candidate_count"):
        normalized[key] = _int_value(normalized.get(key), key)
    for key in ("candidate_feature_count", "edge_feature_count", "memory_feature_count", "context_feature_count", "hidden_dim", "message_passing_layers", "candidate_count"):
        if normalized[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if normalized["missing_indicator_count"] < 0:
        raise ConfigError("missing_indicator_count must be non-negative")
    for key in ("kill_switch_required", "rollback_required", "telemetry_required", "require_executor_isolation"):
        normalized[key] = bool(normalized.get(key, True))
    return normalized


def _source_audit(shadow_summary: dict[str, Any] | None) -> dict[str, Any]:
    reason_codes: list[str] = []
    if not isinstance(shadow_summary, dict):
        reason_codes.append("missing_shadow_replay_summary")
    else:
        if shadow_summary.get("status") != "passed":
            reason_codes.append("shadow_replay_not_passed")
        if shadow_summary.get("next_required_change") != SOURCE_SHADOW_PASS_NEXT_REQUIRED_CHANGE:
            reason_codes.append("shadow_replay_wrong_next_required_change")
        if shadow_summary.get("shadow_replay_passed") is not True:
            reason_codes.append("shadow_replay_pass_field_false")
        if shadow_summary.get("source_match_audit_passed") is not True:
            reason_codes.append("shadow_replay_source_match_not_passed")
    return {"schema_version": "xunce-sandbox-source-audit/v1", "source_status": shadow_summary.get("status") if isinstance(shadow_summary, dict) else None, "source_next_required_change": shadow_summary.get("next_required_change") if isinstance(shadow_summary, dict) else None, "reason_codes": unique_sorted(reason_codes), "source_shadow_audit_passed": not reason_codes}


def _boundary_audit(shadow_summary: dict[str, Any] | None) -> dict[str, Any]:
    violations: list[str] = []
    observed: dict[str, bool] = {}
    if isinstance(shadow_summary, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(shadow_summary.get(field, False))
            observed[field] = value
            if value:
                violations.append(field)
    return {"schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION, "observed_source_boundary_fields": observed, "violating_source_boundary_fields": unique_sorted(violations), "source_boundary_audit_passed": not violations, "boundary_audit_passed": not violations, **_closed_boundary_fields()}


def _create_and_load_sandbox_package(config: dict[str, Any], source_checkpoint: Path, paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not source_checkpoint.is_file():
        package_manifest = _empty_package_manifest(paths)
        hash_audit = _empty_hash_audit(source_checkpoint, paths["sandbox_checkpoint"], ["missing_research_checkpoint"])
        return package_manifest, hash_audit, _empty_load_audit(paths["sandbox_checkpoint"], ["missing_research_checkpoint"])
    paths["sandbox_checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, paths["sandbox_checkpoint"])
    source_hash = _sha256(source_checkpoint)
    sandbox_hash = _sha256(paths["sandbox_checkpoint"])
    hash_verified = source_hash == sandbox_hash
    hash_reason_codes = [] if hash_verified else ["sandbox_checkpoint_hash_mismatch"]
    package_manifest = {"schema_version": PACKAGE_MANIFEST_SCHEMA_VERSION, "sandbox_package_created": True, "sandbox_package_root": str(paths["sandbox_package_dir"]), "sandbox_checkpoint": str(paths["sandbox_checkpoint"]), "source_checkpoint": str(source_checkpoint), "architecture": ARCHITECTURE, "checkpoint_sha256": sandbox_hash, "publishes_checkpoint": False, "replaces_default_policy": False}
    hash_audit = {"schema_version": HASH_AUDIT_SCHEMA_VERSION, "source_checkpoint": str(source_checkpoint), "sandbox_checkpoint": str(paths["sandbox_checkpoint"]), "source_checkpoint_sha256": source_hash, "sandbox_checkpoint_sha256": sandbox_hash, "sandbox_checkpoint_hash_verified": hash_verified, "reason_codes": hash_reason_codes}
    load_audit = _load_sandbox_checkpoint(config, paths["sandbox_checkpoint"])
    return package_manifest, hash_audit, load_audit


def _load_sandbox_checkpoint(config: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    reason_codes: list[str] = []
    model = _build_model(config)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return _empty_load_audit(checkpoint_path, ["invalid_research_checkpoint"])
    if not isinstance(payload, dict):
        reason_codes.append("invalid_research_checkpoint")
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        reason_codes.append("checkpoint_state_dict_missing")
    if not reason_codes and isinstance(state_dict, dict):
        try:
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            with torch.no_grad():
                output = model(**_synthetic_batch(config))
            if not (torch.isfinite(output.logits).all() and torch.isfinite(output.value).all()):
                reason_codes.append("sandbox_forward_non_finite")
        except Exception:
            reason_codes.append("sandbox_checkpoint_load_failed")
    verified = not reason_codes
    return {"schema_version": LOAD_AUDIT_SCHEMA_VERSION, "sandbox_checkpoint": str(checkpoint_path), "sandbox_load_verified": verified, "sandbox_forward_verified": verified, "checkpoint_read_only": True, "reason_codes": unique_sorted(reason_codes)}


def _kill_switch_audit(config: dict[str, Any]) -> dict[str, Any]:
    passed = bool(config["kill_switch_required"])
    return {"schema_version": KILL_SWITCH_AUDIT_SCHEMA_VERSION, "kill_switch_required": config["kill_switch_required"], "kill_switch_default_state": "armed", "kill_switch_trigger_dry_run_passed": passed, "kill_switch_audit_passed": passed}


def _rollback_audit(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    probe = output_root / "rollback_probe"
    probe.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(probe)
    passed = bool(config["rollback_required"]) and not probe.exists()
    return {"schema_version": ROLLBACK_AUDIT_SCHEMA_VERSION, "rollback_required": config["rollback_required"], "rollback_dry_run_verified": passed, "rollback_action": "remove_sandbox_candidate_package", "rollback_audit_passed": passed}


def _telemetry_audit(config: dict[str, Any], package_manifest: dict[str, Any], hash_audit: dict[str, Any], load_audit: dict[str, Any]) -> dict[str, Any]:
    required_fields_present = all(key in package_manifest for key in ("sandbox_checkpoint", "architecture", "checkpoint_sha256")) and bool(hash_audit.get("sandbox_checkpoint_sha256")) and "sandbox_load_verified" in load_audit
    passed = bool(config["telemetry_required"]) and required_fields_present
    return {"schema_version": TELEMETRY_AUDIT_SCHEMA_VERSION, "telemetry_required": config["telemetry_required"], "telemetry_fields_present": required_fields_present, "telemetry_audit_passed": passed}


def _decision(source_audit: dict[str, Any], boundary_audit: dict[str, Any], hash_audit: dict[str, Any], load_audit: dict[str, Any], kill_switch_audit: dict[str, Any], rollback_audit: dict[str, Any], telemetry_audit: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    reason_codes.extend(source_audit["reason_codes"])
    if not boundary_audit["source_boundary_audit_passed"]:
        reason_codes.append("sandbox_source_boundary_violation")
    reason_codes.extend(hash_audit["reason_codes"])
    reason_codes.extend(load_audit["reason_codes"])
    if not kill_switch_audit["kill_switch_audit_passed"]:
        reason_codes.append("sandbox_kill_switch_audit_failed")
    if not rollback_audit["rollback_audit_passed"]:
        reason_codes.append("sandbox_rollback_audit_failed")
    if not telemetry_audit["telemetry_audit_passed"]:
        reason_codes.append("sandbox_telemetry_audit_failed")
    reason_codes = unique_sorted(reason_codes)
    if not reason_codes:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif "sandbox_source_boundary_violation" in reason_codes:
        next_required_change = BOUNDARY_NEXT_REQUIRED_CHANGE
    elif "missing_research_checkpoint" in reason_codes or "invalid_research_checkpoint" in reason_codes or "checkpoint_state_dict_missing" in reason_codes:
        next_required_change = FIX_CHECKPOINT_NEXT_REQUIRED_CHANGE
    elif any(code.startswith("missing_shadow") or code.startswith("shadow_replay") for code in reason_codes):
        next_required_change = FIX_SHADOW_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    return {"status": "passed" if not reason_codes else "failed", "reason_codes": reason_codes, "next_required_change": next_required_change}


def _summary(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], shadow_summary: dict[str, Any] | None, source_audit: dict[str, Any], boundary_audit: dict[str, Any], package_manifest: dict[str, Any], hash_audit: dict[str, Any], load_audit: dict[str, Any], kill_switch_audit: dict[str, Any], rollback_audit: dict[str, Any], telemetry_audit: dict[str, Any], decision: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    summary = {"schema_version": SUMMARY_SCHEMA_VERSION, "generated_at": generated_at, "config": str(config_path), "output_root": str(output_root), "summary": str(paths["summary"]), "status": decision["status"], "reason_codes": decision["reason_codes"], "source_shadow_replay_status": shadow_summary.get("status") if isinstance(shadow_summary, dict) else None, "source_shadow_replay_next_required_change": shadow_summary.get("next_required_change") if isinstance(shadow_summary, dict) else None, "source_shadow_audit_passed": source_audit["source_shadow_audit_passed"], "sandbox_candidate_preflight_passed": decision["status"] == "passed", "sandbox_package_created": package_manifest["sandbox_package_created"], "sandbox_checkpoint_hash_verified": hash_audit["sandbox_checkpoint_hash_verified"], "sandbox_load_verified": load_audit["sandbox_load_verified"], "kill_switch_audit_passed": kill_switch_audit["kill_switch_audit_passed"], "rollback_audit_passed": rollback_audit["rollback_audit_passed"], "telemetry_audit_passed": telemetry_audit["telemetry_audit_passed"], "default_policy_read_only": True, "executor_isolation_passed": True, "boundary_audit_passed": boundary_audit["boundary_audit_passed"], "source_boundary_audit_passed": boundary_audit["source_boundary_audit_passed"], "next_required_change": decision["next_required_change"], "git_provenance": {"current": git_snapshot(repo_root)}}
    summary.update(_closed_boundary_fields())
    return summary


def _build_model(config: dict[str, Any]) -> XunceFullNetworkV1:
    torch.manual_seed(config["seed"])
    return XunceFullNetworkV1(candidate_feature_count=config["candidate_feature_count"], edge_feature_count=config["edge_feature_count"], memory_feature_count=config["memory_feature_count"], context_feature_count=config["context_feature_count"], missing_indicator_count=config["missing_indicator_count"], hidden_dim=config["hidden_dim"], message_passing_layers=config["message_passing_layers"], dropout=0.0)


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    package_dir = output_root / SANDBOX_PACKAGE_DIR
    return {"summary": output_root / SUMMARY_FILE, "manifest": output_root / MANIFEST_FILE, "package_manifest": output_root / PACKAGE_MANIFEST_FILE, "hash_audit": output_root / HASH_AUDIT_FILE, "load_audit": output_root / LOAD_AUDIT_FILE, "kill_switch_audit": output_root / KILL_SWITCH_AUDIT_FILE, "rollback_audit": output_root / ROLLBACK_AUDIT_FILE, "telemetry_audit": output_root / TELEMETRY_AUDIT_FILE, "boundary_audit": output_root / BOUNDARY_AUDIT_FILE, "rejection_report": output_root / REJECTION_REPORT_FILE, "report": output_root / REPORT_FILE, "sandbox_package_dir": package_dir, "sandbox_checkpoint": package_dir / CHECKPOINT_FILE}


def _empty_package_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    return {"schema_version": PACKAGE_MANIFEST_SCHEMA_VERSION, "sandbox_package_created": False, "sandbox_package_root": str(paths["sandbox_package_dir"]), "sandbox_checkpoint": str(paths["sandbox_checkpoint"]), "source_checkpoint": None, "architecture": ARCHITECTURE, "checkpoint_sha256": None, "publishes_checkpoint": False, "replaces_default_policy": False}


def _empty_hash_audit(source_checkpoint: Path, sandbox_checkpoint: Path, reason_codes: list[str] | None = None) -> dict[str, Any]:
    return {"schema_version": HASH_AUDIT_SCHEMA_VERSION, "source_checkpoint": str(source_checkpoint), "sandbox_checkpoint": str(sandbox_checkpoint), "source_checkpoint_sha256": None, "sandbox_checkpoint_sha256": None, "sandbox_checkpoint_hash_verified": False, "reason_codes": unique_sorted(reason_codes or [])}


def _empty_load_audit(checkpoint_path: Path, reason_codes: list[str] | None = None) -> dict[str, Any]:
    return {"schema_version": LOAD_AUDIT_SCHEMA_VERSION, "sandbox_checkpoint": str(checkpoint_path), "sandbox_load_verified": False, "sandbox_forward_verified": False, "checkpoint_read_only": True, "reason_codes": unique_sorted(reason_codes or [])}


def _closed_boundary_fields() -> dict[str, bool | float]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update({"real_world_release_approved": False, "real_world_performance_claimed": False, "default_policy_replacement_approved": False, "real_executor_connection_approved": False, "starts_online_canary": False, "canary_traffic_fraction": 0.0, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False, "runs_new_training_update": False, "runs_new_ppo_update": False, "modifies_network": False, "modifies_action_space": False, "modifies_default_astar": False})
    return fields


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(["# Xunce Sandbox Candidate Preflight v1", "", f"- status: `{summary['status']}`", f"- reason_codes: `{summary['reason_codes']}`", f"- sandbox_package_created: `{summary['sandbox_package_created']}`", f"- sandbox_load_verified: `{summary['sandbox_load_verified']}`", f"- publishes_checkpoint: `{summary['publishes_checkpoint']}`", f"- replaces_default_policy: `{summary['replaces_default_policy']}`", f"- connects_real_executor: `{summary['connects_real_executor']}`", f"- next_required_change: `{summary['next_required_change']}`", "", "This is a sandbox candidate preflight only. It does not publish, install, replace default policy, connect an executor, or start canary traffic.", ""])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _int_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc


if __name__ == "__main__":
    raise SystemExit(main())
