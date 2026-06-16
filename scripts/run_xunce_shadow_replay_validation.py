from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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


CONFIG_SCHEMA_VERSION = "xunce-shadow-replay-validation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-shadow-replay-validation-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-shadow-replay-validation-manifest/v1"
RESULT_ROW_SCHEMA_VERSION = "xunce-shadow-replay-result-row/v1"
SOURCE_MATCH_AUDIT_SCHEMA_VERSION = "xunce-shadow-replay-source-match-audit/v1"
CHECKPOINT_LOAD_AUDIT_SCHEMA_VERSION = "xunce-shadow-replay-checkpoint-load-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-shadow-replay-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-shadow-replay-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_shadow_replay_validation_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_shadow_replay_validation_v1"

SUMMARY_FILE = "xunce-shadow-replay-validation-summary.json"
MANIFEST_FILE = "xunce-shadow-replay-validation-manifest.json"
RESULTS_FILE = "xunce-shadow-replay-results.jsonl"
SOURCE_MATCH_AUDIT_FILE = "xunce-shadow-replay-source-match-audit.json"
CHECKPOINT_LOAD_AUDIT_FILE = "xunce-shadow-replay-checkpoint-load-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-shadow-replay-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-shadow-replay-rejection-report.json"
REPORT_FILE = "xunce-shadow-replay-validation-report.md"

ARCHITECTURE = "xunce_full_network_v1"
PASS_NEXT_REQUIRED_CHANGE = "sandbox_candidate_preflight"
FIX_POST_TRAINING_NEXT_REQUIRED_CHANGE = "fix_post_training_offline_evaluation"
FIX_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_controlled_training_candidate_checkpoint"
FIX_DETERMINISM_NEXT_REQUIRED_CHANGE = "fix_xunce_shadow_replay_determinism"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_shadow_replay_boundary_rejections"

SOURCE_SUMMARY_FILE = "xunce-post-training-offline-evaluation-summary.json"
SOURCE_RESULTS_FILE = "xunce-post-training-offline-evaluation-results.jsonl"
SOURCE_PASS_NEXT_REQUIRED_CHANGE = "shadow_replay_validation"
CHECKPOINT_FILE = "xunce-controlled-training-candidate.pt"

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
    parser = argparse.ArgumentParser(description="Run Xunce Shadow Replay Validation v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_shadow_replay_validation(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_shadow_replay_validation(*, config_path: Path, output_root: Path, repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    post_training_root = resolve_path(Path(config["source_post_training_root"]), repo_root)
    controlled_training_root = resolve_path(Path(config["source_controlled_training_root"]), repo_root)
    source_summary = _load_json(post_training_root / SOURCE_SUMMARY_FILE)
    source_rows = _load_jsonl(post_training_root / SOURCE_RESULTS_FILE)
    source_audit = _source_audit(source_summary, source_rows)
    boundary_audit = _boundary_audit(source_summary)
    checkpoint_audit: dict[str, Any]
    replay_rows: list[dict[str, Any]] = []
    source_match_audit: dict[str, Any]
    if source_audit["source_post_training_audit_passed"] and boundary_audit["source_boundary_audit_passed"]:
        checkpoint_audit, model = _load_checkpoint(controlled_training_root / CHECKPOINT_FILE, config)
        if checkpoint_audit["checkpoint_loaded"] and model is not None:
            replay_rows = _replay_rows(config, model)
            source_match_audit = _source_match_audit(source_rows, replay_rows, config)
        else:
            source_match_audit = _empty_source_match_audit()
    else:
        checkpoint_audit = _checkpoint_audit(controlled_training_root / CHECKPOINT_FILE, ["source_blocked"], loaded=False)
        source_match_audit = _empty_source_match_audit()
    decision = _decision(source_audit, boundary_audit, checkpoint_audit, source_match_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at,
        config_path,
        output_root,
        paths,
        source_summary,
        source_audit,
        boundary_audit,
        checkpoint_audit,
        source_match_audit,
        decision,
        repo_root,
    )
    manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "generated_at": generated_at, "config": str(config_path), "output_root": str(output_root), "artifacts": {key: str(path) for key, path in paths.items()}, "summary_status": summary["status"], "next_required_change": summary["next_required_change"]}
    rejection_report = {"schema_version": REJECTION_REPORT_SCHEMA_VERSION, "status": decision["status"], "reason_codes": decision["reason_codes"], "next_required_change": decision["next_required_change"], "source_post_training_audit_passed": source_audit["source_post_training_audit_passed"], "checkpoint_loaded": checkpoint_audit["checkpoint_loaded"], "source_match_audit_passed": source_match_audit["source_match_audit_passed"]}
    _write_jsonl(paths["results"], replay_rows)
    write_json(paths["source_match_audit"], source_match_audit)
    write_json(paths["checkpoint_load_audit"], checkpoint_audit)
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
    for key in ("source_post_training_root", "source_controlled_training_root"):
        if not isinstance(normalized.get(key), str) or not normalized[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    for key in (
        "seed",
        "candidate_feature_count",
        "edge_feature_count",
        "memory_feature_count",
        "context_feature_count",
        "missing_indicator_count",
        "hidden_dim",
        "message_passing_layers",
        "candidate_count",
        "evaluation_case_count",
    ):
        normalized[key] = _int_value(normalized.get(key), key)
    for key in ("candidate_feature_count", "edge_feature_count", "memory_feature_count", "context_feature_count", "hidden_dim", "message_passing_layers", "candidate_count", "evaluation_case_count"):
        if normalized[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if normalized["missing_indicator_count"] < 0:
        raise ConfigError("missing_indicator_count must be non-negative")
    for key in ("source_match_probability_tolerance", "source_match_logit_tolerance", "source_match_loss_tolerance"):
        normalized[key] = _float_value(normalized.get(key), key)
        if normalized[key] < 0.0:
            raise ConfigError(f"{key} must be non-negative")
    return normalized


def _source_audit(source_summary: dict[str, Any] | None, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_codes: list[str] = []
    if not isinstance(source_summary, dict):
        reason_codes.append("missing_post_training_summary")
    else:
        if source_summary.get("status") != "passed":
            reason_codes.append("post_training_not_passed")
        if source_summary.get("next_required_change") != SOURCE_PASS_NEXT_REQUIRED_CHANGE:
            reason_codes.append("post_training_wrong_next_required_change")
        if source_summary.get("checkpoint_loaded") is not True:
            reason_codes.append("post_training_checkpoint_not_loaded")
        if source_summary.get("post_training_offline_evaluation_passed") is not True:
            reason_codes.append("post_training_evaluation_not_passed")
    if not source_rows:
        reason_codes.append("missing_post_training_results")
    return {
        "schema_version": "xunce-shadow-replay-source-audit/v1",
        "source_status": source_summary.get("status") if isinstance(source_summary, dict) else None,
        "source_next_required_change": source_summary.get("next_required_change") if isinstance(source_summary, dict) else None,
        "source_result_count": len(source_rows),
        "reason_codes": unique_sorted(reason_codes),
        "source_post_training_audit_passed": not reason_codes,
    }


def _boundary_audit(source_summary: dict[str, Any] | None) -> dict[str, Any]:
    violations: list[str] = []
    observed: dict[str, bool] = {}
    if isinstance(source_summary, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(source_summary.get(field, False))
            observed[field] = value
            if value:
                violations.append(field)
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "observed_source_boundary_fields": observed,
        "violating_source_boundary_fields": unique_sorted(violations),
        "source_boundary_audit_passed": not violations,
        "boundary_audit_passed": not violations,
        **_closed_boundary_fields(),
    }


def _load_checkpoint(checkpoint_path: Path, config: dict[str, Any]) -> tuple[dict[str, Any], XunceFullNetworkV1 | None]:
    if not checkpoint_path.is_file():
        return _checkpoint_audit(checkpoint_path, ["missing_research_checkpoint"], loaded=False), None
    model = _build_model(config)
    reason_codes: list[str] = []
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # pragma: no cover
        return _checkpoint_audit(checkpoint_path, [f"invalid_research_checkpoint:{type(exc).__name__}"], loaded=False), None
    if not isinstance(payload, dict):
        reason_codes.append("invalid_research_checkpoint")
    state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        reason_codes.append("checkpoint_state_dict_missing")
    if not reason_codes and isinstance(state_dict, dict):
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:  # pragma: no cover
            reason_codes.append(f"checkpoint_state_dict_load_failed:{type(exc).__name__}")
    loaded = not reason_codes
    return _checkpoint_audit(checkpoint_path, reason_codes, loaded=loaded), model if loaded else None


def _checkpoint_audit(checkpoint_path: Path, reason_codes: list[str], *, loaded: bool) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_LOAD_AUDIT_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_loaded": loaded,
        "checkpoint_read_only": True,
        "reason_codes": unique_sorted(reason_codes),
    }


def _replay_rows(config: dict[str, Any], model: XunceFullNetworkV1) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    target_action = torch.tensor([0], dtype=torch.long)
    for case_index in range(config["evaluation_case_count"]):
        batch = _synthetic_batch(config)
        with torch.no_grad():
            output = model(**batch)
            loss = F.cross_entropy(output.masked_logits, target_action)
            probability = output.action_probs[0, 0]
            logit = output.logits[0, 0]
        rows.append(
            {
                "schema_version": RESULT_ROW_SCHEMA_VERSION,
                "case_id": f"contract-case-{case_index:03d}",
                "target_action": 0,
                "trained_loss": float(loss.item()),
                "trained_target_probability": float(probability.item()),
                "trained_target_logit": float(logit.item()),
                "finite_outputs": bool(torch.isfinite(output.logits).all() and torch.isfinite(output.value).all()),
            }
        )
    return rows


def _source_match_audit(source_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    source_by_id = {str(row.get("case_id")): row for row in source_rows}
    mismatches: list[dict[str, Any]] = []
    max_probability_delta = 0.0
    max_logit_delta = 0.0
    max_loss_delta = 0.0
    match_count = 0
    for replay in replay_rows:
        case_id = str(replay["case_id"])
        source = source_by_id.get(case_id)
        if not source:
            mismatches.append({"case_id": case_id, "reason": "missing_source_case"})
            continue
        probability_delta = abs(float(source.get("trained_target_probability", float("nan"))) - replay["trained_target_probability"])
        logit_delta = abs(float(source.get("trained_target_logit", float("nan"))) - replay["trained_target_logit"])
        loss_delta = abs(float(source.get("trained_loss", float("nan"))) - replay["trained_loss"])
        max_probability_delta = max(max_probability_delta, probability_delta)
        max_logit_delta = max(max_logit_delta, logit_delta)
        max_loss_delta = max(max_loss_delta, loss_delta)
        finite_outputs_match = bool(source.get("finite_outputs", False)) is True and replay["finite_outputs"] is True
        case_mismatch = (
            not math.isfinite(probability_delta)
            or not math.isfinite(logit_delta)
            or not math.isfinite(loss_delta)
            or probability_delta > config["source_match_probability_tolerance"]
            or logit_delta > config["source_match_logit_tolerance"]
            or loss_delta > config["source_match_loss_tolerance"]
            or not finite_outputs_match
        )
        if case_mismatch:
            mismatches.append(
                {
                    "case_id": case_id,
                    "probability_delta": probability_delta,
                    "logit_delta": logit_delta,
                    "loss_delta": loss_delta,
                    "finite_outputs_match": finite_outputs_match,
                }
            )
        else:
            match_count += 1
    reason_codes = ["shadow_replay_source_mismatch"] if mismatches else []
    return {
        "schema_version": SOURCE_MATCH_AUDIT_SCHEMA_VERSION,
        "source_case_count": len(source_rows),
        "replay_case_count": len(replay_rows),
        "scenario_match_count": match_count,
        "scenario_mismatch_count": len(mismatches),
        "max_replay_probability_delta": max_probability_delta,
        "max_replay_logit_delta": max_logit_delta,
        "max_replay_loss_delta": max_loss_delta,
        "mismatches": mismatches,
        "reason_codes": reason_codes,
        "source_match_audit_passed": not mismatches,
        "replay_determinism_passed": not mismatches,
    }


def _empty_source_match_audit() -> dict[str, Any]:
    return {
        "schema_version": SOURCE_MATCH_AUDIT_SCHEMA_VERSION,
        "source_case_count": 0,
        "replay_case_count": 0,
        "scenario_match_count": 0,
        "scenario_mismatch_count": 0,
        "max_replay_probability_delta": None,
        "max_replay_logit_delta": None,
        "max_replay_loss_delta": None,
        "mismatches": [],
        "reason_codes": [],
        "source_match_audit_passed": False,
        "replay_determinism_passed": False,
    }


def _decision(source_audit: dict[str, Any], boundary_audit: dict[str, Any], checkpoint_audit: dict[str, Any], source_match_audit: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    reason_codes.extend(source_audit["reason_codes"])
    if not boundary_audit["source_boundary_audit_passed"]:
        reason_codes.append("shadow_replay_source_boundary_violation")
    reason_codes.extend(checkpoint_audit["reason_codes"])
    if checkpoint_audit["checkpoint_loaded"]:
        reason_codes.extend(source_match_audit["reason_codes"])
    reason_codes = unique_sorted(reason_codes)
    if not reason_codes:
        next_required_change = PASS_NEXT_REQUIRED_CHANGE
    elif "shadow_replay_source_boundary_violation" in reason_codes:
        next_required_change = BOUNDARY_NEXT_REQUIRED_CHANGE
    elif any(code == "missing_research_checkpoint" or code.startswith("invalid_research_checkpoint") or code.startswith("checkpoint_state_dict") for code in reason_codes):
        next_required_change = FIX_CHECKPOINT_NEXT_REQUIRED_CHANGE
    elif "shadow_replay_source_mismatch" in reason_codes:
        next_required_change = FIX_DETERMINISM_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FIX_POST_TRAINING_NEXT_REQUIRED_CHANGE
    return {"status": "passed" if not reason_codes else "failed", "reason_codes": reason_codes, "next_required_change": next_required_change}


def _summary(
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    source_summary: dict[str, Any] | None,
    source_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    checkpoint_audit: dict[str, Any],
    source_match_audit: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_post_training_status": source_summary.get("status") if isinstance(source_summary, dict) else None,
        "source_post_training_next_required_change": source_summary.get("next_required_change") if isinstance(source_summary, dict) else None,
        "source_post_training_audit_passed": source_audit["source_post_training_audit_passed"],
        "checkpoint_loaded": checkpoint_audit["checkpoint_loaded"],
        "checkpoint_read_only": checkpoint_audit["checkpoint_read_only"],
        "shadow_replay_passed": decision["status"] == "passed",
        "source_match_audit_passed": source_match_audit["source_match_audit_passed"],
        "replay_determinism_passed": source_match_audit["replay_determinism_passed"],
        "scenario_match_count": source_match_audit["scenario_match_count"],
        "scenario_mismatch_count": source_match_audit["scenario_mismatch_count"],
        "max_replay_probability_delta": source_match_audit["max_replay_probability_delta"],
        "max_replay_logit_delta": source_match_audit["max_replay_logit_delta"],
        "max_replay_loss_delta": source_match_audit["max_replay_loss_delta"],
        "boundary_audit_passed": boundary_audit["boundary_audit_passed"],
        "source_boundary_audit_passed": boundary_audit["source_boundary_audit_passed"],
        "next_required_change": decision["next_required_change"],
        "git_provenance": {"current": git_snapshot(repo_root)},
    }
    summary.update(_closed_boundary_fields())
    return summary


def _build_model(config: dict[str, Any]) -> XunceFullNetworkV1:
    torch.manual_seed(config["seed"])
    return XunceFullNetworkV1(
        candidate_feature_count=config["candidate_feature_count"],
        edge_feature_count=config["edge_feature_count"],
        memory_feature_count=config["memory_feature_count"],
        context_feature_count=config["context_feature_count"],
        missing_indicator_count=config["missing_indicator_count"],
        hidden_dim=config["hidden_dim"],
        message_passing_layers=config["message_passing_layers"],
        dropout=0.0,
    )


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "results": output_root / RESULTS_FILE,
        "source_match_audit": output_root / SOURCE_MATCH_AUDIT_FILE,
        "checkpoint_load_audit": output_root / CHECKPOINT_LOAD_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _closed_boundary_fields() -> dict[str, bool | float]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update(
        {
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "runs_new_training_update": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
    )
    return fields


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Xunce Shadow Replay Validation v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- source_match_audit_passed: `{summary['source_match_audit_passed']}`",
            f"- replay_determinism_passed: `{summary['replay_determinism_passed']}`",
            f"- publishes_checkpoint: `{summary['publishes_checkpoint']}`",
            f"- connects_real_executor: `{summary['connects_real_executor']}`",
            f"- starts_online_canary: `{summary['starts_online_canary']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            "",
            "This is offline shadow/replay validation only. It does not publish, install, connect an executor, or start canary traffic.",
            "",
        ]
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _int_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _float_value(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
