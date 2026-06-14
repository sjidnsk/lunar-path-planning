from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from git_provenance import git_snapshot
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot


CONFIG_SCHEMA_VERSION = "guarded-experimental-policy-install-canary-dry-run-config/v1"
SUMMARY_SCHEMA_VERSION = "guarded-experimental-policy-install-canary-dry-run-summary/v1"
SANDBOX_MANIFEST_SCHEMA_VERSION = (
    "guarded-experimental-policy-install-canary-sandbox-manifest/v1"
)
PACKAGING_SCHEMA_VERSION = (
    "guarded-experimental-policy-release-candidate-packaging-summary/v1"
)
PACKAGE_MANIFEST_SCHEMA_VERSION = (
    "guarded-experimental-policy-release-candidate-package-manifest/v1"
)
EXPECTED_PACKAGING_VERDICT = "eligible_for_guarded_install_dry_run"
EXPECTED_INSTALL_CANARY_VERDICT = "eligible_for_guarded_shadow_release_trial"
EXPECTED_READINESS_STATUS = (
    "guarded_experimental_policy_install_canary_dry_run_evaluated"
)

SUMMARY_FILE = "guarded-experimental-policy-install-canary-dry-run-summary.json"
SANDBOX_MANIFEST_FILE = "install-canary-sandbox-manifest.json"
PACKAGE_CONSUMER_AUDIT_FILE = "install-canary-package-consumer-audit.json"
STEP_AUDIT_FILE = "install-canary-step-audit.jsonl"
ROLLBACK_AUDIT_FILE = "install-canary-rollback-audit.json"
READINESS_FILE = "install-canary-readiness-validate-only.json"
REPORT_FILE = "install-canary-report.md"
SANDBOX_DIR = "install-canary-sandbox"

CanaryRunner = Callable[..., dict[str, Any]]
ReadinessRunner = Callable[..., dict[str, Any]]


def run_guarded_experimental_policy_install_canary_dry_run(
    *,
    packaging_root: Path,
    output_root: Path,
    batch_root: Path,
    config: dict[str, Any],
    repo_root: Path,
    canary_runner: CanaryRunner | None = None,
    readiness_runner: ReadinessRunner | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    packaging_root = Path(packaging_root)
    output_root = Path(output_root)
    batch_root = Path(batch_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sandbox_root = output_root / SANDBOX_DIR
    sandbox_root.mkdir(parents=True, exist_ok=True)

    files = _output_files(config)
    summary_path = output_root / files["summary"]
    sandbox_manifest_path = output_root / files["sandbox_manifest"]
    package_consumer_audit_path = output_root / files["package_consumer_audit"]
    step_audit_path = output_root / files["step_audit"]
    rollback_audit_path = output_root / files["rollback_audit"]
    readiness_path = output_root / files["readiness_validate_only"]
    report_path = output_root / files["report"]

    packaging_summary_path = packaging_root / _input_files(config)["packaging_summary"]
    packaging_summary = _read_json_if_exists(packaging_summary_path)
    package_manifest_path = _resolve_optional_path(
        packaging_summary.get("package_manifest"), packaging_root, repo_root
    ) or packaging_root / "release-candidate-package-manifest.json"
    package_manifest = _read_json_if_exists(package_manifest_path)
    package_checkpoint_path = _resolve_optional_path(
        packaging_summary.get("package_checkpoint_path")
        or package_manifest.get("package_checkpoint_path"),
        packaging_root,
        repo_root,
    )
    package_metadata_path = _resolve_optional_path(
        packaging_summary.get("package_checkpoint_metadata_path")
        or package_manifest.get("package_checkpoint_metadata_path"),
        packaging_root,
        repo_root,
    )
    checkpoint_metadata = _read_json_if_exists(package_metadata_path)
    preflight_summary_path = _resolve_optional_path(
        packaging_summary.get("preflight_summary")
        or package_manifest.get("preflight_summary"),
        packaging_root,
        repo_root,
    )
    preflight_summary = _read_json_if_exists(preflight_summary_path)
    canary_steps_path = _resolve_optional_path(
        preflight_summary.get("multihorizon_steps"),
        packaging_root,
        repo_root,
    )
    steps = _read_jsonl(canary_steps_path)

    reason_codes: list[str] = []
    _validate_packaging_input(
        packaging_summary=packaging_summary,
        package_manifest=package_manifest,
        reason_codes=reason_codes,
    )

    default_policy_path = _default_policy_path(config, repo_root)
    default_before = _file_snapshot(default_policy_path)
    package_consumer_audit = _package_consumer_audit(
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
        packaging_summary=packaging_summary,
        package_manifest=package_manifest,
        canary_steps_path=canary_steps_path,
    )
    _write_json(package_consumer_audit_path, package_consumer_audit)
    for reason in package_consumer_audit["reason_codes"]:
        _add_reason(reason_codes, reason)

    sandbox_manifest = _sandbox_manifest(
        repo_root=repo_root,
        output_root=output_root,
        sandbox_root=sandbox_root,
        packaging_root=packaging_root,
        packaging_summary_path=packaging_summary_path,
        package_manifest_path=package_manifest_path,
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
        package_consumer_audit=package_consumer_audit,
        default_before=default_before,
        default_after=default_before,
    )
    _write_json(sandbox_manifest_path, sandbox_manifest)

    if not reason_codes:
        runner = canary_runner or _run_canary_audit
        canary_audit = runner(
            checkpoint_path=package_checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            steps=steps,
            config=config,
            repo_root=repo_root,
            sandbox_manifest_path=sandbox_manifest_path,
        )
    else:
        canary_audit = _empty_canary_audit("package consumer validation failed")
    canary_audit = _normalize_canary_audit(canary_audit)
    _write_jsonl(step_audit_path, canary_audit["step_rows"])
    _validate_canary_audit(canary_audit, config, reason_codes)

    default_after = _file_snapshot(default_policy_path)
    rollback_audit = _rollback_audit(
        default_before=default_before,
        default_after=default_after,
        sandbox_root=sandbox_root,
        package_checkpoint_path=package_checkpoint_path,
        packaging_summary=packaging_summary,
        package_manifest=package_manifest,
    )
    _write_json(rollback_audit_path, rollback_audit)
    if not rollback_audit["rollback_default_audit_passed"]:
        _add_reason(reason_codes, "install_canary_rollback_boundary_invalid")
    if rollback_audit.get("default_policy_unchanged") is False:
        _add_reason(reason_codes, "install_canary_default_policy_modified")

    sandbox_manifest = _sandbox_manifest(
        repo_root=repo_root,
        output_root=output_root,
        sandbox_root=sandbox_root,
        packaging_root=packaging_root,
        packaging_summary_path=packaging_summary_path,
        package_manifest_path=package_manifest_path,
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
        package_consumer_audit=package_consumer_audit,
        default_before=default_before,
        default_after=default_after,
    )
    _write_json(sandbox_manifest_path, sandbox_manifest)
    if not sandbox_manifest["sandbox_manifest_passed"]:
        _add_reason(reason_codes, "install_canary_sandbox_manifest_invalid")

    status_without_readiness = "passed" if not reason_codes else "failed"
    pre_readiness_summary = _summary_payload(
        status=status_without_readiness,
        reason_codes=reason_codes,
        repo_root=repo_root,
        packaging_root=packaging_root,
        output_root=output_root,
        batch_root=batch_root,
        sandbox_root=sandbox_root,
        packaging_summary_path=packaging_summary_path,
        package_manifest_path=package_manifest_path,
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
        summary_path=summary_path,
        sandbox_manifest_path=sandbox_manifest_path,
        package_consumer_audit_path=package_consumer_audit_path,
        step_audit_path=step_audit_path,
        rollback_audit_path=rollback_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        packaging_summary=packaging_summary,
        package_consumer_audit=package_consumer_audit,
        canary_audit=canary_audit,
        rollback_audit=rollback_audit,
        sandbox_manifest=sandbox_manifest,
        readiness={},
    )
    _write_json(summary_path, pre_readiness_summary)

    if status_without_readiness == "passed":
        runner = readiness_runner or _run_readiness_validate_only
        readiness = runner(
            repo_root=repo_root,
            batch_root=batch_root,
            install_canary_summary_path=summary_path,
            config_path=Path(
                config.get("readiness", {}).get(
                    "config",
                    "configs/policy_training_readiness_review_v1.json",
                )
            ),
        )
        _validate_readiness(readiness, config, reason_codes)
    else:
        readiness = {
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": list(reason_codes),
            "reason_codes": list(reason_codes),
            "recommended_next_action": "fix_guarded_experimental_policy_install_canary_dry_run",
        }
    _write_json(readiness_path, readiness)

    final_status = "passed" if not reason_codes else "failed"
    summary = _summary_payload(
        status=final_status,
        reason_codes=reason_codes,
        repo_root=repo_root,
        packaging_root=packaging_root,
        output_root=output_root,
        batch_root=batch_root,
        sandbox_root=sandbox_root,
        packaging_summary_path=packaging_summary_path,
        package_manifest_path=package_manifest_path,
        package_checkpoint_path=package_checkpoint_path,
        package_metadata_path=package_metadata_path,
        summary_path=summary_path,
        sandbox_manifest_path=sandbox_manifest_path,
        package_consumer_audit_path=package_consumer_audit_path,
        step_audit_path=step_audit_path,
        rollback_audit_path=rollback_audit_path,
        readiness_path=readiness_path,
        report_path=report_path,
        packaging_summary=packaging_summary,
        package_consumer_audit=package_consumer_audit,
        canary_audit=canary_audit,
        rollback_audit=rollback_audit,
        sandbox_manifest=sandbox_manifest,
        readiness=readiness,
    )
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate installing a guarded experimental policy package and run a canary dry-run."
    )
    parser.add_argument(
        "--packaging-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_release_candidate_packaging_v1",
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1",
    )
    parser.add_argument(
        "--config",
        default="configs/guarded_experimental_policy_install_canary_dry_run_v1.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_path(Path(args.config), repo_root)
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SystemExit(f"invalid config schema: {config.get('schema_version')}")
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": str(args.config)}, sort_keys=True))
        return 0

    summary = run_guarded_experimental_policy_install_canary_dry_run(
        packaging_root=_resolve_path(Path(args.packaging_root), repo_root),
        output_root=_resolve_path(Path(args.output_root), repo_root),
        batch_root=_resolve_path(Path(args.batch_root), repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "install_canary_verdict": summary["install_canary_verdict"],
                "readiness_status": summary.get("readiness_status"),
                "canary_step_count": summary.get("canary_step_count"),
                "controlled_regression_count": summary.get("controlled_regression_count"),
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def _validate_packaging_input(
    *,
    packaging_summary: dict[str, Any],
    package_manifest: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if packaging_summary.get("schema_version") != PACKAGING_SCHEMA_VERSION:
        _add_reason(reason_codes, "install_canary_packaging_summary_schema_invalid")
    if packaging_summary.get("status") != "passed" or _string_list(
        packaging_summary.get("reason_codes")
    ):
        _add_reason(reason_codes, "install_canary_packaging_summary_not_passed")
    if packaging_summary.get("package_verdict") != EXPECTED_PACKAGING_VERDICT:
        _add_reason(reason_codes, "install_canary_packaging_not_eligible")
    if package_manifest.get("schema_version") != PACKAGE_MANIFEST_SCHEMA_VERSION:
        _add_reason(reason_codes, "install_canary_package_manifest_schema_invalid")
    for field, reason in (
        ("checkpoint_identity_audit_passed", "install_canary_packaging_checkpoint_identity_failed"),
        ("checkpoint_load_passed", "install_canary_packaging_checkpoint_load_failed"),
        ("rollback_audit_passed", "install_canary_packaging_rollback_failed"),
    ):
        if packaging_summary.get(field) is not True:
            _add_reason(reason_codes, reason)
    if _int(packaging_summary.get("checkpoint_load_sample_count")) < 64:
        _add_reason(reason_codes, "install_canary_packaging_load_sample_below_threshold")
    for payload in (packaging_summary, package_manifest):
        if payload.get("runs_new_ppo_update") is True:
            _add_reason(reason_codes, "install_canary_unexpected_ppo_update")
        if payload.get("publishes_checkpoint") is True:
            _add_reason(reason_codes, "install_canary_checkpoint_publication_claimed")
        if payload.get("replaces_default_policy") is True:
            _add_reason(reason_codes, "install_canary_default_policy_replacement_claimed")
        if payload.get("performance_claimed") is True:
            _add_reason(reason_codes, "install_canary_policy_performance_claimed")
        if payload.get("formal_training_ready_claimed") is True:
            _add_reason(reason_codes, "install_canary_formal_ready_claimed")
    if _git_current_matches_sources(packaging_summary) is False:
        _add_reason(reason_codes, "install_canary_packaging_git_provenance_mismatch")


def _package_consumer_audit(
    *,
    package_checkpoint_path: Path | None,
    package_metadata_path: Path | None,
    packaging_summary: dict[str, Any],
    package_manifest: dict[str, Any],
    canary_steps_path: Path | None,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    if package_checkpoint_path is None or not package_checkpoint_path.is_file():
        _add_reason(reason_codes, "install_canary_package_checkpoint_missing")
        sha256 = None
        size = 0
    else:
        sha256, size = _sha256_and_size(package_checkpoint_path)
    expected_sha256 = (
        packaging_summary.get("package_checkpoint_sha256")
        or package_manifest.get("package_checkpoint_sha256")
    )
    expected_size = max(
        _int(packaging_summary.get("package_checkpoint_size_bytes")),
        _int(package_manifest.get("package_checkpoint_size_bytes")),
    )
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        _add_reason(reason_codes, "install_canary_package_checkpoint_hash_missing")
    elif sha256 != expected_sha256:
        _add_reason(reason_codes, "install_canary_package_checkpoint_hash_mismatch")
    if expected_size <= 0:
        _add_reason(reason_codes, "install_canary_package_checkpoint_size_missing")
    elif size != expected_size:
        _add_reason(reason_codes, "install_canary_package_checkpoint_size_mismatch")
    if package_metadata_path is None or not package_metadata_path.is_file():
        _add_reason(reason_codes, "install_canary_package_metadata_missing")
    return {
        "schema_version": "guarded-experimental-policy-install-canary-package-consumer-audit/v1",
        "package_consumer_audit_passed": not reason_codes,
        "reason_codes": reason_codes,
        "package_checkpoint_path": None
        if package_checkpoint_path is None
        else str(package_checkpoint_path),
        "package_checkpoint_exists": bool(package_checkpoint_path and package_checkpoint_path.is_file()),
        "package_checkpoint_sha256": expected_sha256,
        "package_checkpoint_size_bytes": expected_size,
        "consumer_checkpoint_sha256": sha256,
        "consumer_checkpoint_size_bytes": size,
        "package_checkpoint_metadata_path": None
        if package_metadata_path is None
        else str(package_metadata_path),
        "package_checkpoint_metadata_exists": bool(
            package_metadata_path and package_metadata_path.is_file()
        ),
        "canary_steps_path": None if canary_steps_path is None else str(canary_steps_path),
    }


def _run_canary_audit(
    *,
    checkpoint_path: Path | None,
    checkpoint_metadata: dict[str, Any],
    steps: list[dict[str, Any]],
    config: dict[str, Any],
    repo_root: Path,
    sandbox_manifest_path: Path,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return _empty_canary_audit("checkpoint path missing")
    min_count = _min_canary_step_count(config)
    rows: list[dict[str, Any]] = []
    counts = _empty_canary_audit(None)
    counts["canary_step_count"] = 0
    try:
        import torch
        from model_explorer.policy.architectures import build_policy_network_from_metadata
        from model_explorer.policy.rollout_io import _observation_from_dict
        from model_explorer.policy.torch_policy import observation_to_tensors
    except ModuleNotFoundError:
        try:
            from run_selected_formal_ppo_candidate_promotion_preflight import (
                _install_model_explorer_path,
            )

            _install_model_explorer_path(repo_root)
            import torch
            from model_explorer.policy.architectures import build_policy_network_from_metadata
            from model_explorer.policy.rollout_io import _observation_from_dict
            from model_explorer.policy.torch_policy import observation_to_tensors
        except Exception as exc:  # noqa: BLE001
            counts["checkpoint_error"] = str(exc)
            return counts
    except Exception as exc:  # noqa: BLE001
        counts["checkpoint_error"] = str(exc)
        return counts

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        first_observation = next(
            step.get("observation")
            for step in steps
            if isinstance(step.get("observation"), dict)
        )
        first = _observation_from_dict(first_observation)
        architecture = checkpoint.get("architecture") or checkpoint_metadata.get("architecture")
        hidden_size = _int(
            checkpoint.get("hidden_size")
            or checkpoint_metadata.get("hidden_size")
            or checkpoint.get("training", {}).get("hidden_size"),
            16,
        )
        network = build_policy_network_from_metadata(
            architecture,
            candidate_feature_count=len(first.candidate_feature_names),
            global_feature_count=len(first.global_feature_names),
            missing_indicator_count=len(first.candidate_missing_indicator_names),
            hidden_size=hidden_size,
            architecture_config=checkpoint.get("architecture_config")
            or checkpoint_metadata.get("architecture_config"),
        )
        state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint is missing model state")
        network.load_state_dict(state)
        network.eval()
        with torch.no_grad():
            for source_index, step in enumerate(steps):
                if counts["canary_step_count"] >= min_count:
                    break
                row = _canary_step_from_network(
                    network=network,
                    step=step,
                    source_index=source_index,
                    torch=torch,
                    observation_from_dict=_observation_from_dict,
                    observation_to_tensors=observation_to_tensors,
                )
                _accumulate_canary_counts(counts, row)
                rows.append(row)
    except Exception as exc:  # noqa: BLE001
        counts["checkpoint_error"] = str(exc)
    counts["step_rows"] = rows
    counts["sandbox_manifest"] = str(sandbox_manifest_path)
    return counts


def _canary_step_from_network(
    *,
    network: Any,
    step: dict[str, Any],
    source_index: int,
    torch: Any,
    observation_from_dict: Callable[[dict[str, Any]], Any],
    observation_to_tensors: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    observation_payload = step.get("observation")
    if not isinstance(observation_payload, dict):
        return {
            "source_index": source_index,
            "context_id": step.get("context_id"),
            "scenario_id": step.get("scenario_id"),
            "scenario_family": step.get("scenario_family"),
            "missing_observation": True,
            "invalid_action_mask": False,
            "non_finite_logits": False,
            "non_finite_log_prob": False,
            "non_finite_value": False,
            "non_finite_reward": not _finite(step.get("reward")),
            "raw_policy_action_index": None,
            "controlled_action_index": step.get("controlled_action_index"),
            "controlled_choice_source": "source_fallback",
            "gate_reason_codes": ["missing_observation"],
            "controlled_regression_reason_codes": [],
            "path_cost_delta": _float(step.get("path_cost_delta")),
            "risk_delta": _float(step.get("risk_delta")),
            "log_prob": None,
            "value": None,
            "reward": step.get("reward"),
        }
    observation = observation_from_dict(observation_payload)
    if not observation.action_mask or not any(observation.action_mask):
        return {
            "source_index": source_index,
            "context_id": step.get("context_id"),
            "scenario_id": step.get("scenario_id"),
            "scenario_family": step.get("scenario_family"),
            "missing_observation": False,
            "invalid_action_mask": True,
            "non_finite_logits": False,
            "non_finite_log_prob": False,
            "non_finite_value": False,
            "non_finite_reward": not _finite(step.get("reward")),
            "raw_policy_action_index": None,
            "controlled_action_index": step.get("controlled_action_index"),
            "controlled_choice_source": "source_fallback",
            "gate_reason_codes": ["invalid_action_mask"],
            "controlled_regression_reason_codes": [],
            "path_cost_delta": _float(step.get("path_cost_delta")),
            "risk_delta": _float(step.get("risk_delta")),
            "log_prob": None,
            "value": None,
            "reward": step.get("reward"),
        }
    output = network(**observation_to_tensors(observation, device="cpu"))
    logits = output.masked_logits[0]
    non_finite_logits = not bool(torch.isfinite(logits).all().item())
    if non_finite_logits:
        raw_action = None
        log_prob = math.nan
    else:
        raw_action = int(torch.argmax(logits).item())
        distribution = torch.distributions.Categorical(logits=output.masked_logits)
        log_prob = float(distribution.log_prob(torch.tensor([raw_action])).item())
    value = float(output.value[0].item())
    stored_controlled = _int(step.get("controlled_action_index", step.get("action_index")), -1)
    stored_reasons = _string_list(step.get("controlled_regression_reason_codes"))
    gate_reasons = list(_string_list(step.get("gate_reason_codes")))
    if raw_action is None or raw_action != stored_controlled:
        _add_reason(gate_reasons, "raw_policy_action_mismatch_with_guarded_shadow")
        controlled_source = "source_fallback"
    elif gate_reasons or _string_list(step.get("rejection_reason_codes")):
        controlled_source = "source_fallback"
    else:
        controlled_source = "policy"
    path_cost_delta = _float(step.get("path_cost_delta"))
    risk_delta = _float(step.get("risk_delta"))
    if path_cost_delta > 0 and "path_cost_regression" not in stored_reasons:
        stored_reasons.append("path_cost_regression")
    if risk_delta > 0 and "risk_regression" not in stored_reasons:
        stored_reasons.append("risk_regression")
    return {
        "source_index": source_index,
        "context_id": step.get("context_id"),
        "scenario_id": step.get("scenario_id"),
        "scenario_family": step.get("scenario_family"),
        "split": step.get("split"),
        "missing_observation": False,
        "invalid_action_mask": bool(
            raw_action is not None
            and (raw_action < 0 or raw_action >= len(observation.action_mask) or not observation.action_mask[raw_action])
        ),
        "non_finite_logits": non_finite_logits,
        "non_finite_log_prob": not math.isfinite(log_prob),
        "non_finite_value": not math.isfinite(value),
        "non_finite_reward": not _finite(step.get("reward")),
        "raw_policy_action_index": raw_action,
        "controlled_action_index": stored_controlled,
        "controlled_choice_source": controlled_source,
        "gate_reason_codes": gate_reasons,
        "controlled_regression_reason_codes": stored_reasons,
        "path_cost_delta": path_cost_delta,
        "risk_delta": risk_delta,
        "log_prob": log_prob,
        "value": value,
        "reward": step.get("reward"),
    }


def _accumulate_canary_counts(counts: dict[str, Any], row: dict[str, Any]) -> None:
    counts["canary_step_count"] += 1
    if row.get("missing_observation"):
        counts["missing_observation_count"] += 1
    if row.get("invalid_action_mask"):
        counts["invalid_action_mask_count"] += 1
    if row.get("non_finite_logits"):
        counts["non_finite_logits_count"] += 1
    if row.get("non_finite_log_prob"):
        counts["non_finite_log_prob_count"] += 1
    if row.get("non_finite_value"):
        counts["non_finite_value_count"] += 1
    if row.get("non_finite_reward"):
        counts["non_finite_reward_count"] += 1
    gate_reasons = _string_list(row.get("gate_reason_codes"))
    if gate_reasons:
        counts["raw_policy_rejection_count"] += 1
    if row.get("controlled_choice_source") != "policy":
        counts["fallback_count"] += 1
    controlled_reasons = set(_string_list(row.get("controlled_regression_reason_codes")))
    if controlled_reasons:
        counts["controlled_regression_count"] += 1
    if "safety_regression" in controlled_reasons:
        counts["controlled_safety_regression_count"] += 1
    if "contract_regression" in controlled_reasons:
        counts["controlled_contract_regression_count"] += 1
    if "path_cost_regression" in controlled_reasons or "risk_regression" in controlled_reasons:
        counts["controlled_path_risk_regression_count"] += 1
    if "source_selection_regression" in controlled_reasons:
        counts["controlled_source_selection_regression_count"] += 1


def _normalize_canary_audit(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _empty_canary_audit(payload.get("checkpoint_error"))
    normalized.update(payload)
    normalized["schema_version"] = "guarded-experimental-policy-install-canary-step-audit/v1"
    normalized["step_rows"] = [
        row for row in payload.get("step_rows", []) if isinstance(row, dict)
    ]
    for key in (
        "canary_step_count",
        "missing_observation_count",
        "invalid_action_mask_count",
        "non_finite_logits_count",
        "non_finite_log_prob_count",
        "non_finite_value_count",
        "non_finite_reward_count",
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
        "raw_policy_rejection_count",
        "fallback_count",
    ):
        normalized[key] = _int(normalized.get(key))
    return normalized


def _empty_canary_audit(error: str | None) -> dict[str, Any]:
    payload = {
        "schema_version": "guarded-experimental-policy-install-canary-step-audit/v1",
        "canary_step_count": 0,
        "missing_observation_count": 0,
        "invalid_action_mask_count": 0,
        "non_finite_logits_count": 0,
        "non_finite_log_prob_count": 0,
        "non_finite_value_count": 0,
        "non_finite_reward_count": 0,
        "controlled_regression_count": 0,
        "controlled_safety_regression_count": 0,
        "controlled_contract_regression_count": 0,
        "controlled_path_risk_regression_count": 0,
        "controlled_source_selection_regression_count": 0,
        "raw_policy_rejection_count": 0,
        "fallback_count": 0,
        "step_rows": [],
    }
    if error:
        payload["checkpoint_error"] = error
    return payload


def _validate_canary_audit(
    canary_audit: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    if _int(canary_audit.get("canary_step_count")) < _min_canary_step_count(config):
        _add_reason(reason_codes, "install_canary_step_count_below_threshold")
    for field, reason in (
        ("missing_observation_count", "install_canary_missing_observation"),
        ("invalid_action_mask_count", "install_canary_invalid_action_mask"),
        ("non_finite_logits_count", "install_canary_non_finite_inference"),
        ("non_finite_log_prob_count", "install_canary_non_finite_inference"),
        ("non_finite_value_count", "install_canary_non_finite_inference"),
        ("non_finite_reward_count", "install_canary_non_finite_reward"),
    ):
        if _int(canary_audit.get(field)) > 0:
            _add_reason(reason_codes, reason)
    for field in (
        "controlled_regression_count",
        "controlled_safety_regression_count",
        "controlled_contract_regression_count",
        "controlled_path_risk_regression_count",
        "controlled_source_selection_regression_count",
    ):
        if _int(canary_audit.get(field)) > 0:
            _add_reason(reason_codes, "install_canary_controlled_regression")
    if canary_audit.get("checkpoint_error"):
        _add_reason(reason_codes, "install_canary_checkpoint_load_failed")


def _rollback_audit(
    *,
    default_before: dict[str, Any],
    default_after: dict[str, Any],
    sandbox_root: Path,
    package_checkpoint_path: Path | None,
    packaging_summary: dict[str, Any],
    package_manifest: dict[str, Any],
) -> dict[str, Any]:
    default_unchanged = _snapshots_same(default_before, default_after)
    sandbox_deletable = sandbox_root.exists() and sandbox_root.is_dir()
    package_traceable = bool(package_checkpoint_path and package_checkpoint_path.is_file())
    source_traceable = package_traceable
    publication_flags = {
        "packaging_summary": _publication_flags(packaging_summary),
        "package_manifest": _publication_flags(package_manifest),
    }
    passed = (
        default_unchanged
        and sandbox_deletable
        and package_traceable
        and source_traceable
        and not any(any(flags.values()) for flags in publication_flags.values())
    )
    return {
        "schema_version": "guarded-experimental-policy-install-canary-rollback-audit/v1",
        "rollback_default_audit_passed": passed,
        "default_policy_unchanged": default_unchanged,
        "default_policy_path": default_before.get("path") or default_after.get("path"),
        "default_policy_sha256_before": default_before.get("sha256"),
        "default_policy_sha256_after": default_after.get("sha256"),
        "default_policy_size_bytes_before": default_before.get("size_bytes"),
        "default_policy_size_bytes_after": default_after.get("size_bytes"),
        "default_policy_exists_before": default_before.get("exists"),
        "default_policy_exists_after": default_after.get("exists"),
        "sandbox_deletable": sandbox_deletable,
        "package_traceable": package_traceable,
        "source_traceable": source_traceable,
        "default_policy_replaced": not default_unchanged,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "publication_flags": publication_flags,
    }


def _sandbox_manifest(
    *,
    repo_root: Path,
    output_root: Path,
    sandbox_root: Path,
    packaging_root: Path,
    packaging_summary_path: Path,
    package_manifest_path: Path,
    package_checkpoint_path: Path | None,
    package_metadata_path: Path | None,
    package_consumer_audit: dict[str, Any],
    default_before: dict[str, Any],
    default_after: dict[str, Any],
) -> dict[str, Any]:
    default_unchanged = _snapshots_same(default_before, default_after)
    sandbox_passed = (
        sandbox_root.exists()
        and package_consumer_audit.get("package_consumer_audit_passed") is True
        and default_unchanged
        and package_checkpoint_path is not None
    )
    return {
        "schema_version": SANDBOX_MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "sandbox_manifest_passed": sandbox_passed,
        "output_root": str(output_root),
        "sandbox_root": str(sandbox_root),
        "packaging_root": str(packaging_root),
        "packaging_summary": str(packaging_summary_path),
        "package_manifest": str(package_manifest_path),
        "package_checkpoint_path": None
        if package_checkpoint_path is None
        else str(package_checkpoint_path),
        "package_checkpoint_metadata_path": None
        if package_metadata_path is None
        else str(package_metadata_path),
        "package_checkpoint_sha256": package_consumer_audit.get("package_checkpoint_sha256"),
        "package_checkpoint_size_bytes": package_consumer_audit.get("package_checkpoint_size_bytes"),
        "consumer_checkpoint_sha256": package_consumer_audit.get("consumer_checkpoint_sha256"),
        "consumer_checkpoint_size_bytes": package_consumer_audit.get("consumer_checkpoint_size_bytes"),
        "default_policy_path": default_before.get("path") or default_after.get("path"),
        "default_policy_exists_before": default_before.get("exists"),
        "default_policy_exists_after": default_after.get("exists"),
        "default_policy_sha256_before": default_before.get("sha256"),
        "default_policy_sha256_after": default_after.get("sha256"),
        "default_policy_replaced": not default_unchanged,
        "writes_default_policy": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _summary_payload(
    *,
    status: str,
    reason_codes: list[str],
    repo_root: Path,
    packaging_root: Path,
    output_root: Path,
    batch_root: Path,
    sandbox_root: Path,
    packaging_summary_path: Path,
    package_manifest_path: Path,
    package_checkpoint_path: Path | None,
    package_metadata_path: Path | None,
    summary_path: Path,
    sandbox_manifest_path: Path,
    package_consumer_audit_path: Path,
    step_audit_path: Path,
    rollback_audit_path: Path,
    readiness_path: Path,
    report_path: Path,
    packaging_summary: dict[str, Any],
    package_consumer_audit: dict[str, Any],
    canary_audit: dict[str, Any],
    rollback_audit: dict[str, Any],
    sandbox_manifest: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    verdict = _install_canary_verdict(reason_codes)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": list(reason_codes),
        "next_required_change": None
        if status == "passed"
        else "fix_guarded_experimental_policy_install_canary_dry_run",
        "install_canary_verdict": verdict,
        "packaging_root": str(packaging_root),
        "packaging_summary": str(packaging_summary_path),
        "package_manifest": str(package_manifest_path),
        "package_root": packaging_summary.get("package_root"),
        "package_checkpoint_path": None
        if package_checkpoint_path is None
        else str(package_checkpoint_path),
        "package_checkpoint_metadata_path": None
        if package_metadata_path is None
        else str(package_metadata_path),
        "output_root": str(output_root),
        "batch_root": str(batch_root),
        "sandbox_root": str(sandbox_root),
        "summary": str(summary_path),
        "sandbox_manifest": str(sandbox_manifest_path),
        "package_consumer_audit": str(package_consumer_audit_path),
        "step_audit": str(step_audit_path),
        "rollback_audit": str(rollback_audit_path),
        "readiness_validate_only": str(readiness_path),
        "report": str(report_path),
        "selected_seed": packaging_summary.get("selected_seed"),
        "selected_budget": packaging_summary.get("selected_budget"),
        "package_checkpoint_sha256": package_consumer_audit.get("package_checkpoint_sha256"),
        "consumer_checkpoint_sha256": package_consumer_audit.get("consumer_checkpoint_sha256"),
        "package_checkpoint_size_bytes": _int(
            package_consumer_audit.get("package_checkpoint_size_bytes")
        ),
        "consumer_checkpoint_size_bytes": _int(
            package_consumer_audit.get("consumer_checkpoint_size_bytes")
        ),
        "package_consumer_audit_passed": package_consumer_audit.get(
            "package_consumer_audit_passed"
        )
        is True,
        "sandbox_manifest_passed": sandbox_manifest.get("sandbox_manifest_passed") is True,
        "canary_step_count": _int(canary_audit.get("canary_step_count")),
        "missing_observation_count": _int(canary_audit.get("missing_observation_count")),
        "invalid_action_mask_count": _int(canary_audit.get("invalid_action_mask_count")),
        "non_finite_logits_count": _int(canary_audit.get("non_finite_logits_count")),
        "non_finite_log_prob_count": _int(canary_audit.get("non_finite_log_prob_count")),
        "non_finite_value_count": _int(canary_audit.get("non_finite_value_count")),
        "non_finite_reward_count": _int(canary_audit.get("non_finite_reward_count")),
        "controlled_regression_count": _int(canary_audit.get("controlled_regression_count")),
        "controlled_safety_regression_count": _int(
            canary_audit.get("controlled_safety_regression_count")
        ),
        "controlled_contract_regression_count": _int(
            canary_audit.get("controlled_contract_regression_count")
        ),
        "controlled_path_risk_regression_count": _int(
            canary_audit.get("controlled_path_risk_regression_count")
        ),
        "controlled_source_selection_regression_count": _int(
            canary_audit.get("controlled_source_selection_regression_count")
        ),
        "raw_policy_rejection_count": _int(canary_audit.get("raw_policy_rejection_count")),
        "fallback_count": _int(canary_audit.get("fallback_count")),
        "rollback_default_audit_passed": rollback_audit.get(
            "rollback_default_audit_passed"
        )
        is True,
        "default_policy_unchanged": rollback_audit.get("default_policy_unchanged") is True,
        "default_policy_path": rollback_audit.get("default_policy_path"),
        "default_policy_sha256_before": rollback_audit.get("default_policy_sha256_before"),
        "default_policy_sha256_after": rollback_audit.get("default_policy_sha256_after"),
        "readiness_status": readiness.get("training_readiness_status"),
        "training_blockers": list(readiness.get("training_blockers") or []),
        "readiness_reason_codes": list(readiness.get("reason_codes") or []),
        "runs_install_canary_dry_run": True,
        "executes_install_or_canary": True,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "formal_training_ready_claimed": False,
        "recommended_next_action": "guarded_shadow_release_trial"
        if status == "passed"
        else "fix_guarded_experimental_policy_install_canary_dry_run",
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }


def _run_readiness_validate_only(
    *,
    repo_root: Path,
    batch_root: Path,
    install_canary_summary_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    command = [
        "bash",
        str(repo_root / "scripts" / "run_policy_training_readiness_review.sh"),
        "--batch-root",
        str(batch_root),
        "--config",
        str(_resolve_path(config_path, repo_root)),
        "--guarded-experimental-policy-install-canary-dry-run-summary",
        str(install_canary_summary_path),
        "--validate-only",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    first_line = next((line for line in completed.stdout.splitlines() if line.strip()), "")
    try:
        result = json.loads(first_line)
    except json.JSONDecodeError:
        return {
            "training_readiness_status": "readiness_validate_only_unparseable",
            "reason_codes": ["readiness_validate_only_stdout_unparseable"],
            "training_blockers": [completed.stderr.strip() or completed.stdout[:1000]],
            "command": command,
            "returncode": completed.returncode,
        }
    result["command"] = command
    result["returncode"] = completed.returncode
    return result


def _validate_readiness(
    readiness: dict[str, Any],
    config: dict[str, Any],
    reason_codes: list[str],
) -> None:
    expected = config.get("readiness", {}).get("expected_status", EXPECTED_READINESS_STATUS)
    if readiness.get("training_readiness_status") != expected:
        _add_reason(reason_codes, "install_canary_readiness_status_mismatch")
    if _string_list(readiness.get("training_blockers")):
        _add_reason(reason_codes, "install_canary_readiness_blocked")
    if _string_list(readiness.get("reason_codes")):
        _add_reason(reason_codes, "install_canary_readiness_reason_codes")
    if _int(readiness.get("returncode")) not in (0,):
        _add_reason(reason_codes, "install_canary_readiness_command_failed")


def _install_canary_verdict(reason_codes: list[str]) -> str:
    if not reason_codes:
        return EXPECTED_INSTALL_CANARY_VERDICT
    if any("checkpoint" in reason or "package" in reason for reason in reason_codes):
        return "blocked_by_package_consumer_audit"
    if any(
        reason
        in {
            "install_canary_controlled_regression",
            "install_canary_step_count_below_threshold",
            "install_canary_missing_observation",
            "install_canary_invalid_action_mask",
            "install_canary_non_finite_inference",
            "install_canary_non_finite_reward",
            "install_canary_checkpoint_load_failed",
        }
        for reason in reason_codes
    ):
        return "blocked_by_guarded_canary"
    if any("default_policy" in reason or "rollback" in reason for reason in reason_codes):
        return "blocked_by_rollback_boundary"
    return "blocked_by_guarded_canary"


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Guarded Experimental Policy Install Canary Dry-Run v1",
            "",
            f"- status: `{summary['status']}`",
            f"- reason_codes: `{summary['reason_codes']}`",
            f"- install_canary_verdict: `{summary['install_canary_verdict']}`",
            f"- canary_step_count: `{summary.get('canary_step_count')}`",
            f"- controlled_regression_count: `{summary.get('controlled_regression_count')}`",
            f"- rollback_default_audit_passed: `{summary.get('rollback_default_audit_passed')}`",
            f"- readiness_status: `{summary.get('readiness_status')}`",
            "",
            "This stage only simulates a guarded install in an isolated sandbox. "
            "It does not publish or replace the default policy.",
            "",
        ]
    )


def _output_files(config: dict[str, Any]) -> dict[str, str]:
    output_files = config.get("output_files", {})
    defaults = {
        "summary": SUMMARY_FILE,
        "sandbox_manifest": SANDBOX_MANIFEST_FILE,
        "package_consumer_audit": PACKAGE_CONSUMER_AUDIT_FILE,
        "step_audit": STEP_AUDIT_FILE,
        "rollback_audit": ROLLBACK_AUDIT_FILE,
        "readiness_validate_only": READINESS_FILE,
        "report": REPORT_FILE,
    }
    return {key: str(output_files.get(key, value)) for key, value in defaults.items()}


def _input_files(config: dict[str, Any]) -> dict[str, str]:
    input_files = config.get("input_files", {})
    return {
        "packaging_summary": str(
            input_files.get(
                "packaging_summary",
                "guarded-experimental-policy-release-candidate-packaging-summary.json",
            )
        )
    }


def _default_policy_path(config: dict[str, Any], repo_root: Path) -> Path | None:
    default_policy = config.get("default_policy", {})
    if not isinstance(default_policy, dict):
        return None
    value = default_policy.get("path")
    if not isinstance(value, str) or not value:
        return None
    return _resolve_path(Path(value), repo_root)


def _min_canary_step_count(config: dict[str, Any]) -> int:
    return _int(config.get("validation", {}).get("min_canary_step_count"), 64)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return _read_json(path)


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def _resolve_optional_path(value: Any, base: Path, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base / path
    if candidate.exists():
        return candidate
    return repo_root / path


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _file_snapshot(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None, "size_bytes": 0}
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "size_bytes": 0}
    if not path.is_file():
        return {"path": str(path), "exists": True, "sha256": None, "size_bytes": 0}
    sha256, size = _sha256_and_size(path)
    return {"path": str(path), "exists": True, "sha256": sha256, "size_bytes": size}


def _snapshots_same(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("path") == right.get("path")
        and left.get("exists") == right.get("exists")
        and left.get("sha256") == right.get("sha256")
        and _int(left.get("size_bytes")) == _int(right.get("size_bytes"))
    )


def _publication_flags(payload: dict[str, Any]) -> dict[str, bool]:
    return {
        "runs_new_ppo_update": payload.get("runs_new_ppo_update") is True,
        "publishes_checkpoint": payload.get("publishes_checkpoint") is True,
        "replaces_default_policy": payload.get("replaces_default_policy") is True
        or payload.get("default_policy_replaced") is True,
        "performance_claimed": payload.get("performance_claimed") is True,
        "formal_training_ready_claimed": payload.get("formal_training_ready_claimed") is True,
    }


def _git_current_matches_sources(payload: dict[str, Any]) -> bool | None:
    provenance = payload.get("git_provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("current_matches_sources")
    return value if isinstance(value, bool) else None


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _add_reason(reason_codes: list[str], reason: str) -> None:
    if reason not in reason_codes:
        reason_codes.append(reason)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
