from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

from model_explorer.policy.canonical_reward import load_canonical_reward_profile


CONFIG_SCHEMA_VERSION = "xunce-stage21-0-pure-ppo-readiness-audit-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage21-0-pure-ppo-readiness-summary/v1"
CAPABILITY_SCHEMA_VERSION = "xunce-stage21-0-capability-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage21-0-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage21-0-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage21_0_pure_ppo_readiness_audit_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage21_pure_ppo_coverage_first/"
    "outputs/path_feedback_batch_xunce_stage21_0_pure_ppo_readiness_audit_v1"
)

SUMMARY_FILE = "xunce-stage21-0-pure-ppo-readiness-summary.json"
CAPABILITY_FILE = "xunce-stage21-0-capability-audit.json"
ROUTING_FILE = "xunce-stage21-0-next-stage-routing.json"
REPORT_FILE = "xunce-stage21-0-report.md"
MANIFEST_FILE = "xunce-stage21-0-manifest.json"

ROUTE_BOUNDARY = "resolve_stage21_0_pure_ppo_readiness_boundary_rejections"
ROUTE_REPAIR_INPUTS = "repair_stage21_0_pure_ppo_readiness_inputs"
ROUTE_STAGE21_1 = "implement_stage21_1_xunce_on_policy_ppo_rollout_collector"

BOUNDARY_FIELDS = (
    "stage21_0_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


class ConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 21.0 pure PPO readiness audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    try:
        summary = run_xunce_stage21_0_pure_ppo_readiness_audit(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
            repo_root=Path(args.repo_root),
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": summary["status"],
                "next_required_change": summary["next_required_change"],
                "existing_capability_count": summary["existing_capability_count"],
                "missing_capability_count": summary["missing_capability_count"],
                "stage21_0_authorized": summary["stage21_0_authorized"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage21_0_pure_ppo_readiness_audit(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root=repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    boundary_reasons = _boundary_rejections(config)
    profile = load_canonical_reward_profile(Path(config["canonical_reward_profile"]))
    capabilities = _capability_audit(repo_root)
    available = [name for name, row in capabilities["capabilities"].items() if row["status"] == "available"]
    partial = [name for name, row in capabilities["capabilities"].items() if row["status"] == "partial"]
    missing = [name for name, row in capabilities["capabilities"].items() if row["status"] == "missing"]
    recent_evidence = _recent_evidence(config)
    registry = capabilities["stage_registry"]

    blocking = list(boundary_reasons)
    min_available = int(config["min_required_existing_capability_count"])
    if len(available) < min_available:
        blocking.append("stage21_0_existing_capability_count_below_minimum")
    if registry["stage21_0_registered"] is not True:
        blocking.append("stage21_0_registry_entry_missing")
    if profile.profile_version != "v3":
        blocking.append("stage21_0_requires_canonical_reward_profile_v3")

    route = ROUTE_BOUNDARY if boundary_reasons else (ROUTE_REPAIR_INPUTS if blocking else ROUTE_STAGE21_1)
    status = "passed" if route == ROUTE_STAGE21_1 else "failed"
    generated_at = _utc_now()

    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "primary_route": route,
        "stage21_0_authorized": False,
        "stage21_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    stage21_1_blockers = _stage21_1_blockers(capabilities)
    xunce_checkpoint_format_verdict = _capability_verdict(
        capabilities,
        "xunce_checkpoint_loader",
        passed="xunce_checkpoint_format_readable_for_evaluation",
        failed="xunce_checkpoint_format_not_readable",
    )
    high_fidelity_rollout_contract_verdict = (
        "high_fidelity_rollout_contract_partial_requires_on_policy_stochastic_sampling"
        if "high_fidelity_dynamic_rollout_artifacts" in partial
        else _capability_verdict(
            capabilities,
            "high_fidelity_dynamic_rollout_artifacts",
            passed="high_fidelity_rollout_contract_available",
            failed="high_fidelity_rollout_contract_missing",
        )
    )
    xunce_ppo_reuse_verdict = (
        "generic_ppo_reusable_only_after_xunce_batch_adapter"
        if "generic_masked_ppo_loss" in available and "xunce_specific_ppo_batch_adapter" in missing
        else "xunce_ppo_reuse_blocked"
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "profile_hash": profile.profile_hash,
        "stage21_output_base": config["stage21_output_base"],
        "existing_capabilities": available,
        "needs_improvement_capabilities": partial,
        "missing_capabilities": missing,
        "existing_capability_count": len(available),
        "needs_improvement_capability_count": len(partial),
        "missing_capability_count": len(missing),
        "recent_evidence_roots": recent_evidence,
        "stage_registry": registry,
        "stage20_oracle_imitation_role": "diagnostic_only_not_pure_ppo_gate",
        "stage21_1_blockers": stage21_1_blockers,
        "xunce_checkpoint_format_verdict": xunce_checkpoint_format_verdict,
        "high_fidelity_rollout_contract_verdict": high_fidelity_rollout_contract_verdict,
        "xunce_ppo_reuse_verdict": xunce_ppo_reuse_verdict,
        "stage21_readiness_conclusion": _readiness_conclusion(available, partial, missing),
        "blocking_reason_codes": _unique(blocking),
        "reason_codes": _unique(blocking + capabilities["reason_codes"]),
        "next_required_change": route,
        "next_stage_routing": routing,
        "stage21_0_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str((output_root / SUMMARY_FILE).resolve()),
        "capability_audit": str((output_root / CAPABILITY_FILE).resolve()),
        "routing": str((output_root / ROUTING_FILE).resolve()),
        "report": str((output_root / REPORT_FILE).resolve()),
        "manifest": str((output_root / MANIFEST_FILE).resolve()),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "summary": str((output_root / SUMMARY_FILE).resolve()),
        "capability_audit": str((output_root / CAPABILITY_FILE).resolve()),
        "routing": str((output_root / ROUTING_FILE).resolve()),
        "report": str((output_root / REPORT_FILE).resolve()),
        "config": str(_resolve_path(config_path, repo_root)),
    }

    _write_json(output_root / CAPABILITY_FILE, capabilities)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, capabilities), encoding="utf-8")
    return summary


def _load_config(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    for key in ("stage21_output_base", "canonical_reward_profile"):
        if not isinstance(config.get(key), str) or not str(config[key]).strip():
            raise ConfigError(f"{key} must be a non-empty string")
    config["canonical_reward_profile"] = str(_resolve_path(Path(config["canonical_reward_profile"]), repo_root))
    config["min_required_existing_capability_count"] = _positive_int(
        config.get("min_required_existing_capability_count", 6),
        "min_required_existing_capability_count",
    )
    fraction = config.get("canary_traffic_fraction", 0.0)
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or float(fraction) != 0.0:
        raise ConfigError("canary_traffic_fraction must be 0.0")
    return config


def _capability_audit(repo_root: Path) -> dict[str, Any]:
    checks = {
        "xunce_full_network_masked_policy_value": _check(
            repo_root,
            "scripts/xunce_full_network_common.py",
            ["class XunceFullNetworkV1", "value_head", "masked_logits", "action_probs", "action_mask"],
            "Xunce full network exposes masked policy logits and a value head.",
        ),
        "xunce_checkpoint_loader": _check(
            repo_root,
            "scripts/run_xunce_high_fidelity_real_map_comparison.py",
            ["def _load_xunce_checkpoint", "model_state_dict", "load_state_dict", "_resolve_xunce_model_config"],
            "Existing evaluator can load the Xunce checkpoint format read-only.",
        ),
        "high_fidelity_dynamic_rollout_artifacts": _check(
            repo_root,
            "scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py",
            ["def _run_coverage_rollouts", "candidate_metric_rows", "paired_decision_rows", "coverage_metric_mode"],
            "High-fidelity rollout can produce coverage, path, risk, pair and candidate audit artifacts.",
        ),
        "xunce_model_batch_adapter": _check(
            repo_root,
            "scripts/run_xunce_high_fidelity_real_map_comparison.py",
            ["def _scenario_to_model_inputs", "xunce_batch", "candidate_features", "edge_features", "memory_features", "context_features"],
            "Existing evaluation adapter can build Xunce full-network tensors from high-fidelity candidates.",
        ),
        "generic_masked_ppo_loss": _check(
            repo_root,
            "model-explorer/src/model_explorer/policy/ppo.py",
            ["def compute_masked_ppo_loss", "old_log_probs", "returns", "advantages", "Categorical", "action_mask"],
            "Generic masked PPO loss exists for the older policy architecture.",
        ),
        "generic_rollout_transition_schema": _check(
            repo_root,
            "model-explorer/src/model_explorer/policy/rollout.py",
            ["class RolloutTransition", "log_prob", "value", "next_observation", "done"],
            "Generic rollout transition schema already records PPO-critical fields.",
        ),
        "limited_ppo_update_smoke": _check(
            repo_root,
            "scripts/run_limited_ppo_update_smoke.py",
            ["old_log_prob_max_abs_error", "old_value_max_abs_error", "approx_kl", "clip_fraction", "parameter_l2_delta"],
            "Existing smoke runner validates old log prob/value and PPO stability metrics.",
        ),
        "canonical_reward_v3_profile": _check(
            repo_root,
            "configs/xunce_canonical_reward_guard_profile_v3.json",
            ["xunce-canonical-reward-guard-profile/v3", "path_cost_includes_risk_proxy", "trajectory_guards"],
            "Canonical v3 profile exists for trajectory-level reward/guard lineage.",
        ),
        "xunce_on_policy_stochastic_collector": _missing(
            "Stage 21.1 still needs a Xunce-specific collector that samples actions stochastically and records old_log_prob/value.",
        ),
        "xunce_specific_ppo_batch_adapter": _missing(
            "Generic PPO batch code does not yet consume Xunce edge/memory/context tensors directly.",
        ),
        "coverage_first_ppo_reward_contract": _missing(
            "Stage 21.2 still needs a coverage-first PPO reward contract with final coverage and 99pct bonuses.",
        ),
        "post_update_trajectory_eval": _missing(
            "Stage 21.5 still needs a post-update high-fidelity trajectory evaluation gate.",
        ),
        "multi_seed_ppo_pilot": _missing(
            "Stage 21.6 still needs multi-seed pilot orchestration and holdout aggregation.",
        ),
    }
    _mark_partial_if(
        checks,
        "high_fidelity_dynamic_rollout_artifacts",
        repo_root / "scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py",
        ["torch.multinomial", "Categorical("],
        "High-fidelity runner is available for evaluation artifacts but does not yet prove stochastic on-policy PPO sampling.",
    )
    _mark_partial_if(
        checks,
        "limited_ppo_update_smoke",
        repo_root / "scripts/run_limited_ppo_update_smoke.py",
        ["XunceFullNetworkV1", "edge_features", "memory_features", "context_features"],
        "Existing PPO smoke is useful but targets generic masked-policy batches, not the Xunce full-network tensor contract.",
    )
    registry = _stage_registry_status(repo_root)
    reason_codes = []
    if checks["high_fidelity_dynamic_rollout_artifacts"]["status"] == "partial":
        reason_codes.append("high_fidelity_runner_not_yet_on_policy_stochastic_collector")
    if checks["limited_ppo_update_smoke"]["status"] == "partial":
        reason_codes.append("ppo_update_smoke_not_xunce_full_network_specific")
    if not registry["stage21_0_registered"]:
        reason_codes.append("stage21_0_registry_entry_missing")
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "capabilities": checks,
        "stage_registry": registry,
        "reason_codes": _unique(reason_codes),
    }


def _check(repo_root: Path, relative: str, patterns: list[str], description: str) -> dict[str, Any]:
    path = repo_root / relative
    missing = []
    text = ""
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="ignore")
        missing = [pattern for pattern in patterns if pattern not in text]
    else:
        missing = list(patterns)
    return {
        "status": "available" if path.is_file() and not missing else "missing",
        "description": description,
        "path": str(path),
        "required_patterns": patterns,
        "missing_patterns": missing,
    }


def _missing(description: str) -> dict[str, Any]:
    return {
        "status": "missing",
        "description": description,
        "path": None,
        "required_patterns": [],
        "missing_patterns": ["not_implemented"],
    }


def _mark_partial_if(
    checks: dict[str, dict[str, Any]],
    key: str,
    path: Path,
    positive_patterns: list[str],
    description: str,
) -> None:
    if checks[key]["status"] != "available":
        return
    text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
    if not any(pattern in text for pattern in positive_patterns):
        checks[key]["status"] = "partial"
        checks[key]["description"] = description
        checks[key]["missing_patterns"] = list(positive_patterns)


def _stage_registry_status(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs/stage_registry.json"
    payload = _read_json(path) if path.is_file() else {}
    stages = payload.get("stages") if isinstance(payload, dict) else {}
    stage_ids = set(stages) if isinstance(stages, dict) else set()
    return {
        "stage_registry_path": str(path),
        "stage21_0_registered": "xunce-stage21-0-pure-ppo-readiness-audit" in stage_ids,
        "stage21_1_registered": "xunce-stage21-1-xunce-on-policy-ppo-rollout-collector" in stage_ids,
        "stage21_2_registered": "xunce-stage21-2-coverage-first-ppo-reward-contract" in stage_ids,
        "stage21_3_registered": "xunce-stage21-3-ppo-batch-validation" in stage_ids,
        "stage21_4_registered": "xunce-stage21-4-tiny-ppo-update-smoke" in stage_ids,
        "stage21_5_registered": "xunce-stage21-5-post-update-offline-trajectory-evaluation" in stage_ids,
        "stage21_6_registered": "xunce-stage21-6-multi-seed-ppo-pilot" in stage_ids,
    }


def _recent_evidence(config: dict[str, Any]) -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for key in (
        "stage18_11_path_cost_weight_calibration_root",
        "stage19_evaluator_critic_preflight_root",
        "stage20_oracle_imitation_dataset_root",
        "stage20_1_same_candidate_oracle_imitation_root",
    ):
        raw = config.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        roots[key] = {
            "path": str(path),
            "exists": path.exists(),
            "diagnostic_only_for_stage21": key.startswith("stage20"),
        }
    return roots


def _readiness_conclusion(available: list[str], partial: list[str], missing: list[str]) -> str:
    return (
        "PPO foundations exist, but pure Xunce PPO is not ready until Stage 21.1 adds an on-policy stochastic "
        "collector and later stages add coverage-first reward, batch validation, Xunce-specific update, "
        "post-update evaluation, and multi-seed pilot."
    )


def _stage21_1_blockers(capabilities: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    rows = capabilities.get("capabilities", {})
    if rows.get("xunce_on_policy_stochastic_collector", {}).get("status") != "available":
        blockers.append("missing_xunce_on_policy_stochastic_collector")
    if rows.get("xunce_specific_ppo_batch_adapter", {}).get("status") != "available":
        blockers.append("missing_xunce_specific_ppo_batch_adapter")
    if rows.get("high_fidelity_dynamic_rollout_artifacts", {}).get("status") == "partial":
        blockers.append("high_fidelity_runner_still_argmax_evaluation_not_stochastic_collector")
    if rows.get("limited_ppo_update_smoke", {}).get("status") == "partial":
        blockers.append("generic_ppo_smoke_not_xunce_full_network_update")
    return blockers


def _capability_verdict(capabilities: dict[str, Any], key: str, *, passed: str, failed: str) -> str:
    row = capabilities.get("capabilities", {}).get(key, {})
    return passed if row.get("status") == "available" else failed


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        value = config.get(field)
        if value is not False:
            reasons.append(field)
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _render_report(summary: dict[str, Any], capabilities: dict[str, Any]) -> str:
    rows = []
    for name, row in capabilities["capabilities"].items():
        rows.append(f"- `{name}`: `{row['status']}` - {row['description']}")
    blockers = [f"- `{item}`" for item in summary["stage21_1_blockers"]]
    return "\n".join(
        [
            "# Xunce Stage 21.0 Pure PPO Readiness Audit",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- profile: `{summary['profile_id']}` / `{summary['profile_hash']}`",
            f"- existing_capability_count: `{summary['existing_capability_count']}`",
            f"- needs_improvement_capability_count: `{summary['needs_improvement_capability_count']}`",
            f"- missing_capability_count: `{summary['missing_capability_count']}`",
            "",
            "## Conclusion",
            "",
            summary["stage21_readiness_conclusion"],
            "",
            "## Stage 21.1 Blockers",
            "",
            *(blockers or ["- None"]),
            "",
            "## Verdicts",
            "",
            f"- xunce_checkpoint_format_verdict: `{summary['xunce_checkpoint_format_verdict']}`",
            f"- high_fidelity_rollout_contract_verdict: `{summary['high_fidelity_rollout_contract_verdict']}`",
            f"- xunce_ppo_reuse_verdict: `{summary['xunce_ppo_reuse_verdict']}`",
            "",
            "## Capabilities",
            "",
            *rows,
            "",
            "## Boundaries",
            "",
            "- No PPO update was run.",
            "- No checkpoint was published or installed as default policy.",
            "- No real executor or canary was connected.",
        ]
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"JSON root must be an object: {path}")
    return payload


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a positive integer") from exc
    if numeric <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return numeric


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
