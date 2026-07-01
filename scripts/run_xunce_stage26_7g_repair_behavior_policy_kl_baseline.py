from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage21_4_tiny_ppo_update_smoke as stage21_4
    import run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as stage26_7f
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_4_tiny_ppo_update_smoke as stage21_4
    import scripts.run_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep as stage26_7f


STAGE_ID = "xunce-stage26-7g-repair-behavior-policy-kl-baseline"
CONFIG_SCHEMA_VERSION = "xunce-stage26-7g-repair-behavior-policy-kl-baseline-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-7g-summary/v1"
ZERO_AUDIT_SCHEMA_VERSION = "xunce-stage26-7g-zero-update-kl-baseline-audit/v1"
KL_AUDIT_SCHEMA_VERSION = "xunce-stage26-7g-policy-vs-behavior-kl-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-7g-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-7g-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_7g_repair_behavior_policy_kl_baseline_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7g_repair_behavior_policy_kl_baseline_v1"
)
DEFAULT_STAGE26_7F_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep_v1"
)

SUMMARY_FILE = "xunce-stage26-7g-summary.json"
ZERO_AUDIT_FILE = "xunce-stage26-7g-zero-update-kl-baseline-audit.json"
KL_AUDIT_FILE = "xunce-stage26-7g-policy-vs-behavior-kl-audit.json"
SWEEP_FILE = "xunce-stage26-7g-update-sweep-results.jsonl"
CHECKPOINT_AUDIT_FILE = "xunce-stage26-7g-checkpoint-audit.json"
POST_EVAL_AUDIT_FILE = "xunce-stage26-7g-post-update-eval-audit.json"
ROUTING_FILE = "xunce-stage26-7g-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-7g-report.md"
MANIFEST_FILE = "xunce-stage26-7g-manifest.json"

ROUTE_INPUTS = "rerun_stage26_7g_required_inputs"
ROUTE_POLICY_CAPTURE = "repair_stage26_7_policy_old_logprob_capture"
ROUTE_RATIO = "repair_stage26_7_behavior_ratio_contract"
ROUTE_KL_SOURCE = "repair_stage26_7_kl_gate_source"
ROUTE_REDUCE = "reduce_stage26_7_credit_update_strength"
ROUTE_CHECKPOINT = "repair_stage26_7_credit_checkpoint_boundary"
ROUTE_EVAL_BINDING = "repair_stage26_7_credit_post_update_eval_binding"
ROUTE_MARGIN = "calibrate_stage26_synthetic_discrete_margin_crossing_after_credit"
ROUTE_CREDIT = "repair_stage26_synthetic_credit_assignment"
ROUTE_MULTI_SEED = "run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot"
ROUTE_BOUNDARY = "resolve_stage26_7g_boundary_rejections"

ROUTE_FROM_STAGE26_7F = "repair_stage26_7_behavior_policy_kl_baseline"
MAX_ALLOWED_KL_LIMIT = 1.5
BOUNDARY_FIELDS = ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.7G behavior-policy KL baseline repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_7g_repair_behavior_policy_kl_baseline(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_7g_repair_behavior_policy_kl_baseline(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_7f_summary = _read_json_if_exists(Path(config["stage26_7f_root"]) / stage26_7f.SUMMARY_FILE)
    primary_stage26_1_root = _primary_stage26_1_root(config, stage26_7f_summary)
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_7f_summary, primary_stage26_1_root)

    zero_audit: dict[str, Any] = {"schema_version": ZERO_AUDIT_SCHEMA_VERSION, "status": "skipped"}
    combo_details: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    if not boundary_rejections and not input_rejections and bool(config["run_stage26_chain"]):
        zero_audit = _zero_update_kl_baseline_audit(config, primary_stage26_1_root, repo_root)
        if _zero_update_policy_kl_ok(zero_audit, float(config["zero_update_policy_kl_abs_tolerance"])):
            existing_sweep_path = output_root / SWEEP_FILE
            if bool(config.get("reuse_existing_outputs")) and existing_sweep_path.exists():
                sweep_rows = _read_jsonl(existing_sweep_path)
                combo_details = _existing_combo_details(output_root, sweep_rows)
            else:
                for combo in config["update_sweep"]:
                    detail = stage26_7f._run_combo(
                        config=config,
                        combo=combo,
                        primary_stage26_1_root=primary_stage26_1_root,
                        output_root=output_root,
                        repo_root=repo_root,
                    )
                    row = dict(detail["sweep_row"])
                    row.update(_combo_policy_behavior_kl_fields(detail))
                    detail["sweep_row"] = row
                    combo_details.append(detail)
                    sweep_rows.append(row)

    stable_rows = [row for row in sweep_rows if stage26_7f._combo_is_stable(row, float(config["max_abs_approx_kl"]))]
    best_combo = stage26_7f._best_stable_combo(combo_details, float(config["max_abs_approx_kl"]))
    kl_audit = _policy_vs_behavior_kl_audit(zero_audit, sweep_rows, float(config["max_abs_approx_kl"]))
    checkpoint_audit = dict(stage26_7f._checkpoint_audit(sweep_rows), schema_version="xunce-stage26-7g-checkpoint-audit/v1")
    post_eval_audit = dict(stage26_7f._post_eval_audit(best_combo), schema_version="xunce-stage26-7g-post-update-eval-audit/v1")
    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        zero_audit=zero_audit,
        sweep_rows=sweep_rows,
        stable_rows=stable_rows,
        best_combo=best_combo,
        kl_audit=kl_audit,
        post_eval_audit=post_eval_audit,
    )
    status = "passed" if route == ROUTE_MULTI_SEED else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_7f_root": config["stage26_7f_root"],
        "stage26_7f_status": stage26_7f_summary.get("status"),
        "stage26_7f_next_required_change": stage26_7f_summary.get("next_required_change"),
        "primary_stage26_1_root": str(primary_stage26_1_root) if primary_stage26_1_root else None,
        "zero_update_policy_approx_kl": zero_audit.get("policy_approx_kl"),
        "zero_update_behavior_approx_kl": zero_audit.get("behavior_approx_kl"),
        "zero_update_policy_kl_ok": _zero_update_policy_kl_ok(zero_audit, float(config["zero_update_policy_kl_abs_tolerance"])),
        "behavior_policy_kl_diagnostic_only": True,
        "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
        "kl_gate_source": "policy_old_logprob_when_available/v1",
        "combo_count": len(sweep_rows),
        "stable_combo_count": len(stable_rows),
        "best_stable_combo_id": best_combo.get("combo_id"),
        "best_stable_final_post_update_policy_approx_kl": best_combo.get("final_post_update_policy_approx_kl"),
        "best_stable_final_post_update_behavior_approx_kl": best_combo.get("final_post_update_behavior_approx_kl"),
        "selected_action_changed_count": best_combo.get("selected_action_changed_count", 0),
        "main_final_coverage_delta": best_combo.get("main_final_coverage_delta", 0.0),
        "main_coverage_auc_delta": best_combo.get("main_coverage_auc_delta", 0.0),
        "main_coverage_per_100m_delta": best_combo.get("main_coverage_per_100m_delta", 0.0),
        "hybrid_astar_path_cost_delta": best_combo.get("hybrid_astar_path_cost_delta", 0.0),
        "hybrid_astar_path_cost_delta_is_diagnostic_only": True,
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "coverage_denominator_source": "main_coverable_cells/v1",
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "max_traversable_slope_deg": 30.0,
        "max_abs_approx_kl": config["max_abs_approx_kl"],
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
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary": str(output_root / SUMMARY_FILE),
        "zero_update_kl_baseline_audit": str(output_root / ZERO_AUDIT_FILE),
        "policy_vs_behavior_kl_audit": str(output_root / KL_AUDIT_FILE),
        "update_sweep_results": str(output_root / SWEEP_FILE),
        "checkpoint_audit": str(output_root / CHECKPOINT_AUDIT_FILE),
        "post_update_eval_audit": str(output_root / POST_EVAL_AUDIT_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / ZERO_AUDIT_FILE, zero_audit)
    _write_json(output_root / KL_AUDIT_FILE, kl_audit)
    stage26_7f._write_jsonl(output_root / SWEEP_FILE, sweep_rows)
    _write_json(output_root / CHECKPOINT_AUDIT_FILE, checkpoint_audit)
    _write_json(output_root / POST_EVAL_AUDIT_FILE, post_eval_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, sweep_rows), encoding="utf-8")
    return summary


def _zero_update_kl_baseline_audit(config: dict[str, Any], stage26_1_root: Path, repo_root: Path) -> dict[str, Any]:
    stage21_3_root = stage26_1_root / "s21_3"
    summary21_3 = _read_json(stage21_3_root / "xunce-stage21-3-ppo-batch-validation-summary.json")
    batch_rows = _read_jsonl(stage21_3_root / "xunce-stage21-3-ppo-trainable-batch.jsonl")
    train_rows = [row for row in batch_rows if row.get("stage21_3_split") == "train"]
    stage21_1_root = Path(str(summary21_3.get("stage21_1_collector_root", "")))
    stage21_1_summary = _read_json_if_exists(stage21_1_root / "xunce-stage21-1-on-policy-ppo-rollout-collector-summary.json")
    high_fidelity_config = _read_json(Path(config["high_fidelity_config"]))
    if any(stage21_4._row_uses_continuous_theta(row) for row in train_rows):
        init_seed = config.get("continuous_theta_head_init_seed", stage21_1_summary.get("sampling_seed"))
        if init_seed is not None:
            high_fidelity_config["continuous_theta_head_init_seed"] = int(init_seed)
    _, model, _ = stage21_4._load_xunce_checkpoint(
        Path(config["xunce_candidate_checkpoint"]),
        config=high_fidelity_config,
        repo_root=repo_root,
    )
    if model is None:
        return {"schema_version": ZERO_AUDIT_SCHEMA_VERSION, "status": "failed", "reason": "source_checkpoint_not_loadable"}
    sampling_temperature = stage21_4._resolve_sampling_temperature(train_rows, stage21_1_summary, config)
    behavior_values: list[float] = []
    policy_values: list[float] = []
    new_values: list[float] = []
    model.eval()
    with stage21_4.torch.no_grad():
        for row in train_rows:
            tensors = stage21_4._deserialize_xunce_batch(row.get("xunce_batch"))
            action_index = stage21_4._required_int(row.get("action_index"), "action_index")
            sampling_mask = stage21_4._sampling_mask(row, tensors["action_mask"], action_index)
            output = model(**tensors)
            logits = output.masked_logits[0] / float(sampling_temperature)
            logits = logits.masked_fill(~sampling_mask, -1.0e9)
            if stage21_4._row_uses_continuous_theta(row):
                theta_detail = stage21_4.continuous_theta_torch_log_prob(
                    point_logits=logits,
                    theta_mu_rad=output.theta_mu_rad[0],
                    theta_kappa=output.theta_kappa[0],
                    action_index=action_index,
                    theta_rad=stage21_4._required_float(row.get("selected_theta_rad"), "selected_theta_rad"),
                )
                new_log_prob = float(theta_detail["total_log_prob"])
            else:
                distribution = stage21_4.torch.distributions.Categorical(logits=logits)
                new_log_prob = float(distribution.log_prob(stage21_4.torch.tensor(action_index, dtype=stage21_4.torch.long)))
            behavior_old = stage21_4._required_float(row.get("old_log_prob"), "old_log_prob")
            policy_old = stage21_4._old_policy_log_prob_for_kl(row, fallback=behavior_old)
            behavior_values.append(behavior_old)
            policy_values.append(policy_old)
            new_values.append(new_log_prob)
    behavior_kl = _mean_delta(behavior_values, new_values)
    policy_kl = _mean_delta(policy_values, new_values)
    return {
        "schema_version": ZERO_AUDIT_SCHEMA_VERSION,
        "status": "passed",
        "train_row_count": len(train_rows),
        "synthetic_credit_behavior_policy_row_count": sum(1 for row in train_rows if stage21_4._row_uses_synthetic_credit_behavior_policy(row)),
        "behavior_approx_kl": behavior_kl,
        "policy_approx_kl": policy_kl,
        "policy_kl_abs_tolerance": config["zero_update_policy_kl_abs_tolerance"],
        "policy_kl_within_tolerance": abs(policy_kl) <= float(config["zero_update_policy_kl_abs_tolerance"]),
        "behavior_policy_kl_diagnostic_only": True,
        "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
        "kl_gate_source": "policy_old_logprob_when_available/v1",
    }


def _combo_policy_behavior_kl_fields(detail: dict[str, Any]) -> dict[str, Any]:
    root = detail["combo_root"] / "s26_2" / "s21_4"
    summary = _read_json_if_exists(root / "xunce-stage21-4-tiny-ppo-update-smoke-summary.json")
    loss_path = Path(str(summary.get("loss_audit") or root / "xunce-stage21-4-ppo-loss-audit.jsonl"))
    rows = _read_jsonl_if_exists(loss_path)
    if not rows:
        return {}
    first = rows[0]
    last = rows[-1]
    return {
        "pre_update_behavior_approx_kl": _finite(first.get("pre_update_behavior_approx_kl")),
        "post_update_behavior_approx_kl": _finite(last.get("post_update_behavior_approx_kl")),
        "pre_update_policy_approx_kl": _finite(first.get("pre_update_policy_approx_kl")),
        "post_update_policy_approx_kl": _finite(last.get("post_update_policy_approx_kl")),
        "final_post_update_policy_approx_kl": _finite(summary.get("final_post_update_policy_approx_kl")),
        "final_post_update_behavior_approx_kl": _finite(summary.get("final_post_update_behavior_approx_kl")),
        "ppo_ratio_old_log_prob_source": summary.get("ppo_ratio_old_log_prob_source"),
        "kl_gate_source": summary.get("kl_gate_source"),
        "behavior_policy_kl_diagnostic_only": summary.get("behavior_policy_kl_diagnostic_only"),
    }


def _policy_vs_behavior_kl_audit(zero_audit: dict[str, Any], rows: list[dict[str, Any]], max_abs_approx_kl: float) -> dict[str, Any]:
    policy_values = [_finite(row.get("final_post_update_policy_approx_kl")) for row in rows]
    behavior_values = [_finite(row.get("final_post_update_behavior_approx_kl")) for row in rows]
    policy_clean = [value for value in policy_values if value is not None]
    behavior_clean = [value for value in behavior_values if value is not None]
    return {
        "schema_version": KL_AUDIT_SCHEMA_VERSION,
        "max_abs_approx_kl": max_abs_approx_kl,
        "zero_update_policy_approx_kl": zero_audit.get("policy_approx_kl"),
        "zero_update_behavior_approx_kl": zero_audit.get("behavior_approx_kl"),
        "policy_kl_exceeded_count": sum(1 for value in policy_clean if abs(value) > max_abs_approx_kl),
        "behavior_kl_exceeded_count": sum(1 for value in behavior_clean if abs(value) > max_abs_approx_kl),
        "min_post_update_policy_approx_kl": min(policy_clean) if policy_clean else None,
        "max_post_update_policy_approx_kl": max(policy_clean) if policy_clean else None,
        "min_post_update_behavior_approx_kl": min(behavior_clean) if behavior_clean else None,
        "max_post_update_behavior_approx_kl": max(behavior_clean) if behavior_clean else None,
        "behavior_policy_kl_diagnostic_only": True,
        "ppo_ratio_old_log_prob_source": "behavior_policy_when_present",
        "kl_gate_source": "policy_old_logprob_when_available/v1",
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    zero_audit: dict[str, Any],
    sweep_rows: list[dict[str, Any]],
    stable_rows: list[dict[str, Any]],
    best_combo: dict[str, Any],
    kl_audit: dict[str, Any],
    post_eval_audit: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if not _zero_update_policy_kl_ok(zero_audit, float(zero_audit.get("policy_kl_abs_tolerance") or 1.0e-3)):
        return ROUTE_POLICY_CAPTURE
    if any(row.get("ppo_ratio_old_log_prob_source") != "behavior_policy_when_present" for row in sweep_rows):
        return ROUTE_RATIO
    if any(row.get("kl_gate_source") != "policy_old_logprob_when_available/v1" for row in sweep_rows):
        return ROUTE_KL_SOURCE
    if not stable_rows:
        return ROUTE_REDUCE
    if any(row.get("checkpoint_reload_passed") is not True or row.get("experimental_only") is not True for row in stable_rows):
        return ROUTE_CHECKPOINT
    if _post_eval_binding_failed(best_combo, post_eval_audit):
        return ROUTE_EVAL_BINDING
    if int(post_eval_audit.get("selected_action_changed_count") or 0) <= 0:
        return ROUTE_MARGIN
    if float(post_eval_audit.get("main_coverage_per_100m_delta") or 0.0) < 0.0:
        return ROUTE_CREDIT
    if stage26_7f._post_eval_success(post_eval_audit, best_combo):
        return ROUTE_MULTI_SEED
    return ROUTE_CREDIT


def _post_eval_binding_failed(best_combo: dict[str, Any], post_eval_audit: dict[str, Any]) -> bool:
    if not best_combo:
        return False
    if best_combo.get("stage26_3_status") != "failed":
        return False
    route = str(best_combo.get("stage26_3_next_required_change") or "")
    if route in {
        "rerun_stage26_3_required_inputs",
        "repair_stage26_3_synthetic_inference_binding",
        "repair_stage26_3_synthetic_lineage_binding",
        "repair_stage26_5_hybrid_path_inference_binding",
    }:
        return True
    return (
        int(best_combo.get("hybrid_path_contract_mismatch_count") or post_eval_audit.get("hybrid_path_contract_mismatch_count") or 0) > 0
        or int(best_combo.get("synthetic_inference_required_field_missing_count") or post_eval_audit.get("synthetic_inference_required_field_missing_count") or 0) > 0
        or bool(best_combo.get("stage21_5_execution_incomplete"))
    )


def _zero_update_policy_kl_ok(audit: dict[str, Any], tolerance: float) -> bool:
    return audit.get("status") == "passed" and _finite(audit.get("policy_approx_kl")) is not None and abs(float(audit["policy_approx_kl"])) <= tolerance


def _input_rejections(summary: dict[str, Any], primary_stage26_1_root: Path | None) -> list[str]:
    if not summary:
        return ["missing_stage26_7f_summary"]
    reasons: list[str] = []
    if summary.get("stage_id") != stage26_7f.STAGE_ID:
        reasons.append("stage26_7f_wrong_stage_id")
    if summary.get("next_required_change") != ROUTE_FROM_STAGE26_7F:
        reasons.append("stage26_7f_route_mismatch")
    if summary.get("pre_update_kl_baseline_exceeded") is not True:
        reasons.append("stage26_7f_pre_update_kl_baseline_not_exceeded")
    if primary_stage26_1_root is None or not primary_stage26_1_root.exists():
        reasons.append("primary_stage26_1_root_missing")
    else:
        stage26_1_summary = _read_json_if_exists(primary_stage26_1_root / "xunce-stage26-1-summary.json")
        if stage26_1_summary.get("status") != "passed":
            reasons.append("primary_stage26_1_status_not_passed")
    return reasons


def _primary_stage26_1_root(config: dict[str, Any], stage26_7f_summary: dict[str, Any]) -> Path | None:
    if config.get("primary_stage26_1_root"):
        return Path(str(config["primary_stage26_1_root"]))
    if stage26_7f_summary.get("primary_stage26_1_root"):
        return Path(str(stage26_7f_summary["primary_stage26_1_root"]))
    return None


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    config = dict(payload)
    config["stage26_7f_root"] = str(_resolve_path(Path(config.get("stage26_7f_root") or DEFAULT_STAGE26_7F_ROOT), repo_root))
    config["stage26_2_base_config"] = str(_resolve_path(Path(config.get("stage26_2_base_config") or stage26_7f.stage26_2.DEFAULT_CONFIG), repo_root))
    config["stage26_3_base_config"] = str(_resolve_path(Path(config.get("stage26_3_base_config") or stage26_7f.stage26_3.DEFAULT_CONFIG), repo_root))
    base26_2 = _read_json(Path(config["stage26_2_base_config"]))
    config["xunce_candidate_checkpoint"] = str(_resolve_path(Path(config.get("xunce_candidate_checkpoint") or base26_2["xunce_candidate_checkpoint"]), repo_root))
    config["high_fidelity_config"] = str(_resolve_path(Path(config.get("high_fidelity_config") or base26_2["high_fidelity_config"]), repo_root))
    config["run_stage26_chain"] = bool(config.get("run_stage26_chain", True))
    config["reuse_existing_outputs"] = bool(config.get("reuse_existing_outputs", False))
    if config.get("primary_stage26_1_root"):
        config["primary_stage26_1_root"] = str(_resolve_path(Path(str(config["primary_stage26_1_root"])), repo_root))
    config["required_scenario_count"] = int(config.get("required_scenario_count", 2))
    config["eval_rollout_steps"] = int(config.get("eval_rollout_steps", 4))
    config["hybrid_astar_candidate_eval_workers"] = int(config.get("hybrid_astar_candidate_eval_workers", 4))
    config["clip_ratio"] = float(config.get("clip_ratio", 0.2))
    config["entropy_coefficient"] = float(config.get("entropy_coefficient", 0.01))
    config["advantage_clip_abs"] = float(config.get("advantage_clip_abs", 5.0))
    config["max_grad_norm"] = float(config.get("max_grad_norm", 1.0))
    config["max_abs_approx_kl"] = float(config.get("max_abs_approx_kl", MAX_ALLOWED_KL_LIMIT))
    if not math.isfinite(config["max_abs_approx_kl"]) or config["max_abs_approx_kl"] > MAX_ALLOWED_KL_LIMIT:
        raise ValueError("max_abs_approx_kl must be finite and must not exceed 1.5 for Stage26.7G")
    config["zero_update_policy_kl_abs_tolerance"] = float(config.get("zero_update_policy_kl_abs_tolerance", 1.0e-3))
    config["update_sweep"] = stage26_7f._update_sweep(config.get("update_sweep"))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    return config


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _existing_combo_details(output_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row in rows:
        work_dir = str(row.get("combo_work_dir") or row.get("work_dir") or row.get("combo_id") or "")
        combo_root = output_root / work_dir if work_dir else output_root
        details.append(
            {
                "combo": {"combo_id": row.get("combo_id"), "work_dir": work_dir},
                "combo_root": combo_root,
                "stage26_2_config": _read_json_if_exists(combo_root / "xunce-stage26-7f-stage26-2-config.json"),
                "stage26_3_config": _read_json_if_exists(combo_root / "xunce-stage26-7f-stage26-3-config.json"),
                "stage26_2_summary": _read_json_if_exists(combo_root / "s26_2" / "xunce-stage26-2-summary.json"),
                "stage26_3_summary": _read_json_if_exists(combo_root / "s26_3" / "xunce-stage26-3-summary.json"),
                "sweep_row": row,
            }
        )
    return details


def _render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage26.7G Repair Behavior-Policy KL Baseline",
        "",
        f"- status: `{summary['status']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        f"- zero_update_policy_approx_kl: `{summary.get('zero_update_policy_approx_kl')}`",
        f"- zero_update_behavior_approx_kl: `{summary.get('zero_update_behavior_approx_kl')}`",
        f"- stable_combo_count: `{summary.get('stable_combo_count')}`",
        f"- best_stable_combo_id: `{summary.get('best_stable_combo_id')}`",
        f"- main_coverage_per_100m_delta: `{summary.get('main_coverage_per_100m_delta')}`",
        "",
        "Combo results:",
    ]
    for row in rows:
        lines.append(
            "- "
            f"{row.get('combo_id')}: s26_2={row.get('stage26_2_status')}, "
            f"policy_kl={row.get('final_post_update_policy_approx_kl')}, "
            f"behavior_kl={row.get('final_post_update_behavior_approx_kl')}, "
            f"checkpoint_reload={row.get('checkpoint_reload_passed')}, "
            f"s26_3={row.get('stage26_3_status')}, "
            f"main_per_100m_delta={row.get('main_coverage_per_100m_delta')}"
        )
    lines.extend(
        [
            "",
            "This is a bounded offline smoke. Behavior-policy KL is diagnostic only; PPO ratio still uses behavior old logprob.",
            "",
        ]
    )
    return "\n".join(lines)


def _mean_delta(old_values: list[float], new_values: list[float]) -> float:
    if not old_values:
        return float("nan")
    return float(sum(old - new for old, new in zip(old_values, new_values)) / len(old_values))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        return _read_json(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    try:
        return _read_jsonl(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return []


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
