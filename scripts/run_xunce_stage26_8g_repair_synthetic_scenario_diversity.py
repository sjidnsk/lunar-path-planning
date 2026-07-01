from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    import run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
    import run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8
    import run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as stage26_8f
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_1_synthetic_terrain_collector_smoke as stage26_1
    import scripts.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke as stage26_2
    import scripts.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke as stage26_3
    import scripts.run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot as stage26_8
    import scripts.run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit as stage26_8f


STAGE_ID = "xunce-stage26-8g-repair-synthetic-scenario-diversity"
CONFIG_SCHEMA_VERSION = "xunce-stage26-8g-repair-synthetic-scenario-diversity-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-8g-summary/v1"
ROOT_CAUSE_SCHEMA_VERSION = "xunce-stage26-8g-root-cause-audit/v1"
RECHECK_SCHEMA_VERSION = "xunce-stage26-8g-scenario-diversity-recheck-audit/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-8g-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-8g-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_8g_repair_synthetic_scenario_diversity_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8g_repair_synthetic_scenario_diversity_v1"
)
DEFAULT_STAGE26_8F_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/"
    "outputs/path_feedback_batch_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit_v1"
)

SUMMARY_FILE = "xunce-stage26-8g-summary.json"
ROOT_CAUSE_FILE = "xunce-stage26-8g-root-cause-audit.json"
FIXTURE_FILE = "xunce-stage26-8g-scenario-fixture-catalog.jsonl"
STAGE26_1_SUMMARY_FILE = "xunce-stage26-8g-stage26-1-summary.json"
STAGE26_2_SUMMARY_FILE = "xunce-stage26-8g-stage26-2-summary.json"
STAGE26_3_SUMMARY_FILE = "xunce-stage26-8g-stage26-3-summary.json"
RECHECK_FILE = "xunce-stage26-8g-scenario-diversity-recheck-audit.json"
ROUTING_FILE = "xunce-stage26-8g-next-stage-routing.json"
REPORT_FILE = "xunce-stage26-8g-report.md"
MANIFEST_FILE = "xunce-stage26-8g-manifest.json"

COVERAGE_SOURCE = stage26_8.COVERAGE_SOURCE
PATH_COST_SOURCE = stage26_8.PATH_COST_SOURCE
SYNTHETIC_SOURCE_KIND = stage26_8.SYNTHETIC_SOURCE_KIND
ACTION_SPACE_TYPE = stage26_8.ACTION_SPACE_TYPE
COVERAGE_DENOMINATOR_SOURCE = stage26_8.COVERAGE_DENOMINATOR_SOURCE
SUCCESS_METRIC = stage26_8.SUCCESS_METRIC
SCENARIO_DIVERSITY_SOURCE = stage26_1.SCENARIO_DIVERSITY_SOURCE

ROUTE_INPUTS = "rerun_stage26_8g_required_inputs"
ROUTE_START_POOL = "expand_stage26_synthetic_scenario_start_pool"
ROUTE_COLLECTOR_DIVERSITY = "repair_stage26_8g_collector_scenario_diversity_binding"
ROUTE_EVAL_DIVERSITY = "repair_stage26_8g_eval_scenario_diversity_binding"
ROUTE_LINEAGE = "repair_stage26_8g_scenario_lineage_binding"
ROUTE_CHAIN = "repair_stage26_8g_chain_binding_or_safety"
ROUTE_POLICY_SIGNAL = "repair_stage26_synthetic_policy_update_signal_strength"
ROUTE_RESUME = "resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios"
ROUTE_BOUNDARY = "resolve_stage26_8g_boundary_rejections"

BOUNDARY_FIELDS = ("publishes_checkpoint", "replaces_default_policy", "connects_real_executor", "starts_online_canary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.8G scenario diversity repair.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_8g_repair_synthetic_scenario_diversity(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_8g_repair_synthetic_scenario_diversity(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config = _load_config(config_path, repo_root)
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage26_8f_root = Path(config["stage26_8f_root"])
    stage26_8f_summary = _read_json_if_exists(stage26_8f_root / stage26_8f.SUMMARY_FILE)
    prior_scenario_audit = _read_json_if_exists(stage26_8f_root / stage26_8f.SCENARIO_AUDIT_FILE)
    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(stage26_8f_summary)
    root_cause_audit = _root_cause_audit(prior_scenario_audit)

    stage26_1_summary: dict[str, Any] = {}
    stage26_2_summary: dict[str, Any] = {}
    stage26_3_summary: dict[str, Any] = {}
    fixture_rows: list[dict[str, Any]] = []
    recheck_audit = _empty_recheck_audit()
    run_root = output_root / "h16_seed260801"
    if not boundary_rejections and not input_rejections:
        try:
            stage26_1_summary = _existing_summary_if_passed(run_root / "s26_1" / stage26_1.SUMMARY_FILE) or _run_stage26_1(config, run_root, repo_root)
            fixture_rows = _read_jsonl_if_exists(Path(stage26_1_summary.get("scenario_fixture_catalog") or ""))
            _write_jsonl(output_root / FIXTURE_FILE, fixture_rows)
        except stage26_1.ConfigError as exc:
            root_cause_audit["fixture_generation_error"] = str(exc)
            stage26_1_summary = {"status": "failed", "next_required_change": ROUTE_START_POOL, "error": str(exc)}

        if stage26_1_summary.get("status") == "passed" and _fixture_diverse(fixture_rows):
            stage26_2_summary = _existing_summary_if_passed(run_root / "s26_2" / stage26_2.SUMMARY_FILE) or _run_stage26_2(config, run_root, repo_root)
        if stage26_2_summary.get("status") == "passed":
            stage26_3_summary = _existing_summary_if_passed(run_root / "s26_3" / stage26_3.SUMMARY_FILE) or _run_stage26_3(config, run_root, repo_root)
            if stage26_3_summary:
                recheck_audit = _recheck_stage26_3(run_root / "s26_3", config)

    route = _route(
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        root_cause_audit=root_cause_audit,
        stage26_1_summary=stage26_1_summary,
        stage26_2_summary=stage26_2_summary,
        stage26_3_summary=stage26_3_summary,
        fixture_rows=fixture_rows,
        recheck_audit=recheck_audit,
    )
    status = "passed" if route in {ROUTE_POLICY_SIGNAL, ROUTE_RESUME} else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "stage26_8f_root": str(stage26_8f_root),
        "stage26_8f_status": stage26_8f_summary.get("status"),
        "stage26_8f_next_required_change": stage26_8f_summary.get("next_required_change"),
        "root_cause": root_cause_audit.get("root_cause"),
        "scenario_fixture_count": len(fixture_rows),
        "scenario_fixture_unique_start_count": len({_cell_key(row.get("scenario_start_cell")) for row in fixture_rows}),
        "scenario_fixture_catalog": str(output_root / FIXTURE_FILE),
        "scenario_diversity_primary_mechanism": "deterministic_safe_start_pool/v1",
        "scenario_candidate_seed_generated": bool(fixture_rows),
        "scenario_candidate_seed_applied_to_candidate_generation": False,
        "continuous_theta_eval_reachability_fallback_out_of_scope": True,
        "stage26_8g_changes_reward_ppo_network_hybrid_astar_search": False,
        "stage26_1_root": str(run_root / "s26_1"),
        "stage26_1_status": stage26_1_summary.get("status"),
        "stage26_1_next_required_change": stage26_1_summary.get("next_required_change"),
        "stage26_2_root": str(run_root / "s26_2"),
        "stage26_2_status": stage26_2_summary.get("status"),
        "stage26_2_next_required_change": stage26_2_summary.get("next_required_change"),
        "stage26_3_root": str(run_root / "s26_3"),
        "stage26_3_status": stage26_3_summary.get("status"),
        "stage26_3_next_required_change": stage26_3_summary.get("next_required_change"),
        "scenario_signature_duplicate_group_count": recheck_audit.get("scenario_signature_duplicate_group_count"),
        "scenario_signature_duplicate_ratio": recheck_audit.get("scenario_signature_duplicate_ratio"),
        "scenario_diversity_repaired": recheck_audit.get("scenario_diversity_repaired"),
        "selected_action_changed_count": stage26_3_summary.get("selected_action_changed_count"),
        "main_coverage_per_100m_delta": _float(
            stage26_3_summary.get("main_coverage_per_100m_delta")
            or stage26_3_summary.get("coverage_per_100m_delta")
        ),
        "coverage_denominator_source": COVERAGE_DENOMINATOR_SOURCE,
        "post_update_success_metric": SUCCESS_METRIC,
        "coverage_source": COVERAGE_SOURCE,
        "path_cost_source": PATH_COST_SOURCE,
        "synthetic_source_kind": SYNTHETIC_SOURCE_KIND,
        "action_space_type": ACTION_SPACE_TYPE,
        "scenario_diversity_source": SCENARIO_DIVERSITY_SOURCE,
        "max_traversable_slope_deg": 30.0,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "summary": str(output_root / SUMMARY_FILE),
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "summary_status": status,
        "next_required_change": route,
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "root_cause_audit": str(output_root / ROOT_CAUSE_FILE),
            "scenario_fixture_catalog": str(output_root / FIXTURE_FILE),
            "stage26_1_summary": str(output_root / STAGE26_1_SUMMARY_FILE),
            "stage26_2_summary": str(output_root / STAGE26_2_SUMMARY_FILE),
            "stage26_3_summary": str(output_root / STAGE26_3_SUMMARY_FILE),
            "scenario_diversity_recheck_audit": str(output_root / RECHECK_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "report": str(output_root / REPORT_FILE),
        },
    }
    _write_json(output_root / ROOT_CAUSE_FILE, root_cause_audit)
    if not (output_root / FIXTURE_FILE).exists():
        _write_jsonl(output_root / FIXTURE_FILE, fixture_rows)
    _write_json(output_root / STAGE26_1_SUMMARY_FILE, stage26_1_summary)
    _write_json(output_root / STAGE26_2_SUMMARY_FILE, stage26_2_summary)
    _write_json(output_root / STAGE26_3_SUMMARY_FILE, stage26_3_summary)
    _write_json(output_root / RECHECK_FILE, recheck_audit)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary, root_cause_audit, recheck_audit), encoding="utf-8")
    return summary


def _run_stage26_1(config: dict[str, Any], run_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = stage26_8._build_stage26_1_config(config, int(config["seed"]), run_root, repo_root)
    cfg.update(
        {
            "scenario_diversity_contract_enabled": True,
            "scenario_diversity_source": SCENARIO_DIVERSITY_SOURCE,
            "scenario_seed_base": int(config["seed"]),
            "min_scenario_start_separation_cells": int(config["min_scenario_start_separation_cells"]),
            "min_scenario_start_clearance_cells": int(config["min_scenario_start_clearance_cells"]),
        }
    )
    cfg_path = run_root / "xunce-stage26-8g-stage26-1-config.json"
    _write_json(cfg_path, cfg)
    return stage26_1.run_xunce_stage26_1_synthetic_terrain_collector_smoke(
        config_path=cfg_path,
        output_root=run_root / "s26_1",
        repo_root=repo_root,
    )


def _existing_summary_if_passed(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    summary = _read_json_if_exists(path)
    return summary if summary.get("status") == "passed" else None


def _run_stage26_2(config: dict[str, Any], run_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = stage26_8._build_stage26_2_config(config, seed=int(config["seed"]), stage26_1_root=run_root / "s26_1", repo_root=repo_root)
    cfg_path = run_root / "xunce-stage26-8g-stage26-2-config.json"
    _write_json(cfg_path, cfg)
    return stage26_2.run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke(
        config_path=cfg_path,
        output_root=run_root / "s26_2",
        repo_root=repo_root,
    )


def _run_stage26_3(config: dict[str, Any], run_root: Path, repo_root: Path) -> dict[str, Any]:
    cfg = stage26_8._build_stage26_3_config(config, seed=int(config["seed"]), stage26_2_root=run_root / "s26_2", repo_root=repo_root)
    cfg_path = run_root / "xunce-stage26-8g-stage26-3-config.json"
    _write_json(cfg_path, cfg)
    return stage26_3.run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke(
        config_path=cfg_path,
        output_root=run_root / "s26_3",
        repo_root=repo_root,
    )


def _recheck_stage26_3(stage26_3_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    summary_path = stage26_3_root / stage26_3.SUMMARY_FILE
    if not summary_path.is_file():
        return _empty_recheck_audit() | {"missing_stage26_3_summary": True}
    eval_source = {
        "job_id": "stage26_8g_h16_s260801",
        "horizon_steps": int(config["collector_rollout_steps"]),
        "seed": int(config["seed"]),
        "seed_index": 0,
        "stage26_3_root": str(stage26_3_root),
        "summary_path": str(summary_path),
    }
    payload = stage26_8f._load_eval_source(eval_source)
    audit = stage26_8f._scenario_diversity_audit(
        [payload],
        duplicate_ratio_threshold=float(config["scenario_duplicate_ratio_threshold"]),
    )
    return {
        "schema_version": RECHECK_SCHEMA_VERSION,
        **audit,
        "scenario_diversity_repaired": int(audit.get("scenario_signature_duplicate_group_count") or 0) == 0,
    }


def _root_cause_audit(prior: dict[str, Any]) -> dict[str, Any]:
    signatures = prior.get("scenario_signatures") if isinstance(prior.get("scenario_signatures"), list) else []
    first_cells = {_json_key(row.get("first_current_cell")) for row in signatures if isinstance(row, dict)}
    candidate_hashes = {_json_key(row.get("candidate_set_sequence_hash")) for row in signatures if isinstance(row, dict)}
    covered_hashes = {_json_key(row.get("covered_cells_sequence_hash")) for row in signatures if isinstance(row, dict)}
    action_hashes = {_json_key(row.get("selected_action_sequence_hash")) for row in signatures if isinstance(row, dict)}
    episode_hashes = {_json_key(row.get("episode_metrics_hash")) for row in signatures if isinstance(row, dict)}
    root_cause = "undetermined"
    if signatures and len(first_cells) == 1 and len(candidate_hashes) == 1:
        root_cause = "legacy_safe_start_cell_reused_and_candidate_sequence_repeated"
    elif signatures and len(candidate_hashes) == 1:
        root_cause = "candidate_seed_or_candidate_context_not_diverse"
    elif signatures and len(episode_hashes) == 1:
        root_cause = "trajectory_metric_signature_repeated"
    return {
        "schema_version": ROOT_CAUSE_SCHEMA_VERSION,
        "prior_scenario_signature_count": int(prior.get("scenario_signature_count") or 0),
        "prior_duplicate_group_count": int(prior.get("scenario_signature_duplicate_group_count") or 0),
        "prior_duplicate_ratio": _float(prior.get("scenario_signature_duplicate_ratio")),
        "unique_first_current_cell_count": len(first_cells),
        "unique_candidate_sequence_hash_count": len(candidate_hashes),
        "unique_covered_sequence_hash_count": len(covered_hashes),
        "unique_selected_action_sequence_hash_count": len(action_hashes),
        "unique_episode_metrics_hash_count": len(episode_hashes),
        "root_cause": root_cause,
        "scenario_diversity_primary_mechanism": "deterministic_safe_start_pool/v1",
        "scenario_candidate_seed_applied_to_candidate_generation": False,
        "continuous_theta_eval_reachability_fallback_out_of_scope": True,
        "stage26_8g_changes_reward_ppo_network_hybrid_astar_search": False,
    }


def _route(
    *,
    boundary_rejections: list[str],
    input_rejections: list[str],
    root_cause_audit: dict[str, Any],
    stage26_1_summary: dict[str, Any],
    stage26_2_summary: dict[str, Any],
    stage26_3_summary: dict[str, Any],
    fixture_rows: list[dict[str, Any]],
    recheck_audit: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if root_cause_audit.get("fixture_generation_error"):
        return ROUTE_START_POOL
    if not _fixture_diverse(fixture_rows):
        return ROUTE_START_POOL
    if stage26_1_summary.get("status") != "passed":
        return ROUTE_COLLECTOR_DIVERSITY
    if stage26_2_summary.get("status") != "passed" or stage26_3_summary.get("status") not in {"passed", "failed"}:
        return ROUTE_CHAIN
    if _lineage_mismatch(stage26_3_summary):
        return ROUTE_LINEAGE
    if not recheck_audit.get("scenario_diversity_repaired"):
        return ROUTE_EVAL_DIVERSITY
    if _float(stage26_3_summary.get("main_coverage_per_100m_delta") or stage26_3_summary.get("coverage_per_100m_delta")) > 0.0:
        return ROUTE_RESUME
    return ROUTE_POLICY_SIGNAL


def _fixture_diverse(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return False
    starts = {_cell_key(row.get("scenario_start_cell")) for row in rows}
    signatures = {str(row.get("scenario_diversity_signature_hash") or "") for row in rows}
    content_hashes = {str(row.get("scenario_diversity_content_hash") or "") for row in rows}
    return (
        len(starts) == len(rows)
        and "" not in signatures
        and "" not in content_hashes
        and len(signatures) == len(rows)
        and len(content_hashes) == len(rows)
    )


def _lineage_mismatch(summary: dict[str, Any]) -> bool:
    if not summary:
        return True
    return (
        summary.get("coverage_denominator_source") != COVERAGE_DENOMINATOR_SOURCE
        or summary.get("coverage_source") != COVERAGE_SOURCE
        or summary.get("path_cost_source") != PATH_COST_SOURCE
        or summary.get("synthetic_source_kind") != SYNTHETIC_SOURCE_KIND
        or summary.get("action_space_type") != ACTION_SPACE_TYPE
    )


def _input_rejections(stage26_8f_summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not stage26_8f_summary:
        return ["missing_stage26_8f_summary"]
    if stage26_8f_summary.get("schema_version") != "xunce-stage26-8f-summary/v1":
        reasons.append("stage26_8f_schema_version_mismatch")
    if stage26_8f_summary.get("stage_id") != "xunce-stage26-8f-scenario-diversity-and-policy-margin-audit":
        reasons.append("stage26_8f_stage_id_mismatch")
    if stage26_8f_summary.get("status") != "passed":
        reasons.append("stage26_8f_not_passed")
    if stage26_8f_summary.get("next_required_change") != "repair_stage26_synthetic_scenario_diversity":
        reasons.append("stage26_8f_route_mismatch")
    return reasons


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons = [field for field in BOUNDARY_FIELDS if config.get(field) is True]
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _empty_recheck_audit() -> dict[str, Any]:
    return {
        "schema_version": RECHECK_SCHEMA_VERSION,
        "scenario_signature_count": 0,
        "scenario_signature_duplicate_group_count": 0,
        "scenario_signature_duplicate_ratio": 0.0,
        "scenario_diversity_repaired": False,
    }


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = dict(payload)
    config.setdefault("stage26_8f_root", DEFAULT_STAGE26_8F_ROOT)
    config.setdefault("stage26_0_root", stage26_8.DEFAULT_STAGE26_0_ROOT)
    config.setdefault("stage26_1_base_config", stage26_1.DEFAULT_CONFIG)
    config.setdefault("stage26_2_base_config", stage26_2.DEFAULT_CONFIG)
    config.setdefault("stage26_3_base_config", stage26_3.DEFAULT_CONFIG)
    for key in ("stage26_8f_root", "stage26_0_root", "stage26_1_base_config", "stage26_2_base_config", "stage26_3_base_config"):
        if key in config:
            config[key] = str(_resolve_path(Path(str(config[key])), repo_root))
    config["seed"] = _positive_int(config.get("seed", 260801), "seed")
    config["required_scenario_count"] = _positive_int(config.get("required_scenario_count", 3), "required_scenario_count")
    config["collector_rollout_steps"] = _positive_int(config.get("collector_rollout_steps", 16), "collector_rollout_steps")
    config["eval_rollout_steps"] = _positive_int(config.get("eval_rollout_steps", 16), "eval_rollout_steps")
    config["dynamic_max_candidates_per_step"] = _positive_int(config.get("dynamic_max_candidates_per_step", 36), "dynamic_max_candidates_per_step")
    config["dynamic_proposal_pool_limit_per_step"] = _positive_int(config.get("dynamic_proposal_pool_limit_per_step", 288), "dynamic_proposal_pool_limit_per_step")
    config["hybrid_astar_candidate_eval_workers"] = _positive_int(config.get("hybrid_astar_candidate_eval_workers", 4), "hybrid_astar_candidate_eval_workers")
    config["min_trainable_transition_count"] = _positive_int(config.get("min_trainable_transition_count", 12), "min_trainable_transition_count")
    config["min_scenario_start_separation_cells"] = _positive_int(config.get("min_scenario_start_separation_cells", 5), "min_scenario_start_separation_cells")
    config["min_scenario_start_clearance_cells"] = _nonnegative_int(config.get("min_scenario_start_clearance_cells", 1), "min_scenario_start_clearance_cells")
    config["scenario_duplicate_ratio_threshold"] = float(config.get("scenario_duplicate_ratio_threshold", 0.01))
    for key, default in {
        "theta_bin_count": 8,
        "theta_step_deg": 45,
        "sensor_fov_deg": 90.0,
        "sensor_range_cells": 2,
        "synthetic_credit_mixture_probability": 1.0,
        "synthetic_credit_score_version": "path_efficiency_v2",
        "path_efficiency_max_cost_norm": 0.7,
        "epochs": 4,
        "learning_rate": 0.00001,
        "clip_ratio": 0.2,
        "policy_loss_coefficient": 1.0,
        "value_loss_coefficient": 0.02,
        "loss_scale": 0.25,
        "max_grad_norm": 1.0,
        "max_abs_approx_kl": 1.5,
    }.items():
        config.setdefault(key, default)
    config.setdefault("publishes_checkpoint", False)
    config.setdefault("replaces_default_policy", False)
    config.setdefault("connects_real_executor", False)
    config.setdefault("starts_online_canary", False)
    config.setdefault("canary_traffic_fraction", 0.0)
    return config


def _render_report(summary: dict[str, Any], root_cause: dict[str, Any], recheck: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.8G Scenario Diversity Repair",
            "",
            f"- status: `{summary.get('status')}`",
            f"- next_required_change: `{summary.get('next_required_change')}`",
            f"- root_cause: `{root_cause.get('root_cause')}`",
            f"- scenario_fixture_count: `{summary.get('scenario_fixture_count')}`",
            f"- scenario_signature_duplicate_group_count: `{recheck.get('scenario_signature_duplicate_group_count')}`",
            f"- scenario_diversity_repaired: `{recheck.get('scenario_diversity_repaired')}`",
            "",
            "本阶段只修 scenario 多样性和审计绑定，不宣称性能提升。",
        ]
    )


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _nonnegative_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return parsed


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _cell_key(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{int(value[0])},{int(value[1])}"
    return ""


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
