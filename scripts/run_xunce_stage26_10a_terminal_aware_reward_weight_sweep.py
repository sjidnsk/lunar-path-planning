from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_EXPLORER_SRC = SCRIPT_DIR.parent / "model-explorer" / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

try:
    import run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import run_xunce_stage21_3_ppo_batch_validation as stage21_3
    import run_xunce_stage26_10_terminal_aware_reward_shaping as stage26_10
    import xunce_artifact_io as artifact_io
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage21_2_coverage_first_ppo_reward_contract as stage21_2
    import scripts.run_xunce_stage21_3_ppo_batch_validation as stage21_3
    import scripts.run_xunce_stage26_10_terminal_aware_reward_shaping as stage26_10
    import scripts.xunce_artifact_io as artifact_io

from model_explorer.policy.coverage_first_reward import load_coverage_first_reward_profile


STAGE_ID = "xunce-stage26-10a-terminal-aware-reward-weight-sweep"
CONFIG_SCHEMA_VERSION = "xunce-stage26-10a-terminal-aware-reward-weight-sweep-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-10a-terminal-aware-reward-weight-sweep-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-10a-terminal-aware-reward-weight-sweep-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-10a-terminal-aware-reward-weight-sweep-manifest/v1"
MATRIX_ROW_SCHEMA_VERSION = "xunce-stage26-10a-terminal-aware-reward-weight-sweep-matrix-row/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_10a_terminal_aware_reward_weight_sweep_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_10a"
SUMMARY_FILE = "xunce-stage26-10a-summary.json"
ROUTING_FILE = "xunce-stage26-10a-routing.json"
MANIFEST_FILE = "xunce-stage26-10a-manifest.json"
REPORT_FILE = "xunce-stage26-10a-report.md"
MATRIX_FILE = "xunce-stage26-10a-reward-weight-sweep-matrix.jsonl"

ROUTE_BOUNDARY = "resolve_stage26_10a_boundary_rejections"
ROUTE_INPUTS = "repair_stage26_10a_required_inputs"
ROUTE_PROFILE_CONTRACT = "repair_stage26_10a_reward_profile_contract"
ROUTE_CONTINUE = "continue_stage26_10a_topk_bounded_eval"
ROUTE_EVAL = "repair_stage26_10a_eval_binding_or_safety"
ROUTE_CREDIT = "repair_stage26_terminal_credit_assignment"
ROUTE_REVIEW = "review_stage26_10a_terminal_aware_weight_readiness"
STAGE26_9_CONTINUE_ROUTE = "continue_stage26_9_long_horizon_jobs"
STAGE26_9_EVAL_ROUTE = "repair_stage26_9_eval_binding_or_safety"
STAGE26_9_CREDIT_ROUTE = "repair_stage26_synthetic_credit_assignment"

BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)

SWEEP_VARIANTS = (
    ("baseline", 2.0, 1.0, 2.0, 5.0),
    ("conservative", 1.0, 0.5, 1.0, 3.0),
    ("dead_end_high", 4.0, 1.0, 2.0, 5.0),
    ("incomplete_high", 2.0, 2.0, 2.0, 5.0),
    ("terminal_final_high", 2.0, 1.0, 3.0, 5.0),
    ("success_high", 2.0, 1.0, 2.0, 8.0),
    ("terminal_heavy", 2.0, 2.0, 3.0, 8.0),
    ("dead_end_heavy_success_low", 4.0, 1.0, 2.0, 3.0),
    ("balanced_high", 4.0, 2.0, 3.0, 8.0),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage26.10A terminal-aware reward weight sweep.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_10a_terminal_aware_reward_weight_sweep(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_10a_terminal_aware_reward_weight_sweep(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)

    profiles_dir = output_root / "profiles"
    offline_dir = output_root / "offline"
    topk_dir = output_root / "topk"
    artifact_io.make_dirs(profiles_dir)
    artifact_io.make_dirs(offline_dir)
    artifact_io.make_dirs(topk_dir)

    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(config, repo_root)
    matrix_rows: list[dict[str, Any]] = []
    top_k_profile_ids: list[str] = []

    if not boundary_rejections and not input_rejections:
        profile_rows = _write_profile_variants(config, profiles_dir, repo_root)
        for profile_row in profile_rows:
            matrix_rows.append(_run_offline_profile(config, profile_row, offline_dir, repo_root))
        top_rows = _top_k_rows(matrix_rows, int(config["top_k"]))
        top_k_profile_ids = [str(row["profile_id"]) for row in top_rows]
        _materialize_topk_configs(config, top_rows, topk_dir, repo_root)
        matrix_rows = _refresh_topk_rows(matrix_rows, topk_dir)
        route_before_next_topk = _route(boundary_rejections, input_rejections, matrix_rows, top_k_profile_ids)
        if route_before_next_topk == ROUTE_CONTINUE:
            _run_one_pending_topk(config, top_rows, topk_dir, repo_root)
            matrix_rows = _refresh_topk_rows(matrix_rows, topk_dir)

    route = _route(boundary_rejections, input_rejections, matrix_rows, top_k_profile_ids)
    status = "passed" if route in {ROUTE_CONTINUE, ROUTE_REVIEW} else "failed"
    summary = _summary(
        config=config,
        output_root=output_root,
        route=route,
        status=status,
        boundary_rejections=boundary_rejections,
        input_rejections=input_rejections,
        matrix_rows=matrix_rows,
        top_k_profile_ids=top_k_profile_ids,
    )
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
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
        "summary_status": status,
        "next_required_change": route,
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "manifest": str(output_root / MANIFEST_FILE),
            "report": str(output_root / REPORT_FILE),
            "matrix": str(output_root / MATRIX_FILE),
            "profiles": str(profiles_dir),
            "offline": str(offline_dir),
            "topk": str(topk_dir),
        },
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_jsonl(output_root / MATRIX_FILE, matrix_rows)
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not artifact_io.path_is_file(path):
        raise FileNotFoundError(f"Stage26.10A config not found: {path}")
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if payload.get("stage_id") != STAGE_ID:
        raise ValueError(f"stage_id must be {STAGE_ID}")
    defaults = {
        "stage21_1_collector_root": "",
        "baseline_reward_profile": "configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json",
        "base_stage21_2_config": "configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json",
        "base_stage21_3_config": "configs/xunce_stage21_3_ppo_batch_validation_v1.json",
        "base_stage26_10_config": "configs/xunce_stage26_10_terminal_aware_reward_shaping_v1.json",
        "top_k": 3,
        "run_topk_stage26_10": True,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    config = defaults | payload
    config["top_k"] = _positive_int(config["top_k"], "top_k")
    if not isinstance(config.get("run_topk_stage26_10"), bool):
        raise ValueError("run_topk_stage26_10 must be a JSON boolean")
    for field in BOUNDARY_FIELDS:
        config[field] = bool(config.get(field, False))
    config["canary_traffic_fraction"] = float(config.get("canary_traffic_fraction", 0.0))
    return config


def _write_profile_variants(config: dict[str, Any], profiles_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    baseline_path = _resolve_path(Path(str(config["baseline_reward_profile"])), repo_root)
    baseline_payload = _read_json(baseline_path)
    rows: list[dict[str, Any]] = []
    for index, (variant_id, dead_end, incomplete, terminal_final, success) in enumerate(SWEEP_VARIANTS):
        payload = json.loads(json.dumps(baseline_payload))
        payload["profile_id"] = f"xunce-stage26-10a-{variant_id}"
        payload["profile_version"] = "stage26-10-terminal-aware-v3"
        payload["weights"]["dead_end"] = float(dead_end)
        payload["weights"]["incomplete_terminal"] = float(incomplete)
        payload["weights"]["terminal_final_coverage"] = float(terminal_final)
        payload["weights"]["success_99pct"] = float(success)
        path = profiles_dir / f"{variant_id}.json"
        _write_json(path, payload)
        profile = load_coverage_first_reward_profile(path)
        rows.append(
            {
                "variant_index": index,
                "variant_id": variant_id,
                "profile_id": profile.profile_id,
                "profile_hash": profile.profile_hash,
                "profile_path": str(path.resolve()),
                "weights": dict(profile.weights),
            }
        )
    return rows


def _run_offline_profile(config: dict[str, Any], profile_row: dict[str, Any], offline_dir: Path, repo_root: Path) -> dict[str, Any]:
    profile_id = str(profile_row["profile_id"])
    root = offline_dir / profile_id
    stage21_2_root = root / "stage21_2"
    stage21_3_root = root / "stage21_3"
    artifact_io.make_dirs(root)
    stage21_2_config = root / "stage21_2_config.json"
    stage21_3_config = root / "stage21_3_config.json"
    _write_json(stage21_2_config, _stage21_2_config(config, profile_row, repo_root))
    stage21_2_summary = stage21_2.run_xunce_stage21_2_coverage_first_ppo_reward_contract(
        config_path=stage21_2_config,
        output_root=stage21_2_root,
        repo_root=repo_root,
    )
    _write_json(stage21_3_config, _stage21_3_config(config, stage21_2_root, repo_root))
    stage21_3_summary = stage21_3.run_xunce_stage21_3_ppo_batch_validation(
        config_path=stage21_3_config,
        output_root=stage21_3_root,
        repo_root=repo_root,
    )
    reward_rows = _read_jsonl(stage21_2_root / stage21_2.EVALUATION_FILE)
    hard_risk_positive = int(stage21_2_summary.get("hard_risk_positive_reward_count") or 0)
    reward_trainable_false = int(stage21_2_summary.get("reward_trainable_false_count") or 0)
    reward_reason_counts = _reward_reason_code_counts(reward_rows)
    all_reward_reason_codes_nonblocking = (
        reward_reason_counts["reward_reason_code_total_count"]
        == reward_reason_counts["nonblocking_reward_reason_code_count"]
    )
    offline_passed = (
        stage21_2_summary.get("status") == "passed"
        and stage21_3_summary.get("status") == "passed"
        and hard_risk_positive == 0
        and reward_trainable_false == 0
        and int(stage21_3_summary.get("reward_trainable_false_count") or 0) == 0
        and all_reward_reason_codes_nonblocking
    )
    return {
        "schema_version": MATRIX_ROW_SCHEMA_VERSION,
        **profile_row,
        "offline_status": "passed" if offline_passed else "failed",
        "stage21_2_status": stage21_2_summary.get("status"),
        "stage21_3_status": stage21_3_summary.get("status"),
        "stage21_2_root": str(stage21_2_root),
        "stage21_3_root": str(stage21_3_root),
        "hard_risk_positive_reward_count": hard_risk_positive,
        "reward_trainable_false_count": reward_trainable_false,
        **reward_reason_counts,
        "all_reward_reason_codes_nonblocking": all_reward_reason_codes_nonblocking,
        "dead_end_penalty_component_sum": _component_sum(reward_rows, "dead_end_penalty_component"),
        "incomplete_terminal_penalty_component_sum": _component_sum(reward_rows, "incomplete_terminal_penalty_component"),
        "coverage_gain_component_sum": _component_sum(reward_rows, "coverage_gain_component"),
        "offline_score": _offline_score(profile_row),
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _stage21_2_config(config: dict[str, Any], profile_row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(Path(str(config["base_stage21_2_config"])), repo_root))
    payload["stage21_1_collector_root"] = str(_resolve_path(Path(str(config["stage21_1_collector_root"])), repo_root))
    payload["coverage_first_reward_profile"] = str(profile_row["profile_path"])
    payload["stage21_2_authorized"] = False
    payload["runs_new_ppo_update"] = False
    payload["publishes_checkpoint"] = False
    payload["replaces_default_policy"] = False
    payload["connects_real_executor"] = False
    payload["starts_online_canary"] = False
    payload["canary_traffic_fraction"] = 0.0
    return payload


def _stage21_3_config(config: dict[str, Any], stage21_2_root: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(Path(str(config["base_stage21_3_config"])), repo_root))
    payload["stage21_1_collector_root"] = str(_resolve_path(Path(str(config["stage21_1_collector_root"])), repo_root))
    payload["stage21_2_reward_contract_root"] = str(stage21_2_root)
    payload["stage21_3_authorized"] = False
    payload["runs_new_ppo_update"] = False
    payload["publishes_checkpoint"] = False
    payload["replaces_default_policy"] = False
    payload["connects_real_executor"] = False
    payload["starts_online_canary"] = False
    payload["canary_traffic_fraction"] = 0.0
    return payload


def _top_k_rows(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    passed = [row for row in rows if row.get("offline_status") == "passed"]
    return sorted(passed, key=_ranking_key)[:top_k]


def _ranking_key(row: dict[str, Any]) -> tuple[float, int]:
    return (-float(row.get("offline_score") or 0.0), int(row.get("variant_index") or 0))


def _offline_score(profile_row: dict[str, Any]) -> float:
    weights = profile_row["weights"]
    signal = (
        float(weights["dead_end"])
        + float(weights["incomplete_terminal"])
        + float(weights["terminal_final_coverage"])
        + 0.2 * float(weights["success_99pct"])
    )
    over_penalty = max(0.0, float(weights["dead_end"]) - 2.0) * 0.25
    over_penalty += max(0.0, float(weights["incomplete_terminal"]) - 1.0) * 0.2
    return signal - over_penalty


def _materialize_topk_configs(config: dict[str, Any], top_rows: list[dict[str, Any]], topk_dir: Path, repo_root: Path) -> None:
    base = _read_json(_resolve_path(Path(str(config["base_stage26_10_config"])), repo_root))
    for row in top_rows:
        root = topk_dir / str(row["profile_id"])
        artifact_io.make_dirs(root)
        payload = dict(base)
        payload["terminal_aware_reward_profile"] = str(row["profile_path"])
        payload["release_or_training_authorized"] = False
        payload["publishes_checkpoint"] = False
        payload["replaces_default_policy"] = False
        payload["connects_real_executor"] = False
        payload["starts_online_canary"] = False
        payload["canary_traffic_fraction"] = 0.0
        _write_json(root / "stage26_10_config.json", payload)


def _run_one_pending_topk(config: dict[str, Any], top_rows: list[dict[str, Any]], topk_dir: Path, repo_root: Path) -> None:
    if not bool(config["run_topk_stage26_10"]):
        return
    for row in top_rows:
        root = topk_dir / str(row["profile_id"])
        summary_path = root / stage26_10.SUMMARY_FILE
        if artifact_io.path_is_file(summary_path) and not _topk_summary_needs_continue(_read_json(summary_path)):
            continue
        summary = stage26_10.run_xunce_stage26_10_terminal_aware_reward_shaping(
            config_path=root / "stage26_10_config.json",
            output_root=root / "stage26_10",
            repo_root=repo_root,
        )
        _write_json(summary_path, summary)
        return


def _refresh_topk_rows(rows: list[dict[str, Any]], topk_dir: Path) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        root = topk_dir / str(row["profile_id"])
        summary_path = root / stage26_10.SUMMARY_FILE
        if artifact_io.path_is_file(summary_path):
            summary = _read_json(summary_path)
            is_continuation = _topk_summary_needs_continue(summary)
            updated.update(
                {
                    "stage26_10_status": "pending" if is_continuation else summary.get("status"),
                    "stage26_10_summary_status": summary.get("status"),
                    "stage26_10_next_required_change": summary.get("next_required_change"),
                    "main_coverage_per_100m_delta": _first_float(
                        summary.get("main_coverage_per_100m_delta"),
                        summary.get("mean_main_coverage_per_100m_delta"),
                    ),
                    "selected_action_changed_count": int(summary.get("selected_action_changed_count") or 0),
                    "binding_or_safety_failure": bool(summary.get("binding_or_safety_failure")),
                }
            )
        elif artifact_io.path_is_dir(root):
            updated["stage26_10_status"] = "pending"
        refreshed.append(updated)
    return refreshed


def _topk_summary_needs_continue(summary: dict[str, Any]) -> bool:
    return str(summary.get("next_required_change") or "") == STAGE26_9_CONTINUE_ROUTE


def _route(
    boundary_rejections: list[str],
    input_rejections: list[str],
    rows: list[dict[str, Any]],
    top_k_profile_ids: list[str],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    passed = [row for row in rows if row.get("offline_status") == "passed"]
    if not passed:
        return ROUTE_PROFILE_CONTRACT
    top_rows = [row for row in rows if row.get("profile_id") in set(top_k_profile_ids)]
    if any(bool(row.get("binding_or_safety_failure")) for row in top_rows):
        return ROUTE_EVAL
    if any(row.get("stage26_10_next_required_change") == STAGE26_9_EVAL_ROUTE for row in top_rows):
        return ROUTE_EVAL
    if any(row.get("stage26_10_next_required_change") == STAGE26_9_CREDIT_ROUTE for row in top_rows):
        return ROUTE_CREDIT
    completed = [row for row in top_rows if row.get("stage26_10_status") == "passed"]
    if any(float(row.get("main_coverage_per_100m_delta") or 0.0) > 0.0 for row in completed):
        return ROUTE_REVIEW
    if len(completed) == len(top_rows) and completed:
        if all(
            int(row.get("selected_action_changed_count") or 0) == 0
            and float(row.get("main_coverage_per_100m_delta") or 0.0) == 0.0
            for row in completed
        ):
            return ROUTE_CREDIT
    return ROUTE_CONTINUE


def _summary(
    *,
    config: dict[str, Any],
    output_root: Path,
    route: str,
    status: str,
    boundary_rejections: list[str],
    input_rejections: list[str],
    matrix_rows: list[dict[str, Any]],
    top_k_profile_ids: list[str],
) -> dict[str, Any]:
    passed = [row for row in matrix_rows if row.get("offline_status") == "passed"]
    best = next((row for row in matrix_rows if row.get("profile_id") in set(top_k_profile_ids[:1])), {})
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "profile_count": len(matrix_rows),
        "offline_passed_count": len(passed),
        "top_k": int(config["top_k"]),
        "top_k_profile_ids": top_k_profile_ids,
        "best_profile_id": best.get("profile_id"),
        "best_profile_hash": best.get("profile_hash"),
        "matrix": str(output_root / MATRIX_FILE),
        "summary": str(output_root / SUMMARY_FILE),
        "routing": str(output_root / ROUTING_FILE),
        "manifest": str(output_root / MANIFEST_FILE),
        "report": str(output_root / REPORT_FILE),
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if config.get(field) is not False:
            reasons.append(field)
    if float(config.get("canary_traffic_fraction", 0.0)) != 0.0:
        reasons.append("canary_traffic_fraction")
    return reasons


def _input_rejections(config: dict[str, Any], repo_root: Path) -> list[str]:
    reasons: list[str] = []
    for field in (
        "stage21_1_collector_root",
        "baseline_reward_profile",
        "base_stage21_2_config",
        "base_stage21_3_config",
        "base_stage26_10_config",
    ):
        value = str(config.get(field) or "")
        if not value or not artifact_io.path_exists(_resolve_path(Path(value), repo_root)):
            reasons.append(f"missing_{field}")
    return reasons


def _component_sum(rows: list[dict[str, Any]], component: str) -> float:
    total = 0.0
    for row in rows:
        components = row.get("components") if isinstance(row.get("components"), dict) else {}
        total += float(components.get(component) or 0.0)
    return total


def _reward_reason_code_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = {str(code) for code in stage21_3.NONBLOCKING_REWARD_REASON_CODES}
    total = 0
    nonblocking = 0
    unknown: list[str] = []
    for row in rows:
        codes = row.get("reason_codes") if isinstance(row.get("reason_codes"), list) else []
        for code in codes:
            text = str(code)
            total += 1
            if text in allowed:
                nonblocking += 1
            elif text not in unknown:
                unknown.append(text)
    return {
        "reward_reason_code_total_count": total,
        "nonblocking_reward_reason_code_count": nonblocking,
        "unknown_or_blocking_reward_reason_codes": unknown,
    }


def _first_float(*values: Any) -> float | None:
    for value in values:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.10A Terminal-Aware Reward Weight Sweep",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- profile_count: `{summary['profile_count']}`",
            f"- offline_passed_count: `{summary['offline_passed_count']}`",
            f"- best_profile_id: `{summary.get('best_profile_id')}`",
            f"- matrix: `{summary['matrix']}`",
            "- publishes_checkpoint: `False`",
            "- replaces_default_policy: `False`",
            "- connects_real_executor: `False`",
            "- starts_online_canary: `False`",
            "",
        ]
    )


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return artifact_io.read_jsonl(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    artifact_io.write_jsonl(path, rows)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
