from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as stage26_9
    import xunce_artifact_io as artifact_io
except ModuleNotFoundError:  # pragma: no cover
    import scripts.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot as stage26_9
    import scripts.xunce_artifact_io as artifact_io


STAGE_ID = "xunce-stage26-10-terminal-aware-reward-shaping"
CONFIG_SCHEMA_VERSION = "xunce-stage26-10-terminal-aware-reward-shaping-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage26-10-terminal-aware-reward-shaping-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage26-10-terminal-aware-reward-shaping-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage26-10-terminal-aware-reward-shaping-manifest/v1"

DEFAULT_CONFIG = "configs/xunce_stage26_10_terminal_aware_reward_shaping_v1.json"
DEFAULT_OUTPUT_ROOT = "D:/xunce/out/s26_10"
SUMMARY_FILE = "xunce-stage26-10-summary.json"
ROUTING_FILE = "xunce-stage26-10-routing.json"
MANIFEST_FILE = "xunce-stage26-10-manifest.json"
REPORT_FILE = "xunce-stage26-10-report.md"
DERIVED_STAGE26_1_CONFIG_FILE = "xunce-stage26-10-derived-stage26-1-config.json"
DERIVED_STAGE26_9_CONFIG_FILE = "xunce-stage26-10-derived-stage26-9-config.json"

ROUTE_BOUNDARY = "resolve_stage26_10_boundary_rejections"
ROUTE_INPUTS = "repair_stage26_10_required_inputs"
ROUTE_RUN_STAGE26_9 = "run_stage26_9_terminal_aware_long_horizon_pilot"

TERMINAL_AWARE_REWARD_PROFILE = "configs/xunce_stage26_10_terminal_aware_ppo_reward_profile_v3.json"
BOUNDARY_FIELDS = (
    "release_or_training_authorized",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage26.10 terminal-aware reward shaping orchestrator.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    summary = run_xunce_stage26_10_terminal_aware_reward_shaping(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_stage26_10_terminal_aware_reward_shaping(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config = _load_config(_resolve_path(config_path, repo_root), repo_root)
    output_root = _resolve_path(output_root, repo_root)
    artifact_io.make_dirs(output_root)

    boundary_rejections = _boundary_rejections(config)
    input_rejections = _input_rejections(config, repo_root)
    derived_stage26_1_config = output_root / DERIVED_STAGE26_1_CONFIG_FILE
    derived_stage26_9_config = output_root / DERIVED_STAGE26_9_CONFIG_FILE
    stage26_9_summary: dict[str, Any] = {}

    if not boundary_rejections and not input_rejections:
        _write_json(derived_stage26_1_config, _derived_stage26_1_config(config, repo_root))
        _write_json(derived_stage26_9_config, _derived_stage26_9_config(config, repo_root, derived_stage26_1_config))
        if bool(config["run_stage26_9"]):
            stage26_9_summary = stage26_9.run_xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot(
                config_path=derived_stage26_9_config,
                output_root=output_root / "s26_9",
                repo_root=repo_root,
            )

    route = _route(boundary_rejections, input_rejections, stage26_9_summary, config)
    status = "passed" if not boundary_rejections and not input_rejections else "failed"
    if stage26_9_summary:
        status = "passed" if stage26_9_summary.get("status") == "passed" else "failed"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "boundary_rejections": boundary_rejections,
        "input_rejections": input_rejections,
        "terminal_aware_reward_profile": config["terminal_aware_reward_profile"],
        "base_stage26_1_config": config["base_stage26_1_config"],
        "base_stage26_9_config": config["base_stage26_9_config"],
        "derived_stage26_1_config": str(derived_stage26_1_config),
        "derived_stage26_9_config": str(derived_stage26_9_config),
        "stage26_9_status": stage26_9_summary.get("status"),
        "stage26_9_next_required_change": stage26_9_summary.get("next_required_change"),
        "coverage_source": "endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source": "hybrid_astar_pose_path/v1",
        "action_space_type": "hybrid_discrete_xy_continuous_theta/v1",
        "candidate_reachability_gate_source": "hybrid_astar_pose_reachability/v1",
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
        "terminal_aware_reward_profile": config["terminal_aware_reward_profile"],
        "artifacts": {
            "summary": str(output_root / SUMMARY_FILE),
            "routing": str(output_root / ROUTING_FILE),
            "manifest": str(output_root / MANIFEST_FILE),
            "report": str(output_root / REPORT_FILE),
            "derived_stage26_1_config": str(derived_stage26_1_config),
            "derived_stage26_9_config": str(derived_stage26_9_config),
        },
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    artifact_io.write_text(output_root / REPORT_FILE, _render_report(summary))
    return summary


def _load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    if not artifact_io.path_is_file(path):
        raise FileNotFoundError(f"Stage26.10 config not found: {path}")
    payload = _read_json(path)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = {**_default_config(), **payload}
    for field in ("base_stage26_1_config", "base_stage26_9_config", "terminal_aware_reward_profile"):
        value = config.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    if not isinstance(config.get("run_stage26_9"), bool):
        raise ValueError("run_stage26_9 must be a JSON boolean")
    config["terminal_aware_reward_profile"] = _relative_if_under_repo(
        _resolve_path(Path(str(config["terminal_aware_reward_profile"])), repo_root),
        repo_root,
    )
    return config


def _default_config() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "base_stage26_1_config": "configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json",
        "base_stage26_9_config": "configs/xunce_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot_v1.json",
        "terminal_aware_reward_profile": TERMINAL_AWARE_REWARD_PROFILE,
        "run_stage26_9": True,
        "release_or_training_authorized": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }


def _derived_stage26_1_config(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(Path(config["base_stage26_1_config"]), repo_root))
    payload["coverage_first_reward_profile"] = config["terminal_aware_reward_profile"]
    payload["publishes_checkpoint"] = False
    payload["replaces_default_policy"] = False
    payload["connects_real_executor"] = False
    payload["starts_online_canary"] = False
    payload["canary_traffic_fraction"] = 0.0
    return payload


def _derived_stage26_9_config(config: dict[str, Any], repo_root: Path, derived_stage26_1_config: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(Path(config["base_stage26_9_config"]), repo_root))
    payload["base_stage26_1_config"] = str(derived_stage26_1_config)
    payload["coverage_source"] = "endpoint_theta_slope_obstacle_los/v1"
    payload["path_cost_source"] = "hybrid_astar_pose_path/v1"
    payload["action_space_type"] = "hybrid_discrete_xy_continuous_theta/v1"
    payload["candidate_reachability_gate_source"] = "hybrid_astar_pose_reachability/v1"
    payload["release_or_training_authorized"] = False
    payload["publishes_checkpoint"] = False
    payload["replaces_default_policy"] = False
    payload["connects_real_executor"] = False
    payload["starts_online_canary"] = False
    payload["canary_traffic_fraction"] = 0.0
    return payload


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
    for field in ("base_stage26_1_config", "base_stage26_9_config", "terminal_aware_reward_profile"):
        if not artifact_io.path_is_file(_resolve_path(Path(str(config[field])), repo_root)):
            reasons.append(f"missing_{field}")
    if config.get("stage_id") != STAGE_ID:
        reasons.append("stage_id_mismatch")
    return reasons


def _route(
    boundary_rejections: list[str],
    input_rejections: list[str],
    stage26_9_summary: dict[str, Any],
    config: dict[str, Any],
) -> str:
    if boundary_rejections:
        return ROUTE_BOUNDARY
    if input_rejections:
        return ROUTE_INPUTS
    if stage26_9_summary:
        return str(stage26_9_summary.get("next_required_change") or ROUTE_RUN_STAGE26_9)
    return ROUTE_RUN_STAGE26_9 if bool(config["run_stage26_9"]) else "review_stage26_10_generated_terminal_aware_configs"


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage26.10 Terminal-Aware Reward Shaping",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- terminal_aware_reward_profile: `{summary['terminal_aware_reward_profile']}`",
            f"- derived_stage26_1_config: `{summary['derived_stage26_1_config']}`",
            f"- derived_stage26_9_config: `{summary['derived_stage26_9_config']}`",
            "- publishes_checkpoint: `False`",
            "- replaces_default_policy: `False`",
            "- connects_real_executor: `False`",
            "- starts_online_canary: `False`",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    return artifact_io.read_json(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    artifact_io.write_json(path, payload)


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _relative_if_under_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
