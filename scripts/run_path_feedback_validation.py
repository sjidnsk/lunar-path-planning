from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from platform_command import display_command


SCENARIO_SETS = {
    "smoke",
    "stress",
    "holdout",
    "raw_align_train",
    "raw_align_val",
    "raw_align_test",
    "policy_canary",
    "policy_canary_diversity",
    "policy_canary_opportunity_quality",
    "policy_canary_dense_choke_opportunity",
    "policy_canary_value_stability",
    "policy_canary_sequential_multi_step_opportunity",
    "all",
}
DIAGNOSTIC_PROFILES = {"baseline", "execution", "iris", "all"}
MODULES = ("path-planner", "model-explorer", "dev-platform-constraints")
CONTROL_POINT_VALUE_ARGS = {
    "--gcs-control-point-terrain-weight",
    "--gcs-control-point-second-difference-weight",
    "--gcs-control-point-high-cost-exposure-weight",
    "--gcs-control-point-direction-cone-max-error-deg",
    "--gcs-control-point-direction-cone-rho-floor-m",
    "--gcs-control-point-direction-cone-seed-rho-ratio",
}
CHANNEL_AWARE_VALUE_ARGS = {
    "--channel-aware-neighborhood-radius-cells",
    "--channel-aware-center-weight",
    "--channel-aware-neighborhood-mean-weight",
    "--channel-aware-neighborhood-max-weight",
    "--channel-aware-high-cost-exposure-weight",
    "--channel-aware-blocked-nearby-weight",
    "--channel-aware-clearance-weight",
    "--channel-aware-smoothness-weight",
    "--channel-aware-high-cost-threshold",
}


class PathFeedbackConfigError(ValueError):
    pass


def main(argv: list[str] | None = None) -> int:
    try:
        config = _parse_args(argv)
        _run(config)
    except PathFeedbackConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)
    return 0


def _parse_args(argv: list[str] | None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run the cross-platform semi-real path feedback validation chain."
    )
    parser.add_argument("--output-root", default="outputs/path_feedback_validation")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--scenario-set", choices=sorted(SCENARIO_SETS), default="smoke")
    parser.add_argument("--scenario-spec-json", default="")
    parser.add_argument("--diagnostic-profile", choices=sorted(DIAGNOSTIC_PROFILES), default="baseline")
    parser.add_argument("--simulate-tracking", action="append_const", const="--simulate-tracking", dest="planner_flags")
    parser.add_argument("--optimize-trajectory", action="append_const", const="--optimize-trajectory", dest="planner_flags")
    parser.add_argument("--drake-iris-regions", action="append_const", const="--drake-iris-regions", dest="planner_flags")
    parser.add_argument("--gcs-trajectory-smoke", action="append_const", const="--gcs-trajectory-smoke", dest="planner_flags")
    parser.add_argument("--gcs-geometric-candidate", action="append_const", const="--gcs-geometric-candidate", dest="planner_flags")
    parser.add_argument("--gcs-motion-feasibility", action="append_const", const="--gcs-motion-feasibility", dest="planner_flags")
    parser.add_argument(
        "--gcs-curvature-constrained-candidate",
        action="append_const",
        const="--gcs-curvature-constrained-candidate",
        dest="planner_flags",
    )
    parser.add_argument("--gcs-control-point-candidate", action="append_const", const="--gcs-control-point-candidate", dest="planner_flags")
    parser.add_argument("--anchor-projection-candidate-generation", action="store_true")
    parser.add_argument("--anchor-projection-selection-path-cost-bonus", default="0.0")
    parser.add_argument("--anchor-projection-max-selection-path-cost-regression", default="6.0")
    parser.add_argument("--anchor-projection-max-selection-risk-regression", default="0.5")
    parser.add_argument("--anchor-projection-contract-aware-trainable-target-generation", action="store_true")
    parser.add_argument("--anchor-projection-prefer-contract-safe-trainable-targets", action="store_true")
    parser.add_argument("--anchor-projection-max-trainable-distance-cells", default="2")
    parser.add_argument("--anchor-projection-max-trainable-distance-m", default="1.0")
    parser.add_argument("--anchor-projection-planner-validated-trainable-target-mining", action="store_true")
    parser.add_argument("--anchor-projection-allow-planner-validated-distance-exception", action="store_true")
    parser.add_argument("--anchor-projection-max-planner-validated-distance-cells", default="3")
    parser.add_argument("--anchor-projection-max-planner-validated-distance-m", default="1.5")
    for option in sorted(CONTROL_POINT_VALUE_ARGS | CHANNEL_AWARE_VALUE_ARGS):
        parser.add_argument(option, dest=_dest_from_option(option))
    parser.add_argument("--planning-backend", choices=("astar", "region_graph_guided", "channel_aware_astar"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.top_k < 1:
        raise PathFeedbackConfigError("--top-k must be a positive integer")

    planner_extra_args = list(args.planner_flags or [])
    for profile_arg in _profile_default_args(args.diagnostic_profile):
        _append_unique(planner_extra_args, profile_arg)
    if args.planning_backend:
        planner_extra_args.extend(["--planning-backend", args.planning_backend])
    for option in sorted(CONTROL_POINT_VALUE_ARGS | CHANNEL_AWARE_VALUE_ARGS):
        value = getattr(args, _dest_from_option(option))
        if value is not None:
            planner_extra_args.extend([option, value])

    for option in CONTROL_POINT_VALUE_ARGS:
        if option in planner_extra_args and "--gcs-control-point-candidate" not in planner_extra_args:
            raise PathFeedbackConfigError(f"{option} requires --gcs-control-point-candidate")

    anchor_numbers = {
        "anchor_projection_selection_path_cost_bonus": args.anchor_projection_selection_path_cost_bonus,
        "anchor_projection_max_selection_path_cost_regression": args.anchor_projection_max_selection_path_cost_regression,
        "anchor_projection_max_selection_risk_regression": args.anchor_projection_max_selection_risk_regression,
        "anchor_projection_max_trainable_distance_cells": args.anchor_projection_max_trainable_distance_cells,
        "anchor_projection_max_trainable_distance_m": args.anchor_projection_max_trainable_distance_m,
        "anchor_projection_max_planner_validated_distance_cells": args.anchor_projection_max_planner_validated_distance_cells,
        "anchor_projection_max_planner_validated_distance_m": args.anchor_projection_max_planner_validated_distance_m,
    }
    parsed_anchor_numbers = {key: _non_negative_float(value, key) for key, value in anchor_numbers.items()}
    if not args.anchor_projection_candidate_generation:
        if parsed_anchor_numbers["anchor_projection_selection_path_cost_bonus"] != 0.0:
            raise PathFeedbackConfigError(
                "--anchor-projection-selection-path-cost-bonus requires --anchor-projection-candidate-generation"
            )
        if (
            args.anchor_projection_contract_aware_trainable_target_generation
            or args.anchor_projection_prefer_contract_safe_trainable_targets
            or args.anchor_projection_planner_validated_trainable_target_mining
            or args.anchor_projection_allow_planner_validated_distance_exception
        ):
            raise PathFeedbackConfigError(
                "contract-aware anchor-projection options require --anchor-projection-candidate-generation"
            )

    repo_root = Path(__file__).resolve().parents[1]
    output_root = _resolve_path(args.output_root, repo_root)
    python_bin = os.environ.get("PYTHON") or sys.executable
    acceptance_gate = (
        "semi-real-closed-loop"
        if args.scenario_set == "all" and args.diagnostic_profile == "all" and args.top_k == 3
        else "custom"
    )

    return {
        "repo_root": repo_root,
        "output_root": output_root,
        "top_k": args.top_k,
        "scenario_set": args.scenario_set,
        "scenario_spec_json": args.scenario_spec_json,
        "diagnostic_profile": args.diagnostic_profile,
        "acceptance_gate": acceptance_gate,
        "dry_run": args.dry_run,
        "python_bin": python_bin,
        "planner_extra_args": planner_extra_args,
        "anchor_projection_candidate_generation": args.anchor_projection_candidate_generation,
        "anchor_projection_selection_path_cost_bonus": parsed_anchor_numbers["anchor_projection_selection_path_cost_bonus"],
        "anchor_projection_max_selection_path_cost_regression": parsed_anchor_numbers[
            "anchor_projection_max_selection_path_cost_regression"
        ],
        "anchor_projection_max_selection_risk_regression": parsed_anchor_numbers[
            "anchor_projection_max_selection_risk_regression"
        ],
        "anchor_projection_contract_aware_trainable_target_generation": args.anchor_projection_contract_aware_trainable_target_generation,
        "anchor_projection_prefer_contract_safe_trainable_targets": args.anchor_projection_prefer_contract_safe_trainable_targets,
        "anchor_projection_max_trainable_distance_cells": int(
            parsed_anchor_numbers["anchor_projection_max_trainable_distance_cells"]
        ),
        "anchor_projection_max_trainable_distance_m": parsed_anchor_numbers["anchor_projection_max_trainable_distance_m"],
        "anchor_projection_planner_validated_trainable_target_mining": args.anchor_projection_planner_validated_trainable_target_mining,
        "anchor_projection_allow_planner_validated_distance_exception": args.anchor_projection_allow_planner_validated_distance_exception,
        "anchor_projection_max_planner_validated_distance_cells": int(
            parsed_anchor_numbers["anchor_projection_max_planner_validated_distance_cells"]
        ),
        "anchor_projection_max_planner_validated_distance_m": parsed_anchor_numbers[
            "anchor_projection_max_planner_validated_distance_m"
        ],
    }


def _run(config: dict[str, Any]) -> None:
    repo_root: Path = config["repo_root"]
    output_root: Path = config["output_root"]
    map_dir = output_root / "maps"
    scenario_config = output_root / "npz_validation_scenarios.json"
    export_dir = output_root / "path_planner_sidecars"
    manifest_path = output_root / "path-feedback-manifest.json"
    summary_path = output_root / "path-feedback-summary.json"
    report_path = output_root / "path-feedback-summary.md"
    dev_root = repo_root / "dev-platform-constraints"
    model_root = repo_root / "model-explorer"
    path_planner_root = repo_root / "path-planner"

    print(f"Repository: {repo_root}")
    print(f"Output root: {output_root}")
    print(f"Python executable: {config['python_bin']}")
    print(f"Acceptance gate: {config['acceptance_gate']}")
    print(f"Top-K: {config['top_k']}")
    print(f"Scenario set: {config['scenario_set']}")
    print(f"Scenario spec JSON: {config['scenario_spec_json'] or '(none)'}")
    print(f"Diagnostic profile: {config['diagnostic_profile']}")
    print(f"Planner extra args: {_format_extra_args(config['planner_extra_args'])}")
    print(
        "Anchor projection candidate generation: "
        f"{'enabled' if config['anchor_projection_candidate_generation'] else 'disabled'}"
    )
    print(f"Anchor projection selection path-cost bonus: {config['anchor_projection_selection_path_cost_bonus']}")
    print(
        "Anchor projection max selection path-cost regression: "
        f"{config['anchor_projection_max_selection_path_cost_regression']}"
    )
    print(
        "Anchor projection max selection risk regression: "
        f"{config['anchor_projection_max_selection_risk_regression']}"
    )
    print(
        "Anchor projection contract-aware trainable target generation: "
        f"{'enabled' if config['anchor_projection_contract_aware_trainable_target_generation'] else 'disabled'}"
    )
    print(
        "Anchor projection prefer contract-safe trainable targets: "
        f"{'enabled' if config['anchor_projection_prefer_contract_safe_trainable_targets'] else 'disabled'}"
    )
    print(f"Anchor projection max trainable distance cells: {config['anchor_projection_max_trainable_distance_cells']}")
    print(f"Anchor projection max trainable distance m: {config['anchor_projection_max_trainable_distance_m']}")

    _ensure_submodules(repo_root, dry_run=config["dry_run"])
    if config["dry_run"]:
        print(f"[DRY RUN] mkdir -p {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    generator_args = [
        config["python_bin"],
        "scripts/generate_npz_validation_maps.py",
        "--scenario-set",
        config["scenario_set"],
        "--output-dir",
        str(map_dir),
        "--scenario-config",
        str(scenario_config),
    ]
    if config["scenario_spec_json"]:
        generator_args.extend(["--scenario-spec-json", config["scenario_spec_json"]])
    _run_command(generator_args, cwd=dev_root, dry_run=config["dry_run"])

    _run_pythonpath_command(
        [
            config["python_bin"],
            "scripts/export_path_planner_sidecars.py",
            "--scenario-config",
            str(scenario_config),
            "--output-dir",
            str(export_dir),
            "--top-k",
            str(config["top_k"]),
        ],
        cwd=dev_root,
        dry_run=config["dry_run"],
    )
    _write_manifest(
        config,
        scenario_config=scenario_config,
        export_dir=export_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        report_path=report_path,
        path_planner_root=path_planner_root,
    )
    _run_pythonpath_command(
        [config["python_bin"], "-m", "model_explorer", "path-feedback", "validate", str(manifest_path)],
        cwd=model_root,
        dry_run=config["dry_run"],
    )
    _run_pythonpath_command(
        [config["python_bin"], "-m", "model_explorer", "path-feedback", "run", str(manifest_path)],
        cwd=model_root,
        dry_run=config["dry_run"],
    )
    if config["dry_run"]:
        print(f"[DRY RUN] validate summary gates: {summary_path}")
    else:
        _assert_output_files(manifest_path, summary_path, report_path)
        _validate_summary(config, summary_path=summary_path, manifest_path=manifest_path, scenario_config_path=scenario_config)


def _write_manifest(
    config: dict[str, Any],
    *,
    scenario_config: Path,
    export_dir: Path,
    manifest_path: Path,
    summary_path: Path,
    report_path: Path,
    path_planner_root: Path,
) -> None:
    if config["dry_run"]:
        print(f"[DRY RUN] write path-feedback manifest: {manifest_path}")
        return

    payload = json.loads(scenario_config.read_text(encoding="utf-8"))
    scenarios = []
    for item in payload["scenarios"]:
        scenario_id = item["scenario_id"]
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_group": item.get("scenario_group", "unknown"),
                "scenario_seed": item.get("seed"),
                "scenario_variant_id": item.get("scenario_variant_id"),
                "contract": str(export_dir / f"{scenario_id}.contract.json"),
                "sidecar": str(export_dir / f"{scenario_id}.path-planner-sidecar.json"),
                "current_cell": item["start_cell"],
            }
        )

    manifest = {
        "schema_version": "path-feedback-manifest/v1",
        "scenario_set": config["scenario_set"],
        "diagnostic_profile": config["diagnostic_profile"],
        "acceptance_gate": config["acceptance_gate"],
        "top_k": config["top_k"],
        "planner_extra_args": list(config["planner_extra_args"]),
        "acceptance_metadata": {
            "schema_version": "path-feedback-acceptance-metadata/v1",
            "scenario_set": config["scenario_set"],
            "diagnostic_profile": config["diagnostic_profile"],
            "acceptance_gate": config["acceptance_gate"],
            "top_k": config["top_k"],
            "python_executable": config["python_bin"],
            "planner_extra_args": list(config["planner_extra_args"]),
            "anchor_projection_candidate_generation_enabled": config["anchor_projection_candidate_generation"],
            "anchor_projection_selection_path_cost_bonus": config["anchor_projection_selection_path_cost_bonus"],
            "anchor_projection_max_selection_path_cost_regression": config[
                "anchor_projection_max_selection_path_cost_regression"
            ],
            "anchor_projection_max_selection_risk_regression": config["anchor_projection_max_selection_risk_regression"],
            "anchor_projection_contract_aware_trainable_target_generation": config[
                "anchor_projection_contract_aware_trainable_target_generation"
            ],
            "anchor_projection_prefer_contract_safe_trainable_targets": config[
                "anchor_projection_prefer_contract_safe_trainable_targets"
            ],
            "anchor_projection_max_trainable_distance_cells": config["anchor_projection_max_trainable_distance_cells"],
            "anchor_projection_max_trainable_distance_m": config["anchor_projection_max_trainable_distance_m"],
            "anchor_projection_planner_validated_trainable_target_mining": config[
                "anchor_projection_planner_validated_trainable_target_mining"
            ],
            "anchor_projection_allow_planner_validated_distance_exception": config[
                "anchor_projection_allow_planner_validated_distance_exception"
            ],
            "anchor_projection_max_planner_validated_distance_cells": config[
                "anchor_projection_max_planner_validated_distance_cells"
            ],
            "anchor_projection_max_planner_validated_distance_m": config[
                "anchor_projection_max_planner_validated_distance_m"
            ],
            "open_grid_fallback_used": None,
            "open_grid_fallback_used_gate": {
                "status": "pending",
                "expected": False,
                "actual": None,
                "reason_codes": ["open_grid_fallback_gate_pending"],
            },
        },
        "open_grid_fallback_used_gate": {
            "status": "pending",
            "expected": False,
            "actual": None,
            "reason_codes": ["open_grid_fallback_gate_pending"],
        },
        "planner": {
            "backend": "path_planner_route",
            "path_planner_root": str(path_planner_root),
            "python_executable": config["python_bin"],
        },
        "scenarios": scenarios,
        "outputs": {
            "summary": str(summary_path),
            "report": str(report_path),
        },
    }
    if config["planner_extra_args"]:
        manifest["planner"]["extra_args"] = list(config["planner_extra_args"])
    if config["anchor_projection_candidate_generation"]:
        manifest["planner"]["anchor_projection_candidate_generation"] = {
            "enabled": True,
            "require_anchor_reachable": True,
            "source_selection_path_cost_bonus": config["anchor_projection_selection_path_cost_bonus"],
            "max_source_selection_path_cost_regression": config[
                "anchor_projection_max_selection_path_cost_regression"
            ],
            "max_source_selection_risk_regression": config["anchor_projection_max_selection_risk_regression"],
            "contract_aware_trainable_target_generation": config[
                "anchor_projection_contract_aware_trainable_target_generation"
            ],
            "prefer_contract_safe_trainable_targets": config["anchor_projection_prefer_contract_safe_trainable_targets"],
            "max_trainable_projection_distance_cells": config["anchor_projection_max_trainable_distance_cells"],
            "max_trainable_projection_distance_m": config["anchor_projection_max_trainable_distance_m"],
            "planner_validated_trainable_target_mining": config[
                "anchor_projection_planner_validated_trainable_target_mining"
            ],
            "allow_planner_validated_distance_exception": config[
                "anchor_projection_allow_planner_validated_distance_exception"
            ],
            "max_planner_validated_distance_cells": config["anchor_projection_max_planner_validated_distance_cells"],
            "max_planner_validated_distance_m": config["anchor_projection_max_planner_validated_distance_m"],
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "scenario_count": len(scenarios)}, ensure_ascii=False))


def _validate_summary(
    config: dict[str, Any],
    *,
    summary_path: Path,
    manifest_path: Path,
    scenario_config_path: Path,
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario_config = json.loads(scenario_config_path.read_text(encoding="utf-8"))
    expected_ids = {item["scenario_id"] for item in scenario_config["scenarios"]}
    expected_count = len(expected_ids)

    required_values = {
        "schema_version": "path-feedback-summary/v1",
        "scenario_count": expected_count,
        "open_grid_fallback_used": False,
    }
    for key, expected in required_values.items():
        if summary.get(key) != expected:
            raise PathFeedbackConfigError(f"{summary_path}: expected {key}={expected!r}, got {summary.get(key)!r}")

    expected_metadata = {
        "scenario_set": config["scenario_set"],
        "diagnostic_profile": config["diagnostic_profile"],
        "acceptance_gate": config["acceptance_gate"],
        "top_k": config["top_k"],
        "planner_extra_args": list(config["planner_extra_args"]),
    }
    for key, expected in expected_metadata.items():
        if summary.get(key) != expected:
            raise PathFeedbackConfigError(f"{summary_path}: expected {key}={expected!r}, got {summary.get(key)!r}")
        if manifest.get(key) != expected:
            raise PathFeedbackConfigError(f"{manifest_path}: expected {key}={expected!r}, got {manifest.get(key)!r}")

    acceptance_metadata = summary.get("acceptance_metadata")
    if not isinstance(acceptance_metadata, dict):
        raise PathFeedbackConfigError(f"{summary_path}: acceptance_metadata must be an object")
    open_grid_gate = acceptance_metadata.get("open_grid_fallback_used_gate")
    if not isinstance(open_grid_gate, dict) or open_grid_gate.get("status") != "passed":
        raise PathFeedbackConfigError(f"{summary_path}: open_grid_fallback_used_gate must pass")
    if summary.get("open_grid_fallback_used_gate") != open_grid_gate:
        raise PathFeedbackConfigError(f"{summary_path}: top-level open_grid_fallback_used_gate must mirror acceptance metadata")

    manifest["open_grid_fallback_used_gate"] = dict(open_grid_gate)
    manifest_metadata = manifest.get("acceptance_metadata")
    manifest_metadata = manifest_metadata if isinstance(manifest_metadata, dict) else {}
    manifest_metadata.update(
        {
            "open_grid_fallback_used": summary.get("open_grid_fallback_used"),
            "open_grid_fallback_used_gate": dict(open_grid_gate),
        }
    )
    manifest["acceptance_metadata"] = manifest_metadata
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    candidate_count = summary.get("candidate_count")
    if not isinstance(candidate_count, int) or candidate_count < 3:
        raise PathFeedbackConfigError(f"{summary_path}: expected candidate_count >= 3, got {candidate_count!r}")

    required_keys = (
        "coverage_per_path_cost",
        "path_planning_failure_count",
        "replan_count",
        "tracking_safety_violation_count",
        "trajectory_optimization_fallback_count",
        "region_graph_disconnected_count",
        "iris_requested_count",
        "iris_report_count",
        "iris_status_counts",
        "iris_fallback_count",
        "iris_failure_count",
        "iris_region_count_total",
        "iris_fallback_reasons",
        "region_graph_source_counts",
        "region_graph_fallback_count",
        "region_graph_fallback_reasons",
        "region_graph_start_goal_disconnected_count",
        "scenario_group_summary",
        "diagnostic_interpretation",
        "selection_changed_count",
        "selection_changed_rate",
    )
    missing = [key for key in required_keys if key not in summary]
    if missing:
        raise PathFeedbackConfigError(f"{summary_path}: missing required summary keys: {', '.join(missing)}")

    scenario_ids = {item.get("scenario_id") for item in summary.get("scenarios", [])}
    if scenario_ids != expected_ids:
        raise PathFeedbackConfigError(f"{summary_path}: expected scenarios {sorted(expected_ids)}, got {sorted(scenario_ids)}")

    if config["scenario_set"] in {"stress", "all"}:
        stress_items = [
            item
            for item in summary.get("scenarios", [])
            if str(item.get("scenario_id", "")).startswith("npz_")
            and item.get("scenario_id")
            not in {
                "npz_shadow_corridor",
                "npz_rock_field_multi_pose",
                "npz_low_confidence_risk_band",
            }
        ]
        stress_replan_or_failure = sum(
            int(item.get("path_feedback", {}).get("failure_count", 0))
            + int(item.get("path_feedback", {}).get("replan_count", 0))
            for item in stress_items
        )
        stress_sampled_region_decision_diagnostics = sum(
            int(item.get("sampled_region_path_diagnostics", {}).get("selected_count", 0))
            + int(item.get("sampled_region_path_diagnostics", {}).get("fallback_count", 0))
            + int(item.get("sampled_region_path_diagnostics", {}).get("terminal_adjusted_count", 0))
            + int(item.get("sampled_region_path_diagnostics", {}).get("reachable_terminal_rescue_count", 0))
            + int(item.get("sampled_region_path_diagnostics", {}).get("proxy_goal_anchor_selected_count", 0))
            for item in stress_items
        )
        if stress_sampled_region_decision_diagnostics < 1:
            group_summary = summary.get("scenario_group_summary", {})
            group_summary = group_summary if isinstance(group_summary, dict) else {}
            for group_name in ("stress", "mixed_stress"):
                group_payload = group_summary.get(group_name, {})
                if not isinstance(group_payload, dict):
                    continue
                stress_sampled_region_decision_diagnostics += (
                    int(group_payload.get("sampled_region_path_selected_count", 0))
                    + int(group_payload.get("sampled_region_path_fallback_count", 0))
                    + int(group_payload.get("sampled_region_path_terminal_adjusted_count", 0))
                    + int(group_payload.get("sampled_region_path_reachable_terminal_rescue_count", 0))
                    + int(group_payload.get("sampled_region_path_proxy_goal_anchor_selected_count", 0))
                )
        if stress_replan_or_failure + stress_sampled_region_decision_diagnostics < 1:
            raise PathFeedbackConfigError(
                f"{summary_path}: stress scenarios must produce failure, replan, or sampled-region diagnostics"
            )

        mixed_items = [item for item in summary.get("scenarios", []) if item.get("scenario_group") == "mixed_stress"]
        if mixed_items:
            mixed_reachable = sum(int(item.get("path_feedback", {}).get("reachable_count", 0)) for item in mixed_items)
            mixed_replan_or_failure = sum(
                int(item.get("path_feedback", {}).get("failure_count", 0))
                + int(item.get("path_feedback", {}).get("replan_count", 0))
                for item in mixed_items
            )
            mixed_group_summary = summary.get("scenario_group_summary", {}).get("mixed_stress", {})
            if not isinstance(mixed_group_summary, dict):
                mixed_group_summary = {}
            mixed_sampled_region_decision_diagnostics = int(
                mixed_group_summary.get("sampled_region_path_selected_count", 0)
            ) + int(mixed_group_summary.get("sampled_region_path_terminal_adjusted_count", 0))
            if mixed_sampled_region_decision_diagnostics < 1:
                mixed_sampled_region_decision_diagnostics = sum(
                    int(item.get("sampled_region_path_diagnostics", {}).get("selected_count", 0))
                    + int(item.get("sampled_region_path_diagnostics", {}).get("terminal_adjusted_count", 0))
                    for item in mixed_items
                )
            if mixed_reachable < 1:
                raise PathFeedbackConfigError(f"{summary_path}: mixed stress scenarios must include at least one reachable candidate")
            if mixed_replan_or_failure + mixed_sampled_region_decision_diagnostics < 1:
                raise PathFeedbackConfigError(
                    f"{summary_path}: mixed stress scenarios must produce failure, replan, or sampled-region decision diagnostics"
                )

    print(
        json.dumps(
            {
                "status": "valid",
                "summary": str(summary_path),
                "scenario_count": summary["scenario_count"],
                "candidate_count": summary["candidate_count"],
                "selection_changed_count": summary["selection_changed_count"],
                "open_grid_fallback_used": summary["open_grid_fallback_used"],
                "acceptance_gate": summary["acceptance_gate"],
                "scenario_set": summary["scenario_set"],
                "diagnostic_profile": summary["diagnostic_profile"],
            },
            ensure_ascii=False,
        )
    )


def _profile_default_args(profile: str) -> list[str]:
    if profile == "execution":
        return ["--simulate-tracking", "--optimize-trajectory"]
    if profile == "iris":
        return [
            "--drake-iris-regions",
            "--gcs-trajectory-smoke",
            "--gcs-geometric-candidate",
            "--gcs-motion-feasibility",
            "--gcs-curvature-constrained-candidate",
        ]
    if profile == "all":
        return [
            "--simulate-tracking",
            "--optimize-trajectory",
            "--drake-iris-regions",
            "--gcs-trajectory-smoke",
            "--gcs-geometric-candidate",
            "--gcs-motion-feasibility",
            "--gcs-curvature-constrained-candidate",
        ]
    return []


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _dest_from_option(option: str) -> str:
    return option.removeprefix("--").replace("-", "_")


def _non_negative_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise PathFeedbackConfigError(f"--{label.replace('_', '-')} must be a non-negative finite number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise PathFeedbackConfigError(f"--{label.replace('_', '-')} must be a non-negative finite number")
    return parsed


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _format_extra_args(values: list[str]) -> str:
    return " ".join(values) if values else "(none)"


def _run_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY RUN] (cd {cwd} && {display_command(command)})")
        return
    print(f"==> (cd {cwd} && {display_command(command)})")
    subprocess.run(command, cwd=cwd, check=True)


def _run_pythonpath_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY RUN] (cd {cwd} && PYTHONPATH=src {display_command(command)})")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd / "src")
    print(f"==> (cd {cwd} && PYTHONPATH=src {display_command(command)})")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _ensure_submodules(repo_root: Path, *, dry_run: bool) -> None:
    command = ["git", "-C", str(repo_root), "submodule", "update", "--init", "--recursive", *MODULES]
    if dry_run:
        print(f"[DRY RUN] {display_command(command)}")
        return
    if (repo_root / ".git").exists():
        subprocess.run(command, check=True)
    for module in MODULES:
        if not (repo_root / module / "src").is_dir():
            raise PathFeedbackConfigError(
                f"Missing initialized submodule: {module}. Run: git submodule update --init --recursive {' '.join(MODULES)}"
            )


def _assert_output_files(*paths: Path) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise PathFeedbackConfigError(f"Expected non-empty output file was not created: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
