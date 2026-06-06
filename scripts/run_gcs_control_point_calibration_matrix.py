from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TRIAGE_SCHEMA_VERSION = "gcs-control-point-candidate-triage-summary/v1"
MATRIX_SCHEMA_VERSION = "gcs-control-point-calibration-matrix/v1"
SUMMARY_SCHEMA_VERSION = "gcs-control-point-calibration-matrix-summary/v1"
DEFAULT_PYTHON = Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python")


DEFAULT_MATRIX = [
    {
        "matrix_id": "baseline",
        "terrain_objective_weight": 0.05,
        "second_difference_weight": 0.2,
        "direction_cone_max_error_deg": 45.0,
        "direction_cone_rho_floor_m": 0.0001,
        "direction_cone_seed_rho_ratio": 0.05,
    },
    {
        "matrix_id": "terrain_up",
        "terrain_objective_weight": 0.08,
        "second_difference_weight": 0.2,
        "direction_cone_max_error_deg": 45.0,
        "direction_cone_rho_floor_m": 0.0001,
        "direction_cone_seed_rho_ratio": 0.05,
    },
    {
        "matrix_id": "smoothness_up",
        "terrain_objective_weight": 0.05,
        "second_difference_weight": 0.35,
        "direction_cone_max_error_deg": 45.0,
        "direction_cone_rho_floor_m": 0.0001,
        "direction_cone_seed_rho_ratio": 0.05,
    },
    {
        "matrix_id": "direction_cone_relaxed",
        "terrain_objective_weight": 0.05,
        "second_difference_weight": 0.2,
        "direction_cone_max_error_deg": 55.0,
        "direction_cone_rho_floor_m": 0.0001,
        "direction_cone_seed_rho_ratio": 0.03,
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a solver-rerun matrix for GCS control-point calibration."
    )
    parser.add_argument("--triage-json", required=True, help="Baseline control-point triage JSON.")
    parser.add_argument("--output-root", required=True, help="Directory for matrix artifacts.")
    parser.add_argument("--matrix-json", help="Optional calibration matrix JSON.")
    parser.add_argument("--planner-python", help="Python executable for path_planner.cli.")
    parser.add_argument("--planner-command", help="Optional fake or alternate planner script for tests.")
    parser.add_argument("--path-planner-root", help="Path to path-planner checkout.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    path_planner_root = Path(args.path_planner_root) if args.path_planner_root else repo_root / "path-planner"
    planner_python = Path(args.planner_python) if args.planner_python else _default_python()
    output_root = _resolve_path(Path(args.output_root), repo_root)
    triage_path = _resolve_path(Path(args.triage_json), repo_root)
    matrix_path = _resolve_path(Path(args.matrix_json), repo_root) if args.matrix_json else None

    triage = _load_triage(triage_path)
    matrix = _load_matrix(matrix_path)
    output_root.mkdir(parents=True, exist_ok=True)
    cases = _run_matrix(
        triage,
        matrix,
        output_root=output_root,
        planner_python=planner_python,
        planner_command=Path(args.planner_command) if args.planner_command else None,
        path_planner_root=path_planner_root,
    )
    summary = _build_summary(
        triage,
        matrix,
        cases,
        source_triage=triage_path,
        output_root=output_root,
    )
    summary_path = output_root / "gcs-control-point-calibration-matrix-summary.json"
    report_path = output_root / "gcs-control-point-calibration-matrix-summary.md"
    _write_json(summary_path, summary)
    report_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "calibration matrix complete",
                "summary": str(summary_path),
                "report": str(report_path),
                "matrix_count": summary["matrix_count"],
                "case_count": summary["case_count"],
                "safety_regression_count": summary["safety_regression_count"],
                "default_change_recommended": summary["default_change_recommended"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _default_python() -> Path:
    configured = os.environ.get("PYTHON")
    if configured:
        return Path(configured)
    return DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)


def _resolve_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _load_triage(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("triage JSON must be an object")
    if payload.get("schema_version") != TRIAGE_SCHEMA_VERSION:
        raise ValueError(f"unsupported triage schema: {payload.get('schema_version')!r}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("triage JSON requires candidates")
    return payload


def _load_matrix(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(item) for item in DEFAULT_MATRIX]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("matrix JSON must be an object")
    if payload.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError(f"unsupported matrix schema: {payload.get('schema_version')!r}")
    matrix = payload.get("matrix")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("matrix JSON requires a non-empty matrix list")
    return [_matrix_entry(item) for item in matrix if isinstance(item, dict)]


def _matrix_entry(item: dict[str, Any]) -> dict[str, Any]:
    matrix_id = str(item.get("matrix_id") or "").strip()
    if not matrix_id:
        raise ValueError("matrix entry requires matrix_id")
    return {
        "matrix_id": matrix_id,
        "terrain_objective_weight": _float_required(item, "terrain_objective_weight"),
        "second_difference_weight": _float_required(item, "second_difference_weight"),
        "direction_cone_max_error_deg": _float_required(item, "direction_cone_max_error_deg"),
        "direction_cone_rho_floor_m": _float_required(item, "direction_cone_rho_floor_m"),
        "direction_cone_seed_rho_ratio": _float_required(item, "direction_cone_seed_rho_ratio"),
    }


def _float_required(item: dict[str, Any], key: str) -> float:
    value = item.get(key)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"matrix entry requires numeric {key}")
    return float(value)


def _run_matrix(
    triage: dict[str, Any],
    matrix: list[dict[str, Any]],
    *,
    output_root: Path,
    planner_python: Path,
    planner_command: Path | None,
    path_planner_root: Path,
) -> list[dict[str, Any]]:
    candidates = [item for item in triage["candidates"] if isinstance(item, dict)]
    cases: list[dict[str, Any]] = []
    for entry in matrix:
        matrix_id = _safe_path(entry["matrix_id"])
        for candidate in candidates:
            scenario_id = str(candidate.get("scenario_id") or "unknown")
            action_index = int(candidate.get("action_index") or 0)
            route_path = (
                output_root
                / "route_artifacts"
                / matrix_id
                / _safe_path(scenario_id)
                / f"action-{action_index:03d}"
                / "path-planner-route.json"
            )
            report_dir = route_path.parent / "diagnostics"
            request_artifact = candidate.get("request_artifact")
            if not request_artifact:
                cases.append(
                    _case_without_rerun(
                        candidate,
                        entry,
                        route_artifact=route_path,
                        reason="request_artifact_missing",
                    )
                )
                continue
            command = _planner_argv(
                planner_python,
                planner_command=planner_command,
                request_artifact=Path(str(request_artifact)),
                route_path=route_path,
                report_dir=report_dir,
                matrix_entry=entry,
            )
            route_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if planner_command is None:
                env["PYTHONPATH"] = str(path_planner_root / "src")
            completed = subprocess.run(
                command,
                cwd=path_planner_root if planner_command is None else Path.cwd(),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                cases.append(
                    _case_without_rerun(
                        candidate,
                        entry,
                        route_artifact=route_path,
                        reason="planner_command_failed",
                        command=command,
                        stderr=completed.stderr,
                    )
                )
                continue
            route = json.loads(route_path.read_text(encoding="utf-8"))
            cases.append(_case_from_route(candidate, entry, route, route_artifact=route_path, command=command))
    return cases


def _planner_argv(
    planner_python: Path,
    *,
    planner_command: Path | None,
    request_artifact: Path,
    route_path: Path,
    report_dir: Path,
    matrix_entry: dict[str, Any],
) -> list[str]:
    base = [str(planner_python), str(planner_command)] if planner_command else [str(planner_python), "-m", "path_planner.cli"]
    return [
        *base,
        "--input",
        str(request_artifact),
        "--output-json",
        str(route_path),
        "--output-dir",
        str(report_dir),
        "--drake-iris-regions",
        "--gcs-control-point-candidate",
        "--gcs-motion-feasibility",
        "--gcs-control-point-terrain-weight",
        str(matrix_entry["terrain_objective_weight"]),
        "--gcs-control-point-second-difference-weight",
        str(matrix_entry["second_difference_weight"]),
        "--gcs-control-point-direction-cone-max-error-deg",
        str(matrix_entry["direction_cone_max_error_deg"]),
        "--gcs-control-point-direction-cone-rho-floor-m",
        str(matrix_entry["direction_cone_rho_floor_m"]),
        "--gcs-control-point-direction-cone-seed-rho-ratio",
        str(matrix_entry["direction_cone_seed_rho_ratio"]),
    ]


def _case_from_route(
    baseline: dict[str, Any],
    matrix_entry: dict[str, Any],
    route: dict[str, Any],
    *,
    route_artifact: Path,
    command: list[str],
) -> dict[str, Any]:
    direction = _dict(route.get("gcs_trajectory_constraint_summary"))
    trajectory_cost = _dict(route.get("gcs_trajectory_cost_summary"))
    candidate_cost = _dict(route.get("gcs_candidate_cost_summary"))
    case = {
        "matrix_id": matrix_entry["matrix_id"],
        "parameters": dict(matrix_entry),
        "scenario_id": baseline.get("scenario_id"),
        "action_index": baseline.get("action_index"),
        "cell": baseline.get("cell"),
        "baseline_fallback_reason": baseline.get("candidate_fallback_reason"),
        "new_fallback_reason": route.get("gcs_candidate_fallback_reason"),
        "baseline_selected": bool(baseline.get("candidate_selected")),
        "new_selected": bool(route.get("gcs_candidate_selected")),
        "attempted": route.get("gcs_trajectory_attempted"),
        "success": route.get("gcs_trajectory_success"),
        "cost_delta_vs_baseline": route.get("gcs_candidate_cost_delta_vs_baseline"),
        "cost_delta_vs_postprocess": route.get("gcs_candidate_cost_delta_vs_postprocess"),
        "sampled_terrain_cost": _first_present(
            trajectory_cost.get("sampled_terrain_cost"),
            candidate_cost.get("sampled_terrain_cost"),
        ),
        "baseline_sampled_terrain_cost": baseline.get("sampled_terrain_cost"),
        "high_cost_exposure_delta_vs_baseline": candidate_cost.get(
            "high_cost_exposure_delta_vs_baseline"
        ),
        "baseline_high_cost_exposure_delta_vs_baseline": baseline.get(
            "high_cost_exposure_delta_vs_baseline"
        ),
        "collision_count": _int_value(route.get("gcs_trajectory_collision_count")),
        "baseline_collision_count": _int_value(baseline.get("collision_count")),
        "direction_cone_violation_count": _int_value(direction.get("violation_count")),
        "baseline_direction_cone_violation_count": _int_value(
            baseline.get("direction_cone_violation_count")
        ),
        "direction_cone_risk_flags": direction.get("risk_flags") if isinstance(direction.get("risk_flags"), list) else [],
        "baseline_direction_cone_risk_flags": baseline.get("direction_cone_risk_flags")
        if isinstance(baseline.get("direction_cone_risk_flags"), list)
        else [],
        "direction_cone_eta": direction.get("eta"),
        "direction_cone_rho_min": _first_present(direction.get("rho_min"), direction.get("rho_lower_bound_min_m")),
        "direction_cone_tolerance_deg": direction.get("max_allowed_direction_error_deg"),
        "terrain_objective_weight": _first_present(
            trajectory_cost.get("terrain_objective_weight"),
            candidate_cost.get("terrain_objective_weight"),
        ),
        "second_difference_weight": _dict(direction.get("objective_term_weights")).get(
            "control_point_second_difference_quadratic"
        ),
        "motion_feasibility_status": route.get("gcs_motion_feasibility_feasibility_status"),
        "baseline_motion_feasibility_status": baseline.get("motion_feasibility_status"),
        "motion_feasibility_curvature_violation_count": _int_value(
            route.get("gcs_motion_feasibility_curvature_violation_count")
        ),
        "baseline_motion_feasibility_curvature_violation_count": _int_value(
            baseline.get("motion_feasibility_curvature_violation_count")
        ),
        "motion_feasibility_heading_violation_count": _int_value(
            route.get("gcs_motion_feasibility_heading_violation_count")
        ),
        "baseline_motion_feasibility_heading_violation_count": _int_value(
            baseline.get("motion_feasibility_heading_violation_count")
        ),
        "route_artifact": str(route_artifact),
        "command": command,
    }
    reasons = _safety_regression_reasons(case)
    case["safety_regression_reasons"] = reasons
    case["safety_regression"] = bool(reasons)
    return case


def _case_without_rerun(
    baseline: dict[str, Any],
    matrix_entry: dict[str, Any],
    *,
    route_artifact: Path,
    reason: str,
    command: list[str] | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    return {
        "matrix_id": matrix_entry["matrix_id"],
        "parameters": dict(matrix_entry),
        "scenario_id": baseline.get("scenario_id"),
        "action_index": baseline.get("action_index"),
        "cell": baseline.get("cell"),
        "baseline_fallback_reason": baseline.get("candidate_fallback_reason"),
        "new_fallback_reason": reason,
        "baseline_selected": bool(baseline.get("candidate_selected")),
        "new_selected": False,
        "attempted": False,
        "success": False,
        "safety_regression": False,
        "safety_regression_reasons": [],
        "route_artifact": str(route_artifact),
        "command": command or [],
        "stderr": stderr,
    }


def _safety_regression_reasons(case: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _float_gt(case.get("sampled_terrain_cost"), case.get("baseline_sampled_terrain_cost")):
        reasons.append("terrain_cost_regression")
    if _float_gt(
        case.get("high_cost_exposure_delta_vs_baseline"),
        case.get("baseline_high_cost_exposure_delta_vs_baseline"),
    ):
        reasons.append("high_cost_exposure_regression")
    if _int_value(case.get("collision_count")) > _int_value(case.get("baseline_collision_count")):
        reasons.append("collision_regression")
    if _int_value(case.get("direction_cone_violation_count")) > _int_value(
        case.get("baseline_direction_cone_violation_count")
    ):
        reasons.append("direction_cone_violation_regression")
    baseline_flags = set(case.get("baseline_direction_cone_risk_flags") or [])
    new_flags = set(case.get("direction_cone_risk_flags") or [])
    if not new_flags.issubset(baseline_flags):
        reasons.append("direction_cone_risk_flag_regression")
    if case.get("baseline_motion_feasibility_status") == "feasible" and case.get(
        "motion_feasibility_status"
    ) not in {None, "feasible"}:
        reasons.append("motion_feasibility_status_regression")
    if _int_value(case.get("motion_feasibility_curvature_violation_count")) > _int_value(
        case.get("baseline_motion_feasibility_curvature_violation_count")
    ):
        reasons.append("motion_curvature_regression")
    if _int_value(case.get("motion_feasibility_heading_violation_count")) > _int_value(
        case.get("baseline_motion_feasibility_heading_violation_count")
    ):
        reasons.append("motion_heading_regression")
    return reasons


def _build_summary(
    triage: dict[str, Any],
    matrix: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    source_triage: Path,
    output_root: Path,
) -> dict[str, Any]:
    baseline_counts = _counter_payload(triage.get("fallback_reason_counts"))
    matrix_summaries = [_matrix_summary(entry, cases, baseline_counts=baseline_counts) for entry in matrix]
    acceptable = [
        item
        for item in matrix_summaries
        if item["safety_regression_count"] == 0
        and item["selected_count"] > int(triage.get("selected_count") or 0)
        and item["target_blocker_count"] < (
            baseline_counts.get("cost_dominated", 0)
            + baseline_counts.get("direction_cone_constraint_violation", 0)
        )
    ]
    default_change_recommended = bool(acceptable)
    safety_regression_count = sum(1 for case in cases if case.get("safety_regression"))
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source_triage": str(source_triage),
        "output_root": str(output_root),
        "matrix_count": len(matrix),
        "candidate_count": len([item for item in triage.get("candidates", []) if isinstance(item, dict)]),
        "case_count": len(cases),
        "route_artifact_count": sum(1 for case in cases if case.get("route_artifact")),
        "baseline_fallback_reason_counts": dict(sorted(baseline_counts.items())),
        "matrix_summaries": matrix_summaries,
        "safety_regression_count": safety_regression_count,
        "default_change_recommended": default_change_recommended,
        "recommendation": "default_change_recommended" if default_change_recommended else "no_default_change_recommended",
        "recommended_matrix_id": acceptable[0]["matrix_id"] if acceptable else None,
        "default_change_reason": (
            "matrix_passed_safety_and_blocker_gates"
            if default_change_recommended
            else "no_matrix_passed_safety_and_blocker_gates"
        ),
        "cases": cases,
    }


def _matrix_summary(
    entry: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    baseline_counts: Counter[str],
) -> dict[str, Any]:
    matrix_cases = [case for case in cases if case.get("matrix_id") == entry["matrix_id"]]
    new_counts: Counter[str] = Counter()
    for case in matrix_cases:
        fallback = case.get("new_fallback_reason")
        if fallback:
            new_counts[str(fallback)] += 1
    blocker_deltas = {
        reason: int(new_counts.get(reason, 0) - baseline_counts.get(reason, 0))
        for reason in sorted(set(new_counts) | set(baseline_counts))
    }
    return {
        "matrix_id": entry["matrix_id"],
        "parameters": dict(entry),
        "case_count": len(matrix_cases),
        "attempted_count": sum(1 for case in matrix_cases if case.get("attempted") is True),
        "success_count": sum(1 for case in matrix_cases if case.get("success") is True),
        "selected_count": sum(1 for case in matrix_cases if case.get("new_selected") is True),
        "new_fallback_reason_counts": dict(sorted(new_counts.items())),
        "blocker_deltas": blocker_deltas,
        "target_blocker_count": int(
            new_counts.get("cost_dominated", 0)
            + new_counts.get("direction_cone_constraint_violation", 0)
        ),
        "safety_regression_count": sum(1 for case in matrix_cases if case.get("safety_regression")),
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# GCS Control-Point Calibration Matrix",
        "",
        f"- schema_version: `{summary['schema_version']}`",
        f"- source_triage: `{summary['source_triage']}`",
        f"- matrix_count: {summary['matrix_count']}",
        f"- case_count: {summary['case_count']}",
        f"- safety_regression_count: {summary['safety_regression_count']}",
        f"- recommendation: `{summary['recommendation']}`",
        f"- default_change_reason: `{summary['default_change_reason']}`",
        "",
        "## Matrix Summary",
        "",
        "| matrix | selected | safety regressions | target blockers |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in summary["matrix_summaries"]:
        lines.append(
            f"| {item['matrix_id']} | {item['selected_count']} | "
            f"{item['safety_regression_count']} | {item['target_blocker_count']} |"
        )
    lines.extend(["", "## Blocker Deltas", ""])
    for item in summary["matrix_summaries"]:
        deltas = ", ".join(
            f"{reason}={delta:+d}" for reason, delta in item["blocker_deltas"].items()
        )
        lines.append(f"- `{item['matrix_id']}`: {deltas or 'none'}")
    unsupported_count = sum(
        1
        for case in summary["cases"]
        if case.get("new_fallback_reason") == "unsupported_route_replacement"
    )
    lines.extend(
        [
            "",
            "## Expected Not Evaluated",
            "",
            f"- unsupported_route_replacement cases: {unsupported_count}",
        ]
    )
    regressions = [case for case in summary["cases"] if case.get("safety_regression")]
    lines.extend(["", "## Safety Regression Cases", ""])
    if not regressions:
        lines.append("- none")
    for case in regressions:
        reasons = ", ".join(case.get("safety_regression_reasons") or [])
        lines.append(
            "- `{matrix}` {scenario}/action-{action:03d}: {baseline} -> {new}; {reasons}".format(
                matrix=case.get("matrix_id"),
                scenario=case.get("scenario_id"),
                action=int(case.get("action_index") or 0),
                baseline=case.get("baseline_fallback_reason"),
                new=case.get("new_fallback_reason"),
                reasons=reasons,
            )
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _counter_payload(value: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counts
    for key, count in value.items():
        if isinstance(count, int) and not isinstance(count, bool):
            counts[str(key)] += count
    return counts


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _float_gt(value: Any, baseline: Any, *, tolerance: float = 1.0e-9) -> bool:
    left = _float_value(value)
    right = _float_value(baseline)
    return left is not None and right is not None and left > right + tolerance


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_path(value: Any) -> str:
    text = str(value)
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)
    return safe or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
