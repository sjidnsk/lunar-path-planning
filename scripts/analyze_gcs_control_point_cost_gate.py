from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGETED_SWEEP_SCHEMA_VERSION = "gcs-control-point-targeted-sweep-summary/v1"
COST_GATE_SCHEMA_VERSION = "gcs-control-point-cost-gate-decomposition-summary/v1"
COST_GATE_CLASSES = (
    "true_cost_dominated",
    "high_cost_exposure_blocked",
    "terrain_proxy_mismatch",
    "baseline_overlap_or_duplicate",
    "direction_cone_fixed_but_cost_blocked",
    "insufficient_cost_diagnostics",
    "safety_regression_excluded",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decompose GCS control-point cost gate blockers from a targeted sweep summary."
    )
    parser.add_argument("--targeted-sweep-summary", required=True, help="Input targeted sweep summary JSON.")
    parser.add_argument("--output-root", required=True, help="Directory for cost-gate decomposition artifacts.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    source_path = _resolve_path(Path(args.targeted_sweep_summary), repo_root)
    output_root = _resolve_path(Path(args.output_root), repo_root)

    try:
        targeted = _load_targeted_summary(source_path)
        summary = _build_summary(targeted, source_path=source_path, output_root=output_root)
    except ValueError as exc:
        print(f"cost-gate decomposition error: {exc}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "gcs-control-point-cost-gate-decomposition-summary.json"
    report_path = output_root / "gcs-control-point-cost-gate-decomposition-summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(_render_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "cost-gate decomposition complete",
                "summary": str(summary_path),
                "report": str(report_path),
                "case_count": summary["case_count"],
                "default_change_recommended": summary["default_change_recommended"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _resolve_path(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _load_targeted_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"targeted sweep summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("targeted sweep summary must be a JSON object")
    if payload.get("schema_version") != TARGETED_SWEEP_SCHEMA_VERSION:
        raise ValueError(f"unsupported targeted sweep schema: {payload.get('schema_version')!r}")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("targeted sweep summary requires cases")
    return payload


def _build_summary(
    targeted: dict[str, Any],
    *,
    source_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    rows = [
        _decompose_case(case)
        for case in targeted["cases"]
        if isinstance(case, dict) and _is_cost_gate_case(case)
    ]
    class_counts = Counter(row["cost_gate_class"] for row in rows)
    matrix_summaries = _matrix_summaries(rows)
    scenario_summaries = _scenario_summaries(rows)
    return {
        "schema_version": COST_GATE_SCHEMA_VERSION,
        "source_targeted_sweep_summary": str(source_path),
        "output_root": str(output_root),
        "input_case_count": len([case for case in targeted["cases"] if isinstance(case, dict)]),
        "case_count": len(rows),
        "cost_gate_class_counts": {name: class_counts.get(name, 0) for name in COST_GATE_CLASSES},
        "matrix_summaries": matrix_summaries,
        "scenario_summaries": scenario_summaries,
        "default_change_recommended": False,
        "recommendation": "no_default_change_recommended",
        "default_change_reason": "cost_gate_decomposition_required_before_default_update",
        "cases": rows,
    }


def _is_cost_gate_case(case: dict[str, Any]) -> bool:
    return (
        case.get("new_fallback_reason") == "cost_dominated"
        or case.get("transition_class") in {"direction_cone_to_cost_dominated", "cost_dominated_persistent"}
    )


def _decompose_case(case: dict[str, Any]) -> dict[str, Any]:
    route = _load_route(case.get("route_artifact"))
    trajectory_cost = _dict(route.get("gcs_trajectory_cost_summary"))
    candidate_cost = _dict(route.get("gcs_candidate_cost_summary"))
    direction = _dict(route.get("gcs_trajectory_constraint_summary"))
    sampled_terrain_cost = _number(
        _first_present(
            case.get("sampled_terrain_cost"),
            trajectory_cost.get("sampled_terrain_cost"),
            candidate_cost.get("sampled_terrain_cost"),
        )
    )
    control_point_terrain_cost = _number(
        _first_present(
            case.get("control_point_terrain_cost"),
            trajectory_cost.get("control_point_terrain_cost"),
            candidate_cost.get("control_point_terrain_cost"),
        )
    )
    cost_delta_vs_baseline = _number(
        _first_present(case.get("cost_delta_vs_baseline"), route.get("gcs_candidate_cost_delta_vs_baseline"))
    )
    cost_delta_vs_postprocess = _number(
        _first_present(case.get("cost_delta_vs_postprocess"), route.get("gcs_candidate_cost_delta_vs_postprocess"))
    )
    high_cost_delta = _number(
        _first_present(
            case.get("high_cost_exposure_delta_vs_baseline"),
            candidate_cost.get("high_cost_exposure_delta_vs_baseline"),
            trajectory_cost.get("high_cost_exposure_delta_vs_baseline"),
        )
    )
    baseline_overlap_ratio = _number(
        _first_present(case.get("baseline_overlap_ratio"), route.get("gcs_candidate_baseline_overlap_ratio"))
    )
    row = {
        "matrix_id": case.get("matrix_id"),
        "scenario_id": case.get("scenario_id"),
        "action_index": case.get("action_index"),
        "baseline_fallback_reason": case.get("baseline_fallback_reason"),
        "new_fallback_reason": case.get("new_fallback_reason"),
        "transition_class": case.get("transition_class"),
        "sampled_terrain_cost": sampled_terrain_cost,
        "control_point_terrain_cost": control_point_terrain_cost,
        "path_cost_delta": _first_present(cost_delta_vs_baseline, cost_delta_vs_postprocess),
        "high_cost_exposure_delta_vs_baseline": high_cost_delta,
        "baseline_overlap_ratio": baseline_overlap_ratio,
        "cost_delta_vs_baseline": cost_delta_vs_baseline,
        "cost_delta_vs_postprocess": cost_delta_vs_postprocess,
        "collision": {
            "count": _int_value(_first_present(case.get("collision_count"), route.get("gcs_trajectory_collision_count"))),
            "baseline_count": _int_value(case.get("baseline_collision_count")),
        },
        "direction_cone": {
            "violation_count": _int_value(
                _first_present(case.get("direction_cone_violation_count"), direction.get("violation_count"))
            ),
            "baseline_violation_count": _int_value(case.get("baseline_direction_cone_violation_count")),
            "risk_flags": _list_value(_first_present(case.get("direction_cone_risk_flags"), direction.get("risk_flags"))),
            "baseline_risk_flags": _list_value(case.get("baseline_direction_cone_risk_flags")),
        },
        "motion_feasibility": {
            "status": _first_present(
                case.get("motion_feasibility_status"),
                route.get("gcs_motion_feasibility_feasibility_status"),
            ),
            "baseline_status": case.get("baseline_motion_feasibility_status"),
            "curvature_violation_count": _int_value(
                _first_present(
                    case.get("motion_feasibility_curvature_violation_count"),
                    route.get("gcs_motion_feasibility_curvature_violation_count"),
                )
            ),
            "heading_violation_count": _int_value(
                _first_present(
                    case.get("motion_feasibility_heading_violation_count"),
                    route.get("gcs_motion_feasibility_heading_violation_count"),
                )
            ),
        },
        "safety_regression": bool(case.get("safety_regression")),
        "safety_regression_reasons": _list_value(case.get("safety_regression_reasons")),
        "route_artifact": case.get("route_artifact"),
        "request_artifact": _request_artifact(case),
        "parameters": case.get("parameters") if isinstance(case.get("parameters"), dict) else {},
    }
    row["missing_diagnostics"] = _missing_diagnostics(row)
    row["cost_gate_class"] = _cost_gate_class(row)
    row["recommended_next_action"] = _recommended_next_action(row["cost_gate_class"])
    return row


def _load_route(route_artifact: Any) -> dict[str, Any]:
    if not route_artifact:
        return {}
    path = Path(str(route_artifact))
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _request_artifact(case: dict[str, Any]) -> str | None:
    direct = case.get("request_artifact")
    if direct:
        return str(direct)
    command = case.get("command")
    if not isinstance(command, list):
        return None
    for index, value in enumerate(command):
        if value == "--input" and index + 1 < len(command):
            return str(command[index + 1])
    return None


def _missing_diagnostics(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if row.get("cost_delta_vs_baseline") is None and row.get("cost_delta_vs_postprocess") is None:
        missing.append("cost_delta")
    if row.get("sampled_terrain_cost") is None and row.get("control_point_terrain_cost") is None:
        missing.append("terrain_cost")
    return missing


def _cost_gate_class(row: dict[str, Any]) -> str:
    if row["safety_regression"]:
        return "safety_regression_excluded"
    if row["missing_diagnostics"]:
        return "insufficient_cost_diagnostics"
    high_cost_delta = row.get("high_cost_exposure_delta_vs_baseline")
    if high_cost_delta is not None and high_cost_delta > 0.0:
        return "high_cost_exposure_blocked"
    if _terrain_proxy_mismatch(row):
        return "terrain_proxy_mismatch"
    overlap = row.get("baseline_overlap_ratio")
    if overlap is not None and overlap >= 0.95:
        return "baseline_overlap_or_duplicate"
    if (
        row.get("transition_class") == "direction_cone_to_cost_dominated"
        or row.get("baseline_fallback_reason") == "direction_cone_constraint_violation"
        and row.get("new_fallback_reason") == "cost_dominated"
    ):
        return "direction_cone_fixed_but_cost_blocked"
    if _positive(row.get("cost_delta_vs_baseline")) or _positive(row.get("cost_delta_vs_postprocess")):
        return "true_cost_dominated"
    return "insufficient_cost_diagnostics"


def _terrain_proxy_mismatch(row: dict[str, Any]) -> bool:
    sampled = row.get("sampled_terrain_cost")
    control_point = row.get("control_point_terrain_cost")
    if sampled is None or control_point is None:
        return False
    return abs(float(control_point) - float(sampled)) / max(1.0, abs(float(sampled))) >= 0.5


def _recommended_next_action(cost_gate_class: str) -> str:
    if cost_gate_class in {
        "terrain_proxy_mismatch",
        "direction_cone_fixed_but_cost_blocked",
        "insufficient_cost_diagnostics",
    }:
        return "candidate for terrain proxy or quality-gate review"
    if cost_gate_class == "baseline_overlap_or_duplicate":
        return "review duplicate or baseline comparison before gate change"
    return "must remain rejected"


def _matrix_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("matrix_id"))].append(row)
    summaries = []
    for matrix_id, matrix_rows in sorted(grouped.items()):
        counts = Counter(row["cost_gate_class"] for row in matrix_rows)
        summaries.append(
            {
                "matrix_id": matrix_id,
                "case_count": len(matrix_rows),
                "cost_gate_class_counts": {name: counts.get(name, 0) for name in COST_GATE_CLASSES},
                "top_blocking_cases": _top_blocking_cases(matrix_rows),
            }
        )
    return summaries


def _scenario_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scenario_id"))].append(row)
    summaries = []
    for scenario_id, scenario_rows in sorted(grouped.items()):
        counts = Counter(row["cost_gate_class"] for row in scenario_rows)
        summaries.append(
            {
                "scenario_id": scenario_id,
                "case_count": len(scenario_rows),
                "cost_gate_class_counts": {name: counts.get(name, 0) for name in COST_GATE_CLASSES},
            }
        )
    return summaries


def _top_blocking_cases(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["recommended_next_action"] != "candidate for terrain proxy or quality-gate review",
            str(row.get("scenario_id")),
            int(row.get("action_index") or 0),
        ),
    )
    return [
        {
            "scenario_id": row.get("scenario_id"),
            "action_index": row.get("action_index"),
            "cost_gate_class": row["cost_gate_class"],
            "recommended_next_action": row["recommended_next_action"],
            "cost_delta_vs_baseline": row.get("cost_delta_vs_baseline"),
            "high_cost_exposure_delta_vs_baseline": row.get("high_cost_exposure_delta_vs_baseline"),
            "route_artifact": row.get("route_artifact"),
        }
        for row in sorted_rows[:limit]
    ]


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# GCS Control-Point Cost-Gate Decomposition",
        "",
        f"- schema_version: `{summary['schema_version']}`",
        f"- source_targeted_sweep_summary: `{summary['source_targeted_sweep_summary']}`",
        f"- case_count: {summary['case_count']}",
        f"- recommendation: `{summary['recommendation']}`",
        f"- default_change_reason: `{summary['default_change_reason']}`",
        "",
        "## Cost Gate Classes",
        "",
    ]
    for name, count in summary["cost_gate_class_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Matrix Groups", ""])
    cases_by_matrix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary["cases"]:
        cases_by_matrix[str(row.get("matrix_id"))].append(row)
    for matrix_id, rows in sorted(cases_by_matrix.items()):
        lines.extend([f"## Matrix: {matrix_id}", ""])
        rows_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_scenario[str(row.get("scenario_id"))].append(row)
        for scenario_id, scenario_rows in sorted(rows_by_scenario.items()):
            lines.extend([f"### Scenario: {scenario_id}", ""])
            for row in sorted(scenario_rows, key=lambda item: int(item.get("action_index") or 0)):
                lines.append(
                    "- action={action} class=`{klass}` transition=`{baseline}->{new}` "
                    "delta={delta} high_cost_delta={high} next={next_action}".format(
                        action=row.get("action_index"),
                        klass=row["cost_gate_class"],
                        baseline=row.get("baseline_fallback_reason"),
                        new=row.get("new_fallback_reason"),
                        delta=row.get("cost_delta_vs_baseline"),
                        high=row.get("high_cost_exposure_delta_vs_baseline"),
                        next_action=row["recommended_next_action"],
                    )
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0.0


def _int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
