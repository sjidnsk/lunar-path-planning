"""Deterministic Stage 5 CSV and Markdown report rendering."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np

from lunar_exploration_ppo.eval.baselines import ALL_METHODS
from lunar_exploration_ppo.eval.metrics import (
    BOOTSTRAP_METRICS,
    EpisodeResult,
    validate_episode_result,
)


CLAIM_BOUNDARY: Final = (
    "fair_baseline_evaluator_system_closure_no_task_advantage/v1"
)
_SUMMARY_FIELDS: Final = frozenset(
    {"method", "scale_profile", "episode_count", *BOOTSTRAP_METRICS}
)
COMPARISON_COLUMNS: Final = tuple(
    column
    for metric in BOOTSTRAP_METRICS
    for column in (metric, f"{metric}_ci95_low", f"{metric}_ci95_high")
)
COMPARISON_COLUMNS = (
    "method",
    "scale_profile",
    "episode_count",
    *COMPARISON_COLUMNS,
)
COVERAGE_CURVE_COLUMNS: Final = (
    "method",
    "scale_profile",
    "step",
    "mean_coverage",
)


class ReportError(ValueError):
    """Report inputs cannot produce a comparable deterministic artifact."""


def comparison_table_csv(
    summaries: Sequence[Mapping[str, object]],
    bootstrap_audits: Sequence[Mapping[str, object]],
) -> bytes:
    ordered = _ordered_summaries(summaries)
    audit_by_method = _audits_by_method(bootstrap_audits)
    rows: list[tuple[str, ...]] = [COMPARISON_COLUMNS]
    for summary in ordered:
        method = str(summary["method"])
        audit = audit_by_method[method]
        metric_audits = audit.get("metrics")
        if not isinstance(metric_audits, Mapping) or set(metric_audits) != set(
            BOOTSTRAP_METRICS
        ):
            raise ReportError("bootstrap audit metric schema drift")
        values: list[str] = [
            method,
            str(summary["scale_profile"]),
            _format_value(summary["episode_count"]),
        ]
        for metric in BOOTSTRAP_METRICS:
            interval = metric_audits[metric]
            if not isinstance(interval, Mapping) or set(interval) != {
                "estimate",
                "ci95_low",
                "ci95_high",
                "valid_resample_count",
            }:
                raise ReportError("bootstrap interval schema drift")
            if interval["estimate"] != summary[metric]:
                raise ReportError("bootstrap estimate does not match summary")
            values.extend(
                (
                    _format_value(summary[metric]),
                    _format_value(interval["ci95_low"]),
                    _format_value(interval["ci95_high"]),
                )
            )
        rows.append(tuple(values))
    return _csv_bytes(rows)


def coverage_curves_csv(episodes: Sequence[EpisodeResult]) -> bytes:
    rows = tuple(episodes)
    if not rows:
        raise ReportError("coverage curves require episodes")
    for row in rows:
        validate_episode_result(row)
    groups = {
        method: tuple(row for row in rows if row.method == method)
        for method in ALL_METHODS
    }
    if any(not group for group in groups.values()) or {
        row.method for row in rows
    } != set(ALL_METHODS):
        raise ReportError("coverage curves require the exact Stage 5 method set")
    output: list[tuple[str, ...]] = [COVERAGE_CURVE_COLUMNS]
    for method in ALL_METHODS:
        group = tuple(sorted(groups[method], key=_episode_order_key))
        scales = {row.scale_profile for row in group}
        lengths = {len(row.coverage_curve) for row in group}
        if len(scales) != 1 or len(lengths) != 1:
            raise ReportError("coverage curve schema differs within a method")
        matrix = np.asarray([row.coverage_curve for row in group], dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ReportError("coverage curves contain non-finite values")
        mean_curve = np.mean(matrix, axis=0, dtype=np.float64)
        for step, value in enumerate(mean_curve):
            output.append(
                (
                    method,
                    group[0].scale_profile,
                    str(step),
                    _format_value(float(value)),
                )
            )
    return _csv_bytes(output)


def baseline_eval_report(
    summaries: Sequence[Mapping[str, object]],
    *,
    claim_boundary: str,
) -> bytes:
    if claim_boundary != CLAIM_BOUNDARY:
        raise ReportError("Stage 5 claim boundary drift")
    ordered = _ordered_summaries(summaries)
    lines = [
        "# Stage 5 Fair Baseline Evaluator",
        "",
        f"Claim boundary: `{claim_boundary}`.",
        "",
        (
            "This report records Smoke evaluator system acceptance only and does not "
            "establish PPO task advantage."
        ),
        "",
        "All methods use the same environment, ordered scenario/seed schedule, candidate "
        "set, planner validation, sensor model, exact coverable denominator, success "
        "threshold, and fixed step budget. The method-specific action rule is the only "
        "comparison difference: non-learning baselines select a frontier by their fixed "
        "rule and execute that candidate's `recommended_theta`; PPO selects the lowest-index "
        "masked-logit argmax and executes that candidate's policy `theta_mu`.",
        "",
        "| method | success rate | mean final coverage | steps to 99% (success only) | "
        "path to 99% (success only) | coverage AUC | coverage per meter | invalid mean | "
        "planner failure mean | safety violations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in ordered:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(summary["method"]),
                    _format_value(summary["success_rate_under_fixed_step_budget"]),
                    _format_value(summary["mean_final_coverage"]),
                    _format_value(summary["steps_to_99_success_only"]),
                    _format_value(summary["path_length_to_99_success_only"]),
                    _format_value(summary["coverage_auc_over_steps"]),
                    _format_value(summary["coverage_per_meter"]),
                    _format_value(summary["invalid_action_count_mean"]),
                    _format_value(summary["planner_failure_count_mean"]),
                    _format_value(summary["safety_violation_count"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Success is `coverage_rate >= 0.99` at or before `max_steps`. Unsuccessful "
            "episodes have null success-only step and path metrics. Confidence intervals "
            "are fixed-seed episode bootstrap intervals, never step-resampled intervals.",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _ordered_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    rows = tuple(summaries)
    if len(rows) != len(ALL_METHODS):
        raise ReportError("summary schema requires exactly one row per Stage 5 method")
    by_method: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _SUMMARY_FIELDS:
            raise ReportError("summary schema drift")
        method = row.get("method")
        if not isinstance(method, str) or method in by_method:
            raise ReportError("summary method schema drift")
        _validate_summary_values(row)
        by_method[method] = row
    if set(by_method) != set(ALL_METHODS):
        raise ReportError("summary method set drift")
    return tuple(by_method[method] for method in ALL_METHODS)


def _audits_by_method(
    audits: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    rows = tuple(audits)
    by_method: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReportError("bootstrap audit schema drift")
        method = row.get("method")
        if not isinstance(method, str) or method in by_method:
            raise ReportError("bootstrap audit method schema drift")
        by_method[method] = row
    if len(rows) != len(ALL_METHODS) or set(by_method) != set(ALL_METHODS):
        raise ReportError("bootstrap audit method set drift")
    return by_method


def _validate_summary_values(summary: Mapping[str, object]) -> None:
    if not isinstance(summary["scale_profile"], str) or not summary["scale_profile"]:
        raise ReportError("summary scale profile is invalid")
    if type(summary["episode_count"]) is not int or summary["episode_count"] <= 0:
        raise ReportError("summary episode count is invalid")
    for name in BOOTSTRAP_METRICS:
        value = summary[name]
        if value is None and name in {
            "steps_to_99_success_only",
            "path_length_to_99_success_only",
        }:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ReportError("summary metric must be finite or an allowed null")


def _episode_order_key(result: EpisodeResult) -> tuple[object, ...]:
    return (
        result.scenario_key,
        result.scenario_seed,
        result.terrain_seed,
        result.start_pose_seed,
        result.evaluation_seed,
    )


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if type(value) is bool:
        raise ReportError("boolean is not a report number")
    if type(value) is int:
        return str(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ReportError("report numbers must be finite")
        return format(number, ".17g")
    raise ReportError("unsupported report value")


def _csv_bytes(rows: Sequence[Sequence[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


__all__ = [
    "CLAIM_BOUNDARY",
    "COMPARISON_COLUMNS",
    "COVERAGE_CURVE_COLUMNS",
    "ReportError",
    "baseline_eval_report",
    "comparison_table_csv",
    "coverage_curves_csv",
]
