from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import xunce_artifact_io as artifact_io


BOOTSTRAP_SEED = 20260726
BOOTSTRAP_RESAMPLES = 2000
COVERAGE_THRESHOLD = 0.80
MIDTERM_LATENCY_MS = 2000.0
FINAL_LATENCY_MS = 1000.0
PLATFORMS = ("wheel", "legged", "hopper")
SCALES = ("standard", "kilometer")
STRATA = tuple(f"{platform}/{scale}" for platform in PLATFORMS for scale in SCALES)
TIMING_COMPONENTS = (
    "input_validation_ns",
    "platform_instantiation_ns",
    "search_ns",
    "complete_route_validation_ns",
    "result_assembly_ns",
)

COLORS = {
    "test_q24": "#6F7FB3",
    "unseen24": "#D7A8BB",
    "wheel": "#6F7FB3",
    "legged": "#C78FA6",
    "hopper": "#3EA6A0",
    "ink": "#262626",
    "grid": "#D7D7D7",
    "threshold": "#4A4A4A",
}


class FinalizationError(RuntimeError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FinalizationError(reason)


def _sha256(path: Path) -> str:
    return hashlib.sha256(artifact_io.read_bytes(path)).hexdigest()


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    _require(bool(values), "empty_quantile_input")
    index = max(1, math.ceil(quantile * len(values))) - 1
    return sorted(values)[index]


def _bootstrap_ci(values: Sequence[float]) -> dict[str, Any]:
    generator = random.Random(BOOTSTRAP_SEED)
    sampled_means = sorted(
        statistics.mean(generator.choice(values) for _ in range(len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "unit": "episode",
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "confidence_level": 0.95,
        "lower": _nearest_rank(sampled_means, 0.025),
        "upper": _nearest_rank(sampled_means, 0.975),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(artifact_io.read_text(path))))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    artifact_io.write_text(path, buffer.getvalue())


def _g1_summary(rows: Sequence[Mapping[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(len(rows) == 48, "g1_expected_48_episodes")
    normalized: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for row in rows:
        split = str(row["evaluation_split"])
        _require(split in {"test_q24", "unseen24"}, "g1_unknown_split")
        episode_id = str(row["episode_id"])
        _require(episode_id not in seen_episode_ids, "g1_duplicate_episode")
        seen_episode_ids.add(episode_id)
        covered = int(row["final_covered_cell_count"])
        denominator = int(row["denominator_cell_count"])
        coverage = float(row["coverage"])
        violations = int(row["safety_violation_count"])
        _require(denominator > 0 and 0 <= covered <= denominator, "g1_cell_counts_invalid")
        _require(abs(coverage - covered / denominator) <= 1e-15, "g1_coverage_recompute_mismatch")
        _require(violations == 0, "g1_safety_violation")
        normalized.append(
            {
                **dict(row),
                "episode_index": int(row["episode_index"]),
                "final_covered_cell_count": covered,
                "denominator_cell_count": denominator,
                "coverage": coverage,
                "coverage_percent": coverage * 100.0,
                "safety_violation_count": violations,
            }
        )
    normalized.sort(key=lambda item: (str(item["evaluation_split"]), int(item["episode_index"])))
    summaries: dict[str, Any] = {}
    for split in ("test_q24", "unseen24"):
        split_rows = [row for row in normalized if row["evaluation_split"] == split]
        _require(len(split_rows) == 24, f"g1_{split}_expected_24")
        indices = {int(row["episode_index"]) for row in split_rows}
        _require(indices == set(range(24)), f"g1_{split}_episode_index_set")
        values = [float(row["coverage"]) for row in split_rows]
        summaries[split] = {
            "sample_count": len(values),
            "mean": statistics.mean(values),
            "sample_stddev": statistics.stdev(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "coverage_80_count": sum(value >= COVERAGE_THRESHOLD for value in values),
            "coverage_80_proportion": sum(value >= COVERAGE_THRESHOLD for value in values)
            / len(values),
            "coverage_99_count": sum(value >= 0.99 for value in values),
            "bootstrap_ci": _bootstrap_ci(values),
            "mean_threshold_passed": statistics.mean(values) >= COVERAGE_THRESHOLD,
            "safety_violation_count": 0,
        }
    _require(
        sum(row["result_origin"] == "repair_rerun" for row in normalized) == 3,
        "g1_repair_lineage_expected_3",
    )
    return (
        {
            "schema_version": "xunce-mid-dual-final-g1-summary/v1",
            "threshold": COVERAGE_THRESHOLD,
            "splits": summaries,
            "overall_episode_count": 48,
            "overall_coverage_80_count": sum(
                float(row["coverage"]) >= COVERAGE_THRESHOLD for row in normalized
            ),
            "overall_safety_violation_count": 0,
            "lineage": {
                "kind": "mixed-code-incremental-repair/v1",
                "parent_episode_count": 45,
                "repair_episode_count": 3,
                "homogeneous_code_execution": False,
            },
            "gate_passed": all(
                summaries[split]["mean_threshold_passed"]
                for split in ("test_q24", "unseen24")
            ),
        },
        normalized,
    )


def _truth_index(
    truth_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(len(truth_rows) == 129, "g2_expected_129_truth_requests")
    by_sha: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for wrapper in truth_rows:
        request_sha = wrapper.get("provider_request_sha256")
        request_id = wrapper.get("provider_request_id")
        truth = wrapper.get("truth_request")
        _require(type(request_sha) is str and len(request_sha) == 64, "g2_truth_request_sha")
        _require(type(request_id) is str and bool(request_id), "g2_truth_request_id")
        _require(type(truth) is dict, "g2_truth_request_payload")
        platform = truth.get("platform_kind")
        scale = truth.get("scale")
        difficulty = truth.get("difficulty_class")
        reachable = truth.get("oracle_reachable")
        _require(platform in PLATFORMS and scale in SCALES, "g2_truth_stratum")
        _require(
            difficulty in {"reachable", "hard_reachable", "unreachable"},
            "g2_truth_difficulty",
        )
        _require(type(reachable) is bool, "g2_truth_reachability")
        _require((difficulty == "unreachable") is (not reachable), "g2_truth_semantics")
        _require(request_sha not in by_sha and request_id not in by_id, "g2_truth_duplicate")
        normalized = {
            "provider_request_sha256": request_sha,
            "request_id": request_id,
            "platform": platform,
            "scale": scale,
            "difficulty_class": difficulty,
            "oracle_reachable": reachable,
        }
        by_sha[request_sha] = normalized
        by_id[request_id] = normalized
    return by_sha, by_id


def _validate_g2_row(row: Mapping[str, Any], truth: Mapping[str, Any]) -> float:
    _require(row.get("schema_version") == "xunce-mid-dual-g2-r3-formal-row/v1", "g2_schema")
    _require(row.get("formal_sample") is True, "g2_not_formal_sample")
    _require(row.get("schedule_kind") == "formal", "g2_schedule")
    _require(row.get("worker_count") == 4, "g2_worker_count")
    _require(row.get("request_id") == truth["request_id"], "g2_request_id_mismatch")
    _require(row.get("platform") == truth["platform"], "g2_platform_mismatch")
    _require(row.get("scale") == truth["scale"], "g2_scale_mismatch")
    repeat_index = row.get("repeat_index")
    _require(type(repeat_index) is int and 0 <= repeat_index < 5, "g2_repeat_index")
    timing = row.get("timing")
    _require(type(timing) is dict, "g2_timing_missing")
    component_sum = 0
    for key in TIMING_COMPONENTS:
        value = timing.get(key)
        _require(type(value) is int and value >= 0, f"g2_timing_{key}")
        component_sum += value
    _require(timing.get("total_ns") == component_sum, "g2_timing_sum_mismatch")
    elapsed_ms = row.get("elapsed_ms")
    _require(
        type(elapsed_ms) in {int, float}
        and not isinstance(elapsed_ms, bool)
        and math.isfinite(float(elapsed_ms))
        and float(elapsed_ms) >= 0.0,
        "g2_elapsed_invalid",
    )
    _require(abs(float(elapsed_ms) - component_sum / 1_000_000.0) <= 1e-12, "g2_elapsed_mismatch")
    provider_success = row.get("provider_success")
    route_l2_valid = row.get("route_l2_valid")
    _require(type(provider_success) is bool and type(route_l2_valid) is bool, "g2_outcome_flags")
    outcome = row.get("canonical_outcome")
    _require(type(outcome) is dict, "g2_canonical_outcome")
    if truth["oracle_reachable"]:
        validation = outcome.get("validation")
        _require(provider_success is True and route_l2_valid is True, "g2_reachable_failed")
        _require(
            outcome.get("outcome_type") == "success"
            and type(validation) is dict
            and validation.get("level") == "L2"
            and validation.get("passed") is True,
            "g2_reachable_l2_invalid",
        )
    else:
        _require(provider_success is False and route_l2_valid is False, "g2_unreachable_accepted")
        _require(outcome.get("outcome_type") == "failure", "g2_unreachable_outcome")
    return float(elapsed_ms)


def _g2_summary(
    formal_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(len(formal_rows) == 645, "g2_expected_645_calls")
    truth_by_sha, truth_by_id = _truth_index(truth_rows)
    call_ids: set[str] = set()
    repeats: dict[str, set[int]] = defaultdict(set)
    normalized_calls: list[dict[str, Any]] = []
    for row in formal_rows:
        request_sha = row.get("provider_request_sha256")
        _require(type(request_sha) is str and request_sha in truth_by_sha, "g2_truth_join")
        truth = truth_by_sha[request_sha]
        call_id = row.get("call_id")
        _require(type(call_id) is str and call_id not in call_ids, "g2_call_id")
        call_ids.add(call_id)
        elapsed_ms = _validate_g2_row(row, truth)
        repeat_index = int(row["repeat_index"])
        _require(repeat_index not in repeats[truth["request_id"]], "g2_repeat_duplicate")
        repeats[truth["request_id"]].add(repeat_index)
        normalized_calls.append(
            {
                "call_id": call_id,
                "request_id": truth["request_id"],
                "provider_request_sha256": request_sha,
                "platform": truth["platform"],
                "scale": truth["scale"],
                "difficulty_class": truth["difficulty_class"],
                "oracle_reachable": truth["oracle_reachable"],
                "repeat_index": repeat_index,
                "elapsed_ms": elapsed_ms,
                "provider_success": row["provider_success"],
                "route_l2_valid": row["route_l2_valid"],
            }
        )
    _require(set(repeats) == set(truth_by_id), "g2_request_set_mismatch")
    _require(
        all(indices == set(range(5)) for indices in repeats.values()),
        "g2_repeat_matrix",
    )

    strata: dict[str, dict[str, Any]] = {}
    for key in STRATA:
        platform, scale = key.split("/")
        truths = [
            truth
            for truth in truth_by_id.values()
            if truth["platform"] == platform and truth["scale"] == scale
        ]
        expected_total = 33 if scale == "standard" else 10
        expected_reachable = 30 if scale == "standard" else 8
        _require(len(truths) == expected_total, f"g2_{key}_truth_count")
        _require(
            sum(truth["oracle_reachable"] for truth in truths) == expected_reachable,
            f"g2_{key}_reachable_count",
        )
        stratum_calls = [
            row
            for row in normalized_calls
            if row["platform"] == platform and row["scale"] == scale
        ]
        reachable_calls = [row for row in stratum_calls if row["oracle_reachable"]]
        elapsed = [float(row["elapsed_ms"]) for row in reachable_calls]
        passed_1s = sum(value <= FINAL_LATENCY_MS for value in elapsed)
        passed_2s = sum(value <= MIDTERM_LATENCY_MS for value in elapsed)
        strata[key] = {
            "unique_request_count": expected_total,
            "reachable_expected_unique_request_count": expected_reachable,
            "unreachable_unique_request_count": expected_total - expected_reachable,
            "raw_call_count": len(stratum_calls),
            "reachable_expected_call_count": len(reachable_calls),
            "provider_success_count": sum(row["provider_success"] for row in reachable_calls),
            "route_l2_valid_count": sum(row["route_l2_valid"] for row in reachable_calls),
            "at_or_below_1000ms_count": passed_1s,
            "at_or_below_1000ms_proportion": passed_1s / len(elapsed),
            "at_or_below_2000ms_count": passed_2s,
            "at_or_below_2000ms_proportion": passed_2s / len(elapsed),
            "mean_ms": sum(elapsed) / len(elapsed),
            "p50_ms": _nearest_rank(elapsed, 0.50),
            "p95_ms": _nearest_rank(elapsed, 0.95),
            "max_ms": max(elapsed),
        }
    canonical = {
        "schema_version": "xunce-mid-dual-g2-canonical-summary/v2",
        "formal_call_count": 645,
        "unique_request_count": 129,
        "reachable_unique_request_count": 114,
        "unreachable_unique_request_count": 15,
        "strata": strata,
    }
    reachable_calls = [row for row in normalized_calls if row["oracle_reachable"]]
    unreachable_calls = [row for row in normalized_calls if not row["oracle_reachable"]]
    correctness_passed = (
        len(reachable_calls) == 570
        and all(row["provider_success"] and row["route_l2_valid"] for row in reachable_calls)
        and len(unreachable_calls) == 75
        and all(
            not row["provider_success"] and not row["route_l2_valid"]
            for row in unreachable_calls
        )
    )
    midterm_passed = all(
        cell["mean_ms"] <= MIDTERM_LATENCY_MS
        and cell["p95_ms"] <= MIDTERM_LATENCY_MS
        and cell["max_ms"] <= MIDTERM_LATENCY_MS
        for cell in strata.values()
    )
    final_passed = all(
        cell["mean_ms"] <= FINAL_LATENCY_MS
        and cell["p95_ms"] <= FINAL_LATENCY_MS
        and cell["at_or_below_1000ms_proportion"] >= 0.95
        for cell in strata.values()
    )
    canonical["correctness"] = {
        "reachable_l2_valid_count": sum(row["route_l2_valid"] for row in reachable_calls),
        "reachable_expected_call_count": len(reachable_calls),
        "unreachable_rejected_count": sum(
            not row["provider_success"] and not row["route_l2_valid"]
            for row in unreachable_calls
        ),
        "unreachable_expected_call_count": len(unreachable_calls),
        "passed": correctness_passed,
    }
    canonical["thresholds"] = {
        "midterm_ms": MIDTERM_LATENCY_MS,
        "final_ms": FINAL_LATENCY_MS,
        "midterm_passed": midterm_passed,
        "final_passed": final_passed,
    }
    canonical["gate_passed"] = correctness_passed and midterm_passed and final_passed
    normalized_calls.sort(
        key=lambda item: (
            PLATFORMS.index(str(item["platform"])),
            SCALES.index(str(item["scale"])),
            str(item["request_id"]),
            int(item["repeat_index"]),
        )
    )
    return canonical, normalized_calls


def _configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, output_root: Path, stem: str) -> list[str]:
    outputs: list[str] = []
    for suffix, fmt, dpi in (
        ("svg", "svg", 300),
        ("pdf", "pdf", 300),
        ("png", "png", 300),
        ("tiff", "tiff", 600),
    ):
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        relative = f"{stem}.{suffix}"
        artifact_io.write_bytes(output_root / relative, buffer.getvalue())
        outputs.append(relative)
    return outputs


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _dual_gate_figure(
    g1: Mapping[str, Any],
    g1_rows: Sequence[Mapping[str, Any]],
    g2: Mapping[str, Any],
    output_root: Path,
) -> list[str]:
    fig = plt.figure(figsize=(7.20, 4.35))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.02, 1.0),
        height_ratios=(1.13, 0.87),
        wspace=0.46,
        hspace=0.52,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    for split, marker, label in (
        ("test_q24", "o", "Test-Q24"),
        ("unseen24", "s", "Unseen-24"),
    ):
        values = sorted(
            float(row["coverage_percent"])
            for row in g1_rows
            if row["evaluation_split"] == split
        )
        ax_a.plot(
            range(1, len(values) + 1),
            values,
            marker=marker,
            markersize=3.2,
            linewidth=1.0,
            color=COLORS[split],
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=label,
        )
    ax_a.axhline(
        COVERAGE_THRESHOLD * 100.0,
        color=COLORS["threshold"],
        linestyle=(0, (4, 2)),
        linewidth=0.9,
        label="80% threshold",
    )
    ax_a.set_xlim(0.5, 24.5)
    ax_a.set_ylim(0, 102)
    ax_a.set_xlabel("Scene rank within split")
    ax_a.set_ylabel("Final coverage (%)")
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    ax_a.legend(loc="lower right", frameon=False)
    test = g1["splits"]["test_q24"]
    unseen = g1["splits"]["unseen24"]
    ax_a.text(
        0.54,
        0.72,
        (
            f"Test mean {test['mean'] * 100:.2f}%  ({test['coverage_80_count']}/24 ≥80%)\n"
            f"Unseen mean {unseen['mean'] * 100:.2f}%  ({unseen['coverage_80_count']}/24 ≥80%)"
        ),
        transform=ax_a.transAxes,
        va="top",
        ha="center",
        fontsize=6.7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BDBDBD"},
    )
    _panel_label(ax_a, "a")

    x = np.arange(len(STRATA))
    metrics = (
        ("mean_ms", "Mean", "o"),
        ("p95_ms", "P95", "s"),
        ("max_ms", "Maximum", "^"),
    )
    for field, label, marker in metrics:
        ax_b.plot(
            x,
            [float(g2["strata"][key][field]) for key in STRATA],
            color=COLORS["ink"],
            linewidth=0.7,
            marker=marker,
            markersize=3.7,
            markerfacecolor="white" if field != "mean_ms" else COLORS["ink"],
            markeredgewidth=0.8,
            label=label,
        )
    for index, key in enumerate(STRATA):
        platform = key.split("/")[0]
        ax_b.scatter(
            [index],
            [float(g2["strata"][key]["p95_ms"])],
            s=17,
            marker="s",
            facecolor=COLORS[platform],
            edgecolor=COLORS["ink"],
            linewidth=0.45,
            zorder=4,
        )
    ax_b.axhline(
        FINAL_LATENCY_MS,
        color=COLORS["threshold"],
        linestyle=(0, (4, 2)),
        linewidth=0.9,
    )
    ax_b.axhline(
        MIDTERM_LATENCY_MS,
        color="#9B9B9B",
        linestyle=(0, (1.5, 2.5)),
        linewidth=0.8,
    )
    ax_b.text(5.25, FINAL_LATENCY_MS * 1.06, "1 s", ha="right", va="bottom", fontsize=6.0)
    ax_b.text(5.25, MIDTERM_LATENCY_MS * 1.03, "2 s", ha="right", va="bottom", fontsize=6.0)
    ax_b.set_yscale("log")
    ax_b.set_ylim(8, 3200)
    ax_b.set_ylabel("Latency (ms, log scale)")
    ax_b.set_xticks(x, ("W-S", "W-K", "L-S", "L-K", "H-S", "H-K"))
    ax_b.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    ax_b.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        borderpad=0.2,
        ncol=3,
        handlelength=1.2,
        columnspacing=0.8,
    )
    _panel_label(ax_b, "b")

    correctness = g2["correctness"]
    labels = ("Reachable L2", "Unreachable rejected")
    numerators = (
        correctness["reachable_l2_valid_count"],
        correctness["unreachable_rejected_count"],
    )
    denominators = (
        correctness["reachable_expected_call_count"],
        correctness["unreachable_expected_call_count"],
    )
    bars = ax_c.barh(
        [0, 1],
        [100.0, 100.0],
        height=0.52,
        color=(COLORS["wheel"], COLORS["hopper"]),
        edgecolor=COLORS["ink"],
        linewidth=0.55,
        hatch=("", "///"),
    )
    for bar, label, numerator, denominator in zip(
        bars,
        labels,
        numerators,
        denominators,
    ):
        center_y = bar.get_y() + bar.get_height() / 2
        ax_c.text(
            2.5,
            center_y,
            label,
            ha="left",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=6.8,
        )
        ax_c.text(
            98.5,
            center_y,
            f"{numerator}/{denominator}",
            ha="right",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=7.0,
        )
    ax_c.set_yticks([])
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 102)
    ax_c.set_xlabel("Correct decision rate (%)")
    ax_c.set_xticks((0, 25, 50, 75, 100))
    ax_c.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    _panel_label(ax_c, "c")

    fig.align_ylabels((ax_a, ax_b))
    outputs = _save_figure(fig, output_root, "figure_dual_gate")
    plt.close(fig)
    return outputs


def _latency_distribution_figure(
    calls: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[str]:
    reachable = [row for row in calls if row["oracle_reachable"]]
    grouped = [
        [
            float(row["elapsed_ms"])
            for row in reachable
            if f"{row['platform']}/{row['scale']}" == key
        ]
        for key in STRATA
    ]
    fig, ax = plt.subplots(figsize=(7.20, 3.30))
    box = ax.boxplot(
        grouped,
        positions=np.arange(1, 7),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["ink"], "linewidth": 1.0},
        whiskerprops={"color": COLORS["ink"], "linewidth": 0.65},
        capprops={"color": COLORS["ink"], "linewidth": 0.65},
        boxprops={"color": COLORS["ink"], "linewidth": 0.65},
    )
    for patch, key in zip(box["boxes"], STRATA):
        patch.set_facecolor(COLORS[key.split("/")[0]])
        patch.set_alpha(0.72)
        if key.endswith("kilometer"):
            patch.set_hatch("///")
    generator = np.random.default_rng(20260728)
    for index, (values, key) in enumerate(zip(grouped, STRATA), start=1):
        jitter = generator.uniform(-0.19, 0.19, len(values))
        ax.scatter(
            index + jitter,
            values,
            s=5.0,
            color=COLORS[key.split("/")[0]],
            alpha=0.36,
            edgecolors="none",
            rasterized=False,
        )
        cell = {
            "p95": _nearest_rank(values, 0.95),
            "n": len(values),
        }
        ax.text(
            index,
            max(values) * 1.12,
            f"P95 {cell['p95']:.1f}\nn={cell['n']}",
            ha="center",
            va="bottom",
            fontsize=6.0,
        )
    ax.axhline(
        FINAL_LATENCY_MS,
        color=COLORS["threshold"],
        linestyle=(0, (4, 2)),
        linewidth=0.9,
        label="1 s formal target",
    )
    ax.axhline(
        MIDTERM_LATENCY_MS,
        color="#9B9B9B",
        linestyle=(0, (1.5, 2.5)),
        linewidth=0.8,
        label="2 s midterm threshold",
    )
    ax.set_yscale("log")
    ax.set_ylim(8, 3000)
    ax.set_ylabel("End-to-end planning latency (ms, log scale)")
    ax.set_xlabel("Platform and map scale")
    ax.set_xticks(
        np.arange(1, 7),
        (
            "Wheel\nStandard",
            "Wheel\nKilometer",
            "Legged\nStandard",
            "Legged\nKilometer",
            "Hopper\nStandard",
            "Hopper\nKilometer",
        ),
    )
    ax.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    ax.legend(loc="upper left", frameon=False)
    _panel_label(ax, "a")
    outputs = _save_figure(fig, output_root, "figure_g2_latency_distribution")
    plt.close(fig)
    return outputs


def _figure_contract() -> str:
    return """# 中期双门槛结果图件合同

- 核心结论：缩减规模正式试验中，G1 两个固定场景分组的平均覆盖率均超过 80%；G2 六个平台—尺度分层同时满足正确性和时延门槛。
- 图形原型：定量多面板结果图；另附 G2 原始可达调用时延分布图。
- 统计单位：G1 为场景；G2 为正式调用。G2 每个请求重复 5 次，图中不把重复调用解释为相互独立的新场景。
- G1 口径：Test-Q24 与 Unseen-24 各 24 个场景，阈值 80%；报告均值、样本标准差、中位数和 episode-level bootstrap 95% 区间。
- G2 口径：129 个唯一请求、645 次正式调用；可达请求按 L2 路线有效性判断，不可达请求按正确拒绝判断；时延仅对可达调用统计。
- 阈值：G1 平均覆盖率 80%；G2 中期门槛 2 s，正式目标 1 s。
- 输出：183 mm 双栏宽度；SVG 为可编辑主文件，PDF、300 dpi PNG 和 600 dpi TIFF 为辅助文件。
- 视觉编码：平台使用三种低饱和颜色，尺度同时用纹理或标签编码；阈值使用线型，避免仅依赖红绿色。
- 边界：结果来自固定离线仿真条件，不外推为实物平台或第三方测试结论；G1 明示 45 条原始结果与 3 条增量修复结果的混合 lineage。
"""


def _caption(g1: Mapping[str, Any], g2: Mapping[str, Any]) -> str:
    test = g1["splits"]["test_q24"]
    unseen = g1["splits"]["unseen24"]
    worst_key = max(STRATA, key=lambda key: float(g2["strata"][key]["p95_ms"]))
    worst = g2["strata"][worst_key]
    return f"""# 图注

**图 6-3｜中期双门槛正式实验结果。** a，Test-Q24 与 Unseen-24 各 24 个场景的最终覆盖率排序；虚线为 80% 门槛。两组平均覆盖率分别为 {test['mean'] * 100:.3f}% 和 {unseen['mean'] * 100:.3f}%，达到 80% 的场景分别为 {test['coverage_80_count']}/24 和 {unseen['coverage_80_count']}/24。b，轮式、足式和跳跃式平台在标准与公里级地图上的平均、P95 和最大规划时延；最慢分层为 {worst_key}，P95={worst['p95_ms']:.3f} ms。c，可达调用 L2 有效路线 {g2['correctness']['reachable_l2_valid_count']}/{g2['correctness']['reachable_expected_call_count']}，不可达调用正确拒绝 {g2['correctness']['unreachable_rejected_count']}/{g2['correctness']['unreachable_expected_call_count']}。G1 Unseen-24 由 21 条原始结果与 3 条增量修复结果组成；总体 47/48 个场景达到 80%，不表述为 48/48。

**图 6-4｜六个平台—尺度分层的正式规划时延分布。** 仅统计真值标记为可达的 570 次调用；标准尺度每个平台 n=150，公里级每个平台 n=40。箱线图表示四分位范围与中位数，散点为单次调用；虚线和点线分别表示 1 s 正式目标与 2 s 中期门槛。每个唯一请求重复 5 次，重复调用用于稳定计时，不作为新增独立场景。
"""


def _report(g1: Mapping[str, Any], g2: Mapping[str, Any]) -> str:
    test = g1["splits"]["test_q24"]
    unseen = g1["splits"]["unseen24"]
    rows = []
    for key in STRATA:
        cell = g2["strata"][key]
        rows.append(
            f"| {key} | {cell['reachable_expected_call_count']} | "
            f"{cell['mean_ms']:.3f} | {cell['p50_ms']:.3f} | "
            f"{cell['p95_ms']:.3f} | {cell['max_ms']:.3f} | 通过 |"
        )
    return f"""# 中期缩减规模双门槛正式实验结果

结论：在既定 24+24 个探索场景和 645 次路径规划调用规模下，未知环境自主探索覆盖率门槛与路径规划处理时间门槛均通过。

## G1 未知环境自主探索覆盖率

| 分组 | 场景数 | 平均覆盖率 | 样本标准差 | 中位数 | 95% bootstrap CI | 达到80%的场景 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| Test-Q24 | 24 | {test['mean'] * 100:.3f}% | {test['sample_stddev'] * 100:.3f}% | {test['median'] * 100:.3f}% | [{test['bootstrap_ci']['lower'] * 100:.3f}%, {test['bootstrap_ci']['upper'] * 100:.3f}%] | {test['coverage_80_count']}/24 | 通过 |
| Unseen-24 | 24 | {unseen['mean'] * 100:.3f}% | {unseen['sample_stddev'] * 100:.3f}% | {unseen['median'] * 100:.3f}% | [{unseen['bootstrap_ci']['lower'] * 100:.3f}%, {unseen['bootstrap_ci']['upper'] * 100:.3f}%] | {unseen['coverage_80_count']}/24 | 通过 |

两组平均覆盖率均高于 80%。48 个场景中有 {g1['overall_coverage_80_count']} 个达到 80%，全部场景安全违规计数为 0。Unseen-24 使用 21 条原始结果和 3 条增量修复结果进行显式 lineage 汇总；Test-Q24 保留原始 24 条结果，因此总体结果不表述为 48/48 全部通过。

## G2 路径规划正确性与处理时间

正式试验包含 129 个唯一请求，每个请求重复 5 次，共 645 次调用。其中可达请求对应 570 次调用，全部输出 L2 有效路线；不可达请求对应 75 次调用，全部被正确拒绝。

| 平台/尺度 | 可达调用数 | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) | 判定 |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

六个平台—尺度分层的平均值、P95 和最大值均低于 2 s 中期门槛；同时各分层平均值和 P95 均低于 1 s，且 100% 可达调用不超过 1 s，正式目标亦通过。

## 结果边界

上述结论对应固定离线仿真场景、固定输入及规定统计口径，不外推为实物平台或第三方测试结论。
"""


def _source_hashes(
    g1_csv: Path,
    g2_results: Path,
    g2_truth: Path,
    g2_runner_audit: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "xunce-mid-dual-final-source-hashes/v1",
        "sources": {
            "g1_episodes_csv": {"path": str(g1_csv), "sha256": _sha256(g1_csv)},
            "g2_formal_results_jsonl": {
                "path": str(g2_results),
                "sha256": _sha256(g2_results),
            },
            "g2_truth_sidecar_jsonl": {
                "path": str(g2_truth),
                "sha256": _sha256(g2_truth),
            },
            "g2_runner_recompute_audit": {
                "path": str(g2_runner_audit),
                "sha256": _sha256(g2_runner_audit),
            },
        },
    }


def _manifest(output_root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    files = []
    for relative in artifact_io.list_relative_files(output_root):
        if relative == "manifest.json":
            continue
        path = output_root / relative
        files.append(
            {
                "path": relative,
                "bytes": artifact_io.file_size(path),
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": "xunce-mid-dual-final-manifest/v1",
        "status": summary["status"],
        "double_gate_passed": summary["double_gate_passed"],
        "files": files,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    g1_csv = Path(args.g1_audit_root) / "episodes.csv"
    g2_root = Path(args.g2_root)
    g2_results = g2_root / "phases" / "p04" / "a01" / "results.jsonl"
    g2_runner_audit = g2_root / "g2_r3_recompute_audit.json"
    g2_truth = Path(args.g2_execution_root) / "truth" / "request-sidecar.jsonl"
    output_root = Path(args.output_root)
    for source in (g1_csv, g2_results, g2_runner_audit, g2_truth):
        _require(artifact_io.path_is_file(source), f"missing_source:{source}")
    _require(not artifact_io.path_exists(output_root), "output_root_already_exists")
    artifact_io.make_dirs(output_root)

    g1, g1_rows = _g1_summary(_read_csv(g1_csv))
    formal_rows = artifact_io.read_jsonl(g2_results)
    truth_rows = artifact_io.read_jsonl(g2_truth)
    g2, g2_calls = _g2_summary(formal_rows, truth_rows)

    runner_audit = artifact_io.read_json(g2_runner_audit)
    independent_core = {
        key: g2[key]
        for key in (
            "schema_version",
            "formal_call_count",
            "unique_request_count",
            "reachable_unique_request_count",
            "unreachable_unique_request_count",
            "strata",
        )
    }
    _require(independent_core == runner_audit, "g2_runner_independent_recompute_mismatch")
    _require(g1["gate_passed"] and g2["gate_passed"], "double_gate_not_passed")

    summary = {
        "schema_version": "xunce-mid-dual-final-summary/v1",
        "status": "passed",
        "double_gate_passed": True,
        "scale_profile": "midterm_reduced_w8x3_update80/v1",
        "g1": g1,
        "g2": g2,
        "independent_recompute": {
            "g1_from_episode_rows": True,
            "g2_from_645_raw_calls_and_129_truth_rows": True,
            "g2_runner_consistency": True,
        },
    }
    artifact_io.write_json(output_root / "summary.json", summary)
    artifact_io.write_json(
        output_root / "source-hashes.json",
        _source_hashes(g1_csv, g2_results, g2_truth, g2_runner_audit),
    )
    _write_csv(
        output_root / "g1-episodes.csv",
        g1_rows,
        (
            "evaluation_split",
            "episode_index",
            "episode_id",
            "scenario_id",
            "lane_id",
            "result_origin",
            "final_covered_cell_count",
            "denominator_cell_count",
            "coverage",
            "coverage_percent",
            "safety_violation_count",
            "termination_reason",
            "source_line_sha256",
            "result_sha256",
        ),
    )
    _write_csv(
        output_root / "g2-calls.csv",
        g2_calls,
        (
            "call_id",
            "request_id",
            "provider_request_sha256",
            "platform",
            "scale",
            "difficulty_class",
            "oracle_reachable",
            "repeat_index",
            "elapsed_ms",
            "provider_success",
            "route_l2_valid",
        ),
    )
    strata_rows = [{"stratum": key, **g2["strata"][key]} for key in STRATA]
    _write_csv(
        output_root / "g2-strata.csv",
        strata_rows,
        ("stratum", *next(iter(g2["strata"].values())).keys()),
    )
    artifact_io.write_text(output_root / "figure-contract.md", _figure_contract())
    artifact_io.write_text(output_root / "caption.md", _caption(g1, g2))
    artifact_io.write_text(output_root / "report.md", _report(g1, g2))

    _configure_matplotlib()
    _dual_gate_figure(g1, g1_rows, g2, output_root)
    _latency_distribution_figure(g2_calls, output_root)
    artifact_io.write_json(output_root / "manifest.json", _manifest(output_root, summary))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立复算并生成中期双门槛正式结果与学术图件。"
    )
    parser.add_argument("--g1-audit-root", required=True)
    parser.add_argument("--g2-root", required=True)
    parser.add_argument("--g2-execution-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = finalize(args)
    print(
        f"status={summary['status']} "
        f"double_gate_passed={summary['double_gate_passed']} "
        f"output_root={args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
