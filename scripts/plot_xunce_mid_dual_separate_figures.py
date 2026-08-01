from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import xunce_artifact_io as artifact_io


COVERAGE_THRESHOLD_PERCENT = 80.0
FINAL_LATENCY_MS = 1000.0
MIDTERM_LATENCY_MS = 2000.0
STRATA = (
    "wheel/standard",
    "wheel/kilometer",
    "legged/standard",
    "legged/kilometer",
    "hopper/standard",
    "hopper/kilometer",
)
STRATUM_LABELS = (
    "Wheel\nStandard",
    "Wheel\nKilometer",
    "Legged\nStandard",
    "Legged\nKilometer",
    "Hopper\nStandard",
    "Hopper\nKilometer",
)
COLORS = {
    "test_q24": "#6F7FB3",
    "unseen24": "#C78FA6",
    "wheel": "#6F7FB3",
    "legged": "#C78FA6",
    "hopper": "#3EA6A0",
    "ink": "#252525",
    "grid": "#D8D8D8",
    "threshold": "#555555",
}


class FigureDataError(RuntimeError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FigureDataError(reason)


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(artifact_io.read_text(path))))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(artifact_io.read_text(path))
    _require(isinstance(payload, dict), f"expected_json_object:{path.name}")
    return payload


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.2,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _save_figure(fig: plt.Figure, output_root: Path, stem: str) -> list[Path]:
    artifact_io.make_dirs(output_root)
    outputs = [
        output_root / f"{stem}.png",
        output_root / f"{stem}.svg",
        output_root / f"{stem}.pdf",
        output_root / f"{stem}.tiff",
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.05)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        outputs[3],
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return outputs


def _g1_figure(
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
    output_root: Path,
) -> list[Path]:
    _require(len(rows) == 48, "g1_expected_48_episode_rows")
    split_specs = (
        ("test_q24", "Test-Q24", "o", "-"),
        ("unseen24", "Unseen-24", "s", (0, (4, 1.8))),
    )
    split_values: dict[str, list[float]] = {}
    for split, _, _, _ in split_specs:
        values = sorted(
            float(row["coverage_percent"])
            for row in rows
            if row["evaluation_split"] == split
        )
        _require(len(values) == 24, f"g1_{split}_expected_24_rows")
        split_values[split] = values

    g1_summary = summary["g1"]
    fig = plt.figure(figsize=(7.20, 3.25), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(2.15, 1.0), wspace=0.15)
    ax_rank = fig.add_subplot(grid[0, 0])
    ax_mean = fig.add_subplot(grid[0, 1])

    for split, label, marker, linestyle in split_specs:
        values = split_values[split]
        ax_rank.plot(
            np.arange(1, 25),
            values,
            color=COLORS[split],
            linestyle=linestyle,
            linewidth=1.15,
            marker=marker,
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=label,
            zorder=3,
        )
    ax_rank.axhline(
        COVERAGE_THRESHOLD_PERCENT,
        color=COLORS["threshold"],
        linestyle=(0, (4, 2.2)),
        linewidth=1.0,
        label="80% threshold",
        zorder=2,
    )
    ax_rank.set_xlim(0.5, 24.5)
    ax_rank.set_ylim(0.0, 102.0)
    ax_rank.set_xticks((1, 6, 12, 18, 24))
    ax_rank.set_yticks((0, 20, 40, 60, 80, 100))
    ax_rank.set_xlabel("Scene rank within split")
    ax_rank.set_ylabel("Final coverage (%)")
    ax_rank.set_title("Scene-level final coverage", loc="left", pad=7)
    ax_rank.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.85)
    ax_rank.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.055),
        borderaxespad=0.0,
        handlelength=2.4,
    )
    _panel_label(ax_rank, "a")

    x_positions = np.arange(2)
    split_keys = ("test_q24", "unseen24")
    split_labels = ("Test-Q24", "Unseen-24")
    means = np.array(
        [float(g1_summary["splits"][key]["mean"]) * 100.0 for key in split_keys]
    )
    ci_lower = np.array(
        [
            float(g1_summary["splits"][key]["bootstrap_ci"]["lower"]) * 100.0
            for key in split_keys
        ]
    )
    ci_upper = np.array(
        [
            float(g1_summary["splits"][key]["bootstrap_ci"]["upper"]) * 100.0
            for key in split_keys
        ]
    )
    yerr = np.vstack((means - ci_lower, ci_upper - means))
    for index, key in enumerate(split_keys):
        ax_mean.errorbar(
            index,
            means[index],
            yerr=yerr[:, index : index + 1],
            fmt="o" if key == "test_q24" else "s",
            color=COLORS["ink"],
            markerfacecolor=COLORS[key],
            markeredgecolor=COLORS["ink"],
            markeredgewidth=0.6,
            markersize=6.4,
            elinewidth=1.1,
            capsize=4.0,
            capthick=1.1,
            zorder=3,
        )
        passed = int(g1_summary["splits"][key]["coverage_80_count"])
        text_x = index - 0.12 if key == "test_q24" else index + 0.12
        text_ha = "right" if key == "test_q24" else "left"
        ax_mean.text(
            text_x,
            means[index],
            f"{means[index]:.2f}%\n{passed}/24 ≥80%",
            ha=text_ha,
            va="center",
            fontsize=6.7,
            color=COLORS["ink"],
        )
    ax_mean.axhline(
        COVERAGE_THRESHOLD_PERCENT,
        color=COLORS["threshold"],
        linestyle=(0, (4, 2.2)),
        linewidth=1.0,
        zorder=1,
    )
    ax_mean.text(
        1.45,
        80.65,
        "80% threshold",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=COLORS["threshold"],
    )
    ax_mean.set_xlim(-0.58, 1.58)
    ax_mean.set_ylim(78.0, 104.5)
    ax_mean.set_xticks(x_positions, split_labels)
    ax_mean.set_yticks((80, 85, 90, 95, 100))
    ax_mean.set_ylabel("Mean final coverage (%)")
    ax_mean.set_title("Split mean and 95% bootstrap CI", loc="left", pad=7)
    ax_mean.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.85)
    _panel_label(ax_mean, "b")

    return _save_figure(fig, output_root, "figure_g1_exploration_results")


def _g2_figure(
    calls: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
    output_root: Path,
) -> list[Path]:
    _require(len(calls) == 645, "g2_expected_645_call_rows")
    reachable = [row for row in calls if _as_bool(row["oracle_reachable"])]
    _require(len(reachable) == 570, "g2_expected_570_reachable_calls")
    grouped: list[list[float]] = []
    for stratum in STRATA:
        platform, scale = stratum.split("/")
        values = [
            float(row["elapsed_ms"])
            for row in reachable
            if row["platform"] == platform and row["scale"] == scale
        ]
        expected = 150 if scale == "standard" else 40
        _require(len(values) == expected, f"g2_{stratum}_unexpected_reachable_count")
        grouped.append(values)

    g2_summary = summary["g2"]
    correctness = g2_summary["correctness"]
    fig = plt.figure(figsize=(7.20, 3.25), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(2.25, 1.0), wspace=0.18)
    ax_latency = fig.add_subplot(grid[0, 0])
    ax_correctness = fig.add_subplot(grid[0, 1])

    positions = np.arange(1, 7)
    box = ax_latency.boxplot(
        grouped,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["ink"], "linewidth": 1.0},
        whiskerprops={"color": COLORS["ink"], "linewidth": 0.65},
        capprops={"color": COLORS["ink"], "linewidth": 0.65},
        boxprops={"color": COLORS["ink"], "linewidth": 0.65},
    )
    for patch, stratum in zip(box["boxes"], STRATA):
        patch.set_facecolor(COLORS[stratum.split("/")[0]])
        patch.set_alpha(0.72)

    generator = np.random.default_rng(20260728)
    for position, values, stratum in zip(positions, grouped, STRATA):
        jitter = generator.uniform(-0.18, 0.18, len(values))
        ax_latency.scatter(
            position + jitter,
            values,
            s=5.0,
            color=COLORS[stratum.split("/")[0]],
            alpha=0.28,
            edgecolors="none",
            zorder=2,
        )
        p95 = float(g2_summary["strata"][stratum]["p95_ms"])
        ax_latency.scatter(
            position,
            p95,
            s=22,
            marker="D",
            facecolor="white",
            edgecolor=COLORS["ink"],
            linewidth=0.75,
            zorder=4,
        )

    ax_latency.axhline(
        FINAL_LATENCY_MS,
        color=COLORS["threshold"],
        linestyle=(0, (4, 2.2)),
        linewidth=1.0,
        zorder=1,
    )
    ax_latency.axhline(
        MIDTERM_LATENCY_MS,
        color="#999999",
        linestyle=(0, (1.5, 2.5)),
        linewidth=0.9,
        zorder=1,
    )
    ax_latency.text(
        6.44,
        FINAL_LATENCY_MS * 1.05,
        "1 s formal target",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=COLORS["threshold"],
    )
    ax_latency.text(
        6.44,
        MIDTERM_LATENCY_MS * 1.03,
        "2 s midterm threshold",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#777777",
    )
    ax_latency.scatter(
        [],
        [],
        s=22,
        marker="D",
        facecolor="white",
        edgecolor=COLORS["ink"],
        linewidth=0.75,
        label="P95",
    )
    ax_latency.set_yscale("log")
    ax_latency.set_xlim(0.5, 6.5)
    ax_latency.set_ylim(8.0, 3000.0)
    ax_latency.set_xticks(positions, STRATUM_LABELS)
    ax_latency.set_xlabel("Platform and map scale")
    ax_latency.set_ylabel("End-to-end latency (ms, log scale)")
    ax_latency.set_title("Reachable-call latency (n=570)", loc="left", pad=7)
    ax_latency.grid(
        axis="y",
        which="major",
        color=COLORS["grid"],
        linewidth=0.55,
        alpha=0.85,
    )
    ax_latency.legend(loc="upper left", handletextpad=0.4)
    _panel_label(ax_latency, "a")

    labels = ("Reachable L2 valid", "Unreachable rejected")
    numerators = (
        int(correctness["reachable_l2_valid_count"]),
        int(correctness["unreachable_rejected_count"]),
    )
    denominators = (
        int(correctness["reachable_expected_call_count"]),
        int(correctness["unreachable_expected_call_count"]),
    )
    rates = [100.0 * numerator / denominator for numerator, denominator in zip(numerators, denominators)]
    bars = ax_correctness.barh(
        [1, 0],
        rates,
        height=0.54,
        color=(COLORS["wheel"], COLORS["hopper"]),
        edgecolor=COLORS["ink"],
        linewidth=0.65,
        hatch=("", "///"),
        zorder=2,
    )
    for bar, label, numerator, denominator in zip(
        bars,
        labels,
        numerators,
        denominators,
    ):
        center_y = bar.get_y() + bar.get_height() / 2.0
        ax_correctness.text(
            3.2,
            center_y,
            label,
            ha="left",
            va="center",
            color="white",
            fontsize=6.8,
            fontweight="semibold",
        )
        ax_correctness.text(
            97.5,
            center_y,
            f"{numerator}/{denominator}",
            ha="right",
            va="center",
            color="white",
            fontsize=7.2,
            fontweight="bold",
        )
    ax_correctness.set_xlim(0.0, 102.0)
    ax_correctness.set_ylim(-0.6, 1.6)
    ax_correctness.set_yticks([])
    ax_correctness.set_xticks((0, 50, 100))
    ax_correctness.set_xlabel("Correct decision rate (%)")
    ax_correctness.set_title("Planning correctness", loc="left", pad=7)
    ax_correctness.grid(
        axis="x",
        color=COLORS["grid"],
        linewidth=0.55,
        alpha=0.85,
        zorder=0,
    )
    ax_correctness.text(
        0.0,
        0.975,
        "129 unique requests · 645 formal calls",
        transform=ax_correctness.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="#555555",
    )
    _panel_label(ax_correctness, "b")

    return _save_figure(fig, output_root, "figure_g2_planning_results")


def _write_captions(output_root: Path, summary: Mapping[str, Any]) -> Path:
    g1 = summary["g1"]
    g2 = summary["g2"]
    test = g1["splits"]["test_q24"]
    unseen = g1["splits"]["unseen24"]
    worst_key = max(
        STRATA,
        key=lambda key: float(g2["strata"][key]["p95_ms"]),
    )
    worst_p95 = float(g2["strata"][worst_key]["p95_ms"])
    text = f"""# G1 与 G2 独立结果图图注

## G1 未知环境自主探索结果

**a，** Test-Q24 与 Unseen-24 各 24 个场景的最终覆盖率排序，虚线表示 80% 门槛。**b，** 两个场景集的平均最终覆盖率及场景级 bootstrap 95% 置信区间。Test-Q24 平均覆盖率为 {float(test['mean']) * 100:.3f}%，23/24 个场景达到 80%；Unseen-24 平均覆盖率为 {float(unseen['mean']) * 100:.3f}%，24/24 个场景达到 80%。总体 47/48 个场景达到 80%，安全违规计数为 0。Unseen-24 由 21 条原始结果与 3 条增量修复结果显式汇总。

## G2 多平台路径规划结果

**a，** 六个平台—地图尺度分层中 570 次可达调用的端到端规划时延分布；箱线图表示四分位范围和中位数，散点表示单次调用，空心菱形表示 P95。虚线和点线分别表示 1 s 正式目标与 2 s 中期门槛。最慢分层为 {worst_key}，P95 为 {worst_p95:.3f} ms。**b，** 可达请求的 L2 有效路径为 570/570，不可达请求的正确拒绝为 75/75。正式实验包括 129 个唯一请求，每个请求重复 5 次，共 645 次调用；重复调用用于稳定计时，不解释为新增独立场景。
"""
    path = output_root / "captions_g1_g2_separate.md"
    artifact_io.write_text(path, text)
    return path


def _copy_report_previews(
    g1_outputs: Sequence[Path],
    g2_outputs: Sequence[Path],
    report_figure_root: Path,
) -> list[Path]:
    artifact_io.make_dirs(report_figure_root)
    copies = [
        report_figure_root / "图6-3_G1未知环境自主探索结果.png",
        report_figure_root / "图6-4_G2多平台路径规划结果.png",
    ]
    artifact_io.write_bytes(copies[0], artifact_io.read_bytes(g1_outputs[0]))
    artifact_io.write_bytes(copies[1], artifact_io.read_bytes(g2_outputs[0]))
    return copies


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从最终汇总包绘制相互独立的 G1 与 G2 学术结果图。"
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report-figure-root", type=Path)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = (args.output_root or input_root).resolve()
    g1_rows = _read_csv(input_root / "g1-episodes.csv")
    g2_calls = _read_csv(input_root / "g2-calls.csv")
    summary = _read_json(input_root / "summary.json")

    _configure_style()
    g1_outputs = _g1_figure(g1_rows, summary, output_root)
    plt.close("all")
    g2_outputs = _g2_figure(g2_calls, summary, output_root)
    plt.close("all")
    caption_path = _write_captions(output_root, summary)

    report_outputs: list[Path] = []
    if args.report_figure_root is not None:
        report_outputs = _copy_report_previews(
            g1_outputs,
            g2_outputs,
            args.report_figure_root.resolve(),
        )

    payload = {
        "g1_outputs": [str(path) for path in g1_outputs],
        "g2_outputs": [str(path) for path in g2_outputs],
        "caption": str(caption_path),
        "report_previews": [str(path) for path in report_outputs],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
