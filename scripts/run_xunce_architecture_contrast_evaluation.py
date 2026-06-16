from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from math import isfinite
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
for path in (SCRIPT_DIR, MODEL_EXPLORER_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import global_99_boundary_defaults
    from xunce_topology_graph_proto_common import TopologyAwareCoverageGraphPrototype, parameter_count
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_topology_graph_proto_common import TopologyAwareCoverageGraphPrototype, parameter_count

from model_explorer.policy.architectures import build_policy_network_from_metadata


CONFIG_SCHEMA_VERSION = "xunce-architecture-contrast-evaluation-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-architecture-contrast-evaluation-summary/v1"
MANIFEST_SCHEMA_VERSION = "xunce-architecture-contrast-evaluation-manifest/v1"
RESULT_ROW_SCHEMA_VERSION = "xunce-architecture-contrast-result-row/v1"
PARAM_LATENCY_AUDIT_SCHEMA_VERSION = "xunce-architecture-contrast-parameter-latency-audit/v1"
RANKING_AUDIT_SCHEMA_VERSION = "xunce-architecture-contrast-ranking-audit/v1"
BOUNDARY_AUDIT_SCHEMA_VERSION = "xunce-architecture-contrast-boundary-audit/v1"
REJECTION_REPORT_SCHEMA_VERSION = "xunce-architecture-contrast-rejection-report/v1"

DEFAULT_CONFIG = "configs/xunce_architecture_contrast_evaluation_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_architecture_contrast_evaluation_v1"

SUMMARY_FILE = "xunce-architecture-contrast-evaluation-summary.json"
MANIFEST_FILE = "xunce-architecture-contrast-evaluation-manifest.json"
RESULTS_FILE = "xunce-architecture-contrast-results.jsonl"
PARAM_LATENCY_AUDIT_FILE = "xunce-architecture-contrast-parameter-latency-audit.json"
RANKING_AUDIT_FILE = "xunce-architecture-contrast-ranking-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-architecture-contrast-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-architecture-contrast-rejection-report.json"
REPORT_FILE = "xunce-architecture-contrast-evaluation-report.md"

XUNCE_PROTO_ARCHITECTURE = "topology_aware_coverage_graph_proto_v1"
REFERENCE_ARCHITECTURE = "candidate_attention_v1"
PASS_NEXT_REQUIRED_CHANGE = "full_xunce_network_v1_design"
FAIL_NEXT_REQUIRED_CHANGE = "fix_architecture_contrast_evaluation"
FIX_STAGE6_NEXT_REQUIRED_CHANGE = "fix_xunce_proto_mechanism_validation"
FIX_STAGE4_NEXT_REQUIRED_CHANGE = "fix_xunce_topology_feature_extraction_audit"

BOUNDARY_FIELDS = tuple(global_99_boundary_defaults()) + (
    "real_world_release_approved",
    "real_world_performance_claimed",
    "default_policy_replacement_approved",
    "real_executor_connection_approved",
    "starts_online_canary",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce Architecture Contrast Evaluation v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    try:
        summary = run_xunce_architecture_contrast_evaluation(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "winner_architecture": summary["winner_architecture"],
                "xunce_proto_rank": summary["xunce_proto_rank"],
                "next_required_change": summary["next_required_change"],
                "summary": summary["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


def run_xunce_architecture_contrast_evaluation(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config_path = Path(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    paths = _artifact_paths(output_root)
    stage6_root = resolve_path(Path(config["source_proto_mechanism_validation_root"]), repo_root)
    stage4_root = resolve_path(Path(config["source_topology_feature_extraction_root"]), repo_root)
    stage6_summary_path = stage6_root / "xunce-proto-mechanism-validation-summary.json"
    stage6_summary = _load_json(stage6_summary_path)
    boundary_audit = _boundary_audit(stage6_summary)

    source_reason_codes = _source_reason_codes(stage6_summary)
    results: list[dict[str, Any]] = []
    ranking_audit = _empty_ranking_audit()
    param_latency_audit = _empty_param_latency_audit()

    if not source_reason_codes:
        source_paths = _source_paths(stage4_root)
        candidates = _load_jsonl(source_paths["candidates"])
        edges = _load_jsonl(source_paths["edges"])
        memory = _load_json(source_paths["memory"])
        if not candidates:
            source_reason_codes.append("missing_topology_feature_candidates")
        if not edges:
            source_reason_codes.append("missing_topology_feature_edges")
        if not isinstance(memory, dict):
            source_reason_codes.append("missing_topology_feature_memory")
        if not source_reason_codes:
            results = _evaluate_architectures(config, candidates, edges, memory)
            ranking_audit = _ranking_audit(config, results)
            param_latency_audit = _param_latency_audit(config, results)

    decision = _decision(source_reason_codes, ranking_audit, param_latency_audit, boundary_audit)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        config=config,
        stage6_summary_path=stage6_summary_path,
        stage6_summary=stage6_summary,
        results=results,
        ranking_audit=ranking_audit,
        param_latency_audit=param_latency_audit,
        boundary_audit=boundary_audit,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = _manifest(generated_at, config_path, output_root, paths, summary)
    rejection_report = _rejection_report(decision, ranking_audit, param_latency_audit, boundary_audit)

    write_jsonl(paths["results"], results)
    write_json(paths["param_latency_audit"], param_latency_audit)
    write_json(paths["ranking_audit"], ranking_audit)
    write_json(paths["boundary_audit"], boundary_audit)
    write_json(paths["rejection_report"], rejection_report)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection_report), encoding="utf-8")
    return summary


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    normalized = dict(payload)
    for key in ("source_proto_mechanism_validation_root", "source_topology_feature_extraction_root"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
    architectures = payload.get("architectures")
    if not isinstance(architectures, list) or not architectures or any(not isinstance(item, str) for item in architectures):
        raise ConfigError("architectures must be a non-empty string list")
    if XUNCE_PROTO_ARCHITECTURE not in architectures:
        raise ConfigError(f"architectures must include {XUNCE_PROTO_ARCHITECTURE}")
    if REFERENCE_ARCHITECTURE not in architectures:
        raise ConfigError(f"architectures must include {REFERENCE_ARCHITECTURE}")
    normalized["architectures"] = list(dict.fromkeys(architectures))
    seeds = payload.get("architecture_seeds")
    if not isinstance(seeds, dict):
        raise ConfigError("architecture_seeds must be an object")
    normalized["architecture_seeds"] = {architecture: _positive_int(seeds.get(architecture, 1), f"architecture_seeds.{architecture}") for architecture in normalized["architectures"]}
    for key in ("hidden_dim", "attention_heads", "message_passing_layers", "latency_repeats", "max_xunce_rank"):
        normalized[key] = _positive_int(payload.get(key), key)
    for key in (
        "dropout",
        "max_parameter_ratio_vs_candidate_attention",
        "max_latency_ratio_vs_candidate_attention",
        "max_forward_latency_ms",
    ):
        normalized[key] = _nonnegative_float(payload.get(key), key)
    if normalized["dropout"] >= 1.0:
        raise ConfigError("dropout must be < 1.0")
    mask = payload.get("action_mask_override")
    if mask is not None:
        if not isinstance(mask, list) or not mask or any(not isinstance(value, bool) for value in mask):
            raise ConfigError("action_mask_override must be a non-empty boolean list")
        if not any(mask):
            raise ConfigError("action_mask_override must contain at least one true value")
        normalized["action_mask_override"] = mask
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "results": output_root / RESULTS_FILE,
        "param_latency_audit": output_root / PARAM_LATENCY_AUDIT_FILE,
        "ranking_audit": output_root / RANKING_AUDIT_FILE,
        "boundary_audit": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _source_paths(stage4_root: Path) -> dict[str, Path]:
    return {
        "candidates": stage4_root / "xunce-topology-feature-extraction-candidates.jsonl",
        "edges": stage4_root / "xunce-topology-feature-extraction-edges.jsonl",
        "memory": stage4_root / "xunce-topology-feature-extraction-memory.json",
    }


def _evaluate_architectures(
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    memory: dict[str, Any],
) -> list[dict[str, Any]]:
    tensors = _fixture_tensors(candidates, edges, memory, config.get("action_mask_override"))
    heuristic_scores = [_heuristic_quality(candidate, memory) for candidate in candidates]
    heuristic_best_candidate = max(range(len(candidates)), key=lambda index: (heuristic_scores[index], -index))
    rows: list[dict[str, Any]] = []
    for architecture in config["architectures"]:
        if architecture == XUNCE_PROTO_ARCHITECTURE:
            row = _evaluate_xunce_proto(config, architecture, tensors, heuristic_scores, heuristic_best_candidate)
        else:
            row = _evaluate_existing_architecture(config, architecture, tensors, heuristic_scores, heuristic_best_candidate)
        rows.append(row)
    return rows


def _evaluate_existing_architecture(
    config: dict[str, Any],
    architecture: str,
    tensors: dict[str, Any],
    heuristic_scores: list[float],
    heuristic_best_candidate: int,
) -> dict[str, Any]:
    seed = config["architecture_seeds"][architecture]
    torch.manual_seed(seed)
    architecture_config: dict[str, Any] = {"hidden_dim": config["hidden_dim"], "dropout": config["dropout"]}
    if architecture == REFERENCE_ARCHITECTURE:
        architecture_config["attention_heads"] = config["attention_heads"]
    network = build_policy_network_from_metadata(
        architecture,
        candidate_feature_count=len(tensors["candidate_feature_names"]),
        global_feature_count=len(tensors["memory_feature_names"]),
        missing_indicator_count=len(tensors["candidate_missing_indicator_names"]),
        hidden_size=config["hidden_dim"],
        architecture_config=architecture_config,
    )
    network.eval()

    def forward_once():
        return network(
            candidate_features=tensors["candidate_features"],
            global_features=tensors["memory_features"],
            action_mask=tensors["action_mask"],
            candidate_missing_indicators=tensors["candidate_missing_indicators"],
        )

    output, median_latency = _timed_forward(forward_once, config["latency_repeats"])
    return _result_row(
        architecture=architecture,
        seed=seed,
        logits=output.masked_logits[0].detach().cpu().tolist(),
        value=float(output.value[0].detach().cpu()),
        parameter_count_value=parameter_count(network),
        median_forward_latency_ms=median_latency,
        heuristic_scores=heuristic_scores,
        heuristic_best_candidate=heuristic_best_candidate,
    )


def _evaluate_xunce_proto(
    config: dict[str, Any],
    architecture: str,
    tensors: dict[str, Any],
    heuristic_scores: list[float],
    heuristic_best_candidate: int,
) -> dict[str, Any]:
    seed = config["architecture_seeds"][architecture]
    torch.manual_seed(seed)
    network = TopologyAwareCoverageGraphPrototype(
        candidate_feature_count=len(tensors["candidate_feature_names"]),
        edge_feature_count=len(tensors["edge_feature_names"]),
        memory_feature_count=len(tensors["memory_feature_names"]),
        hidden_dim=config["hidden_dim"],
        message_passing_layers=config["message_passing_layers"],
        dropout=config["dropout"],
    )
    network.eval()

    def forward_once():
        return network(
            candidate_features=tensors["candidate_features"],
            edge_features=tensors["edge_features"],
            edge_index=tensors["edge_index"],
            memory_features=tensors["memory_features"],
            action_mask=tensors["action_mask"],
        )

    output, median_latency = _timed_forward(forward_once, config["latency_repeats"])
    return _result_row(
        architecture=architecture,
        seed=seed,
        logits=output.masked_logits[0].detach().cpu().tolist(),
        value=float(output.value[0].detach().cpu()),
        parameter_count_value=parameter_count(network),
        median_forward_latency_ms=median_latency,
        heuristic_scores=heuristic_scores,
        heuristic_best_candidate=heuristic_best_candidate,
    )


def _timed_forward(forward_once, repeats: int):
    timings: list[float] = []
    output = None
    with torch.no_grad():
        for _ in range(repeats):
            started = time.perf_counter()
            output = forward_once()
            timings.append((time.perf_counter() - started) * 1000.0)
    if output is None:
        raise ValueError("latency_repeats must be positive")
    return output, float(statistics.median(timings))


def _result_row(
    *,
    architecture: str,
    seed: int,
    logits: list[float],
    value: float,
    parameter_count_value: int,
    median_forward_latency_ms: float,
    heuristic_scores: list[float],
    heuristic_best_candidate: int,
) -> dict[str, Any]:
    selected_candidate = max(range(len(logits)), key=lambda index: (logits[index], -index))
    selected_quality = heuristic_scores[selected_candidate]
    best_quality = heuristic_scores[heuristic_best_candidate]
    non_finite_count = sum(0 if isfinite(float(value)) else 1 for value in logits + [value, median_forward_latency_ms])
    return {
        "schema_version": RESULT_ROW_SCHEMA_VERSION,
        "architecture": architecture,
        "seed": seed,
        "selected_candidate_index": selected_candidate,
        "heuristic_best_candidate_index": heuristic_best_candidate,
        "selected_heuristic_quality": selected_quality,
        "heuristic_best_quality": best_quality,
        "heuristic_regret": max(0.0, best_quality - selected_quality),
        "agrees_with_heuristic_best": selected_candidate == heuristic_best_candidate,
        "parameter_count": int(parameter_count_value),
        "median_forward_latency_ms": median_forward_latency_ms,
        "non_finite_output_count": non_finite_count,
        "value": value,
        "masked_logits": [float(value) for value in logits],
    }


def _fixture_tensors(
    candidates: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    memory: dict[str, Any],
    action_mask_override: list[bool] | None,
) -> dict[str, Any]:
    candidate_feature_names = sorted(candidates[0].get("features", {}))
    candidate_missing_indicator_names = sorted(candidates[0].get("missing_indicators", {}))
    edge_feature_names = sorted(edges[0].get("features", {}))
    memory_feature_names = sorted(memory.get("features", {}))
    mask = action_mask_override or [True] * len(candidates)
    if len(mask) != len(candidates):
        raise ConfigError("action_mask_override length must match candidate count")
    return {
        "candidate_feature_names": candidate_feature_names,
        "candidate_missing_indicator_names": candidate_missing_indicator_names,
        "edge_feature_names": edge_feature_names,
        "memory_feature_names": memory_feature_names,
        "candidate_features": torch.tensor(
            [[[_feature_value(candidate.get("features", {}), name) for name in candidate_feature_names] for candidate in candidates]],
            dtype=torch.float32,
        ),
        "candidate_missing_indicators": torch.tensor(
            [[[1.0 if candidate.get("missing_indicators", {}).get(name, False) else 0.0 for name in candidate_missing_indicator_names] for candidate in candidates]],
            dtype=torch.float32,
        ),
        "edge_features": torch.tensor(
            [[_feature_value(edge.get("features", {}), name) for name in edge_feature_names] for edge in edges],
            dtype=torch.float32,
        ),
        "edge_index": torch.tensor(
            [[int(edge["candidate_left_index"]), int(edge["candidate_right_index"])] for edge in edges],
            dtype=torch.long,
        ),
        "memory_features": torch.tensor(
            [[_feature_value(memory.get("features", {}), name) for name in memory_feature_names]],
            dtype=torch.float32,
        ),
        "action_mask": torch.tensor([mask], dtype=torch.bool),
    }


def _heuristic_quality(candidate: dict[str, Any], memory: dict[str, Any]) -> float:
    features = candidate.get("features", {})
    memory_features = memory.get("features", {})
    return (
        _feature_value(features, "new_coverage_cell_count")
        - _feature_value(features, "bfs_distance_from_current")
        - _feature_value(features, "revisit_path_cell_count")
        - 10.0 * _feature_value(features, "fallback_risk")
        - 0.1 * _feature_value(features, "coverage_overlap_count")
        - _feature_value(features, "budget_fraction_cost")
        + _feature_value(memory_features, "remaining_budget_fraction")
    )


def _feature_value(features: dict[str, Any], name: str) -> float:
    value = features.get(name, 0.0)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _ranking_audit(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        results,
        key=lambda row: (
            -float(row["selected_heuristic_quality"]),
            float(row["heuristic_regret"]),
            int(row["parameter_count"]),
            float(row["median_forward_latency_ms"]),
            str(row["architecture"]),
        ),
    )
    ranks = {row["architecture"]: rank + 1 for rank, row in enumerate(ordered)}
    xunce_rank = ranks.get(XUNCE_PROTO_ARCHITECTURE)
    return {
        "schema_version": RANKING_AUDIT_SCHEMA_VERSION,
        "ranking_order": [row["architecture"] for row in ordered],
        "architecture_ranks": ranks,
        "winner_architecture": ordered[0]["architecture"] if ordered else None,
        "xunce_proto_rank": xunce_rank,
        "max_allowed_xunce_rank": config["max_xunce_rank"],
        "xunce_selected_heuristic_quality": _row_for(results, XUNCE_PROTO_ARCHITECTURE).get("selected_heuristic_quality"),
        "best_selected_heuristic_quality": ordered[0]["selected_heuristic_quality"] if ordered else None,
        "ranking_quality_gate_passed": bool(xunce_rank is not None and xunce_rank <= config["max_xunce_rank"]),
    }


def _param_latency_audit(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    xunce = _row_for(results, XUNCE_PROTO_ARCHITECTURE)
    reference = _row_for(results, REFERENCE_ARCHITECTURE)
    xunce_param = int(xunce.get("parameter_count", 0) or 0)
    reference_param = int(reference.get("parameter_count", 0) or 0)
    xunce_latency = float(xunce.get("median_forward_latency_ms", float("inf")))
    reference_latency = float(reference.get("median_forward_latency_ms", float("inf")))
    parameter_ratio = xunce_param / reference_param if reference_param > 0 else float("inf")
    latency_ratio = xunce_latency / reference_latency if reference_latency > 0 else float("inf")
    parameter_passed = isfinite(parameter_ratio) and parameter_ratio <= config["max_parameter_ratio_vs_candidate_attention"]
    latency_passed = (
        isfinite(latency_ratio)
        and latency_ratio <= config["max_latency_ratio_vs_candidate_attention"]
        and xunce_latency <= config["max_forward_latency_ms"]
    )
    return {
        "schema_version": PARAM_LATENCY_AUDIT_SCHEMA_VERSION,
        "reference_architecture": REFERENCE_ARCHITECTURE,
        "xunce_proto_parameter_count": xunce_param,
        "candidate_attention_parameter_count": reference_param,
        "parameter_ratio_vs_candidate_attention": parameter_ratio,
        "max_parameter_ratio_vs_candidate_attention": config["max_parameter_ratio_vs_candidate_attention"],
        "xunce_proto_latency_ms": xunce_latency,
        "candidate_attention_latency_ms": reference_latency,
        "latency_ratio_vs_candidate_attention": latency_ratio,
        "max_latency_ratio_vs_candidate_attention": config["max_latency_ratio_vs_candidate_attention"],
        "max_forward_latency_ms": config["max_forward_latency_ms"],
        "parameter_efficiency_gate_passed": parameter_passed,
        "latency_gate_passed": latency_passed,
    }


def _row_for(results: list[dict[str, Any]], architecture: str) -> dict[str, Any]:
    for row in results:
        if row.get("architecture") == architecture:
            return row
    return {}


def _source_reason_codes(stage6_summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(stage6_summary, dict):
        return ["missing_proto_mechanism_validation_summary"]
    reason_codes: list[str] = []
    if stage6_summary.get("status") != "passed":
        reason_codes.append("proto_mechanism_validation_not_passed")
    if stage6_summary.get("next_required_change") != "architecture_contrast_evaluation":
        reason_codes.append("proto_mechanism_validation_wrong_next_required_change")
    return reason_codes


def _decision(
    source_reason_codes: list[str],
    ranking_audit: dict[str, Any],
    param_latency_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    reason_codes = list(source_reason_codes)
    source_blocked = bool(source_reason_codes)
    if source_blocked and any(code.startswith("missing_proto") or code.startswith("proto_mechanism") for code in source_reason_codes):
        next_required_change = FIX_STAGE6_NEXT_REQUIRED_CHANGE
    elif source_blocked and any(code.startswith("missing_topology_feature") for code in source_reason_codes):
        next_required_change = FIX_STAGE4_NEXT_REQUIRED_CHANGE
    else:
        next_required_change = FAIL_NEXT_REQUIRED_CHANGE
    if not boundary_audit.get("boundary_audit_passed", False):
        reason_codes.append("architecture_contrast_boundary_violation")
    if not source_blocked:
        if not param_latency_audit.get("parameter_efficiency_gate_passed", False):
            reason_codes.append("xunce_parameter_efficiency_gate_failed")
        if not param_latency_audit.get("latency_gate_passed", False):
            reason_codes.append("xunce_latency_gate_failed")
        if not ranking_audit.get("ranking_quality_gate_passed", False):
            reason_codes.append("xunce_ranking_quality_gate_failed")
    reason_codes = unique_sorted(reason_codes)
    passed = not reason_codes
    return {
        "status": "passed" if passed else "failed",
        "reason_codes": reason_codes,
        "next_required_change": PASS_NEXT_REQUIRED_CHANGE if passed else next_required_change,
    }


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    config: dict[str, Any],
    stage6_summary_path: Path,
    stage6_summary: dict[str, Any] | None,
    results: list[dict[str, Any]],
    ranking_audit: dict[str, Any],
    param_latency_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    boundary_defaults = _closed_boundary_fields()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "summary": str(paths["summary"]),
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_mechanism_validation_summary": str(stage6_summary_path),
        "source_mechanism_validation_status": stage6_summary.get("status") if isinstance(stage6_summary, dict) else None,
        "architecture_count": len(results),
        "evaluated_architectures": [row.get("architecture") for row in results],
        "reference_architecture": REFERENCE_ARCHITECTURE,
        "winner_architecture": ranking_audit.get("winner_architecture"),
        "xunce_proto_rank": ranking_audit.get("xunce_proto_rank"),
        "xunce_proto_parameter_count": param_latency_audit.get("xunce_proto_parameter_count", 0),
        "candidate_attention_parameter_count": param_latency_audit.get("candidate_attention_parameter_count", 0),
        "xunce_proto_latency_ms": param_latency_audit.get("xunce_proto_latency_ms", 0.0),
        "candidate_attention_latency_ms": param_latency_audit.get("candidate_attention_latency_ms", 0.0),
        "parameter_efficiency_gate_passed": param_latency_audit.get("parameter_efficiency_gate_passed", False),
        "latency_gate_passed": param_latency_audit.get("latency_gate_passed", False),
        "ranking_quality_gate_passed": ranking_audit.get("ranking_quality_gate_passed", False),
        "architecture_contrast_passed": decision["status"] == "passed",
        "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False),
        "next_required_change": decision["next_required_change"],
        "git_provenance": {"current": git_snapshot(repo_root)},
    }
    summary.update(boundary_defaults)
    return summary


def _manifest(generated_at: str, config_path: Path, output_root: Path, paths: dict[str, Path], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }


def _boundary_audit(stage6_summary: dict[str, Any] | None) -> dict[str, Any]:
    observed = {}
    violations = []
    if isinstance(stage6_summary, dict):
        for field in BOUNDARY_FIELDS:
            value = bool(stage6_summary.get(field, False))
            observed[field] = value
            if value:
                violations.append(field)
    return {
        "schema_version": BOUNDARY_AUDIT_SCHEMA_VERSION,
        "observed_source_boundary_fields": observed,
        "violating_source_boundary_fields": sorted(set(violations)),
        "boundary_audit_passed": not violations,
        **_closed_boundary_fields(),
    }


def _closed_boundary_fields() -> dict[str, bool | float]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update(
        {
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
    )
    return fields


def _rejection_report(
    decision: dict[str, Any],
    ranking_audit: dict[str, Any],
    param_latency_audit: dict[str, Any],
    boundary_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_REPORT_SCHEMA_VERSION,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "ranking_quality_gate_passed": ranking_audit.get("ranking_quality_gate_passed", False),
        "parameter_efficiency_gate_passed": param_latency_audit.get("parameter_efficiency_gate_passed", False),
        "latency_gate_passed": param_latency_audit.get("latency_gate_passed", False),
        "boundary_audit_passed": boundary_audit.get("boundary_audit_passed", False),
    }


def _render_report(summary: dict[str, Any], rejection_report: dict[str, Any]) -> str:
    lines = [
        "# Xunce Architecture Contrast Evaluation v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- winner_architecture: `{summary.get('winner_architecture')}`",
        f"- xunce_proto_rank: `{summary.get('xunce_proto_rank')}`",
        f"- architecture_contrast_passed: `{summary['architecture_contrast_passed']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "## Boundary",
        "",
        "- This stage is an offline deterministic architecture contrast audit.",
        "- It does not train PPO, publish checkpoints, replace default policy, connect an executor, or claim real-world performance.",
        "",
        "## Rejection Report",
        "",
        f"- parameter_efficiency_gate_passed: `{rejection_report['parameter_efficiency_gate_passed']}`",
        f"- latency_gate_passed: `{rejection_report['latency_gate_passed']}`",
        f"- ranking_quality_gate_passed: `{rejection_report['ranking_quality_gate_passed']}`",
        f"- boundary_audit_passed: `{rejection_report['boundary_audit_passed']}`",
        "",
    ]
    return "\n".join(lines)


def _empty_ranking_audit() -> dict[str, Any]:
    return {
        "schema_version": RANKING_AUDIT_SCHEMA_VERSION,
        "ranking_order": [],
        "architecture_ranks": {},
        "winner_architecture": None,
        "xunce_proto_rank": None,
        "ranking_quality_gate_passed": False,
    }


def _empty_param_latency_audit() -> dict[str, Any]:
    return {
        "schema_version": PARAM_LATENCY_AUDIT_SCHEMA_VERSION,
        "reference_architecture": REFERENCE_ARCHITECTURE,
        "xunce_proto_parameter_count": 0,
        "candidate_attention_parameter_count": 0,
        "parameter_efficiency_gate_passed": False,
        "latency_gate_passed": False,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        rows.append(payload)
    return rows


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a positive integer") from exc
    if numeric <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return numeric


def _nonnegative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be non-negative")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be non-negative") from exc
    if numeric < 0.0:
        raise ConfigError(f"{field_name} must be non-negative")
    return numeric


if __name__ == "__main__":
    raise SystemExit(main())
