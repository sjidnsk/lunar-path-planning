from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if MODEL_EXPLORER_SRC.is_dir() and str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

try:
    from git_provenance import git_snapshot
    from global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from global_99_governance_common import global_99_boundary_defaults
    from xunce_full_network_common import XunceFullNetworkV1, parameter_count
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.global_99_coverage_contract import ConfigError, resolve_path, unique_sorted, utc_now, write_json, write_jsonl
    from scripts.global_99_governance_common import global_99_boundary_defaults
    from scripts.xunce_full_network_common import XunceFullNetworkV1, parameter_count

import torch
from model_explorer.policy.features import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, MISSING_INDICATOR_NAMES, PolicyObservation
from model_explorer.policy.training import load_policy_checkpoint


CONFIG_SCHEMA_VERSION = "xunce-high-fidelity-real-map-comparison-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-high-fidelity-real-map-comparison-summary/v1"
DEFAULT_CONFIG = "configs/xunce_high_fidelity_real_map_comparison_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_high_fidelity_real_map_comparison_v1"

SUMMARY_FILE = "xunce-high-fidelity-real-map-comparison-summary.json"
MANIFEST_FILE = "xunce-high-fidelity-real-map-comparison-manifest.json"
SCENARIO_RESULTS_FILE = "xunce-high-fidelity-real-map-scenario-results.jsonl"
POLICY_DECISIONS_FILE = "xunce-high-fidelity-policy-decisions.jsonl"
ROI_FAMILY_SUMMARY_FILE = "xunce-high-fidelity-roi-family-summary.json"
XUNCE_VS_INCUMBENT_AUDIT_FILE = "xunce-vs-incumbent-audit.json"
EFFICIENCY_AUDIT_FILE = "xunce-efficiency-audit.json"
MODEL_INFERENCE_AUDIT_FILE = "xunce-high-fidelity-model-inference-audit.json"
MODEL_INFERENCE_RESULTS_FILE = "xunce-high-fidelity-model-inference-results.jsonl"
SOURCE_MATCH_AUDIT_FILE = "xunce-source-match-audit.json"
BOUNDARY_AUDIT_FILE = "xunce-high-fidelity-boundary-audit.json"
REJECTION_REPORT_FILE = "xunce-high-fidelity-rejection-report.json"
REPORT_FILE = "xunce-high-fidelity-real-map-comparison-report.md"

EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
EXPANSION_PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"

PASS_ADVANTAGE_NEXT_REQUIRED_CHANGE = "xunce_default_policy_candidate_authorization_preflight"
PASS_NO_ADVANTAGE_NEXT_REQUIRED_CHANGE = "xunce_research_iteration_required"
FIX_EXPANSION_NEXT_REQUIRED_CHANGE = "fix_xunce_high_fidelity_real_map_roi_expansion"
FIX_XUNCE_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_xunce_sandbox_candidate_preflight"
FIX_INCUMBENT_CHECKPOINT_NEXT_REQUIRED_CHANGE = "fix_incumbent_policy_checkpoint"
FIX_COMPARISON_NEXT_REQUIRED_CHANGE = "fix_xunce_high_fidelity_real_map_comparison"
FIX_GUARD_FALLBACK_NEXT_REQUIRED_CHANGE = "fix_xunce_high_fidelity_guard_fallback"
BOUNDARY_NEXT_REQUIRED_CHANGE = "resolve_xunce_high_fidelity_real_map_boundary_rejections"

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

FORBIDDEN_FIELDS = tuple(dict.fromkeys(BOUNDARY_FIELDS + (
    "checkpoint_publication_approved",
    "final_release_approved",
    "performance_claimed",
    "runs_new_training_update",
    "runs_new_ppo_update",
    "modifies_network",
    "modifies_action_space",
    "modifies_default_astar",
)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Xunce High-Fidelity Real-Map Policy Comparison v1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--source-roi-expansion-root")
    parser.add_argument("--xunce-candidate-checkpoint")
    parser.add_argument("--incumbent-policy-checkpoint")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    overrides = {
        key: value
        for key, value in {
            "source_roi_expansion_root": args.source_roi_expansion_root,
            "xunce_candidate_checkpoint": args.xunce_candidate_checkpoint,
            "incumbent_policy_checkpoint": args.incumbent_policy_checkpoint,
        }.items()
        if value is not None
    }
    try:
        summary = run_xunce_high_fidelity_real_map_comparison(
            config_path=resolve_path(Path(args.config), repo_root),
            output_root=resolve_path(Path(args.output_root), repo_root),
            repo_root=repo_root,
            config_overrides=overrides,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"], "xunce_candidate_advantage_established": summary["xunce_candidate_advantage_established"], "next_required_change": summary["next_required_change"], "summary": summary["summary"]}, ensure_ascii=False, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


def run_xunce_high_fidelity_real_map_comparison(*, config_path: Path, output_root: Path, repo_root: Path, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root)
    config = _load_config(config_path, repo_root, config_overrides=config_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(output_root)
    source = _load_source(config, repo_root)
    boundary = _boundary_audit(config, source)
    model_bundle = _load_model_bundle(config, source, repo_root)
    scenario_results, policy_decisions, roi_family, inference_results = _compare_scenarios(config, source, model_bundle)
    model_inference = _model_inference_audit(model_bundle, inference_results)
    comparison = _comparison_audit(config, scenario_results, policy_decisions)
    efficiency = _efficiency_audit(config, policy_decisions, model_bundle)
    source_match = _source_match_audit(config, source, scenario_results)
    decision = _decision(source, boundary, comparison, efficiency, source_match, model_inference)
    generated_at = utc_now()
    summary = _summary(
        generated_at=generated_at,
        config_path=config_path,
        output_root=output_root,
        paths=paths,
        config=config,
        source=source,
        boundary=boundary,
        comparison=comparison,
        efficiency=efficiency,
        model_inference=model_inference,
        source_match=source_match,
        roi_family=roi_family,
        decision=decision,
        repo_root=repo_root,
    )
    manifest = {
        "schema_version": "xunce-high-fidelity-real-map-comparison-manifest/v1",
        "generated_at": generated_at,
        "config": str(config_path),
        "output_root": str(output_root),
        "artifacts": {key: str(value) for key, value in paths.items()},
        "source_roi_expansion_root": config["source_roi_expansion_root"],
        "summary_status": summary["status"],
        "next_required_change": summary["next_required_change"],
    }
    rejection = {
        "schema_version": "xunce-high-fidelity-comparison-rejection-report/v1",
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "next_required_change": decision["next_required_change"],
        "xunce_candidate_advantage_established": decision["xunce_candidate_advantage_established"],
    }
    write_jsonl(paths["scenario_results"], scenario_results)
    write_jsonl(paths["policy_decisions"], policy_decisions)
    write_jsonl(paths["model_inference_results"], inference_results)
    write_json(paths["roi_family"], roi_family)
    write_json(paths["xunce_vs_incumbent"], comparison)
    write_json(paths["efficiency"], efficiency)
    write_json(paths["model_inference"], model_inference)
    write_json(paths["source_match"], source_match)
    write_json(paths["boundary"], boundary)
    write_json(paths["rejection_report"], rejection)
    write_json(paths["manifest"], manifest)
    write_json(paths["summary"], summary)
    paths["report"].write_text(_render_report(summary, rejection), encoding="utf-8")
    return summary


def _load_config(path: Path, repo_root: Path, *, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("config root must be an object")
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    if config_overrides:
        payload = {**payload, **config_overrides}
    normalized = dict(payload)
    for key in ("source_roi_expansion_root", "xunce_candidate_checkpoint", "incumbent_policy_checkpoint"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ConfigError(f"{key} must be a non-empty string")
        normalized[key] = str(resolve_path(Path(payload[key]), repo_root))
    normalized["required_scenario_count"] = _positive_int(payload.get("required_scenario_count", 24), "required_scenario_count")
    normalized["max_xunce_guard_fallback_rate"] = _nonnegative_float(payload.get("max_xunce_guard_fallback_rate", 0.05), "max_xunce_guard_fallback_rate")
    normalized["max_xunce_parameter_count"] = _positive_int(payload.get("max_xunce_parameter_count", 10_000_000), "max_xunce_parameter_count")
    normalized["max_latency_ratio_vs_candidate_attention"] = _nonnegative_float(payload.get("max_latency_ratio_vs_candidate_attention", 1.5), "max_latency_ratio_vs_candidate_attention")
    normalized["max_median_inference_latency_ms"] = _nonnegative_float(payload.get("max_median_inference_latency_ms", 5.0), "max_median_inference_latency_ms")
    normalized["canary_traffic_fraction"] = _nonnegative_float(payload.get("canary_traffic_fraction", 0.0), "canary_traffic_fraction")
    return normalized


def _artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "summary": output_root / SUMMARY_FILE,
        "manifest": output_root / MANIFEST_FILE,
        "scenario_results": output_root / SCENARIO_RESULTS_FILE,
        "policy_decisions": output_root / POLICY_DECISIONS_FILE,
        "roi_family": output_root / ROI_FAMILY_SUMMARY_FILE,
        "xunce_vs_incumbent": output_root / XUNCE_VS_INCUMBENT_AUDIT_FILE,
        "efficiency": output_root / EFFICIENCY_AUDIT_FILE,
        "model_inference": output_root / MODEL_INFERENCE_AUDIT_FILE,
        "model_inference_results": output_root / MODEL_INFERENCE_RESULTS_FILE,
        "source_match": output_root / SOURCE_MATCH_AUDIT_FILE,
        "boundary": output_root / BOUNDARY_AUDIT_FILE,
        "rejection_report": output_root / REJECTION_REPORT_FILE,
        "report": output_root / REPORT_FILE,
    }


def _load_source(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    root = resolve_path(Path(config["source_roi_expansion_root"]), repo_root)
    reasons: list[str] = []
    expansion = _read_json(root / EXPANSION_SUMMARY_FILE, reasons, "roi_expansion")
    slices = _read_jsonl(root / EXPANSION_SLICES_FILE, reasons, "roi_expansion_slices")
    path_feedback = _read_json(root / EXPANSION_PATH_FEEDBACK_AUDIT_FILE, reasons, "roi_expansion_path_feedback")
    xunce_checkpoint = resolve_path(Path(config["xunce_candidate_checkpoint"]), repo_root)
    incumbent_checkpoint = resolve_path(Path(config["incumbent_policy_checkpoint"]), repo_root)
    if not xunce_checkpoint.is_file():
        reasons.append("missing_xunce_candidate_checkpoint")
    if not incumbent_checkpoint.is_file():
        reasons.append("missing_incumbent_policy_checkpoint")
    return {
        "root": root,
        "expansion": expansion,
        "slices": slices,
        "path_feedback": path_feedback,
        "xunce_checkpoint": xunce_checkpoint,
        "incumbent_checkpoint": incumbent_checkpoint,
        "read_reason_codes": unique_sorted(reasons),
    }


def _load_model_bundle(config: dict[str, Any], source: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    xunce_audit, xunce_model, xunce_config = _load_xunce_checkpoint(
        Path(source["xunce_checkpoint"]),
        config=config,
        repo_root=repo_root,
    )
    incumbent_audit, incumbent_scorer = _load_incumbent_checkpoint(Path(source["incumbent_checkpoint"]))
    reason_codes = unique_sorted(
        list(xunce_audit["reason_codes"])
        + list(incumbent_audit["reason_codes"])
    )
    return {
        "xunce_model": xunce_model,
        "incumbent_scorer": incumbent_scorer,
        "xunce_config": xunce_config,
        "xunce_checkpoint_loaded": bool(xunce_audit["checkpoint_loaded"]),
        "incumbent_checkpoint_loaded": bool(incumbent_audit["checkpoint_loaded"]),
        "xunce_parameter_count": xunce_audit["parameter_count"],
        "incumbent_parameter_count": incumbent_audit["parameter_count"],
        "xunce_checkpoint_audit": xunce_audit,
        "incumbent_checkpoint_audit": incumbent_audit,
        "reason_codes": reason_codes,
    }


def _load_xunce_checkpoint(
    checkpoint_path: Path,
    *,
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], XunceFullNetworkV1 | None, dict[str, int]]:
    if not checkpoint_path.is_file():
        return _checkpoint_audit(checkpoint_path, ["missing_xunce_candidate_checkpoint"], loaded=False), None, {}
    reasons: list[str] = []
    payload: dict[str, Any] = {}
    try:
        raw_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        raw_payload = None
        reasons.append("invalid_xunce_candidate_checkpoint")
    if isinstance(raw_payload, dict):
        payload = raw_payload
    elif raw_payload is not None:
        reasons.append("invalid_xunce_candidate_checkpoint")
    state_dict = payload.get("model_state_dict") if payload else None
    if not isinstance(state_dict, dict):
        reasons.append("xunce_checkpoint_state_dict_missing")
    xunce_config = _resolve_xunce_model_config(payload, config=config, repo_root=repo_root)
    if not xunce_config:
        reasons.append("xunce_checkpoint_model_config_missing")

    model: XunceFullNetworkV1 | None = None
    if not reasons and isinstance(state_dict, dict):
        try:
            model = XunceFullNetworkV1(
                candidate_feature_count=xunce_config["candidate_feature_count"],
                edge_feature_count=xunce_config["edge_feature_count"],
                memory_feature_count=xunce_config["memory_feature_count"],
                context_feature_count=xunce_config["context_feature_count"],
                missing_indicator_count=xunce_config["missing_indicator_count"],
                hidden_dim=xunce_config["hidden_dim"],
                message_passing_layers=xunce_config["message_passing_layers"],
                dropout=0.0,
            )
            model.load_state_dict(state_dict, strict=True)
            model.eval()
        except Exception:
            reasons.append("invalid_xunce_candidate_checkpoint")
            model = None
    loaded = model is not None and not reasons
    audit = _checkpoint_audit(
        checkpoint_path,
        reasons,
        loaded=loaded,
        parameter_count_value=parameter_count(model) if model is not None else None,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    )
    return audit, model if loaded else None, xunce_config


def _load_incumbent_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], Any | None]:
    if not checkpoint_path.is_file():
        return _checkpoint_audit(checkpoint_path, ["missing_incumbent_policy_checkpoint"], loaded=False), None
    reasons: list[str] = []
    scorer = None
    try:
        scorer = load_policy_checkpoint(checkpoint_path)
    except Exception:
        reasons.append("incumbent_checkpoint_format_unsupported")
    loaded = scorer is not None and not reasons
    audit = _checkpoint_audit(
        checkpoint_path,
        reasons,
        loaded=loaded,
        parameter_count_value=parameter_count(scorer.network) if scorer is not None else None,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        metadata={"format": "model-explorer-masked-policy"} if scorer is not None else {},
    )
    return audit, scorer if loaded else None


def _checkpoint_audit(
    checkpoint_path: Path,
    reason_codes: list[str],
    *,
    loaded: bool,
    parameter_count_value: int | None = None,
    checkpoint_sha256: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "xunce-high-fidelity-checkpoint-load-audit/v1",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_loaded": bool(loaded),
        "checkpoint_read_only": True,
        "parameter_count": parameter_count_value,
        "metadata": dict(metadata or {}),
        "reason_codes": unique_sorted(reason_codes),
    }


def _resolve_xunce_model_config(payload: dict[str, Any], *, config: dict[str, Any], repo_root: Path) -> dict[str, int]:
    fields = (
        "candidate_feature_count",
        "edge_feature_count",
        "memory_feature_count",
        "context_feature_count",
        "missing_indicator_count",
        "hidden_dim",
        "message_passing_layers",
        "candidate_count",
    )
    sources: list[dict[str, Any]] = []
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if isinstance(metadata, dict):
        sources.append(metadata)
    sources.append(config)
    fallback_config = _read_optional_json(repo_root / "configs" / "xunce_sandbox_candidate_preflight_v1.json")
    if fallback_config:
        sources.append(fallback_config)
    resolved: dict[str, int] = {}
    for field in fields:
        for source in sources:
            value = _nonnegative_int_or_none(source.get(field))
            if value is not None and (field == "missing_indicator_count" or value > 0):
                resolved[field] = value
                break
    required = set(fields) - {"candidate_count"}
    if not required.issubset(resolved):
        return {}
    resolved.setdefault("candidate_count", 0)
    return resolved


def _scenario_to_model_inputs(
    scenario: dict[str, Any],
    candidates: list[dict[str, Any]],
    index: int,
    xunce_config: dict[str, int],
) -> dict[str, Any]:
    action_mask = tuple(_candidate_is_valid(candidate) for candidate in candidates)
    candidate_cells = tuple(_candidate_cell(candidate) for candidate in candidates)
    common_features = tuple(_common_candidate_features(candidate, idx, candidates) for idx, candidate in enumerate(candidates))
    missing_indicators = tuple(tuple(0.0 for _ in MISSING_INDICATOR_NAMES) for _candidate in candidates)
    observation = PolicyObservation(
        candidate_feature_names=CANDIDATE_FEATURE_NAMES,
        candidate_features=common_features,
        global_feature_names=GLOBAL_FEATURE_NAMES,
        global_features=_scenario_global_features(scenario, index),
        action_mask=action_mask,
        candidate_cells=tuple(_normalize_cell(cell) for cell in candidate_cells),
        candidate_missing_feature_names=tuple(() for _candidate in candidates),
        candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
        candidate_missing_indicators=missing_indicators,
    )
    xunce_batch = _xunce_batch_from_common_features(
        common_features,
        action_mask=action_mask,
        scenario=scenario,
        index=index,
        config=xunce_config,
    )
    return {
        "input_source": "high_fidelity_scenario_adapter/v1",
        "action_mask": action_mask,
        "has_valid_action": any(action_mask),
        "candidate_cells": candidate_cells,
        "incumbent_observation": observation,
        "xunce_batch": xunce_batch,
    }


def _xunce_batch_from_common_features(
    common_features: tuple[tuple[float, ...], ...],
    *,
    action_mask: tuple[bool, ...],
    scenario: dict[str, Any],
    index: int,
    config: dict[str, int],
) -> dict[str, torch.Tensor]:
    candidate_features = [_fit_feature_vector(row, config.get("candidate_feature_count", len(row))) for row in common_features]
    context_features = [
        _fit_feature_vector((*row, *_scenario_global_features(scenario, index)), config.get("context_feature_count", len(row)))
        for row in common_features
    ]
    memory_features = _fit_feature_vector(_scenario_memory_features(scenario, common_features, index), config.get("memory_feature_count", 1))
    missing_count = config.get("missing_indicator_count", 0)
    missing = [[0.0 for _ in range(missing_count)] for _row in common_features]
    edge_index_rows = [(candidate_index, candidate_index + 1) for candidate_index in range(max(0, len(common_features) - 1))]
    edge_feature_count = config.get("edge_feature_count", 1)
    edge_features = [
        _fit_feature_vector(
            (
                abs(right - left) / max(1.0, float(len(common_features) - 1)),
                float(left) / max(1.0, float(len(common_features))),
                float(right) / max(1.0, float(len(common_features))),
                1.0 if action_mask[left] and action_mask[right] else 0.0,
                1.0,
            ),
            edge_feature_count,
        )
        for left, right in edge_index_rows
    ]
    if not edge_features:
        edge_features = []
    return {
        "candidate_features": torch.tensor([candidate_features], dtype=torch.float32),
        "edge_features": torch.tensor(edge_features, dtype=torch.float32).reshape(len(edge_features), edge_feature_count),
        "edge_index": torch.tensor(edge_index_rows, dtype=torch.long).reshape(len(edge_index_rows), 2),
        "memory_features": torch.tensor([memory_features], dtype=torch.float32),
        "context_features": torch.tensor([context_features], dtype=torch.float32),
        "action_mask": torch.tensor([action_mask], dtype=torch.bool),
        "candidate_missing_indicators": torch.tensor([missing], dtype=torch.float32),
    }


def _score_xunce_model(model: XunceFullNetworkV1, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        output = model(**batch)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return _network_output_detail(output, latency_ms=latency_ms)


def _network_output_detail(output: Any, *, latency_ms: float) -> dict[str, Any]:
    logits = output.logits[0].detach().cpu()
    masked_logits = output.masked_logits[0].detach().cpu()
    action_probs = output.action_probs[0].detach().cpu()
    value = output.value[0].detach().cpu()
    selected_index = int(torch.argmax(action_probs).item())
    sorted_indices = torch.argsort(action_probs, descending=True)
    rank_matches = (sorted_indices == selected_index).nonzero(as_tuple=False)
    selected_rank = int(rank_matches[0].item()) + 1 if int(rank_matches.numel()) else 0
    finite_outputs = bool(
        torch.isfinite(logits).all()
        and torch.isfinite(masked_logits).all()
        and torch.isfinite(action_probs).all()
        and torch.isfinite(value)
    )
    return {
        "logits": [float(item) for item in logits],
        "masked_logits": [float(item) for item in masked_logits],
        "action_probs": [float(item) for item in action_probs],
        "value": float(value),
        "selected_action_index": selected_index,
        "selected_probability": float(action_probs[selected_index]),
        "selected_rank": selected_rank,
        "finite_outputs": finite_outputs,
        "latency_ms": float(latency_ms),
    }


def _policy_detail_to_dict(detail: Any) -> dict[str, Any]:
    return {
        "logits": list(detail.logits),
        "masked_logits": list(detail.masked_logits),
        "action_probs": list(detail.action_probs),
        "value": detail.value,
        "selected_action_index": detail.selected_action_index,
        "selected_probability": detail.selected_probability,
        "selected_rank": detail.selected_rank,
        "finite_outputs": detail.finite_outputs,
        "latency_ms": detail.latency_ms,
    }


def _model_inference_audit(model_bundle: dict[str, Any], inference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    executed_rows = [row for row in inference_rows if row.get("true_model_inference_executed")]
    finite_count = sum(
        1
        for row in inference_rows
        for model_key in ("xunce", "incumbent")
        if isinstance(row.get(model_key), dict) and row[model_key].get("finite_outputs") is True
    )
    mask_violation_count = sum(_int_value(row.get("model_inference_mask_violation_count")) for row in inference_rows)
    reason_codes = list(model_bundle["reason_codes"])
    if not model_bundle["xunce_checkpoint_loaded"] or not model_bundle["incumbent_checkpoint_loaded"]:
        reason_codes.append("true_model_inference_not_executed")
    if inference_rows and len(executed_rows) != len(inference_rows):
        reason_codes.append("true_model_inference_not_executed")
    if inference_rows and finite_count != len(inference_rows) * 2:
        reason_codes.append("model_inference_non_finite_output")
    if mask_violation_count:
        reason_codes.append("model_inference_mask_violation")
    true_model_inference_executed = bool(
        inference_rows
        and len(executed_rows) == len(inference_rows)
        and model_bundle["xunce_checkpoint_loaded"]
        and model_bundle["incumbent_checkpoint_loaded"]
    )
    reason_codes = unique_sorted(reason_codes)
    return {
        "schema_version": "xunce-high-fidelity-model-inference-audit/v1",
        "true_model_inference_executed": true_model_inference_executed,
        "proxy_selection_used": False,
        "input_source": "high_fidelity_scenario_adapter/v1",
        "xunce_checkpoint_loaded": model_bundle["xunce_checkpoint_loaded"],
        "incumbent_checkpoint_loaded": model_bundle["incumbent_checkpoint_loaded"],
        "xunce_checkpoint_audit": model_bundle["xunce_checkpoint_audit"],
        "incumbent_checkpoint_audit": model_bundle["incumbent_checkpoint_audit"],
        "xunce_model_selected_count": len(executed_rows),
        "incumbent_model_selected_count": len(executed_rows),
        "model_inference_finite_output_count": finite_count,
        "model_inference_mask_violation_count": mask_violation_count,
        "xunce_parameter_count": model_bundle["xunce_parameter_count"],
        "incumbent_parameter_count": model_bundle["incumbent_parameter_count"],
        "passed": not reason_codes,
        "reason_codes": reason_codes,
    }


def _compare_scenarios(
    config: dict[str, Any],
    source: dict[str, Any],
    model_bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    scenarios = source["path_feedback"].get("scenarios", [])
    if not isinstance(scenarios, list):
        scenarios = []
    slice_by_id = {str(row.get("scenario_id")): row for row in source["slices"] if isinstance(row, dict)}
    scenario_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            continue
        scenario_id = str(scenario.get("scenario_id", f"scenario-{index:04d}"))
        slice_row = slice_by_id.get(scenario_id, {})
        roi_group = str(slice_row.get("roi_name") or scenario.get("roi_group") or scenario.get("scenario_group") or "unknown")
        result = _compare_one_scenario(scenario, index, model_bundle)
        row = {
            "schema_version": "xunce-high-fidelity-real-map-scenario-result/v1",
            "scenario_id": scenario_id,
            "roi_group": roi_group,
            "split": slice_row.get("split"),
            "required": True,
            "passed": result["passed"],
            "reason_codes": result["reason_codes"],
            "xunce_selected_cell": result["xunce_cell"],
            "incumbent_selected_cell": result["incumbent_cell"],
            "baseline_selected_cell": scenario.get("selected_cell_after_path_feedback"),
            "xunce_path_cost": result["xunce_path_cost"],
            "incumbent_path_cost": result["incumbent_path_cost"],
            "baseline_path_cost": _finite_or_none(scenario.get("selected_path_cost_after_feedback")),
            "xunce_better_than_incumbent": result["xunce_better"],
            "xunce_worse_than_incumbent": result["xunce_worse"],
            "controlled_regression": result["controlled_regression"],
            "guard_fallback": result["guard_fallback"],
            "coverage_proxy_delta": _float_default(scenario.get("coverage_rate_delta")),
            "true_model_inference_executed": result["true_model_inference_executed"],
            "model_inference_finite": result["model_inference_finite"],
            "model_inference_mask_violation": result["model_inference_mask_violation"],
            "xunce_inference_latency_ms": result["xunce_latency_ms"],
            "incumbent_inference_latency_ms": result["incumbent_latency_ms"],
        }
        scenario_rows.append(row)
        family[roi_group].append(row)
        decision_rows.append({
            "schema_version": "xunce-high-fidelity-policy-decision/v1",
            "scenario_id": scenario_id,
            "roi_group": roi_group,
            "candidate_count": result["candidate_count"],
            "xunce_selected_cell": result["xunce_cell"],
            "incumbent_selected_cell": result["incumbent_cell"],
            "guard_fallback": result["guard_fallback"],
            "xunce_score": result["xunce_score"],
            "incumbent_score": result["incumbent_score"],
            "xunce_selected_action_index": result["xunce_action_index"],
            "incumbent_selected_action_index": result["incumbent_action_index"],
            "xunce_selected_probability": result["xunce_probability"],
            "incumbent_selected_probability": result["incumbent_probability"],
            "xunce_selected_rank": result["xunce_rank"],
            "incumbent_selected_rank": result["incumbent_rank"],
            "true_model_inference_executed": result["true_model_inference_executed"],
            "xunce_inference_latency_ms": result["xunce_latency_ms"],
            "incumbent_inference_latency_ms": result["incumbent_latency_ms"],
        })
        inference_row = dict(result["inference_row"])
        inference_row.update({"scenario_id": scenario_id, "roi_group": roi_group})
        inference_rows.append(inference_row)
    family_rows = []
    for roi_group, rows in sorted(family.items()):
        family_rows.append({
            "roi_group": roi_group,
            "scenario_count": len(rows),
            "passed_scenario_count": sum(1 for row in rows if row["passed"]),
            "failed_scenario_count": sum(1 for row in rows if not row["passed"]),
            "xunce_better_than_incumbent_count": sum(1 for row in rows if row["xunce_better_than_incumbent"]),
            "xunce_worse_than_incumbent_count": sum(1 for row in rows if row["xunce_worse_than_incumbent"]),
            "guard_fallback_count": sum(1 for row in rows if row["guard_fallback"]),
        })
    roi_family = {"schema_version": "xunce-high-fidelity-roi-family-summary/v1", "roi_group_count": len(family_rows), "families": family_rows}
    return scenario_rows, decision_rows, roi_family, inference_rows


def _compare_one_scenario(scenario: dict[str, Any], index: int, model_bundle: dict[str, Any]) -> dict[str, Any]:
    candidates = _candidate_rows(scenario)
    adapter = _scenario_to_model_inputs(scenario, candidates, index, model_bundle.get("xunce_config") or {})
    reasons: list[str] = []
    xunce_detail: dict[str, Any] | None = None
    incumbent_detail: dict[str, Any] | None = None
    true_model_inference_executed = False

    if not model_bundle["xunce_checkpoint_loaded"] or not model_bundle["incumbent_checkpoint_loaded"]:
        reasons.append("true_model_inference_not_executed")
    elif not adapter["has_valid_action"]:
        reasons.append("xunce_guard_fallback")
    else:
        try:
            xunce_detail = _score_xunce_model(model_bundle["xunce_model"], adapter["xunce_batch"])
            incumbent_detail = _policy_detail_to_dict(model_bundle["incumbent_scorer"].score_detail(adapter["incumbent_observation"]))
            true_model_inference_executed = True
        except Exception as exc:  # pragma: no cover - exercised through unsupported checkpoint tests before inference.
            reasons.append(f"true_model_inference_failed:{type(exc).__name__}")
            reasons.append("true_model_inference_not_executed")

    xunce_index = _selected_index(xunce_detail)
    incumbent_index = _selected_index(incumbent_detail)
    xunce = _candidate_at(candidates, xunce_index)
    incumbent = _candidate_at(candidates, incumbent_index)
    xunce_cost = _candidate_cost(xunce) if xunce is not None else None
    incumbent_cost = _candidate_cost(incumbent) if incumbent is not None else None
    xunce_mask_violation = _mask_violation(adapter["action_mask"], xunce_index)
    incumbent_mask_violation = _mask_violation(adapter["action_mask"], incumbent_index)
    model_inference_finite = bool(
        true_model_inference_executed
        and xunce_detail is not None
        and incumbent_detail is not None
        and xunce_detail["finite_outputs"]
        and incumbent_detail["finite_outputs"]
    )
    if true_model_inference_executed and not model_inference_finite:
        reasons.append("model_inference_non_finite_output")
    if xunce_mask_violation or incumbent_mask_violation:
        reasons.append("model_inference_mask_violation")
    guard_fallback = (
        "xunce_guard_fallback" in reasons
        or not true_model_inference_executed
        or xunce is None
        or xunce_cost is None
        or xunce_mask_violation
    )
    xunce_better = xunce_cost is not None and incumbent_cost is not None and xunce_cost < incumbent_cost - 1.0e-12
    xunce_worse = xunce_cost is not None and incumbent_cost is not None and xunce_cost > incumbent_cost + 1.0e-12
    controlled_regression = bool(
        xunce_worse
        or scenario.get("open_grid_fallback_used") is True
        or _int_value(scenario.get("tracking_safety_violation_count")) > 0
        or xunce_mask_violation
    )
    passed = true_model_inference_executed and model_inference_finite and not guard_fallback and not controlled_regression
    if guard_fallback and "xunce_guard_fallback" not in reasons:
        reasons.append("xunce_guard_fallback")
    if controlled_regression:
        reasons.append("xunce_controlled_regression")
    xunce_detail_payload = xunce_detail or _empty_model_detail()
    incumbent_detail_payload = incumbent_detail or _empty_model_detail()
    return {
        "passed": passed,
        "reason_codes": unique_sorted(reasons),
        "candidate_count": len(candidates),
        "xunce_cell": _candidate_cell(xunce) if xunce is not None else None,
        "incumbent_cell": _candidate_cell(incumbent) if incumbent is not None else None,
        "xunce_path_cost": xunce_cost,
        "incumbent_path_cost": incumbent_cost,
        "xunce_score": _model_score(xunce_detail_payload),
        "incumbent_score": _model_score(incumbent_detail_payload),
        "xunce_action_index": xunce_index,
        "incumbent_action_index": incumbent_index,
        "xunce_probability": xunce_detail_payload["selected_probability"],
        "incumbent_probability": incumbent_detail_payload["selected_probability"],
        "xunce_rank": xunce_detail_payload["selected_rank"],
        "incumbent_rank": incumbent_detail_payload["selected_rank"],
        "xunce_latency_ms": xunce_detail_payload["latency_ms"],
        "incumbent_latency_ms": incumbent_detail_payload["latency_ms"],
        "true_model_inference_executed": true_model_inference_executed,
        "model_inference_finite": model_inference_finite,
        "model_inference_mask_violation": xunce_mask_violation or incumbent_mask_violation,
        "xunce_better": xunce_better,
        "xunce_worse": xunce_worse,
        "controlled_regression": controlled_regression,
        "guard_fallback": guard_fallback,
        "inference_row": {
            "schema_version": "xunce-high-fidelity-model-inference-result/v1",
            "input_source": "high_fidelity_scenario_adapter/v1",
            "true_model_inference_executed": true_model_inference_executed,
            "candidate_cells": [_candidate_cell(candidate) for candidate in candidates],
            "action_mask": list(adapter["action_mask"]),
            "model_inference_mask_violation_count": int(xunce_mask_violation) + int(incumbent_mask_violation),
            "reason_codes": unique_sorted(reasons),
            "xunce": xunce_detail_payload,
            "incumbent": incumbent_detail_payload,
        },
    }


def _candidate_rows(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    feedback = scenario.get("path_feedback")
    candidates = feedback.get("candidates") if isinstance(feedback, dict) else None
    rows = [item for item in candidates if isinstance(item, dict)] if isinstance(candidates, list) else []
    if rows:
        return rows
    return [
        {"cell": scenario.get("selected_cell_after_path_feedback", [1, 1]), "reachable": True, "path_cost": scenario.get("selected_path_cost_after_feedback", 8.0), "risk": 0.1},
        {"cell": scenario.get("selected_cell_before_path_feedback", [2, 2]), "reachable": True, "path_cost": scenario.get("selected_path_cost_before_feedback", 10.0), "risk": 0.2},
    ]


def _candidate_at(candidates: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    if index is None or index < 0 or index >= len(candidates):
        return None
    return candidates[index]


def _candidate_is_valid(candidate: dict[str, Any]) -> bool:
    cost = _candidate_cost(candidate)
    return bool(candidate.get("reachable", True) is True and cost is not None and math.isfinite(float(cost)))


def _common_candidate_features(
    candidate: dict[str, Any],
    index: int,
    candidates: list[dict[str, Any]],
) -> tuple[float, ...]:
    cell = _normalize_cell(_candidate_cell(candidate)) or (0, 0)
    max_x = max(1.0, float(max(((_normalize_cell(_candidate_cell(item)) or (0, 0))[0] for item in candidates), default=1)))
    max_y = max(1.0, float(max(((_normalize_cell(_candidate_cell(item)) or (0, 0))[1] for item in candidates), default=1)))
    cost = _candidate_cost(candidate)
    max_cost = max(1.0, max((_candidate_cost(item) or 0.0 for item in candidates), default=1.0))
    risk = _float_default(candidate.get("risk"))
    utility = _float_default(candidate.get("utility"))
    coverage_delta = _float_default(candidate.get("expected_coverage_rate_delta"))
    value = _float_default(candidate.get("value"))
    energy_cost = _float_default(candidate.get("energy_cost"))
    normalized_cost = 0.0 if cost is None else _clip_unit(float(cost) / max_cost)
    return (
        _clip_unit(float(cell[0]) / max_x),
        _clip_unit(float(cell[1]) / max_y),
        _clip_signed_unit(float(cell[0]) / max_x),
        _clip_signed_unit(float(cell[1]) / max_y),
        _clip_unit(math.hypot(float(cell[0]), float(cell[1])) / max(1.0, math.hypot(max_x, max_y))),
        _clip_unit(utility),
        1.0 if candidate.get("reachable", True) is True else 0.0,
        _clip_unit(coverage_delta),
        _clip_unit(_float_default(candidate.get("expected_new_coverage_area"))),
        _clip_unit(_float_default(candidate.get("information_gain"))),
        _clip_unit(_float_default(candidate.get("confidence_gain"))),
        _clip_unit(value),
        _clip_unit(risk),
        normalized_cost,
        _clip_unit(energy_cost),
    )


def _scenario_global_features(scenario: dict[str, Any], index: int) -> tuple[float, ...]:
    return (
        0.01,
        0.01,
        1.0,
        1.0,
        _clip_unit(float(_int_value(scenario.get("tracking_safety_violation_count"))) / 10.0),
        _clip_unit(_float_default(scenario.get("coverage_rate"))),
        _clip_unit(float(index) / 24.0),
        _clip_unit(max(0.0, 24.0 - float(index)) / 24.0),
    )


def _scenario_memory_features(
    scenario: dict[str, Any],
    common_features: tuple[tuple[float, ...], ...],
    index: int,
) -> tuple[float, ...]:
    candidate_count = max(1, len(common_features))
    mean_path_cost = statistics.mean(row[13] for row in common_features) if common_features else 0.0
    mean_risk = statistics.mean(row[12] for row in common_features) if common_features else 0.0
    valid_fraction = statistics.mean(row[6] for row in common_features) if common_features else 0.0
    return (
        _clip_unit(float(candidate_count) / 32.0),
        _clip_unit(mean_path_cost),
        _clip_unit(mean_risk),
        _clip_unit(valid_fraction),
        _clip_unit(_float_default(scenario.get("coverage_rate_delta"))),
        _clip_unit(float(index) / 24.0),
    )


def _fit_feature_vector(values: tuple[float, ...], target_count: int) -> list[float]:
    target_count = max(0, int(target_count))
    fitted = [float(value) for value in values[:target_count]]
    if len(fitted) < target_count:
        fitted.extend(0.0 for _ in range(target_count - len(fitted)))
    return fitted


def _selected_index(detail: dict[str, Any] | None) -> int | None:
    if not isinstance(detail, dict):
        return None
    value = detail.get("selected_action_index")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mask_violation(action_mask: tuple[bool, ...], selected_index: int | None) -> bool:
    return selected_index is None or selected_index < 0 or selected_index >= len(action_mask) or not action_mask[selected_index]


def _model_score(detail: dict[str, Any]) -> float | None:
    probabilities = detail.get("action_probs")
    selected_index = _selected_index(detail)
    if not isinstance(probabilities, list) or selected_index is None or selected_index >= len(probabilities):
        return None
    return _finite_or_none(probabilities[selected_index])


def _empty_model_detail() -> dict[str, Any]:
    return {
        "logits": [],
        "masked_logits": [],
        "action_probs": [],
        "value": None,
        "selected_action_index": None,
        "selected_probability": None,
        "selected_rank": None,
        "finite_outputs": False,
        "latency_ms": None,
    }


def _normalize_cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _nonnegative_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


def _clip_unit(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def _clip_signed_unit(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return min(max(float(value), -1.0), 1.0)


def _select_xunce_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate.get("policy") == "xunce":
            cost = _candidate_cost(candidate)
            if candidate.get("reachable", True) is True and cost is not None and math.isfinite(float(cost)):
                return candidate
            return None
    valid = [candidate for candidate in candidates if candidate.get("reachable", True) is True and _candidate_cost(candidate) is not None and math.isfinite(float(_candidate_cost(candidate)))]
    if not valid:
        return None
    return min(valid, key=lambda item: (_candidate_cost(item), _float_default(item.get("risk")), str(item.get("cell"))))


def _select_incumbent_candidate(candidates: list[dict[str, Any]], scenario: dict[str, Any]) -> dict[str, Any]:
    for candidate in candidates:
        if candidate.get("policy") == "incumbent":
            return candidate
    before = scenario.get("selected_cell_before_path_feedback")
    for candidate in candidates:
        if candidate.get("cell") == before:
            return candidate
    if len(candidates) > 1:
        return candidates[1]
    return candidates[0] if candidates else {"cell": before or [0, 0], "reachable": True, "path_cost": scenario.get("selected_path_cost_before_feedback", 10.0), "risk": 0.1}


def _comparison_audit(config: dict[str, Any], scenario_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_count = len(scenario_rows)
    failed_required_count = sum(1 for row in scenario_rows if not row["passed"])
    fallback_count = sum(1 for row in scenario_rows if row["guard_fallback"])
    guided_count = len(decision_rows)
    fallback_rate = fallback_count / guided_count if guided_count else 0.0
    better = sum(1 for row in scenario_rows if row["xunce_better_than_incumbent"])
    worse = sum(1 for row in scenario_rows if row["xunce_worse_than_incumbent"])
    regressions = sum(1 for row in scenario_rows if row["controlled_regression"])
    cost_deltas = []
    for row in scenario_rows:
        if row["xunce_path_cost"] is not None and row["incumbent_path_cost"] is not None:
            cost_deltas.append(float(row["incumbent_path_cost"]) - float(row["xunce_path_cost"]))
    reasons: list[str] = []
    if required_count < config["required_scenario_count"]:
        reasons.append("high_fidelity_comparison_required_scenario_count_short")
    if fallback_rate > config["max_xunce_guard_fallback_rate"]:
        reasons.append("xunce_guard_fallback_rate_exceeded")
    return {
        "schema_version": "xunce-vs-incumbent-audit/v1",
        "passed": not reasons,
        "reason_codes": unique_sorted(reasons),
        "required_scenario_count": required_count,
        "failed_required_scenario_count": failed_required_count,
        "xunce_better_than_incumbent_count": better,
        "xunce_worse_than_incumbent_count": worse,
        "controlled_regression_count": regressions,
        "xunce_guard_fallback_count": fallback_count,
        "xunce_guided_decision_count": guided_count,
        "xunce_guard_fallback_rate": fallback_rate,
        "mean_path_budget_efficiency_delta": statistics.mean(cost_deltas) if cost_deltas else 0.0,
    }


def _efficiency_audit(config: dict[str, Any], decision_rows: list[dict[str, Any]], model_bundle: dict[str, Any]) -> dict[str, Any]:
    xunce_latencies = [
        float(row["xunce_inference_latency_ms"])
        for row in decision_rows
        if isinstance(row.get("xunce_inference_latency_ms"), (int, float))
    ]
    incumbent_latencies = [
        float(row["incumbent_inference_latency_ms"])
        for row in decision_rows
        if isinstance(row.get("incumbent_inference_latency_ms"), (int, float))
    ]
    median_latency = statistics.median(xunce_latencies) if xunce_latencies else 0.0
    incumbent_median_latency = statistics.median(incumbent_latencies) if incumbent_latencies else 0.0
    xunce_params = int(model_bundle["xunce_parameter_count"] or 0)
    reference_params = int(model_bundle["incumbent_parameter_count"] or 0)
    latency_ratio = median_latency / max(incumbent_median_latency, 1.0e-9) if incumbent_latencies else 0.0
    parameter_gate = xunce_params <= config["max_xunce_parameter_count"]
    latency_gate = median_latency <= config["max_median_inference_latency_ms"]
    ratio_gate = latency_ratio <= config["max_latency_ratio_vs_candidate_attention"]
    return {
        "schema_version": "xunce-efficiency-audit/v1",
        "xunce_parameter_count": xunce_params,
        "incumbent_parameter_count": reference_params,
        "xunce_parameter_count_proxy": xunce_params,
        "candidate_attention_parameter_count_proxy": reference_params,
        "max_xunce_parameter_count": config["max_xunce_parameter_count"],
        "xunce_median_inference_latency_ms": median_latency,
        "incumbent_median_inference_latency_ms": incumbent_median_latency,
        "candidate_attention_median_inference_latency_ms_proxy": incumbent_median_latency,
        "latency_ratio_vs_incumbent": latency_ratio,
        "latency_ratio_vs_candidate_attention": latency_ratio,
        "parameter_gate_passed": parameter_gate,
        "latency_gate_passed": latency_gate,
        "latency_ratio_gate_passed": ratio_gate,
        "passed": parameter_gate and latency_gate and ratio_gate,
        "reason_codes": [] if parameter_gate and latency_gate and ratio_gate else ["xunce_efficiency_budget_not_met"],
    }


def _source_match_audit(config: dict[str, Any], source: dict[str, Any], scenario_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expansion = source["expansion"]
    reasons = list(source["read_reason_codes"])
    allowed_next_changes = {
        "xunce_high_fidelity_real_map_policy_comparison",
        "rerun_true_model_inference_and_binding",
    }
    if expansion.get("status") != "passed" or expansion.get("next_required_change") not in allowed_next_changes:
        reasons.append("xunce_high_fidelity_roi_expansion_not_ready")
    if _int_value(expansion.get("slice_count")) < config["required_scenario_count"]:
        reasons.append("xunce_high_fidelity_roi_expansion_slice_count_short")
    return {"schema_version": "xunce-source-match-audit/v1", "passed": not reasons, "reason_codes": unique_sorted(reasons), "source_roi_expansion_status": expansion.get("status"), "source_roi_expansion_next_required_change": expansion.get("next_required_change"), "source_slice_count": _int_value(expansion.get("slice_count")), "scenario_match_count": len(scenario_rows), "scenario_mismatch_count": max(0, config["required_scenario_count"] - len(scenario_rows))}


def _boundary_audit(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for field in FORBIDDEN_FIELDS:
        if source["expansion"].get(field) is True:
            violations.append({"source": "roi_expansion", "field": field, "value": True})
    if _float_default(source["expansion"].get("canary_traffic_fraction")) > 0:
        violations.append({"source": "roi_expansion", "field": "canary_traffic_fraction", "value": source["expansion"].get("canary_traffic_fraction")})
    if config["canary_traffic_fraction"] > 0:
        violations.append({"source": "config", "field": "canary_traffic_fraction", "value": config["canary_traffic_fraction"]})
    return {"schema_version": "xunce-high-fidelity-boundary-audit/v1", "passed": not violations, "reason_codes": ["xunce_high_fidelity_comparison_boundary_violation"] if violations else [], "violations": violations, **_boundary_fields()}


def _decision(
    source: dict[str, Any],
    boundary: dict[str, Any],
    comparison: dict[str, Any],
    efficiency: dict[str, Any],
    source_match: dict[str, Any],
    model_inference: dict[str, Any],
) -> dict[str, Any]:
    reasons = unique_sorted(
        list(source_match["reason_codes"])
        + list(boundary["reason_codes"])
        + list(comparison["reason_codes"])
        + list(model_inference["reason_codes"])
    )
    if "missing_xunce_candidate_checkpoint" in source["read_reason_codes"]:
        reasons.append("missing_xunce_candidate_checkpoint")
    if "missing_incumbent_policy_checkpoint" in source["read_reason_codes"]:
        reasons.append("missing_incumbent_policy_checkpoint")
    reasons = unique_sorted(reasons)
    advantage = (
        not reasons
        and comparison["required_scenario_count"] >= 24
        and comparison["failed_required_scenario_count"] == 0
        and comparison["xunce_worse_than_incumbent_count"] == 0
        and comparison["controlled_regression_count"] == 0
        and comparison["xunce_guard_fallback_rate"] <= 0.05
        and comparison["xunce_better_than_incumbent_count"] >= 1
        and comparison["mean_path_budget_efficiency_delta"] >= 0.0
        and efficiency["passed"]
        and model_inference["true_model_inference_executed"]
        and not model_inference["proxy_selection_used"]
    )
    if reasons:
        status = "failed"
        if (
            "missing_xunce_candidate_checkpoint" in reasons
            or "invalid_xunce_candidate_checkpoint" in reasons
            or "xunce_checkpoint_state_dict_missing" in reasons
            or "xunce_checkpoint_model_config_missing" in reasons
        ):
            next_change = FIX_XUNCE_CHECKPOINT_NEXT_REQUIRED_CHANGE
        elif "missing_incumbent_policy_checkpoint" in reasons or "incumbent_checkpoint_format_unsupported" in reasons:
            next_change = FIX_INCUMBENT_CHECKPOINT_NEXT_REQUIRED_CHANGE
        elif "xunce_guard_fallback_rate_exceeded" in reasons:
            next_change = FIX_GUARD_FALLBACK_NEXT_REQUIRED_CHANGE
        elif "true_model_inference_not_executed" in reasons:
            next_change = FIX_COMPARISON_NEXT_REQUIRED_CHANGE
        elif "xunce_high_fidelity_comparison_boundary_violation" in reasons:
            next_change = BOUNDARY_NEXT_REQUIRED_CHANGE
        else:
            next_change = FIX_EXPANSION_NEXT_REQUIRED_CHANGE
    else:
        status = "passed"
        next_change = PASS_ADVANTAGE_NEXT_REQUIRED_CHANGE if advantage else PASS_NO_ADVANTAGE_NEXT_REQUIRED_CHANGE
    return {"status": status, "reason_codes": unique_sorted(reasons), "xunce_candidate_advantage_established": advantage, "next_required_change": next_change}


def _summary(
    *,
    generated_at: str,
    config_path: Path,
    output_root: Path,
    paths: dict[str, Path],
    config: dict[str, Any],
    source: dict[str, Any],
    boundary: dict[str, Any],
    comparison: dict[str, Any],
    efficiency: dict[str, Any],
    model_inference: dict[str, Any],
    source_match: dict[str, Any],
    roi_family: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": decision["status"],
        "reason_codes": decision["reason_codes"],
        "source_roi_expansion_status": source_match["source_roi_expansion_status"],
        "source_roi_expansion_next_required_change": source_match["source_roi_expansion_next_required_change"],
        "required_scenario_count": comparison["required_scenario_count"],
        "failed_required_scenario_count": comparison["failed_required_scenario_count"],
        "roi_group_count": roi_family["roi_group_count"],
        "xunce_candidate_advantage_established": decision["xunce_candidate_advantage_established"],
        "xunce_better_than_incumbent_count": comparison["xunce_better_than_incumbent_count"],
        "xunce_worse_than_incumbent_count": comparison["xunce_worse_than_incumbent_count"],
        "controlled_regression_count": comparison["controlled_regression_count"],
        "xunce_guard_fallback_count": comparison["xunce_guard_fallback_count"],
        "xunce_guard_fallback_rate": comparison["xunce_guard_fallback_rate"],
        "mean_path_budget_efficiency_delta": comparison["mean_path_budget_efficiency_delta"],
        "true_model_inference_executed": model_inference["true_model_inference_executed"],
        "proxy_selection_used": model_inference["proxy_selection_used"],
        "xunce_checkpoint_loaded": model_inference["xunce_checkpoint_loaded"],
        "incumbent_checkpoint_loaded": model_inference["incumbent_checkpoint_loaded"],
        "xunce_model_selected_count": model_inference["xunce_model_selected_count"],
        "incumbent_model_selected_count": model_inference["incumbent_model_selected_count"],
        "model_inference_finite_output_count": model_inference["model_inference_finite_output_count"],
        "model_inference_mask_violation_count": model_inference["model_inference_mask_violation_count"],
        "xunce_parameter_count": efficiency["xunce_parameter_count"],
        "incumbent_parameter_count": efficiency["incumbent_parameter_count"],
        "xunce_parameter_count_proxy": efficiency["xunce_parameter_count_proxy"],
        "xunce_median_inference_latency_ms": efficiency["xunce_median_inference_latency_ms"],
        "incumbent_median_inference_latency_ms": efficiency["incumbent_median_inference_latency_ms"],
        "latency_ratio_vs_incumbent": efficiency["latency_ratio_vs_incumbent"],
        "efficiency_audit_passed": efficiency["passed"],
        "model_inference_audit_passed": model_inference["passed"],
        "source_match_audit_passed": source_match["passed"],
        "boundary_audit_passed": boundary["passed"],
        "next_required_change": decision["next_required_change"],
        "summary": str(paths["summary"]),
        "manifest": str(paths["manifest"]),
        "scenario_results": str(paths["scenario_results"]),
        "policy_decisions": str(paths["policy_decisions"]),
        "model_inference_audit": str(paths["model_inference"]),
        "model_inference_results": str(paths["model_inference_results"]),
        "config": str(config_path),
        "output_root": str(output_root),
        "git_provenance": git_snapshot(repo_root),
        **_boundary_fields(),
    }


def _read_json(path: Path, reasons: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        reasons.append(f"missing_{label}_summary")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        reasons.append(f"invalid_{label}_summary")
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, reasons: list[str], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        reasons.append(f"missing_{label}")
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            reasons.append(f"invalid_{label}")
            return []
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _candidate_cost(candidate: dict[str, Any]) -> float | None:
    value = candidate.get("path_cost")
    if value is None:
        value = candidate.get("selected_path_cost_after_feedback")
    return _finite_or_none(value)


def _candidate_cell(candidate: dict[str, Any]) -> Any:
    return candidate.get("cell") or candidate.get("selected_cell") or candidate.get("selected_cell_after_path_feedback")


def _xunce_score(candidate: dict[str, Any]) -> float | None:
    cost = _candidate_cost(candidate)
    if cost is None:
        return None
    return -cost - _float_default(candidate.get("risk"))


def _incumbent_score(candidate: dict[str, Any], scenario: dict[str, Any], index: int) -> float | None:
    cost = _candidate_cost(candidate)
    if cost is None:
        cost = _finite_or_none(scenario.get("selected_path_cost_before_feedback"))
    return None if cost is None else -cost - index * 1.0e-9


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _boundary_fields() -> dict[str, Any]:
    fields = {field: False for field in BOUNDARY_FIELDS}
    fields.update({"canary_traffic_fraction": 0.0, "runs_new_training_update": False, "runs_new_ppo_update": False, "modifies_network": False, "modifies_action_space": False, "modifies_default_astar": False, "real_world_release_approved": False, "real_world_performance_claimed": False, "default_policy_replacement_approved": False, "real_executor_connection_approved": False, "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False, "starts_online_canary": False})
    return fields


def _render_report(summary: dict[str, Any], rejection: dict[str, Any]) -> str:
    return "\n".join([
        "# Xunce High-Fidelity Real-Map Policy Comparison v1",
        "",
        f"- status: `{summary['status']}`",
        f"- reason_codes: `{summary['reason_codes']}`",
        f"- xunce_candidate_advantage_established: `{summary['xunce_candidate_advantage_established']}`",
        f"- true_model_inference_executed: `{summary['true_model_inference_executed']}`",
        f"- proxy_selection_used: `{summary['proxy_selection_used']}`",
        f"- xunce_better_than_incumbent_count: `{summary['xunce_better_than_incumbent_count']}`",
        f"- xunce_worse_than_incumbent_count: `{summary['xunce_worse_than_incumbent_count']}`",
        f"- xunce_guard_fallback_rate: `{summary['xunce_guard_fallback_rate']}`",
        f"- xunce_median_inference_latency_ms: `{summary['xunce_median_inference_latency_ms']}`",
        f"- incumbent_median_inference_latency_ms: `{summary['incumbent_median_inference_latency_ms']}`",
        f"- next_required_change: `{summary['next_required_change']}`",
        "",
        "This is an offline read-only comparison. It does not approve default-policy replacement, real executor connection, checkpoint publication, PPO training, or real-world performance claims.",
        "",
        "## Rejection Report",
        "",
        f"```json\n{json.dumps(rejection, ensure_ascii=False, indent=2)}\n```",
        "",
    ])


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
        raise ConfigError(f"{name} must be a non-negative number")
    return float(value)


def _float_default(value: Any) -> float:
    numeric = _finite_or_none(value)
    return 0.0 if numeric is None else numeric


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
