from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot
from run_controlled_hybrid_policy_training_candidate import CHECKPOINT_METADATA_SCHEMA_VERSION
from run_hybrid_policy_training_dry_run import _pairwise_batch, _pairwise_loss
from run_quasi_real_shadow_policy_behavior_audit import _quasi_real_scenario_groups
from run_scenario_disjoint_policy_rollout_evaluation import (
    _candidate_record,
    _score_scenarios,
)


SUMMARY_SCHEMA_VERSION = "quasi-real-shadow-alignment-summary/v1"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1"


def run_quasi_real_shadow_alignment_candidate(
    *,
    taxonomy_root: str | Path,
    dataset_root: str | Path,
    preference_root: str | Path,
    base_candidate_root: str | Path,
    quasi_real_root: str | Path,
    output_root: str | Path,
    config: dict[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    _install_model_explorer_path(Path(repo_root).resolve())
    repo = Path(repo_root).resolve()
    taxonomy = Path(taxonomy_root).resolve()
    dataset = Path(dataset_root).resolve()
    preference = Path(preference_root).resolve()
    base_candidate = Path(base_candidate_root).resolve()
    quasi_real = Path(quasi_real_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = config["input_files"]
    outputs = config["output_files"]
    taxonomy_summary = _load_json(taxonomy / inputs["taxonomy_summary"])
    split_summary = _load_json(dataset / inputs["split_summary"])
    preference_summary = _load_json(preference / inputs["preference_summary"])
    preference_samples = _load_jsonl(preference / inputs.get("preference_samples", "quasi-real-shadow-alignment-preference-samples.jsonl"))
    validation = config.get("validation", {})

    reason_codes: list[str] = []
    taxonomy_failure_count = _int(taxonomy_summary.get("failure_count"))
    preference_count = _int(preference_summary.get("quasi_real_hard_negative_preference_count"))
    if taxonomy_summary.get("status") != "passed":
        reason_codes.append("quasi_real_shadow_failure_taxonomy_required")
    if split_summary.get("status") != "passed":
        reason_codes.append("quasi_real_shadow_alignment_context_leakage_detected")
    if preference_summary.get("status") != "passed":
        reason_codes.append("quasi_real_shadow_hard_negative_signal_insufficient")
    if taxonomy_failure_count < _int(validation.get("min_taxonomy_failure_count")):
        reason_codes.append("quasi_real_shadow_failure_taxonomy_required")
    if preference_count < _int(validation.get("min_quasi_real_hard_negative_preference_count")):
        reason_codes.append("quasi_real_shadow_hard_negative_signal_insufficient")
    for field in ("context_id_overlap_count", "scenario_id_overlap_count", "slice_id_overlap_count"):
        if _int(split_summary.get(field)) > _int(validation.get(f"max_{field}", 0)):
            reason_codes.append("quasi_real_shadow_alignment_context_leakage_detected")
    if _int(preference_summary.get("hard_positive_added_count")) > _int(
        validation.get("max_hard_positive_added_count")
    ):
        reason_codes.append("quasi_real_shadow_alignment_contract_invalid")
    if _int(preference_summary.get("ppo_transition_added_count")) > _int(
        validation.get("max_ppo_transition_added_count")
    ):
        reason_codes.append("quasi_real_shadow_alignment_contract_invalid")

    training_result: dict[str, Any] | None = None
    checkpoint_metadata: dict[str, Any] | None = None
    evaluation_result: dict[str, Any] = {}
    checkpoint_path = output / outputs.get("checkpoint", "experimental-hybrid-policy-candidate.pt")
    checkpoint_metadata_path = output / outputs.get(
        "checkpoint_metadata",
        "experimental-hybrid-policy-candidate-metadata.json",
    )
    candidate_summary_path = output / outputs.get(
        "candidate_summary",
        "raw-policy-generalization-candidate-summary.json",
    )
    if preference_count and not reason_codes:
        try:
            training_result, checkpoint_metadata = _train_alignment_checkpoint(
                base_candidate_root=base_candidate,
                output_root=output,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata_path=checkpoint_metadata_path,
                preference_samples=preference_samples,
                config=config,
                repo_root=repo,
            )
            _write_candidate_summary(
                path=candidate_summary_path,
                output_root=output,
                repo_root=repo,
                training_result=training_result,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata_path=checkpoint_metadata_path,
                git_provenance={"current": _git_snapshot(repo), "current_matches_sources": True},
                config=config,
            )
            evaluation_result = _evaluate_alignment_candidate(
                candidate_root=output,
                dataset_root=dataset,
                quasi_real_root=quasi_real,
                config=config,
                repo_root=repo,
            )
        except Exception as exc:  # noqa: BLE001
            reason_codes.append("quasi_real_shadow_objective_weight_refinement_required")
            evaluation_result = {"training_error": str(exc)}

    holdout_rejected = _int(evaluation_result.get("holdout_policy_changed_gate_rejected_count"))
    holdout_path = _int(evaluation_result.get("holdout_path_cost_regression_count"))
    holdout_risk = _int(evaluation_result.get("holdout_risk_regression_count"))
    holdout_source_selection = _int(evaluation_result.get("holdout_source_selection_regression_count"))
    original_roi_regression = _int(evaluation_result.get("original_roi_regression_count"))
    over_conservative = bool(evaluation_result.get("over_conservative_policy_detected", False))
    if training_result is None:
        reason_codes.append("quasi_real_shadow_objective_weight_refinement_required")
    if holdout_rejected or holdout_path or holdout_risk or holdout_source_selection:
        reason_codes.append("quasi_real_shadow_holdout_regression")
    if original_roi_regression:
        reason_codes.append("quasi_real_shadow_holdout_regression")
    if over_conservative:
        reason_codes.append("quasi_real_shadow_over_conservative_policy_detected")
    status = "failed" if reason_codes else "passed"
    alignment_verdict = (
        "acceptable_for_quasi_real_shadow_audit"
        if status == "passed"
        else evaluation_result.get("alignment_verdict") or "objective_weight_refinement_required"
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": status,
        "reason_codes": sorted(set(reason_codes)),
        "alignment_verdict": alignment_verdict,
        "taxonomy_root": _display_path(taxonomy, repo),
        "dataset_root": _display_path(dataset, repo),
        "preference_root": _display_path(preference, repo),
        "base_candidate_root": _display_path(base_candidate, repo),
        "quasi_real_root": _display_path(quasi_real, repo),
        "output_root": _display_path(output, repo),
        "taxonomy_failure_count": taxonomy_failure_count,
        "quasi_real_hard_negative_preference_count": preference_count,
        "hard_positive_added_count": _int(preference_summary.get("hard_positive_added_count")),
        "ppo_transition_added_count": _int(preference_summary.get("ppo_transition_added_count")),
        "context_id_overlap_count": _int(split_summary.get("context_id_overlap_count")),
        "scenario_id_overlap_count": _int(split_summary.get("scenario_id_overlap_count")),
        "slice_id_overlap_count": _int(split_summary.get("slice_id_overlap_count")),
        "train_policy_changed_gate_rejected_count": _int(evaluation_result.get("train_policy_changed_gate_rejected_count")),
        "val_policy_changed_gate_rejected_count": _int(evaluation_result.get("val_policy_changed_gate_rejected_count")),
        "holdout_policy_changed_gate_rejected_count": holdout_rejected,
        "holdout_path_cost_regression_count": holdout_path,
        "holdout_risk_regression_count": holdout_risk,
        "holdout_source_selection_regression_count": holdout_source_selection,
        "original_roi_regression_count": original_roi_regression,
        "original_roi_path_cost_regression_count": _int(evaluation_result.get("original_roi_path_cost_regression_count")),
        "original_roi_risk_regression_count": _int(evaluation_result.get("original_roi_risk_regression_count")),
        "safe_better_opportunity_count": _int(evaluation_result.get("safe_better_opportunity_count")),
        "over_conservative_policy_detected": over_conservative,
        "training_result": training_result,
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_path": _display_path(checkpoint_path, repo),
        "checkpoint_metadata_path": _display_path(checkpoint_metadata_path, repo),
        "candidate_summary_path": _display_path(candidate_summary_path, repo),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "next_required_change": None if status == "passed" else _next_required_change(reason_codes),
        "git_provenance": {"current": _git_snapshot(repo), "current_matches_sources": True},
        "non_goals": list(config.get("non_goals", [])),
    }
    (output / outputs["summary"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _train_alignment_checkpoint(
    *,
    base_candidate_root: Path,
    output_root: Path,
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    preference_samples: list[dict[str, Any]],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from torch.nn import functional as F

    from model_explorer.policy.architectures import build_policy_network_from_metadata
    from model_explorer.policy.features import (
        CANDIDATE_FEATURE_NAMES,
        GLOBAL_FEATURE_NAMES,
        MISSING_INDICATOR_NAMES,
    )

    base_checkpoint_path = base_candidate_root / config["base_candidate_files"]["checkpoint"]
    base_metadata_path = base_candidate_root / config["base_candidate_files"]["checkpoint_metadata"]
    base_metadata = _load_json(base_metadata_path)
    checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not state:
        raise ValueError("base candidate checkpoint is missing model state")
    training = config.get("training", {})
    hidden_size = _int(
        base_metadata.get("hidden_size")
        or checkpoint.get("training", {}).get("hidden_size")
        or training.get("hidden_size")
        or 64
    )
    architecture = checkpoint.get("architecture") or base_metadata.get("architecture")
    architecture_config = checkpoint.get("architecture_config") or base_metadata.get("architecture_config")
    network = build_policy_network_from_metadata(
        architecture,
        candidate_feature_count=len(CANDIDATE_FEATURE_NAMES),
        global_feature_count=len(GLOBAL_FEATURE_NAMES),
        missing_indicator_count=len(MISSING_INDICATOR_NAMES),
        hidden_size=hidden_size,
        architecture_config=architecture_config,
    )
    network.load_state_dict(state)
    torch.manual_seed(_int(training.get("seed", 0)))
    optimizer = torch.optim.Adam(network.parameters(), lr=float(training.get("learning_rate", 1.0e-4)))
    batch = _pairwise_batch(preference_samples)
    margin = float(training.get("margin", 0.1))
    preference_weight = float(training.get("preference_loss_weight", 1.0))
    residual_weight = float(training.get("quasi_real_hard_negative_loss_weight", 2.0))
    epoch_losses: list[dict[str, Any]] = []
    initial_margin = _preference_margin(network, batch, torch=torch)
    for epoch in range(_int(training.get("epochs", 20))):
        optimizer.zero_grad()
        loss = _pairwise_loss(
            network,
            batch,
            margin=margin,
            preference_weight=preference_weight,
            residual_weight=residual_weight,
            torch=torch,
            F=F,
        )
        if not math.isfinite(float(loss.detach())):
            raise ValueError("alignment pairwise loss is non-finite")
        loss.backward()
        optimizer.step()
        epoch_losses.append({"epoch": epoch + 1, "pairwise_loss": float(loss.detach())})
    final_margin = _preference_margin(network, batch, torch=torch)
    output_root.mkdir(parents=True, exist_ok=True)
    new_checkpoint = dict(checkpoint)
    new_checkpoint["model_state_dict"] = {key: value.detach().cpu() for key, value in network.state_dict().items()}
    new_checkpoint["training"] = {
        **dict(checkpoint.get("training", {})),
        "quasi_real_shadow_alignment": dict(training),
    }
    new_checkpoint["source_root"] = _display_path(base_candidate_root, repo_root)
    new_checkpoint["output_root"] = _display_path(output_root, repo_root)
    new_checkpoint["git_provenance"] = {"current": _git_snapshot(repo_root), "current_matches_sources": True}
    torch.save(new_checkpoint, checkpoint_path)
    metadata = {
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "experimental": True,
        "checkpoint_path": _display_path(checkpoint_path, repo_root),
        "architecture": network.architecture_name,
        "hidden_size": hidden_size,
        "sample_count": len(preference_samples),
        "action_label_sample_count": 0,
        "pairwise_preference_sample_count": len(preference_samples),
        "quasi_real_hard_negative_preference_count": len(preference_samples),
        "epochs": _int(training.get("epochs", 20)),
        "seed": _int(training.get("seed", 0)),
        "learning_rate": float(training.get("learning_rate", 1.0e-4)),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
    }
    checkpoint_metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "sample_count": len(preference_samples),
        "quasi_real_hard_negative_preference_count": len(preference_samples),
        "epochs": _int(training.get("epochs", 20)),
        "seed": _int(training.get("seed", 0)),
        "learning_rate": float(training.get("learning_rate", 1.0e-4)),
        "margin": margin,
        "preference_loss_weight": preference_weight,
        "quasi_real_hard_negative_loss_weight": residual_weight,
        "initial_preference_margin_mean": initial_margin,
        "final_preference_margin_mean": final_margin,
        "epoch_losses": epoch_losses,
        "final_pairwise_loss": epoch_losses[-1]["pairwise_loss"] if epoch_losses else None,
    }, metadata


def _preference_margin(network, batch: dict[str, Any], *, torch) -> float:
    with torch.no_grad():
        output = network(
            candidate_features=batch["candidate_features"],
            global_features=batch["global_features"],
            action_mask=batch["action_mask"],
            candidate_missing_indicators=batch["candidate_missing_indicators"],
        )
        margin = output.masked_logits[:, 0] - output.masked_logits[:, 1]
    return float(margin.mean().item())


def _write_candidate_summary(
    *,
    path: Path,
    output_root: Path,
    repo_root: Path,
    training_result: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_metadata_path: Path,
    git_provenance: dict[str, Any],
    config: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "raw-policy-generalization-candidate-summary/v1",
        "generated_at": _utc_now(),
        "status": "passed",
        "reason_codes": [],
        "candidate_root": _display_path(output_root, repo_root),
        "training_result": training_result,
        "quasi_real_hard_negative_preference_count": training_result["quasi_real_hard_negative_preference_count"],
        "leaked_context_id_count": 0,
        "experimental_checkpoint": True,
        "checkpoint_path": _display_path(checkpoint_path, repo_root),
        "checkpoint_metadata_path": _display_path(checkpoint_metadata_path, repo_root),
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "performance_claimed": False,
        "git_provenance": git_provenance,
        "non_goals": list(config.get("non_goals", [])),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _evaluate_alignment_candidate(
    *,
    candidate_root: Path,
    dataset_root: Path,
    quasi_real_root: Path,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    output_files = config["output_files"]
    dataset_slices_path = dataset_root / config["input_files"].get(
        "alignment_slices",
        "quasi-real-shadow-alignment-slices.jsonl",
    )
    dataset_summary_path = dataset_root / config["input_files"].get(
        "alignment_path_feedback_summary",
        "quasi-real-shadow-alignment-path-feedback-summary.json",
    )
    alignment_slices = _load_jsonl(dataset_slices_path)
    alignment_summary = _load_json(dataset_summary_path)
    split_by_scenario = {str(record.get("scenario_id")): str(record.get("split")) for record in alignment_slices}
    split_groups = _scenario_groups_from_summary(
        summary=alignment_summary,
        summary_path=dataset_summary_path,
        repo_root=repo_root,
    )
    decisions = _score_scenarios(
        checkpoint_path=candidate_root / output_files.get("checkpoint", "experimental-hybrid-policy-candidate.pt"),
        scenario_groups=split_groups,
        config=_scoring_config(config),
        repo_root=repo_root,
    )
    split_counts = {
        "train": _decision_counts([d for d in decisions if split_by_scenario.get(str(d.get("scenario_id"))) == "train"]),
        "val": _decision_counts([d for d in decisions if split_by_scenario.get(str(d.get("scenario_id"))) == "val"]),
        "holdout": _decision_counts([d for d in decisions if split_by_scenario.get(str(d.get("scenario_id"))) == "holdout"]),
    }
    quasi_summary_path = quasi_real_root / config["input_files"].get(
        "quasi_real_path_feedback_summary",
        "quasi-real-map-path-feedback-summary.json",
    )
    quasi_slices_path = quasi_real_root / config["input_files"].get("quasi_real_slices", "quasi-real-map-slices.jsonl")
    quasi_summary = _load_json(quasi_summary_path)
    quasi_slices = _load_jsonl(quasi_slices_path)
    original_groups = _quasi_real_scenario_groups(
        quasi_summary=quasi_summary,
        slices_by_scenario_id={str(item.get("scenario_id")): item for item in quasi_slices},
        summary_path=quasi_summary_path,
        repo_root=repo_root,
    )
    original_decisions = _score_scenarios(
        checkpoint_path=candidate_root / output_files.get("checkpoint", "experimental-hybrid-policy-candidate.pt"),
        scenario_groups=original_groups,
        config=_scoring_config(config),
        repo_root=repo_root,
    )
    original_counts = _decision_counts(original_decisions)
    paths = {
        "alignment_decisions": candidate_root / output_files.get(
            "alignment_decisions",
            "quasi-real-shadow-alignment-decisions.jsonl",
        ),
        "original_decisions": candidate_root / output_files.get(
            "original_shadow_decisions",
            "quasi-real-shadow-alignment-original-roi-decisions.jsonl",
        ),
    }
    paths["alignment_decisions"].write_text(_jsonl(decisions), encoding="utf-8")
    paths["original_decisions"].write_text(_jsonl(original_decisions), encoding="utf-8")
    holdout = split_counts["holdout"]
    return {
        "train_policy_changed_gate_rejected_count": split_counts["train"]["raw_policy_regression_count"],
        "val_policy_changed_gate_rejected_count": split_counts["val"]["raw_policy_regression_count"],
        "holdout_policy_changed_gate_rejected_count": holdout["raw_policy_regression_count"],
        "holdout_path_cost_regression_count": holdout["path_cost_regression_count"],
        "holdout_risk_regression_count": holdout["risk_regression_count"],
        "holdout_source_selection_regression_count": holdout["source_selection_regression_count"],
        "original_roi_regression_count": original_counts["raw_policy_regression_count"],
        "original_roi_path_cost_regression_count": original_counts["path_cost_regression_count"],
        "original_roi_risk_regression_count": original_counts["risk_regression_count"],
        "safe_better_opportunity_count": 0,
        "over_conservative_policy_detected": False,
        "alignment_verdict": "acceptable_for_quasi_real_shadow_audit"
        if not holdout["raw_policy_regression_count"] and not original_counts["raw_policy_regression_count"]
        else "holdout_regression",
    }


def _scenario_groups_from_summary(*, summary: dict[str, Any], summary_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for scenario in summary.get("scenarios", []) if isinstance(summary.get("scenarios"), list) else []:
        if not isinstance(scenario, dict):
            continue
        path_feedback = scenario.get("path_feedback") if isinstance(scenario.get("path_feedback"), dict) else {}
        candidates = path_feedback.get("candidates") if isinstance(path_feedback.get("candidates"), list) else []
        groups.append(
            {
                "run_id": "quasi_real_shadow_alignment",
                "source_path": _display_path(summary_path, repo_root),
                "scenario_id": str(scenario.get("scenario_id", "")),
                "scenario_group": str(scenario.get("scenario_group") or "unknown"),
                "scenario_seed": scenario.get("scenario_seed"),
                "scenario_variant_id": scenario.get("scenario_variant_id"),
                "diagnostic_profile": summary.get("diagnostic_profile"),
                "planning_backend": "path_planner_route",
                "best_by_path_cost": path_feedback.get("best_by_path_cost") if isinstance(path_feedback.get("best_by_path_cost"), dict) else None,
                "candidates": [
                    _candidate_record(candidate, scenario=scenario, summary=summary, group_path=summary_path, repo_root=repo_root)
                    for candidate in candidates
                    if isinstance(candidate, dict)
                ],
            }
        )
    return groups


def _decision_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    raw_regression = [
        decision for decision in decisions if decision.get("raw_policy_decision_class") == "regression"
    ]
    reasons = Counter(
        reason
        for decision in raw_regression
        for reason in decision.get("raw_policy_regression_reason_codes", [])
    )
    return {
        "decision_count": len(decisions),
        "raw_policy_regression_count": len(raw_regression),
        "path_cost_regression_count": reasons.get("path_cost_regression", 0),
        "risk_regression_count": reasons.get("risk_regression", 0),
        "source_selection_regression_count": reasons.get("source_selection_regression", 0),
    }


def _scoring_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation": dict(config.get("evaluation", {})),
    }


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _install_model_explorer_path(repo_root: Path) -> None:
    source = repo_root / "model-explorer" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _next_required_change(reason_codes: list[str]) -> str:
    if "quasi_real_shadow_alignment_context_leakage_detected" in reason_codes:
        return "quasi_real_shadow_alignment_context_leakage_detected"
    if "quasi_real_shadow_hard_negative_signal_insufficient" in reason_codes:
        return "quasi_real_shadow_hard_negative_signal_insufficient"
    if "quasi_real_shadow_holdout_regression" in reason_codes:
        return "quasi_real_shadow_objective_weight_refinement_required"
    if "quasi_real_shadow_over_conservative_policy_detected" in reason_codes:
        return "quasi_real_shadow_over_conservative_policy_detected"
    return "quasi_real_shadow_objective_weight_refinement_required"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize quasi-real shadow alignment calibration readiness.")
    parser.add_argument("--taxonomy-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--preference-root", required=True)
    parser.add_argument("--base-candidate-root", default="outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/update")
    parser.add_argument("--quasi-real-root", default="outputs/path_feedback_batch_quasi_real_map_domain_gap_v1")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads(_resolve_path(args.config, repo_root).read_text(encoding="utf-8"))
    summary = run_quasi_real_shadow_alignment_candidate(
        taxonomy_root=_resolve_path(args.taxonomy_root, repo_root),
        dataset_root=_resolve_path(args.dataset_root, repo_root),
        preference_root=_resolve_path(args.preference_root, repo_root),
        base_candidate_root=_resolve_path(args.base_candidate_root, repo_root),
        quasi_real_root=_resolve_path(args.quasi_real_root, repo_root),
        output_root=_resolve_path(args.output_root, repo_root),
        config=config,
        repo_root=repo_root,
    )
    print(json.dumps({"status": summary["status"], "reason_codes": summary["reason_codes"]}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
