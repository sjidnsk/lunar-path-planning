from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from git_provenance import git_snapshot as _git_snapshot
from run_limited_ppo_update_smoke import (
    _display_path,
    _install_model_explorer_path,
    _load_config,
    _resolve_path,
    _utc_now,
    run_limited_ppo_update_smoke,
)


SUMMARY_SCHEMA_VERSION = "policy-training-cuda-device-support-summary/v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a CUDA/device smoke for policy training updates.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--base-candidate-root", required=True)
    parser.add_argument("--collector-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    _install_model_explorer_path(repo_root)
    config_path = _resolve_path(args.config, repo_root)
    config = _load_config(config_path)
    if args.validate_only:
        print(json.dumps({"status": "config validated", "config": _display_path(config_path, repo_root)}))
        return 0

    output_root = _resolve_path(args.output_root, repo_root)
    limited_summary = run_limited_ppo_update_smoke(
        source_root=_resolve_path(args.source_root, repo_root),
        base_candidate_root=_resolve_path(args.base_candidate_root, repo_root),
        collector_root=_resolve_path(args.collector_root, repo_root),
        output_root=output_root,
        config=config,
        repo_root=repo_root,
    )
    summary = _device_support_summary(
        limited_summary=limited_summary,
        output_root=output_root,
        repo_root=repo_root,
    )
    summary_path = output_root / "policy-training-cuda-device-support-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reason_codes": summary["reason_codes"],
                "requested_device": summary["requested_device"],
                "resolved_device": summary["resolved_device"],
                "optimizer_train_transition_count": summary["optimizer_train_transition_count"],
                "summary": _display_path(summary_path, repo_root),
            },
            ensure_ascii=False,
        )
    )
    return 1 if summary["status"] == "failed" else 0


def _device_support_summary(
    *,
    limited_summary: dict[str, Any],
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    copied_fields = (
        "status",
        "reason_codes",
        "failure_reason_code_counts",
        "next_required_change",
        "requested_device",
        "resolved_device",
        "cuda_available",
        "cuda_device_name",
        "fallback_to_cpu",
        "device_reason_codes",
        "input_ppo_trainable_transition_count",
        "optimizer_train_transition_count",
        "source_fallback_trainable_count",
        "old_log_prob_max_abs_error",
        "old_value_max_abs_error",
        "loss_non_finite_count",
        "non_finite_gradient_count",
        "non_finite_reward_count",
        "non_finite_return_count",
        "non_finite_advantage_count",
        "parameter_l2_delta",
        "approx_kl",
        "clip_fraction",
        "grad_norm_before_clip",
        "max_grad_norm_after_clip",
        "checkpoint_cpu_loadable",
        "experimental_checkpoint",
        "publishes_checkpoint",
        "replaces_default_policy",
        "performance_claimed",
        "formal_training_ready_claimed",
        "runs_formal_ppo_rollout",
        "non_goals",
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "output_root": _display_path(output_root, repo_root),
        "limited_ppo_update_smoke_summary": _display_path(
            output_root / "limited-ppo-update-smoke-summary.json",
            repo_root,
        ),
        "git_provenance": {"current": _git_snapshot(repo_root), "current_matches_sources": True},
    }
    for field in copied_fields:
        summary[field] = limited_summary.get(field)
    summary.setdefault("status", "failed")
    summary.setdefault("reason_codes", ["cuda_training_smoke_failed"])
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
