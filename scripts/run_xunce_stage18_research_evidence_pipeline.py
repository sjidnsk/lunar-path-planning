from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from global_99_coverage_contract import ConfigError, resolve_path
from xunce_stage18_pipeline import (
    build_stage18_pipeline_summary,
    write_stage18_pipeline_artifacts,
)


DEFAULT_CONFIG = "configs/xunce_stage18_research_evidence_pipeline_v1.json"
DEFAULT_OUTPUT_ROOT = "outputs/path_feedback_batch_xunce_stage18_research_evidence_pipeline_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize the consolidated Xunce Stage 18 research evidence pipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root")
    parser.add_argument("--plan-only", action="store_true", help="Resolve roots, summarize evidence, and write pipeline artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print the stage command plan without executing downstream stages.")
    parser.add_argument("--execute", action="store_true", help="Reserved for a future orchestrated execution mode.")
    parser.add_argument("--candidate-root")
    parser.add_argument("--model-inference-root")
    parser.add_argument("--true-binding-root")
    parser.add_argument("--quantization-root")
    parser.add_argument("--oracle-root")
    parser.add_argument("--coverage-comparison-root")
    parser.add_argument("--stage18-5-attribution-root")
    parser.add_argument("--stage18-6-guard-refinement-root")
    parser.add_argument("--stage18-7-candidate-count-scaling-root")
    parser.add_argument("--stage18-9-trajectory-risk-reward-root")
    parser.add_argument("--stage18-11-path-cost-weight-calibration-root")
    parser.add_argument("--stage19-evaluator-critic-preflight-root")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    output_root = resolve_path(Path(args.output_root), repo_root).resolve()
    overrides = {
        "stage18_2_candidate_root": args.candidate_root,
        "stage18_3_model_inference_root": args.model_inference_root,
        "stage18_3_true_incumbent_binding_root": args.true_binding_root,
        "stage18_4_quantization_root": args.quantization_root,
        "stage18_4_oracle_root": args.oracle_root,
        "stage18_4_coverage_comparison_root": args.coverage_comparison_root,
        "stage18_5_attribution_root": args.stage18_5_attribution_root,
        "stage18_6_guard_refinement_root": args.stage18_6_guard_refinement_root,
        "stage18_7_candidate_count_scaling_root": args.stage18_7_candidate_count_scaling_root,
        "stage18_9_trajectory_risk_reward_root": args.stage18_9_trajectory_risk_reward_root,
        "stage18_11_path_cost_weight_calibration_root": args.stage18_11_path_cost_weight_calibration_root,
        "stage19_evaluator_critic_preflight_root": args.stage19_evaluator_critic_preflight_root,
    }
    try:
        summary = build_stage18_pipeline_summary(
            config_path=resolve_path(Path(args.config), repo_root).resolve(),
            output_root=output_root,
            repo_root=repo_root,
            overrides=overrides,
            plan_only=args.plan_only or not args.execute,
            dry_run=args.dry_run,
            execute=args.execute,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    write_stage18_pipeline_artifacts(output_root, summary)
    if args.dry_run:
        for row in summary["stage_command_plan"]:
            print(f"[DRY RUN] {row['display']}")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "evidence_status": summary["evidence_status"],
                "candidate_validity_status": summary["candidate_validity_status"],
                "comparison_verdict": summary["comparison_verdict"],
                "overall_conclusion": summary["overall_conclusion"],
                "next_required_change": summary["next_required_change"],
                "summary": str(output_root / "xunce-stage18-pipeline-summary.json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if summary["status"] in {"passed", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
