import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PolicyRobustnessApplicationSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_policy_robustness_application_smoke.sh"
        self.config = self.repo_root / "configs" / "policy_robustness_application_smoke_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="policy-robustness-smoke-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_smoke(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return subprocess.run(
            ["bash", str(self.script), *args],
            cwd=self.repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _current_git_snapshot(self) -> dict:
        def git(path: Path, *args: str) -> str | None:
            completed = subprocess.run(
                ["git", "-C", str(path), *args],
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            if completed.returncode != 0:
                return None
            return completed.stdout.strip() or None

        return {
            "parent": {
                "path": ".",
                "sha": git(self.repo_root, "rev-parse", "HEAD") or "unknown",
                "branch": git(self.repo_root, "branch", "--show-current"),
            },
            "submodules": {
                name: {
                    "path": name,
                    "sha": git(self.repo_root / name, "rev-parse", "HEAD") or "unknown",
                    "branch": git(self.repo_root / name, "branch", "--show-current"),
                }
                for name in ("dev-platform-constraints", "model-explorer", "path-planner")
            },
        }

    def _write_sources(
        self,
        *,
        robustness_status: str = "passed",
        comparison_status: str = "passed",
        sample_quality_status: str = "passed",
        training_selection_status: str = "passed",
        open_grid: bool = False,
        bad_robustness_schema: bool = False,
        metadata_mismatch: bool = False,
        git_mismatch: bool = False,
        include_channel_aware_audit: bool = False,
        omit_sources: tuple[str, ...] = (),
    ) -> Path:
        run_id = "all-all-k3"
        scenario_id = "npz_near_blocked_corridor"
        run_root = self.batch_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        path_summary = self._path_feedback_summary(open_grid=open_grid)
        path_summary_path = run_root / "path-feedback-summary.json"
        path_summary_path.write_text(json.dumps(path_summary, indent=2), encoding="utf-8")

        robust_git = self.git_snapshot
        if git_mismatch:
            robust_git = dict(self.git_snapshot)
            robust_git["parent"] = dict(self.git_snapshot["parent"], sha="0" * 40)

        decision_context = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "scenario_group": "stress",
            "scenario_set": "stress" if metadata_mismatch else "all",
            "diagnostic_profile": "execution" if metadata_mismatch else "all",
            "top_k": 3,
            "source_summary_path": str(path_summary_path),
        }
        legacy_decision = {
            **decision_context,
            "profile_id": "legacy",
            "selected_cell_before_path_feedback": [2, 2],
            "selected_cell_after_path_feedback": [1, 1],
            "source_selection_changed_by_path_feedback": True,
            "selected_action_before_profile": 0,
            "selected_cell_before_profile": [2, 2],
            "selected_action_after_profile": 0,
            "selected_cell_after_profile": [2, 2],
            "selection_changed_by_profile": False,
            "sample_quality_record": {},
            "candidate_comparisons": [
                self._candidate_comparison(0, [2, 2], reason_codes=["candidate_passed"], before_rank=1, after_rank=1),
                self._candidate_comparison(1, [1, 1], reason_codes=["candidate_passed"], before_rank=2, after_rank=2),
            ],
        }
        aware_decision = {
            **decision_context,
            "profile_id": "sample_quality_aware",
            "selected_cell_before_path_feedback": [2, 2],
            "selected_cell_after_path_feedback": [1, 1],
            "source_selection_changed_by_path_feedback": True,
            "selected_action_before_profile": 0,
            "selected_cell_before_profile": [2, 2],
            "selected_action_after_profile": 1,
            "selected_cell_after_profile": [1, 1],
            "selection_changed_by_profile": True,
            "sample_quality_record": {
                "action": "downweight",
                "decision": "downweight",
                "sample_weight": 0.5,
                "reason_codes": [
                    "path_planning_failure",
                    "replan_required",
                    "iris_fallback",
                    "region_graph_fallback",
                    "region_graph_disconnected",
                ],
            },
            "candidate_comparisons": [
                self._candidate_comparison(
                    0,
                    [2, 2],
                    reason_codes=[
                        "path_planning_failure",
                        "replan_required",
                        "high_path_cost",
                        "iris_fallback",
                        "region_graph_fallback",
                        "region_graph_disconnected",
                        "sample_quality_downweight",
                    ],
                    before_rank=1,
                    after_rank=2,
                ),
                self._candidate_comparison(1, [1, 1], reason_codes=["sample_quality_passed"], before_rank=2, after_rank=1),
            ],
        }
        feedback_decision = dict(aware_decision, profile_id="feedback_aware")
        feedback_decision["sample_quality_record"] = {}

        robustness = {
            "schema_version": "policy-decision-robustness-summary/v0"
            if bad_robustness_schema
            else "policy-decision-robustness-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": robustness_status,
            "reason_codes": [] if robustness_status == "passed" else ["fixture_robustness_failed"],
            "failure_reason_code_counts": {},
            "batch_root": str(self.batch_root),
            "policy_decision_scope": "candidate_sorting_robustness_audit_only",
            "not_real_world_performance_claim": True,
            "does_not_modify_ppo": True,
            "does_not_modify_network": True,
            "does_not_modify_action_space": True,
            "source_summaries": {
                "path_feedback_summary_paths": [str(path_summary_path)],
                "path_feedback_summaries": {
                    run_id: {
                        "path": str(path_summary_path),
                        "status": "passed",
                        "schema_version": "path-feedback-summary/v1",
                    }
                },
            },
            "acceptance_metadata": {"by_run": {run_id: path_summary["acceptance_metadata"]}},
            "git_provenance": {
                "batch": self.git_snapshot,
                "current": robust_git,
                "runs_match_batch": not git_mismatch,
                "current_matches_batch": not git_mismatch,
            },
            "run_count": 1,
            "scenario_count": 1,
            "candidate_count": 2,
            "by_run": {
                run_id: {
                    "run_id": run_id,
                    "status": "passed",
                    "reason_codes": [],
                    "command_args": {"scenario_set": "all", "diagnostic_profile": "all", "top_k": 3},
                    "source_summary_path": str(path_summary_path),
                    "acceptance_metadata": path_summary["acceptance_metadata"],
                    "open_grid_fallback_used": open_grid,
                }
            },
            "profiles": {
                "legacy": {
                    "profile": {"id": "legacy"},
                    "scenario_count": 1,
                    "candidate_count": 2,
                    "selection_changed_count": 0,
                    "reason_code_counts": {"candidate_passed": 2},
                    "decisions": [legacy_decision],
                },
                "feedback_aware": {
                    "profile": {"id": "feedback_aware"},
                    "scenario_count": 1,
                    "candidate_count": 2,
                    "selection_changed_count": 1,
                    "reason_code_counts": {"path_planning_failure": 1},
                    "decisions": [feedback_decision],
                },
                "sample_quality_aware": {
                    "profile": {"id": "sample_quality_aware"},
                    "scenario_count": 1,
                    "candidate_count": 2,
                    "selection_changed_count": 1,
                    "reason_code_counts": {"sample_quality_downweight": 1},
                    "decisions": [aware_decision],
                },
            },
        }
        comparison = {
            "schema_version": "policy-decision-selection-comparison-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": comparison_status,
            "reason_codes": [] if comparison_status == "passed" else ["fixture_comparison_failed"],
            "batch_root": str(self.batch_root),
            "no_training_metric_evaluated": True,
            "profile_ids": ["legacy", "feedback_aware", "sample_quality_aware"],
            "comparison": {
                "legacy_profile_id": "legacy",
                "sample_quality_aware_profile_id": "sample_quality_aware",
                "scenario_count": 1,
                "selection_changed_scenario_count": 1,
                "reason_codes": ["no_training_metric_evaluated"],
            },
        }
        if include_channel_aware_audit:
            comparison["channel_aware_decision_audit"] = self._channel_aware_decision_audit(
                run_id=run_id,
                scenario_id=scenario_id,
            )
        sample_quality = {
            "schema_version": "sample-quality-training-application-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": sample_quality_status,
            "reason_codes": [] if sample_quality_status == "passed" else ["fixture_sample_quality_failed"],
            "git_provenance": {"batch": self.git_snapshot, "current": self.git_snapshot, "current_matches_batch": True},
            "profile_results": {
                "soft_downweight_diagnostics": {
                    "records": [
                        {
                            "run_id": run_id,
                            "scenario_id": scenario_id,
                            "scenario_group": "stress",
                            "action": "downweight",
                            "decision": "downweight",
                            "sample_weight": 0.5,
                            "reason_codes": aware_decision["sample_quality_record"]["reason_codes"],
                        }
                    ]
                }
            },
        }
        training_selection = {
            "schema_version": "training-selection-stability-summary/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "status": training_selection_status,
            "reason_codes": [] if training_selection_status == "passed" else ["fixture_training_selection_failed"],
            "comparison": {"reason_codes": ["no_training_metric_evaluated"]},
        }
        payloads = {
            "policy-decision-robustness-summary.json": robustness,
            "policy-decision-selection-comparison-summary.json": comparison,
            "sample-quality-training-application-summary.json": sample_quality,
            "training-selection-stability-summary.json": training_selection,
        }
        for filename, payload in payloads.items():
            if filename in omit_sources:
                continue
            (self.batch_root / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.batch_root / "policy-decision-robustness-summary.json"

    def _channel_aware_decision_audit(self, *, run_id: str, scenario_id: str) -> dict:
        def record(action_index: int, recommendation: str, reason_codes: list[str]) -> dict:
            payload = {
                "pair_key": "all-all-k3",
                "astar_run_id": f"{run_id}-astar",
                "channel_aware_run_id": f"{run_id}-channel-aware",
                "scenario_id": scenario_id,
                "scenario_group": "stress",
                "action_index": action_index,
                "cell": [action_index + 1, action_index + 1],
                "astar_selected_cell": [2, 2],
                "channel_aware_selected_cell": [1, 1],
                "selected_candidate_changed": True,
                "present": True,
                "selected": recommendation == "keep",
                "quality_improvement": recommendation == "keep",
                "risk_or_high_cost_improvement": recommendation == "keep",
                "path_cost_tradeoff": recommendation == "keep",
                "blocker_reason": reason_codes[-1] if reason_codes[-1] in {"goal_blocked", "same_as_baseline"} else None,
                "recommendation": recommendation,
                "reason_codes": reason_codes,
                "comparison": {
                    "path_changed": recommendation == "keep",
                    "path_cost_delta": 1.0 if recommendation == "keep" else None,
                    "channel_cost_delta": -2.0 if recommendation == "keep" else None,
                    "high_cost_exposure_delta": -3.0 if recommendation == "keep" else None,
                    "risk_delta": None,
                },
            }
            if "goal_blocked" in reason_codes:
                payload.update(
                    {
                        "upstream_blocker_reason": "channel_search_failed:goal_blocked",
                        "failure_taxonomy": "route_generation_failed",
                        "failure_taxonomy_source": "fallback_reason",
                        "candidate_contrast_status": "missing_candidate_contrast",
                        "has_finite_candidate_comparison": False,
                    }
                )
            return payload

        return {
            "schema_version": "channel-aware-decision-audit/v1",
            "generated_at": "2026-06-02T00:00:00Z",
            "mode": "opt_in_decision_evidence_application",
            "route_replacement_default_changed": False,
            "paired_run_count": 1,
            "paired_scenario_count": 1,
            "channel_aware_candidate_count": 4,
            "selected_candidate_changed_count": 1,
            "selected_candidate_changed_rate": 1.0,
            "risk_high_cost_exposure_improvement_count": 1,
            "risk_high_cost_exposure_improvement_rate": 0.25,
            "path_cost_regression_count": 1,
            "path_cost_regression_rate": 0.25,
            "blocker_reason_counts": {"goal_blocked": 1, "same_as_baseline": 1},
            "recommendation_counts": {
                "keep": 1,
                "downweight": 1,
                "reject": 1,
                "needs_more_evidence": 1,
            },
            "conservative_recommendation": "downweight",
            "records": [
                record(0, "keep", ["channel_aware_quality_improved", "path_cost_tradeoff"]),
                record(1, "downweight", ["same_as_baseline"]),
                record(2, "reject", ["goal_blocked"]),
                record(3, "needs_more_evidence", ["channel_aware_evidence_insufficient"]),
            ],
        }

    def _candidate_comparison(
        self,
        action_index: int,
        cell: list[int],
        *,
        reason_codes: list[str],
        before_rank: int,
        after_rank: int,
    ) -> dict:
        return {
            "action_index": action_index,
            "cell": cell,
            "before_rank": before_rank,
            "after_rank": after_rank,
            "rank_delta": after_rank - before_rank,
            "before_score": 1.0,
            "after_score": 0.5,
            "score_components": {"utility": 1.0, "path_cost": 0.0, "risk": 0.0},
            "penalty_components": {
                "path_feedback_penalty": 1.0 if "path_planning_failure" in reason_codes else 0.0,
                "sample_quality_penalty": 0.5 if "sample_quality_downweight" in reason_codes else 0.0,
                "total_penalty": 1.5 if "path_planning_failure" in reason_codes else 0.0,
            },
            "reason_codes": reason_codes,
        }

    def _path_feedback_summary(self, *, open_grid: bool = False) -> dict:
        gate = {
            "status": "failed" if open_grid else "passed",
            "expected": False,
            "actual": open_grid,
            "reason_codes": ["open_grid_fallback_used"] if open_grid else ["open_grid_fallback_not_used"],
        }
        acceptance_metadata = {
            "schema_version": "path-feedback-acceptance-metadata/v1",
            "scenario_set": "all",
            "diagnostic_profile": "all",
            "acceptance_gate": "semi-real-closed-loop",
            "top_k": 3,
            "planner_extra_args": [],
            "open_grid_fallback_used": open_grid,
            "open_grid_fallback_used_gate": gate,
        }
        return {
            "schema_version": "path-feedback-summary/v1",
            "scenario_set": "all",
            "diagnostic_profile": "all",
            "acceptance_gate": "semi-real-closed-loop",
            "top_k": 3,
            "scenario_count": 1,
            "candidate_count": 2,
            "path_planning_failure_count": 1,
            "replan_count": 1,
            "iris_fallback_count": 1,
            "region_graph_fallback_count": 1,
            "region_graph_disconnected_count": 1,
            "open_grid_fallback_used": open_grid,
            "open_grid_fallback_used_gate": gate,
            "acceptance_metadata": acceptance_metadata,
            "scenarios": [
                {
                    "scenario_id": "npz_near_blocked_corridor",
                    "scenario_group": "stress",
                    "selected_cell_before_path_feedback": [2, 2],
                    "selected_cell_after_path_feedback": [1, 1],
                    "selection_changed_by_path_feedback": True,
                    "open_grid_fallback_used": open_grid,
                    "path_feedback": {
                        "candidate_count": 2,
                        "reachable_count": 1,
                        "failure_count": 1,
                        "replan_count": 1,
                        "failure_reasons": ["goal_blocked"],
                        "candidates": [
                            {
                                "action_index": 0,
                                "cell": [2, 2],
                                "reachable": False,
                                "failure_reason": "goal_blocked",
                                "replan_required": True,
                                "path_cost": 180.0,
                                "diagnostic_interpretation": {
                                    "open_grid_fallback_used": open_grid,
                                    "iris_fallback_used": True,
                                    "region_graph_fallback_used": True,
                                    "region_graph_start_goal_connected": False,
                                },
                            },
                            {
                                "action_index": 1,
                                "cell": [1, 1],
                                "reachable": True,
                                "failure_reason": None,
                                "replan_required": False,
                                "path_cost": 12.0,
                                "diagnostic_interpretation": {
                                    "open_grid_fallback_used": open_grid,
                                    "iris_fallback_used": False,
                                    "region_graph_fallback_used": False,
                                    "region_graph_start_goal_connected": True,
                                },
                            },
                        ],
                    },
                    "diagnostic_interpretation": {
                        "target_replacement_reason": "unchanged",
                        "failure_sources": [
                            "path_planning_failure",
                            "replan_required",
                            "iris_fallback",
                            "region_graph_fallback",
                            "region_graph_disconnected",
                        ],
                        "open_grid_fallback_used": open_grid,
                    },
                }
            ],
        }

    def test_config_validate_and_dry_run_do_not_write_outputs(self) -> None:
        robustness_summary = self._write_sources()

        validate = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--robustness-summary",
            str(robustness_summary),
            "--config",
            str(self.config),
            "--validate-only",
        )
        dry_run = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--robustness-summary",
            str(robustness_summary),
            "--config",
            str(self.config),
            "--dry-run",
        )

        self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)
        self.assertIn("config validated", validate.stdout)
        self.assertEqual(dry_run.returncode, 0, dry_run.stdout + dry_run.stderr)
        self.assertIn("sample_quality_aware", dry_run.stdout)
        self.assertFalse((self.batch_root / "policy-robustness-application-summary.json").exists())
        self.assertFalse((self.batch_root / "policy-robustness-application-comparison-summary.json").exists())

    def test_application_smoke_writes_decision_deltas_and_reason_code_aggregates(self) -> None:
        robustness_summary = self._write_sources()

        completed = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--robustness-summary",
            str(robustness_summary),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        application = json.loads((self.batch_root / "policy-robustness-application-summary.json").read_text(encoding="utf-8"))
        comparison = json.loads(
            (self.batch_root / "policy-robustness-application-comparison-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(application["schema_version"], "policy-robustness-application-summary/v1")
        self.assertEqual(application["status"], "passed")
        self.assertEqual(application["applied_profile_id"], "sample_quality_aware")
        self.assertEqual(application["baseline_profile_id"], "legacy")
        self.assertEqual(application["source_summaries"]["policy_decision_robustness_summary"]["schema_version"], "policy-decision-robustness-summary/v1")
        self.assertEqual(application["git_provenance"]["robustness"]["current"]["parent"]["sha"], self.git_snapshot["parent"]["sha"])
        self.assertTrue(application["git_provenance"]["current_matches_robustness"])

        decision = application["decision_records"][0]
        self.assertEqual(decision["selected_action_before"], 0)
        self.assertEqual(decision["selected_action_after"], 1)
        self.assertEqual(decision["selected_cell_before"], [2, 2])
        self.assertEqual(decision["selected_cell_after"], [1, 1])
        self.assertTrue(decision["decision_changed"])
        self.assertEqual(decision["failure_replan_exposure"]["path_feedback_failure_count"], 1)
        self.assertEqual(decision["failure_replan_exposure"]["path_feedback_replan_count"], 1)
        self.assertIn("path_planning_failure", decision["sample_quality_reason_codes"])
        self.assertIn("region_graph_disconnected", decision["reason_codes"])
        self.assertEqual(application["by_scenario_group"]["stress"]["decision_changed_count"], 1)
        self.assertEqual(application["reason_code_counts"]["replan_required"], 1)

        self.assertEqual(comparison["schema_version"], "policy-robustness-application-comparison-summary/v1")
        self.assertTrue(comparison["no_large_scale_training"])
        self.assertTrue(comparison["no_real_world_performance_claim"])
        self.assertTrue(comparison["no_single_metric_improvement_claim"])
        self.assertEqual(comparison["comparison"]["decision_changed_count"], 1)
        self.assertIn("no_large_scale_training", comparison["comparison"]["reason_codes"])

    def test_application_smoke_maps_channel_aware_recommendations_to_actions(self) -> None:
        robustness_summary = self._write_sources(include_channel_aware_audit=True)

        completed = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--robustness-summary",
            str(robustness_summary),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        application = json.loads((self.batch_root / "policy-robustness-application-summary.json").read_text(encoding="utf-8"))
        comparison = json.loads(
            (self.batch_root / "policy-robustness-application-comparison-summary.json").read_text(encoding="utf-8")
        )
        channel = application["channel_aware_application"]
        self.assertEqual(channel["schema_version"], "channel-aware-application-smoke/v1")
        self.assertEqual(channel["record_count"], 4)
        self.assertEqual(channel["recommendation_counts"]["keep"], 1)
        self.assertEqual(channel["recommendation_counts"]["downweight"], 1)
        self.assertEqual(channel["recommendation_counts"]["reject"], 1)
        self.assertEqual(channel["recommendation_counts"]["needs_more_evidence"], 1)
        self.assertEqual(channel["action_counts"]["keep_quality_evidence"], 1)
        self.assertEqual(channel["action_counts"]["downweight_conservative_application"], 1)
        self.assertEqual(channel["action_counts"]["exclude_blocked_candidate_evidence"], 1)
        self.assertEqual(channel["action_counts"]["downweight_needs_more_evidence"], 1)
        self.assertEqual(channel["reason_code_counts"]["goal_blocked"], 1)
        self.assertEqual(channel["reason_code_counts"]["same_as_baseline"], 1)
        self.assertTrue(channel["no_large_scale_training"])
        self.assertFalse(channel["route_replacement_default_changed"])

        records_by_action = {record["action_index"]: record for record in channel["records"]}
        self.assertEqual(records_by_action[0]["application_action"], "keep_quality_evidence")
        self.assertEqual(records_by_action[1]["application_action"], "downweight_conservative_application")
        self.assertEqual(records_by_action[2]["application_action"], "exclude_blocked_candidate_evidence")
        self.assertEqual(records_by_action[3]["application_action"], "downweight_needs_more_evidence")
        self.assertIn("channel_aware_application_keep_quality_evidence", records_by_action[0]["application_reason_codes"])
        self.assertIn("channel_aware_application_exclude_blocked_candidate_evidence", records_by_action[2]["application_reason_codes"])
        self.assertIn("channel_aware_application_downweight_needs_more_evidence", records_by_action[3]["application_reason_codes"])
        self.assertEqual(records_by_action[2]["upstream_blocker_reason"], "channel_search_failed:goal_blocked")
        self.assertEqual(records_by_action[2]["failure_taxonomy"], "route_generation_failed")
        self.assertEqual(records_by_action[2]["failure_taxonomy_source"], "fallback_reason")
        self.assertEqual(records_by_action[2]["candidate_contrast_status"], "missing_candidate_contrast")
        self.assertFalse(records_by_action[2]["has_finite_candidate_comparison"])

        self.assertEqual(
            comparison["channel_aware_application"]["action_counts"],
            channel["action_counts"],
        )

    def test_application_smoke_propagates_platform_goal_contract_mismatch_fields(self) -> None:
        robustness_summary = self._write_sources(include_channel_aware_audit=True)
        comparison_path = self.batch_root / "policy-decision-selection-comparison-summary.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        blocked = comparison["channel_aware_decision_audit"]["records"][2]
        blocked["reason_codes"] = ["goal_blocked", "platform_inflated_goal_blocked"]
        blocked["failure_taxonomy"] = "platform_inflated_goal_blocked"
        blocked["failure_taxonomy_source"] = "platform_goal_feasibility.classification"
        blocked["platform_goal_classification"] = "platform_inflated_goal_blocked"
        blocked["platform_goal_feasibility"] = {
            "schema_version": "platform-goal-feasibility/v1",
            "classification": "platform_inflated_goal_blocked",
            "contract_reachable": True,
            "original_passable": True,
            "inflated_passable": False,
            "blocked_by_platform_footprint": True,
            "nearest_inflated_passable_anchor": [3, 4],
            "anchor_distance_cells": 1,
            "anchor_distance_m": 1.0,
            "proxy_route_comparison": {
                "scope": "audit_proxy_anchor_not_same_cell",
                "anchor_route_feasible": True,
                "same_cell_positive_evidence": False,
            },
        }
        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

        completed = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--robustness-summary",
            str(robustness_summary),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        application = json.loads(
            (self.batch_root / "policy-robustness-application-summary.json").read_text(
                encoding="utf-8"
            )
        )
        blocked_record = {
            record["action_index"]: record
            for record in application["channel_aware_application"]["records"]
        }[2]
        self.assertEqual(blocked_record["failure_taxonomy"], "platform_inflated_goal_blocked")
        self.assertEqual(
            blocked_record["failure_taxonomy_source"],
            "platform_goal_feasibility.classification",
        )
        self.assertEqual(
            blocked_record["platform_goal_classification"],
            "platform_inflated_goal_blocked",
        )
        self.assertEqual(
            blocked_record["platform_goal_feasibility"]["classification"],
            "platform_inflated_goal_blocked",
        )
        self.assertFalse(
            blocked_record["platform_goal_feasibility"]["proxy_route_comparison"][
                "same_cell_positive_evidence"
            ]
        )

    def test_validation_failures_are_written_with_machine_readable_reason_codes(self) -> None:
        robustness_summary = self._write_sources(
            robustness_status="failed",
            sample_quality_status="failed",
            training_selection_status="failed",
            open_grid=True,
            bad_robustness_schema=True,
            metadata_mismatch=True,
            git_mismatch=True,
            omit_sources=("training-selection-stability-summary.json",),
        )

        completed = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--robustness-summary",
            str(robustness_summary),
            "--config",
            str(self.config),
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        application = json.loads((self.batch_root / "policy-robustness-application-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(application["status"], "failed")
        self.assertIn("policy_decision_robustness_summary_schema_mismatch", application["reason_codes"])
        self.assertIn("policy_decision_robustness_summary_failed", application["reason_codes"])
        self.assertIn("sample_quality_training_application_summary_failed", application["reason_codes"])
        self.assertIn("training_selection_stability_summary_missing", application["reason_codes"])
        self.assertIn("open_grid_fallback_used", application["reason_codes"])
        self.assertIn("acceptance_metadata_mismatch", application["reason_codes"])
        self.assertIn("current_git_provenance_mismatch", application["reason_codes"])
        self.assertIn("open_grid_fallback_used", application["failure_reason_code_counts"])

    def test_missing_robustness_summary_returns_nonzero_and_writes_audit_outputs(self) -> None:
        missing = self.batch_root / "missing-robustness.json"

        completed = self._run_smoke(
            "--batch-root",
            str(self.batch_root),
            "--robustness-summary",
            str(missing),
            "--config",
            str(self.config),
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        application = json.loads((self.batch_root / "policy-robustness-application-summary.json").read_text(encoding="utf-8"))
        self.assertIn("policy_decision_robustness_summary_missing", application["reason_codes"])


class PolicyRobustnessApplicationSmokeCompatibilityTests(unittest.TestCase):
    def test_existing_smoke_related_cli_default_behaviors_remain_unchanged(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        single = subprocess.run(
            ["bash", str(repo_root / "scripts" / "run_path_feedback_validation.sh"), "--dry-run"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        batch = subprocess.run(
            [
                "bash",
                str(repo_root / "scripts" / "run_batch_path_feedback_validation.sh"),
                "--matrix",
                str(repo_root / "configs" / "path_feedback_batch_dataset_v1.json"),
                "--validate-only",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(single.returncode, 0, single.stdout + single.stderr)
        self.assertIn("Scenario set: smoke", single.stdout)
        self.assertEqual(batch.returncode, 0, batch.stdout + batch.stderr)
        self.assertIn("matrix validated", batch.stdout)


if __name__ == "__main__":
    unittest.main()
