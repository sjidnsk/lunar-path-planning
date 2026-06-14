import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PolicyTrainingReadinessReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_policy_training_readiness_review.sh"
        self.config = self.repo_root / "configs" / "policy_training_readiness_review_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="policy-training-readiness-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_review(self, *args: str) -> subprocess.CompletedProcess[str]:
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

    def _git_snapshot(self, *, git_mismatch: bool = False) -> dict:
        if not git_mismatch:
            return self.git_snapshot
        return {
            **self.git_snapshot,
            "parent": {**self.git_snapshot["parent"], "sha": "0" * 40},
        }

    def _write_sources(
        self,
        *,
        source_rate: float = 0.0,
        calibrated_rate: float = 0.5,
        applied_count: int = 2,
        rejected_goal_blocked_count: int = 0,
        safety_regression_count: int = 0,
        smoke_recommended: str = "ready_for_policy_training_readiness_review",
        open_grid_fallback_used_count: int = 0,
        contract_mutation: bool = False,
        platform_goal_contract_mismatch_count: int = 0,
        git_mismatch: bool = False,
    ) -> None:
        git_snapshot = self._git_snapshot(git_mismatch=git_mismatch)
        changed_ids = [
            "npz_blocked_nearby_clearance_detour",
            "npz_high_cost_exposure_rock_detour",
        ]
        contract_guard = not contract_mutation
        smoke = {
            "schema_version": "calibrated-policy-application-smoke-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": source_rate,
            "calibrated_selected_candidate_changed_rate": calibrated_rate,
            "calibrated_selection_rate_delta": calibrated_rate - source_rate,
            "applied_calibrated_candidate_count": applied_count,
            "changed_scenario_ids": changed_ids,
            "rejected_goal_blocked_count": rejected_goal_blocked_count,
            "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
            "platform_goal_anchor_available_count": platform_goal_contract_mismatch_count,
            "platform_goal_unresolved_count": 0,
            "platform_goal_feasibility_class_counts": (
                {"platform_inflated_goal_blocked": platform_goal_contract_mismatch_count}
                if platform_goal_contract_mismatch_count
                else {}
            ),
            "safety_regression_count": safety_regression_count,
            "application_gate_reason_codes": [],
            "recommended_next_action": smoke_recommended,
            "audit_only": True,
            "runs_training": False,
            "no_ppo_training": True,
            "no_large_scale_training": True,
            "channel_aware_backend_opt_in": True,
            "does_not_modify_default_astar": True,
            "does_not_modify_ppo": contract_guard,
            "does_not_modify_network": contract_guard,
            "does_not_modify_action_space": contract_guard,
            "does_not_modify_model_explorer_contract": contract_guard,
            "does_not_modify_path_planner_route_contract": contract_guard,
            "does_not_modify_path_planner_sidecar_contract": contract_guard,
            "no_ackermann_feasible_trajectory_claim": True,
            "open_grid_fallback_used_count": open_grid_fallback_used_count,
            "git_provenance": {"current": git_snapshot, "current_matches_sources": not git_mismatch},
        }
        readiness = {
            "schema_version": "channel-aware-training-readiness-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "readiness_status": "ready_for_calibrated_policy_application_smoke",
            "calibrated_readiness_status": "ready_for_calibrated_policy_application_smoke",
            "source_selected_candidate_changed_rate": source_rate,
            "calibration_selected_candidate_changed_rate": calibrated_rate,
            "calibration_safety_regression_count": safety_regression_count,
            "git_provenance": {"current": git_snapshot},
        }
        coverage = {
            "schema_version": "channel-aware-contrast-coverage-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": source_rate,
            "calibrated_selected_candidate_changed_rate": calibrated_rate,
            "changed_scenario_ids": changed_ids,
            "blocked_candidate_rate": 0.0,
            "safety_regression_count": safety_regression_count,
            "recommended_next_action": "ready_for_calibrated_policy_application_smoke",
            "open_grid_fallback_used_count": open_grid_fallback_used_count,
            "git_provenance": {"current": git_snapshot},
        }
        calibration = {
            "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": source_rate,
            "selected_candidate_changed_count": applied_count,
            "selected_candidate_changed_rate": calibrated_rate,
            "changed_scenario_ids": changed_ids,
            "goal_blocked_count": rejected_goal_blocked_count,
            "platform_goal_contract_mismatch_count": platform_goal_contract_mismatch_count,
            "platform_goal_anchor_available_count": platform_goal_contract_mismatch_count,
            "platform_goal_unresolved_count": 0,
            "platform_goal_feasibility_class_counts": (
                {"platform_inflated_goal_blocked": platform_goal_contract_mismatch_count}
                if platform_goal_contract_mismatch_count
                else {}
            ),
            "safety_regression_count": safety_regression_count,
            "git_provenance": {"current": git_snapshot},
        }
        (self.batch_root / "calibrated-policy-application-smoke-summary.json").write_text(
            json.dumps(smoke, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "channel-aware-training-readiness-summary.json").write_text(
            json.dumps(readiness, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "channel-aware-contrast-coverage-summary.json").write_text(
            json.dumps(coverage, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "channel-aware-selection-contrast-calibration-summary.json").write_text(
            json.dumps(calibration, indent=2),
            encoding="utf-8",
        )

    def _write_anchor_projection_summaries(
        self,
        *,
        candidate_trainable_count: int,
        candidate_nontrainable_count: int,
        contract_trainable_count: int,
        contract_nontrainable_count: int,
        anchor_unreachable_count: int = 0,
        source_candidate_not_selected_count: int = 0,
        quality_regression_count: int = 0,
        max_path_cost_margin: float | None = None,
        max_risk_margin: float | None = None,
        reachable_substitute_anchor_found_count: int = 0,
        anchor_unreachable_repaired_count: int = 0,
        true_geometry_unreachable_count: int = 0,
        source_selected_but_distance_rejected_count: int = 0,
        distance_contract_rejected_source_selected_count: int = 0,
        distance_contract_rejected_by_distance_bin: dict | None = None,
        source_candidate_not_selected_by_best_alternative_reason: dict | None = None,
        source_selection_quality_tradeoff_summary: dict | None = None,
    ) -> tuple[Path, Path]:
        candidate_path = self.batch_root / "anchor-projection-candidate-generation-summary.json"
        contract_path = self.batch_root / "anchor-projection-evidence-contract-summary.json"
        distance_bins = distance_contract_rejected_by_distance_bin or {
            "count": 0,
            "source_selected_count": 0,
            "not_source_selected_count": 0,
            "by_projection_distance_cells": {},
            "by_projection_distance_m": {},
        }
        not_selected_reasons = source_candidate_not_selected_by_best_alternative_reason or {
            "distance_contract_rejected": 0,
            "higher_path_cost": 0,
            "higher_path_cost_and_risk": 0,
            "higher_risk": 0,
            "lower_utility_or_coverage": 0,
            "ranking_weight_tradeoff_or_unobserved_utility": 0,
            "no_selected_candidate_comparison": 0,
        }
        tradeoff_summary = source_selection_quality_tradeoff_summary or {
            "generated_not_source_selected_count": source_candidate_not_selected_count,
            "source_selected_but_distance_rejected_count": source_selected_but_distance_rejected_count,
            "distance_contract_rejected_count": distance_bins["count"],
            "source_candidate_not_selected_reason_counts": not_selected_reasons,
            "distance_contract_relaxation_recommendation": (
                "record_only_keep_current_training_distance_contract"
            ),
        }
        candidate_path.write_text(
            json.dumps(
                {
                    "schema_version": "anchor-projection-candidate-generation-summary/v1",
                    "generated_at": "2026-06-07T00:00:00Z",
                    "status": "passed",
                    "reason_codes": [],
                    "trainable_anchor_projection_count": candidate_trainable_count,
                    "candidate_contract_alignment_gap_count": 0,
                    "nontrainable_blocked_target_count": candidate_nontrainable_count,
                    "platform_goal_contract_mismatch_count": (
                        candidate_trainable_count + candidate_nontrainable_count
                    ),
                    "source_selection_quality_regression_count": quality_regression_count,
                    "reachable_substitute_anchor_found_count": reachable_substitute_anchor_found_count,
                    "anchor_unreachable_repaired_by_reachable_substitute_count": (
                        anchor_unreachable_repaired_count
                    ),
                    "true_geometry_unreachable_count": true_geometry_unreachable_count,
                    "source_selected_but_distance_rejected_count": (
                        source_selected_but_distance_rejected_count
                    ),
                    "distance_contract_rejected_source_selected_count": (
                        distance_contract_rejected_source_selected_count
                    ),
                    "distance_contract_rejected_by_distance_bin": distance_bins,
                    "source_candidate_not_selected_by_best_alternative_reason": not_selected_reasons,
                    "source_selection_quality_tradeoff_summary": tradeoff_summary,
                    "anchor_projection_coverage_diagnosis": {
                        "nontrainable_primary_reason_counts": {
                            "anchor_unreachable": anchor_unreachable_count,
                            "source_candidate_not_selected": source_candidate_not_selected_count,
                        },
                        "anchor_unreachable_not_generated_count": anchor_unreachable_count,
                        "projected_candidate_not_source_selected_count": source_candidate_not_selected_count,
                        "anchor_selection_status_counts": {
                            "reachable_substitute_anchor_found": reachable_substitute_anchor_found_count,
                            "true_geometry_unreachable": true_geometry_unreachable_count,
                        },
                        "reachable_substitute_anchor_found_count": reachable_substitute_anchor_found_count,
                        "anchor_unreachable_repaired_by_reachable_substitute_count": (
                            anchor_unreachable_repaired_count
                        ),
                        "true_geometry_unreachable_count": true_geometry_unreachable_count,
                        "projection_distance_contract_rejected_count": distance_bins["count"],
                        "source_selection_margin": {
                            "max_path_cost_margin": max_path_cost_margin,
                            "max_risk_margin": max_risk_margin,
                        }
                    },
                    "git_provenance": {"current": self.git_snapshot},
                    "runs_training": False,
                    "audit_only": True,
                    "no_ppo_training": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        contract_path.write_text(
            json.dumps(
                {
                    "schema_version": "anchor-projection-evidence-contract-summary/v1",
                    "generated_at": "2026-06-07T00:00:00Z",
                    "status": "passed",
                    "reason_codes": [],
                    "trainable_anchor_projection_count": contract_trainable_count,
                    "nontrainable_blocked_target_count": contract_nontrainable_count,
                    "platform_goal_contract_mismatch_count": (
                        contract_trainable_count + contract_nontrainable_count
                    ),
                    "reachable_substitute_anchor_found_count": reachable_substitute_anchor_found_count,
                    "anchor_unreachable_repaired_by_reachable_substitute_count": (
                        anchor_unreachable_repaired_count
                    ),
                    "true_geometry_unreachable_count": true_geometry_unreachable_count,
                    "source_selected_but_distance_rejected_count": (
                        source_selected_but_distance_rejected_count
                    ),
                    "distance_contract_rejected_source_selected_count": (
                        distance_contract_rejected_source_selected_count
                    ),
                    "distance_contract_rejected_by_distance_bin": distance_bins,
                    "source_candidate_not_selected_by_best_alternative_reason": not_selected_reasons,
                    "source_selection_quality_tradeoff_summary": tradeoff_summary,
                    "positive_training_evidence_contains_audit_proxy_anchor_count": 0,
                    "recommended_next_action": "rerun_policy_training_readiness_review_with_anchor_projection_contract",
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                    "runs_training": False,
                    "audit_only": True,
                    "no_ppo_training": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return candidate_path, contract_path

    def test_review_allows_limited_training_dry_run_when_contract_is_clear(self) -> None:
        self._write_sources()

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["schema_version"], "policy-training-readiness-review-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["training_readiness_status"], "ready_for_limited_policy_training_dry_run")
        self.assertEqual(summary["recommended_next_action"], "ready_for_limited_policy_training_dry_run")
        self.assertEqual(summary["source_selected_candidate_changed_rate"], 0.0)
        self.assertEqual(summary["calibrated_selected_candidate_changed_rate"], 0.5)
        self.assertEqual(summary["applied_calibrated_candidate_count"], 2)
        self.assertEqual(summary["training_positive_candidate_count"], 2)
        self.assertEqual(summary["excluded_candidate_count"], 0)
        self.assertEqual(summary["training_blockers"], [])
        self.assertEqual(summary["contract_impact"]["training_contract_status"], "compatible_audit_only")
        self.assertTrue(summary["git_provenance"]["current_matches_sources"])
        self.assertTrue(summary["audit_only"])
        self.assertTrue(summary["no_ppo_training"])
        self.assertTrue(summary["does_not_modify_default_astar"])
        self.assertTrue(summary["no_ackermann_feasible_trajectory_claim"])

    def test_review_requires_contract_refinement_when_goal_blocked_candidates_exist(self) -> None:
        self._write_sources(rejected_goal_blocked_count=3)

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")
        self.assertEqual(summary["recommended_next_action"], "needs_training_contract_refinement")
        self.assertEqual(summary["training_positive_candidate_count"], 2)
        self.assertEqual(summary["excluded_candidate_count"], 3)
        self.assertIn(
            "goal_blocked_candidates_excluded_from_training_positive_evidence",
            summary["training_blockers"],
        )

    def test_review_reports_platform_goal_contract_mismatch_breakdown(self) -> None:
        self._write_sources(
            rejected_goal_blocked_count=3,
            platform_goal_contract_mismatch_count=3,
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")
        self.assertEqual(summary["rejected_goal_blocked_count"], 3)
        self.assertEqual(summary["platform_goal_contract_mismatch_count"], 3)
        self.assertEqual(summary["platform_goal_trainable_anchor_projection_count"], 0)
        self.assertEqual(summary["platform_goal_nontrainable_blocked_target_count"], 3)
        self.assertEqual(summary["platform_goal_anchor_available_count"], 3)
        self.assertEqual(summary["platform_goal_unresolved_count"], 0)
        self.assertEqual(
            summary["platform_goal_feasibility_class_counts"]["platform_inflated_goal_blocked"],
            3,
        )
        self.assertTrue(summary["no_ppo_training"])
        self.assertFalse(summary["runs_training"])

    def test_review_blocks_fallback_safety_and_contract_mutation_from_training_readiness(self) -> None:
        for kwargs, expected_reason in (
            (
                {"open_grid_fallback_used_count": 1},
                "fallback_or_open_grid_evidence_blocks_training_readiness",
            ),
            (
                {"safety_regression_count": 1},
                "safety_regression_blocks_training_readiness",
            ),
            (
                {"contract_mutation": True},
                "contract_mutation_blocks_training_readiness",
            ),
        ):
            with self.subTest(expected_reason=expected_reason):
                shutil.rmtree(self.batch_root)
                self.batch_root.mkdir(parents=True)
                self._write_sources(**kwargs)

                completed = self._run_review(
                    "--batch-root",
                    str(self.batch_root),
                    "--config",
                    str(self.config),
                )

                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                summary = json.loads(
                    (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")
                self.assertEqual(summary["recommended_next_action"], "needs_training_contract_refinement")
                self.assertIn(expected_reason, summary["training_blockers"])

    def test_validate_only_blocks_current_git_mismatch_without_writing_summary(self) -> None:
        self._write_sources(git_mismatch=True)

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--validate-only",
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        validation = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(validation["status"], "validation failed")
        self.assertIn("current_git_provenance_mismatch", validation["reason_codes"])
        self.assertFalse((self.batch_root / "policy-training-readiness-review-summary.json").exists())

    def test_validate_only_blocks_missing_source_current_git_without_writing_summary(self) -> None:
        self._write_sources()
        coverage_path = self.batch_root / "channel-aware-contrast-coverage-summary.json"
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        coverage["git_provenance"].pop("current")
        coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--validate-only",
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        validation = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(validation["status"], "validation failed")
        self.assertIn("current_git_provenance_missing", validation["reason_codes"])
        self.assertIn(
            "channel_aware_contrast_coverage_summary_current_git_provenance_missing",
            validation["reason_codes"],
        )
        self.assertFalse((self.batch_root / "policy-training-readiness-review-summary.json").exists())

    def test_iterative_summary_validate_only_ignores_stale_unrequired_default_summaries(self) -> None:
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=1,
            candidate_nontrainable_count=0,
            contract_trainable_count=1,
            contract_nontrainable_count=0,
        )
        stale_git = self._git_snapshot(git_mismatch=True)
        for path in (candidate_path, contract_path):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["git_provenance"] = {
                "current": stale_git,
                "current_matches_sources": False,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        iterative_path = self.batch_root / "quasi-real-iterative-ppo-mini-loop-stability-summary.json"
        iterative_path.write_text(
            json.dumps(
                {
                    "schema_version": "iterative-ppo-mini-loop-stability-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "round_count": 3,
                    "failed_round_count": 0,
                    "stability_passed": True,
                    "min_optimizer_train_transition_count": 36,
                    "min_ppo_trainable_transition_count": 36,
                    "max_abs_approx_kl": 0.01,
                    "cumulative_parameter_l2_delta": 0.006,
                    "raw_test_regression_count": 0,
                    "sequential_rejected_count": 0,
                    "collector_regression_count": 0,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {
                        "current": self.git_snapshot,
                        "current_matches_sources": True,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--iterative-ppo-mini-loop-stability-summary",
            str(iterative_path),
            "--validate-only",
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        validation = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(
            validation["training_readiness_status"],
            "iterative_ppo_mini_loop_stability_evaluated",
        )
        self.assertEqual(validation["reason_codes"], [])

    def test_review_consumes_anchor_projection_summaries_and_blocks_regressed_contract(self) -> None:
        self._write_sources()
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=2,
            candidate_nontrainable_count=4,
            contract_trainable_count=1,
            contract_nontrainable_count=5,
            quality_regression_count=1,
            max_path_cost_margin=7.5,
            max_risk_margin=0.4,
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--anchor-projection-candidate-generation-summary",
            str(candidate_path),
            "--anchor-projection-evidence-contract-summary",
            str(contract_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        readiness = summary["anchor_projection_readiness"]
        self.assertEqual(readiness["candidate_generation_trainable_count"], 2)
        self.assertEqual(readiness["candidate_generation_nontrainable_count"], 4)
        self.assertEqual(readiness["contract_trainable_count"], 1)
        self.assertEqual(readiness["contract_nontrainable_count"], 5)
        self.assertEqual(readiness["readiness_trainable_count"], 1)
        self.assertEqual(readiness["candidate_contract_alignment_gap_count"], 1)
        self.assertEqual(readiness["anchor_unreachable_count"], 0)
        self.assertEqual(readiness["source_candidate_not_selected_count"], 0)
        self.assertEqual(readiness["audit_proxy_positive_count"], 0)
        self.assertIn(
            "anchor_projection_contract_trainable_count_below_candidate_generation",
            summary["training_blockers"],
        )
        self.assertIn(
            "anchor_projection_source_selection_quality_regression",
            summary["training_blockers"],
        )
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")

    def test_anchor_projection_readiness_reports_candidate_generation_gap_causes(self) -> None:
        self._write_sources()
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=18,
            candidate_nontrainable_count=60,
            contract_trainable_count=3,
            contract_nontrainable_count=39,
            anchor_unreachable_count=36,
            source_candidate_not_selected_count=24,
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--anchor-projection-candidate-generation-summary",
            str(candidate_path),
            "--anchor-projection-evidence-contract-summary",
            str(contract_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        readiness = summary["anchor_projection_readiness"]
        self.assertEqual(readiness["candidate_generation_trainable_count"], 18)
        self.assertEqual(readiness["contract_trainable_count"], 3)
        self.assertEqual(readiness["readiness_trainable_count"], 3)
        self.assertEqual(readiness["candidate_contract_alignment_gap_count"], 15)
        self.assertEqual(readiness["anchor_unreachable_count"], 36)
        self.assertEqual(readiness["source_candidate_not_selected_count"], 24)
        self.assertIn(
            "anchor_projection_contract_trainable_count_below_candidate_generation",
            readiness["training_blockers"],
        )
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")

    def test_anchor_projection_readiness_reports_reachability_aware_breakdown(self) -> None:
        self._write_sources()
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=20,
            candidate_nontrainable_count=58,
            contract_trainable_count=20,
            contract_nontrainable_count=58,
            anchor_unreachable_count=30,
            source_candidate_not_selected_count=28,
            reachable_substitute_anchor_found_count=6,
            anchor_unreachable_repaired_count=6,
            true_geometry_unreachable_count=30,
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--anchor-projection-candidate-generation-summary",
            str(candidate_path),
            "--anchor-projection-evidence-contract-summary",
            str(contract_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        readiness = summary["anchor_projection_readiness"]
        self.assertEqual(readiness["reachable_substitute_anchor_found_count"], 6)
        self.assertEqual(readiness["anchor_unreachable_repaired_by_reachable_substitute_count"], 6)
        self.assertEqual(readiness["true_geometry_unreachable_count"], 30)
        self.assertEqual(summary["anchor_projection_reachable_substitute_anchor_found_count"], 6)
        self.assertEqual(summary["anchor_projection_true_geometry_unreachable_count"], 30)

    def test_anchor_projection_readiness_preserves_distance_contract_calibration(self) -> None:
        self._write_sources()
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=18,
            candidate_nontrainable_count=60,
            contract_trainable_count=18,
            contract_nontrainable_count=60,
            source_candidate_not_selected_count=48,
            source_selected_but_distance_rejected_count=12,
            distance_contract_rejected_source_selected_count=12,
            distance_contract_rejected_by_distance_bin={
                "count": 36,
                "source_selected_count": 12,
                "not_source_selected_count": 24,
                "by_projection_distance_cells": {
                    "3": {
                        "count": 18,
                        "source_selected_count": 6,
                        "not_source_selected_count": 12,
                    }
                },
                "by_projection_distance_m": {
                    "1.5": {
                        "count": 18,
                        "source_selected_count": 6,
                        "not_source_selected_count": 12,
                    }
                },
            },
            source_candidate_not_selected_by_best_alternative_reason={
                "distance_contract_rejected": 24,
                "higher_path_cost": 24,
                "higher_path_cost_and_risk": 0,
                "higher_risk": 0,
                "lower_utility_or_coverage": 0,
                "ranking_weight_tradeoff_or_unobserved_utility": 0,
                "no_selected_candidate_comparison": 0,
            },
            source_selection_quality_tradeoff_summary={
                "generated_not_source_selected_count": 48,
                "source_selected_but_distance_rejected_count": 12,
                "distance_contract_rejected_count": 36,
                "source_candidate_not_selected_reason_counts": {
                    "distance_contract_rejected": 24,
                    "higher_path_cost": 24,
                    "higher_path_cost_and_risk": 0,
                    "higher_risk": 0,
                    "lower_utility_or_coverage": 0,
                    "ranking_weight_tradeoff_or_unobserved_utility": 0,
                    "no_selected_candidate_comparison": 0,
                },
                "distance_contract_relaxation_recommendation": (
                    "record_only_keep_current_training_distance_contract"
                ),
            },
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--anchor-projection-candidate-generation-summary",
            str(candidate_path),
            "--anchor-projection-evidence-contract-summary",
            str(contract_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        readiness = summary["anchor_projection_readiness"]
        self.assertEqual(readiness["source_selected_but_distance_rejected_count"], 12)
        self.assertEqual(readiness["distance_contract_rejected_source_selected_count"], 12)
        self.assertEqual(readiness["distance_contract_rejected_by_distance_bin"]["count"], 36)
        self.assertEqual(
            readiness["source_candidate_not_selected_by_best_alternative_reason"][
                "distance_contract_rejected"
            ],
            24,
        )
        self.assertEqual(
            readiness["source_selection_quality_tradeoff_summary"][
                "distance_contract_relaxation_recommendation"
            ],
            "record_only_keep_current_training_distance_contract",
        )
        self.assertEqual(summary["anchor_projection_source_selected_but_distance_rejected_count"], 12)
        self.assertEqual(summary["anchor_projection_distance_contract_rejected_source_selected_count"], 12)

    def test_review_can_run_anchor_only_with_candidate_and_contract_summaries(self) -> None:
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=1,
            candidate_nontrainable_count=1,
            contract_trainable_count=1,
            contract_nontrainable_count=1,
            source_candidate_not_selected_count=1,
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--anchor-projection-candidate-generation-summary",
            str(candidate_path),
            "--anchor-projection-evidence-contract-summary",
            str(contract_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["application_scope"], "anchor_projection_readiness_contract_review_only")
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")
        self.assertIn("anchor_projection_nontrainable_contexts_remain", summary["training_blockers"])
        self.assertEqual(summary["anchor_projection_readiness"]["readiness_trainable_count"], 1)
        self.assertEqual(summary["anchor_projection_readiness"]["source_candidate_not_selected_count"], 1)
        self.assertFalse(summary["runs_training"])
        self.assertTrue(summary["no_ppo_training"])

    def test_review_uses_contract_aware_ppo_consumable_summary_to_clear_anchor_blocker(self) -> None:
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=2,
            candidate_nontrainable_count=58,
            contract_trainable_count=2,
            contract_nontrainable_count=58,
            source_candidate_not_selected_count=46,
        )
        contract_aware_path = self.batch_root / "anchor-projection-contract-aware-trainable-target-summary.json"
        contract_aware_path.write_text(
            json.dumps(
                {
                    "schema_version": "anchor-projection-contract-aware-trainable-target-summary/v1",
                    "generated_at": "2026-06-08T00:00:00Z",
                    "status": "passed",
                    "reason_codes": [],
                    "contract_trainable_contrast_count": 2,
                    "ppo_consumable_trainable_target_count": 2,
                    "nontrainable_blocked_target_count": 58,
                    "nontrainable_blocked_target_count_delta": -2,
                    "distance_contract_rejected_count_delta": -2,
                    "source_candidate_not_selected_count_delta": -2,
                    "next_required_change": None,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                    "runs_training": False,
                    "audit_only": True,
                    "no_ppo_training": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--anchor-projection-candidate-generation-summary",
            str(candidate_path),
            "--anchor-projection-evidence-contract-summary",
            str(contract_path),
            "--contract-aware-trainable-target-summary",
            str(contract_aware_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["anchor_projection_ppo_consumable_trainable_target_count"], 2)
        self.assertNotIn("anchor_projection_nontrainable_contexts_remain", summary["training_blockers"])
        self.assertEqual(
            summary["training_readiness_status"],
            "ready_for_limited_policy_training_dry_run",
        )

    def test_review_keeps_anchor_blocker_when_contract_aware_main_success_fails(self) -> None:
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=2,
            candidate_nontrainable_count=60,
            contract_trainable_count=2,
            contract_nontrainable_count=60,
            source_candidate_not_selected_count=48,
        )
        contract_aware_path = self.batch_root / "anchor-projection-contract-aware-trainable-target-summary.json"
        contract_aware_path.write_text(
            json.dumps(
                {
                    "schema_version": "anchor-projection-contract-aware-trainable-target-summary/v1",
                    "generated_at": "2026-06-08T00:00:00Z",
                    "status": "passed",
                    "reason_codes": [],
                    "contract_trainable_contrast_count": 2,
                    "ppo_consumable_trainable_target_count": 2,
                    "nontrainable_blocked_target_count": 60,
                    "nontrainable_blocked_target_count_delta": 0,
                    "candidate_contract_alignment_gap_count": 0,
                    "main_success_gate_failures": [
                        "nontrainable_blocked_target_count_not_reduced",
                    ],
                    "next_required_change": "action_or_target_contract_change_required",
                    "readiness_impact": {
                        "recommended_training_blockers": [
                            "anchor_projection_nontrainable_contexts_remain",
                        ],
                    },
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                    "runs_training": False,
                    "audit_only": True,
                    "no_ppo_training": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--anchor-projection-candidate-generation-summary",
            str(candidate_path),
            "--anchor-projection-evidence-contract-summary",
            str(contract_path),
            "--contract-aware-trainable-target-summary",
            str(contract_aware_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["anchor_projection_ppo_consumable_trainable_target_count"], 2)
        self.assertIn("anchor_projection_nontrainable_contexts_remain", summary["training_blockers"])
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")

    def test_review_uses_planner_validated_mining_summary_to_clear_anchor_blocker(self) -> None:
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=18,
            candidate_nontrainable_count=60,
            contract_trainable_count=18,
            contract_nontrainable_count=60,
            source_candidate_not_selected_count=60,
        )
        mining_path = self.batch_root / "planner-validated-trainable-target-mining-summary.json"
        mining_path.write_text(
            json.dumps(
                {
                    "schema_version": "planner-validated-trainable-target-mining-summary/v1",
                    "generated_at": "2026-06-08T00:00:00Z",
                    "status": "passed",
                    "reason_codes": [],
                    "planner_validated_trainable_target_count": 24,
                    "default_contract_trainable_target_count": 18,
                    "planner_validated_distance_exception_count": 6,
                    "nontrainable_blocked_target_count": 54,
                    "nontrainable_blocked_target_count_delta": -6,
                    "distance_contract_blocked_count": 18,
                    "source_selection_not_selected_count": 30,
                    "quality_regression_rejected_count": 6,
                    "candidate_contract_alignment_gap_count": 0,
                    "main_success_gate_failures": [],
                    "next_required_change": None,
                    "readiness_impact": {
                        "recommended_training_blockers": [],
                    },
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                    "runs_training": False,
                    "audit_only": True,
                    "no_ppo_training": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--anchor-projection-candidate-generation-summary",
            str(candidate_path),
            "--anchor-projection-evidence-contract-summary",
            str(contract_path),
            "--planner-validated-trainable-target-mining-summary",
            str(mining_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["anchor_projection_ppo_consumable_trainable_target_count"], 24)
        self.assertEqual(summary["anchor_projection_planner_validated_trainable_target_count"], 24)
        self.assertEqual(summary["anchor_projection_planner_validated_distance_exception_count"], 6)
        self.assertNotIn("anchor_projection_nontrainable_contexts_remain", summary["training_blockers"])
        self.assertEqual(
            summary["training_readiness_status"],
            "ready_for_limited_policy_training_dry_run",
        )

    def test_review_records_hybrid_training_dry_run_completion_without_formal_training_claim(self) -> None:
        self._write_sources()
        hybrid_path = self.batch_root / "hybrid-policy-training-dry-run-summary.json"
        hybrid_path.write_text(
            json.dumps(
                {
                    "schema_version": "hybrid-policy-training-dry-run-summary/v1",
                    "generated_at": "2026-06-08T00:00:00Z",
                    "status": "passed",
                    "reason_codes": [],
                    "dry_run_status": "passed",
                    "action_label_positive_count": 24,
                    "existing_preference_pair_count": 24,
                    "residual_preference_pair_count": 30,
                    "pairwise_preference_signal_count": 54,
                    "hybrid_train_signal_count": 78,
                    "hard_positive_added_count": 0,
                    "invalid_action_mask_count": 0,
                    "empty_action_mask_count": 0,
                    "publishes_checkpoint": False,
                    "performance_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                    "runs_large_scale_training": False,
                    "dry_run_only": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--hybrid-policy-training-dry-run-summary",
            str(hybrid_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["training_readiness_status"], "hybrid_training_dry_run_completed")
        self.assertEqual(summary["recommended_next_action"], "hybrid_training_dry_run_completed")
        self.assertEqual(summary["hybrid_training_readiness"]["hybrid_train_signal_count"], 78)
        self.assertEqual(summary["hybrid_training_readiness"]["action_label_positive_count"], 24)
        self.assertEqual(summary["hybrid_training_readiness"]["pairwise_preference_signal_count"], 54)
        self.assertFalse(summary["hybrid_training_readiness"]["formal_training_ready_claimed"])
        self.assertTrue(summary["no_ppo_training"])
        self.assertFalse(summary["runs_training"])

    def test_review_auto_detects_anchor_only_mode_when_default_anchor_summaries_exist(self) -> None:
        self._write_anchor_projection_summaries(
            candidate_trainable_count=1,
            candidate_nontrainable_count=1,
            contract_trainable_count=1,
            contract_nontrainable_count=1,
            source_candidate_not_selected_count=1,
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["application_scope"], "anchor_projection_readiness_contract_review_only")
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")
        self.assertEqual(summary["reason_codes"], [])
        self.assertIn("anchor_projection_nontrainable_contexts_remain", summary["training_blockers"])

    def test_review_uses_default_config_when_config_argument_is_omitted(self) -> None:
        self._write_anchor_projection_summaries(
            candidate_trainable_count=1,
            candidate_nontrainable_count=1,
            contract_trainable_count=1,
            contract_nontrainable_count=1,
            source_candidate_not_selected_count=1,
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--validate-only",
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("configs/policy_training_readiness_review_v1.json", completed.stdout)
        self.assertIn("anchor_projection_nontrainable_contexts_remain", completed.stdout)

    def test_anchor_projection_margin_blocker_ignores_unselected_diagnostic_margin(self) -> None:
        self._write_sources()
        candidate_path, contract_path = self._write_anchor_projection_summaries(
            candidate_trainable_count=1,
            candidate_nontrainable_count=0,
            contract_trainable_count=1,
            contract_nontrainable_count=0,
            quality_regression_count=0,
        )
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate.pop("max_source_selection_path_cost_margin_vs_best_alternative", None)
        candidate.pop("max_source_selection_risk_margin_vs_best_alternative", None)
        candidate["context_records"] = [
            {
                "scenario_id": "unselected-diagnostic",
                "source_selection_status": "not_source_selected",
                "source_selection_quality_regression": False,
                "training_use": "not_positive_evidence",
                "source_selection_path_cost_margin_vs_best_alternative": 999.0,
                "source_selection_risk_margin_vs_best_alternative": 99.0,
            }
        ]
        candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        threshold_config = self.temp_dir / "policy-training-readiness-tight-margin.json"
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["readiness_thresholds"]["max_anchor_projection_path_cost_regression"] = 1.0
        config["readiness_thresholds"]["max_anchor_projection_risk_regression"] = 1.0
        threshold_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(threshold_config),
            "--anchor-projection-candidate-generation-summary",
            str(candidate_path),
            "--anchor-projection-evidence-contract-summary",
            str(contract_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        readiness = summary["anchor_projection_readiness"]
        self.assertIsNone(readiness["max_source_selection_path_cost_margin_vs_best_alternative"])
        self.assertIsNone(readiness["max_source_selection_risk_margin_vs_best_alternative"])
        self.assertEqual(readiness["diagnostic_max_source_selection_path_cost_margin_vs_best_alternative"], 999.0)
        self.assertEqual(readiness["diagnostic_max_source_selection_risk_margin_vs_best_alternative"], 99.0)
        self.assertNotIn(
            "anchor_projection_source_selection_path_cost_regression",
            summary["training_blockers"],
        )
        self.assertEqual(summary["training_readiness_status"], "ready_for_limited_policy_training_dry_run")

    def test_review_advances_to_scenario_disjoint_policy_candidate_evaluated(self) -> None:
        self._write_sources()
        candidate_path = self.batch_root / "controlled-hybrid-policy-training-candidate-summary.json"
        candidate_path.write_text(
            json.dumps(
                {
                    "schema_version": "controlled-hybrid-policy-training-candidate-summary/v1",
                    "generated_at": "2026-06-08T00:00:00Z",
                    "status": "passed",
                    "candidate_training_status": "passed",
                    "reason_codes": [],
                    "action_label_positive_count": 24,
                    "pairwise_preference_signal_count": 54,
                    "hybrid_train_signal_count": 78,
                    "hard_positive_added_count": 0,
                    "invalid_action_mask_count": 0,
                    "empty_action_mask_count": 0,
                    "experimental_checkpoint": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        fresh_path = self.batch_root / "fresh-holdout-policy-candidate-evaluation-summary.json"
        fresh_path.write_text(
            json.dumps(
                {
                    "schema_version": "fresh-holdout-policy-candidate-evaluation-summary/v1",
                    "generated_at": "2026-06-08T00:00:00Z",
                    "status": "passed",
                    "reason_codes": [],
                    "fresh_disjoint_context_count": 156,
                    "require_context_id": True,
                    "require_scenario_disjoint": True,
                    "raw_holdout_context_count": 156,
                    "identity_overlap_count": 0,
                    "identity_key_missing_count": 0,
                    "accepted_identity_overlap_count": 0,
                    "accepted_identity_key_missing_count": 0,
                    "scenario_overlap_count": 0,
                    "scenario_disjoint": True,
                    "context_id_missing_count": 0,
                    "legacy_identity_fallback_count": 0,
                    "context_id_coverage_rate": 1.0,
                    "fallback_or_open_grid_count": 0,
                    "safety_regression_count": 0,
                    "contract_violation_count": 0,
                    "path_cost_regression_count": 0,
                    "risk_regression_count": 0,
                    "source_selection_regression_count": 0,
                    "experimental_checkpoint": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "candidate_git_current_matches_sources": True,
                    "checkpoint_metadata_git_current_matches_sources": True,
                    "next_required_change": None,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--controlled-hybrid-policy-training-candidate-summary",
            str(candidate_path),
            "--fresh-holdout-policy-candidate-evaluation-summary",
            str(fresh_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(
            summary["training_readiness_status"],
            "scenario_disjoint_policy_candidate_evaluated",
        )
        self.assertEqual(
            summary["recommended_next_action"],
            "scenario_disjoint_policy_candidate_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])
        self.assertEqual(summary["next_required_change"], None)
        fresh_readiness = summary["fresh_holdout_policy_candidate_readiness"]
        self.assertTrue(fresh_readiness["scenario_disjoint"])
        self.assertEqual(fresh_readiness["legacy_identity_fallback_count"], 0)

    def test_review_blocks_failed_scenario_disjoint_context_id_gate(self) -> None:
        self._write_sources()
        fresh_path = self.batch_root / "fresh-holdout-policy-candidate-evaluation-summary.json"
        fresh_path.write_text(
            json.dumps(
                {
                    "schema_version": "fresh-holdout-policy-candidate-evaluation-summary/v1",
                    "generated_at": "2026-06-08T00:00:00Z",
                    "status": "failed",
                    "reason_codes": ["scenario_overlap", "legacy_identity_fallback_used"],
                    "fresh_disjoint_context_count": 1,
                    "require_context_id": True,
                    "require_scenario_disjoint": True,
                    "identity_overlap_count": 1,
                    "identity_key_missing_count": 0,
                    "accepted_identity_overlap_count": 0,
                    "accepted_identity_key_missing_count": 0,
                    "scenario_overlap_count": 1,
                    "scenario_disjoint": False,
                    "context_id_missing_count": 1,
                    "legacy_identity_fallback_count": 1,
                    "fallback_or_open_grid_count": 0,
                    "safety_regression_count": 0,
                    "contract_violation_count": 0,
                    "path_cost_regression_count": 0,
                    "risk_regression_count": 0,
                    "source_selection_regression_count": 0,
                    "experimental_checkpoint": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "next_required_change": "scenario_disjoint_holdout_generation_required",
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--fresh-holdout-policy-candidate-evaluation-summary",
            str(fresh_path),
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["training_readiness_status"], "blocked_by_validation")
        self.assertIn(
            "fresh_holdout_policy_candidate_evaluation_not_passed",
            summary["training_blockers"],
        )
        self.assertEqual(
            summary["next_required_change"],
            "scenario_disjoint_holdout_generation_required",
        )

    def test_generated_sequential_accounting_audit_refines_quasi_real_smoke_blocker_without_advancing_readiness(self) -> None:
        self._write_sources()
        limited_path = self.batch_root / "limited-quasi-real-ppo-update-smoke-summary.json"
        limited_path.write_text(
            json.dumps(
                {
                    "schema_version": "limited-quasi-real-ppo-update-smoke-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "input_ppo_trainable_transition_count": 36,
                    "optimizer_train_transition_count": 36,
                    "validation_test_optimizer_transition_count": 0,
                    "non_empty_gate_reason_optimizer_transition_count": 0,
                    "disallowed_source_optimizer_transition_count": 0,
                    "source_fallback_trainable_count": 0,
                    "loss_non_finite_count": 0,
                    "non_finite_gradient_count": 0,
                    "non_finite_reward_count": 0,
                    "non_finite_return_count": 0,
                    "non_finite_advantage_count": 0,
                    "old_log_prob_max_abs_error": 0.0,
                    "old_value_max_abs_error": 0.0,
                    "parameter_l2_delta": 0.001,
                    "approx_kl": 0.001,
                    "max_grad_norm_after_clip": 0.5,
                    "experimental_checkpoint": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "post_update_raw_generalization_status": "passed",
                    "post_update_sequential_canary_status": "failed",
                    "post_update_generated_collector_status": "failed",
                    "post_update_quasi_real_teacher_following_status": "passed",
                    "post_update_quasi_real_collector_status": "passed",
                    "post_update_raw_test_regression_count": 0,
                    "post_update_sequential_rejected_choice_count": 0,
                    "post_update_generated_collector_trainable_transition_count": 30,
                    "post_update_quasi_real_collector_trainable_transition_count": 36,
                    "post_update_quasi_real_teacher_agreement_rate": 1.0,
                    "post_update_quasi_real_unsafe_disagreement_count": 0,
                    "next_required_change": None,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        audit_path = self.batch_root / "generated-sequential-gate-metric-accounting-audit-summary.json"
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "generated-sequential-gate-metric-accounting-audit-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "legacy_mismatch_count": 6,
                    "raw_policy_path_cost_regression_count": 6,
                    "raw_policy_risk_regression_count": 2,
                    "controlled_path_cost_regression_count": 0,
                    "controlled_risk_regression_count": 0,
                    "diagnosis_verdict_after_origin_split": "pre_existing_generated_sequential_contract_mismatch",
                    "recommended_next_action": "generated_sequential_contract_alignment_required",
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--limited-quasi-real-ppo-update-smoke-summary",
            str(limited_path),
            "--generated-sequential-gate-metric-accounting-audit-summary",
            str(audit_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["training_readiness_status"], "needs_training_contract_refinement")
        self.assertEqual(
            summary["recommended_next_action"],
            "generated_sequential_contract_alignment_required",
        )
        self.assertIn(
            "generated_sequential_contract_alignment_required",
            summary["training_blockers"],
        )
        self.assertFalse(summary["generated_sequential_gate_metric_accounting_readiness"]["completed"])

    def test_long_horizon_teacher_skill_contract_alignment_unblocks_generated_sequential_contract(self) -> None:
        self._write_sources()
        limited_path = self.batch_root / "limited-quasi-real-ppo-update-smoke-summary.json"
        limited_path.write_text(
            json.dumps(
                {
                    "schema_version": "limited-quasi-real-ppo-update-smoke-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "input_ppo_trainable_transition_count": 36,
                    "optimizer_train_transition_count": 36,
                    "validation_test_optimizer_transition_count": 0,
                    "non_empty_gate_reason_optimizer_transition_count": 0,
                    "disallowed_source_optimizer_transition_count": 0,
                    "source_fallback_trainable_count": 0,
                    "loss_non_finite_count": 0,
                    "non_finite_gradient_count": 0,
                    "non_finite_reward_count": 0,
                    "non_finite_return_count": 0,
                    "non_finite_advantage_count": 0,
                    "old_log_prob_max_abs_error": 0.0,
                    "old_value_max_abs_error": 0.0,
                    "parameter_l2_delta": 0.001,
                    "approx_kl": 0.001,
                    "max_grad_norm_after_clip": 0.5,
                    "experimental_checkpoint": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "post_update_raw_generalization_status": "passed",
                    "post_update_sequential_canary_status": "failed",
                    "post_update_generated_collector_status": "failed",
                    "post_update_quasi_real_teacher_following_status": "passed",
                    "post_update_quasi_real_collector_status": "passed",
                    "post_update_raw_test_regression_count": 0,
                    "post_update_sequential_rejected_choice_count": 0,
                    "post_update_generated_collector_trainable_transition_count": 30,
                    "post_update_quasi_real_collector_trainable_transition_count": 36,
                    "post_update_quasi_real_teacher_agreement_rate": 1.0,
                    "post_update_quasi_real_unsafe_disagreement_count": 0,
                    "next_required_change": None,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        audit_path = self.batch_root / "generated-sequential-gate-metric-accounting-audit-summary.json"
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "generated-sequential-gate-metric-accounting-audit-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "legacy_mismatch_count": 6,
                    "diagnosis_verdict_after_origin_split": "pre_existing_generated_sequential_contract_mismatch",
                    "recommended_next_action": "generated_sequential_contract_alignment_required",
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        long_horizon_path = self.batch_root / "long-horizon-teacher-skill-contract-summary.json"
        long_horizon_path.write_text(
            json.dumps(
                {
                    "schema_version": "generated-sequential-long-horizon-teacher-skill-contract-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "verdict": "long_horizon_teacher_skill_contract_aligned",
                    "teacher_equivalent_episode_count": 12,
                    "beyond_teacher_episode_count": 4,
                    "controlled_regression_episode_count": 0,
                    "dominated_raw_choice_count": 6,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--limited-quasi-real-ppo-update-smoke-summary",
            str(limited_path),
            "--generated-sequential-gate-metric-accounting-audit-summary",
            str(audit_path),
            "--generated-sequential-long-horizon-teacher-skill-contract-summary",
            str(long_horizon_path),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            summary["training_readiness_status"],
            "limited_quasi_real_ppo_update_smoke_evaluated",
        )
        self.assertNotIn(
            "generated_sequential_contract_alignment_required",
            summary["training_blockers"],
        )
        self.assertTrue(
            summary[
                "generated_sequential_long_horizon_teacher_skill_contract_readiness"
            ]["completed"]
        )
        self.assertTrue(
            summary[
                "limited_quasi_real_generated_sequential_blocker_overridden_by_long_horizon_contract"
            ]
        )

    def test_return_aligned_guarded_ppo_update_smoke_summary_advances_readiness(self) -> None:
        update_path = self.batch_root / "return-aligned-guarded-ppo-update-smoke-summary.json"
        update_path.write_text(
            json.dumps(
                {
                    "schema_version": "return-aligned-guarded-ppo-update-smoke-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "input_return_aligned_trainable_transition_count": 30,
                    "optimizer_train_transition_count": 30,
                    "validation_test_optimizer_transition_count": 0,
                    "source_fallback_optimizer_transition_count": 0,
                    "non_empty_gate_reason_optimizer_transition_count": 0,
                    "source_fallback_trainable_count": 0,
                    "materialization_error_count": 0,
                    "old_log_prob_max_abs_error": 0.0,
                    "old_value_max_abs_error": 0.0,
                    "loss_non_finite_count": 0,
                    "non_finite_gradient_count": 0,
                    "non_finite_reward_count": 0,
                    "non_finite_return_count": 0,
                    "non_finite_advantage_count": 0,
                    "parameter_l2_delta": 0.001,
                    "approx_kl": 0.01,
                    "max_grad_norm_after_clip": 0.5,
                    "uses_multistep_discounted_return": True,
                    "not_single_step_best_action": True,
                    "post_update_gates_evaluated": True,
                    "post_update_raw_generalization_status": "passed",
                    "post_update_raw_test_regression_count": 0,
                    "post_update_generated_sequential_status": "failed",
                    "post_update_generated_collector_status": "passed",
                    "post_update_generated_collector_trainable_transition_count": 30,
                    "post_update_quasi_real_teacher_following_status": "passed",
                    "post_update_quasi_real_collector_status": "passed",
                    "post_update_quasi_real_collector_trainable_transition_count": 30,
                    "post_update_controlled_regression_count": 0,
                    "post_update_teacher_agreement_rate": 1.0,
                    "post_update_return_aligned_replay_status": "passed",
                    "post_update_return_aligned_replay_trainable_transition_count": 30,
                    "post_update_long_horizon_status": "passed",
                    "post_update_long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
                    "experimental_checkpoint": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--return-aligned-guarded-ppo-update-smoke-summary",
            str(update_path),
            "--validate-only",
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(
            summary["training_readiness_status"],
            "return_aligned_guarded_ppo_update_smoke_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])

    def test_quasi_real_guarded_ppo_rollout_pilot_summary_advances_readiness(self) -> None:
        pilot_path = self.batch_root / "quasi-real-guarded-ppo-rollout-pilot-summary.json"
        pilot_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-ppo-rollout-pilot-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "episode_count": 36,
                    "step_count": 108,
                    "trainable_transition_count": 30,
                    "ppo_trainable_transition_count": 30,
                    "diagnostic_transition_count": 78,
                    "validation_trainable_count": 0,
                    "test_trainable_count": 0,
                    "source_fallback_trainable_count": 0,
                    "teacher_fallback_trainable_count": 0,
                    "non_empty_gate_reason_trainable_count": 0,
                    "missing_observation_count": 0,
                    "missing_log_prob_count": 0,
                    "missing_value_count": 0,
                    "non_finite_reward_count": 0,
                    "non_finite_return_count": 0,
                    "non_finite_advantage_count": 0,
                    "controlled_regression_count": 0,
                    "controlled_safety_regression_count": 0,
                    "controlled_contract_regression_count": 0,
                    "controlled_path_risk_regression_count": 0,
                    "controlled_source_selection_regression_count": 0,
                    "teacher_agreement_rate": 1.0,
                    "quasi_real_collector_replay_status": "passed",
                    "quasi_real_collector_replay_trainable_transition_count": 30,
                    "post_pilot_long_horizon_status": "passed",
                    "post_pilot_long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
                    "uses_multistep_discounted_return": True,
                    "not_single_step_best_action": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--quasi-real-guarded-ppo-rollout-pilot-summary",
            str(pilot_path),
            "--validate-only",
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(
            summary["training_readiness_status"],
            "quasi_real_guarded_ppo_rollout_pilot_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])

    def test_quasi_real_guarded_ppo_stability_replay_summary_advances_readiness(self) -> None:
        stability_path = self.batch_root / "quasi-real-guarded-ppo-stability-replay-summary.json"
        stability_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-ppo-stability-replay-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "replay_count": 3,
                    "passed_replay_count": 3,
                    "episode_count": 36,
                    "step_count": 108,
                    "ppo_trainable_transition_count": 36,
                    "diagnostic_transition_count": 72,
                    "validation_trainable_count": 0,
                    "test_trainable_count": 0,
                    "source_fallback_trainable_count": 0,
                    "missing_observation_count": 0,
                    "missing_log_prob_count": 0,
                    "missing_value_count": 0,
                    "non_finite_reward_count": 0,
                    "non_finite_return_count": 0,
                    "non_finite_advantage_count": 0,
                    "controlled_regression_count": 0,
                    "controlled_safety_regression_count": 0,
                    "controlled_contract_regression_count": 0,
                    "controlled_path_risk_regression_count": 0,
                    "controlled_source_selection_regression_count": 0,
                    "teacher_agreement_rate": 1.0,
                    "baseline_replay_behavior_drift_count": 0,
                    "quasi_real_collector_replay_status": "passed",
                    "long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
                    "acceptance_contract_refined": True,
                    "runs_ppo_update": False,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--quasi-real-guarded-ppo-stability-replay-summary",
            str(stability_path),
            "--validate-only",
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(
            summary["training_readiness_status"],
            "quasi_real_guarded_ppo_stability_replay_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])

    def test_quasi_real_guarded_ppo_horizon5_batch_expansion_summary_advances_readiness(self) -> None:
        expansion_path = (
            self.batch_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json"
        )
        expansion_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "horizon": 5,
                    "episode_count": 96,
                    "step_count": 480,
                    "ppo_trainable_transition_count": 160,
                    "diagnostic_transition_count": 320,
                    "replay_count": 3,
                    "passed_replay_count": 3,
                    "baseline_replay_behavior_drift_count": 0,
                    "validation_trainable_count": 0,
                    "test_trainable_count": 0,
                    "source_fallback_trainable_count": 0,
                    "teacher_fallback_trainable_count": 0,
                    "missing_observation_count": 0,
                    "missing_log_prob_count": 0,
                    "missing_value_count": 0,
                    "non_finite_reward_count": 0,
                    "non_finite_return_count": 0,
                    "non_finite_advantage_count": 0,
                    "controlled_regression_count": 0,
                    "controlled_safety_regression_count": 0,
                    "controlled_contract_regression_count": 0,
                    "controlled_path_risk_regression_count": 0,
                    "controlled_source_selection_regression_count": 0,
                    "teacher_agreement_rate": 1.0,
                    "quasi_real_collector_replay_status": "passed",
                    "quasi_real_collector_replay_trainable_transition_count": 160,
                    "long_horizon_verdict": "long_horizon_teacher_skill_contract_aligned",
                    "uses_multistep_discounted_return": True,
                    "not_single_step_best_action": True,
                    "runs_ppo_update": False,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--quasi-real-guarded-ppo-horizon5-batch-expansion-summary",
            str(expansion_path),
            "--validate-only",
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(
            summary["training_readiness_status"],
            "quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(
            summary["quasi_real_guarded_ppo_horizon5_batch_expansion_readiness"][
                "completed"
            ]
        )

    def test_quasi_real_guarded_ppo_scale512_multiseed_preflight_summary_advances_readiness(self) -> None:
        preflight_path = (
            self.batch_root / "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json"
        )
        preflight_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "horizon": 5,
                    "ppo_trainable_transition_count": 512,
                    "unique_trainable_context_count": 512,
                    "validation_trainable_count": 0,
                    "test_trainable_count": 0,
                    "source_fallback_trainable_count": 0,
                    "teacher_fallback_trainable_count": 0,
                    "missing_observation_count": 0,
                    "missing_log_prob_count": 0,
                    "missing_value_count": 0,
                    "non_finite_reward_count": 0,
                    "non_finite_return_count": 0,
                    "non_finite_advantage_count": 0,
                    "controlled_regression_count": 0,
                    "controlled_safety_regression_count": 0,
                    "controlled_contract_regression_count": 0,
                    "controlled_path_risk_regression_count": 0,
                    "controlled_source_selection_regression_count": 0,
                    "teacher_agreement_rate": 1.0,
                    "seed_count": 3,
                    "passed_seed_count": 3,
                    "seed_failure_count": 0,
                    "seed_max_old_log_prob_abs_error": 0.0,
                    "seed_max_old_value_abs_error": 0.0,
                    "seed_loss_non_finite_count": 0,
                    "seed_non_finite_gradient_count": 0,
                    "seed_non_finite_reward_count": 0,
                    "seed_non_finite_return_count": 0,
                    "seed_non_finite_advantage_count": 0,
                    "seed_max_abs_approx_kl": 0.01,
                    "seed_max_grad_norm_after_clip": 0.5,
                    "min_post_update_guarded_collector_trainable_transition_count": 512,
                    "runs_formal_ppo_rollout": False,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--quasi-real-guarded-ppo-scale512-multiseed-preflight-summary",
            str(preflight_path),
            "--validate-only",
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(
            summary["training_readiness_status"],
            "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(
            summary["quasi_real_guarded_ppo_scale512_multiseed_preflight_readiness"][
                "completed"
            ]
        )

    def test_quasi_real_guarded_ppo_iterative_miniloop_summary_advances_readiness(self) -> None:
        iterative_path = (
            self.batch_root
            / "quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json"
        )
        iterative_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-ppo-iterative-miniloop-stability-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "input_trainable_transition_count": 684,
                    "ppo_trainable_transition_count": 684,
                    "unique_trainable_context_count": 684,
                    "seed_count": 3,
                    "iteration_count": 3,
                    "passed_iteration_count": 9,
                    "failed_iteration_count": 0,
                    "min_optimizer_train_transition_count": 684,
                    "validation_trainable_count": 0,
                    "test_trainable_count": 0,
                    "source_fallback_trainable_count": 0,
                    "teacher_fallback_trainable_count": 0,
                    "non_empty_gate_reason_trainable_count": 0,
                    "missing_observation_count": 0,
                    "missing_log_prob_count": 0,
                    "missing_value_count": 0,
                    "non_finite_reward_count": 0,
                    "non_finite_return_count": 0,
                    "non_finite_advantage_count": 0,
                    "loss_non_finite_count": 0,
                    "non_finite_gradient_count": 0,
                    "max_old_log_prob_abs_error": 0.0,
                    "max_old_value_abs_error": 0.0,
                    "max_abs_approx_kl": 0.01,
                    "max_grad_norm_after_clip": 0.5,
                    "min_teacher_agreement_rate": 1.0,
                    "controlled_regression_count": 0,
                    "controlled_safety_regression_count": 0,
                    "controlled_contract_regression_count": 0,
                    "controlled_path_risk_regression_count": 0,
                    "controlled_source_selection_regression_count": 0,
                    "behavior_drift_count": 0,
                    "runs_formal_ppo_rollout": False,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--quasi-real-guarded-ppo-iterative-miniloop-stability-summary",
            str(iterative_path),
            "--validate-only",
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(
            summary["training_readiness_status"],
            "quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(
            summary["quasi_real_guarded_ppo_iterative_miniloop_stability_readiness"][
                "completed"
            ]
        )

    def test_quasi_real_guarded_formal_ppo_preflight_summary_advances_readiness(self) -> None:
        preflight_path = (
            self.batch_root
            / "quasi-real-guarded-formal-ppo-preflight-summary.json"
        )
        preflight_path.write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-guarded-formal-ppo-preflight-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "input_trainable_transition_count": 684,
                    "optimizer_train_transition_count": 684,
                    "unique_trainable_context_count": 684,
                    "seed_count": 3,
                    "passed_seed_count": 3,
                    "validation_trainable_count": 0,
                    "test_trainable_count": 0,
                    "fallback_trainable_count": 0,
                    "source_fallback_trainable_count": 0,
                    "teacher_fallback_trainable_count": 0,
                    "non_empty_gate_reason_trainable_count": 0,
                    "missing_observation_count": 0,
                    "missing_log_prob_count": 0,
                    "missing_value_count": 0,
                    "non_finite_reward_count": 0,
                    "non_finite_return_count": 0,
                    "non_finite_advantage_count": 0,
                    "loss_non_finite_count": 0,
                    "non_finite_gradient_count": 0,
                    "max_old_log_prob_abs_error": 0.0,
                    "max_old_value_abs_error": 0.0,
                    "max_abs_approx_kl": 0.01,
                    "max_grad_norm_after_clip": 0.5,
                    "min_parameter_l2_delta": 0.001,
                    "teacher_agreement_rate": 1.0,
                    "controlled_regression_count": 0,
                    "controlled_safety_regression_count": 0,
                    "controlled_contract_regression_count": 0,
                    "controlled_path_risk_regression_count": 0,
                    "controlled_source_selection_regression_count": 0,
                    "rollback_manifest": "formal-preflight-rollback-manifest.json",
                    "runs_formal_ppo_rollout": False,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                    "formal_training_ready_claimed": False,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        completed = self._run_review(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--quasi-real-guarded-formal-ppo-preflight-summary",
            str(preflight_path),
            "--validate-only",
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(
            summary["training_readiness_status"],
            "quasi_real_guarded_formal_ppo_preflight_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(
            summary["quasi_real_guarded_formal_ppo_preflight_readiness"]["completed"]
        )


if __name__ == "__main__":
    unittest.main()
