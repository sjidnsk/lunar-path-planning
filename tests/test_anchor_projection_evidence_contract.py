import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class AnchorProjectionEvidenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_anchor_projection_evidence_contract.sh"
        self.config = self.repo_root / "configs" / "anchor_projection_evidence_contract_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anchor-projection-contract-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_contract(self, *args: str) -> subprocess.CompletedProcess[str]:
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
        records: list[dict],
        *,
        git_mismatch: bool = False,
        safety_regression_count: int = 0,
        fallback_count: int = 0,
    ) -> None:
        git_snapshot = self._git_snapshot(git_mismatch=git_mismatch)
        platform_mismatch_count = sum(
            1
            for record in records
            if record.get("platform_goal_contract_mismatch")
            or record.get("platform_goal_classification")
            in {
                "platform_inflated_goal_blocked",
                "original_goal_blocked",
                "out_of_bounds",
                "unknown_contract_mismatch",
            }
        )
        regeneration = {
            "schema_version": "goal-blocked-evidence-regeneration-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "needs_regeneration_input_count": len(records),
            "regenerated_record_count": len(records),
            "platform_goal_contract_mismatch_count": platform_mismatch_count,
            "platform_goal_trainable_anchor_projection_count": 0,
            "platform_goal_nontrainable_blocked_target_count": platform_mismatch_count,
            "platform_goal_anchor_available_count": sum(
                1 for record in records if record.get("platform_goal_anchor_available")
            ),
            "platform_goal_unresolved_count": sum(
                1
                for record in records
                if record.get("platform_goal_classification") == "unknown_contract_mismatch"
            ),
            "eligible_negative_evidence_candidate_count": 0,
            "still_unresolved_count": 0,
            "safety_regression_count": safety_regression_count,
            "fallback_or_open_grid_count": fallback_count,
            "audit_only": True,
            "no_ppo_training": True,
            "runs_training": False,
            "does_not_modify_default_astar": True,
            "does_not_modify_ppo": True,
            "does_not_modify_network": True,
            "does_not_modify_action_space": True,
            "does_not_modify_model_explorer_contract": True,
            "does_not_modify_path_planner_route_contract": True,
            "does_not_modify_path_planner_sidecar_contract": True,
            "no_ackermann_feasible_trajectory_claim": True,
            "regenerated_records": records,
            "git_provenance": {
                "current": git_snapshot,
                "current_matches_sources": not git_mismatch,
            },
        }
        review = {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "training_readiness_status": "needs_training_contract_refinement",
            "source_selected_candidate_changed_rate": 0.0,
            "calibrated_selected_candidate_changed_rate": 0.5,
            "training_positive_candidate_count": 1,
            "excluded_candidate_count": len(records),
            "safety_regression_count": safety_regression_count,
            "fallback_or_open_grid_count": fallback_count,
            "audit_only": True,
            "no_ppo_training": True,
            "runs_training": False,
            "git_provenance": {
                "current": git_snapshot,
                "current_matches_sources": not git_mismatch,
            },
        }
        (self.batch_root / "goal-blocked-evidence-regeneration-summary.json").write_text(
            json.dumps(regeneration, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "policy-training-readiness-review-summary.json").write_text(
            json.dumps(review, indent=2),
            encoding="utf-8",
        )

    def _platform_record(
        self,
        scenario_id: str,
        *,
        classification: str = "platform_inflated_goal_blocked",
        anchor_reachable: bool = False,
        training_use: str = "not_positive_evidence",
        comparison_scope: str = "audit_proxy_anchor_not_same_cell",
        distance_m: float | None = 0.5,
        distance_cells: int | None = 1,
    ) -> dict:
        anchor = [18, 5] if classification == "platform_inflated_goal_blocked" else None
        feasibility = {
            "schema_version": "platform-goal-feasibility/v1",
            "cell": [17, 5],
            "policy_target_cell": [17, 5],
            "execution_goal_cell": None,
            "contract_reachable": True,
            "original_passable": True,
            "inflated_passable": False,
            "blocked_by_platform_footprint": True,
            "nearest_inflated_passable_anchor": anchor,
            "anchor_distance_cells": distance_cells,
            "anchor_distance_m": distance_m,
            "classification": classification,
            "anchor_projection": {
                "nearest_inflated_passable_anchor": anchor,
                "projection_distance_cells": distance_cells,
                "projection_distance_m": distance_m,
                "anchor_reachable": anchor_reachable,
                "comparison_scope": comparison_scope,
                "scope": comparison_scope,
                "same_cell_positive_evidence": False,
                "training_use": training_use,
                "evidence_boundary": "explicit_anchor_projection_contract"
                if training_use != "not_positive_evidence"
                else "audit_projection_not_same_cell_positive_evidence",
            },
            "proxy_route_comparison": {
                "scope": comparison_scope,
                "anchor_route_feasible": anchor_reachable,
                "same_cell_positive_evidence": False,
            },
        }
        return {
            "scenario_id": scenario_id,
            "pair_key": "all-all-k3",
            "action_index": 0,
            "cell": [17, 5],
            "diagnostic_decision": "platform_goal_contract_mismatch",
            "failure_category": classification,
            "platform_goal_contract_mismatch": True,
            "platform_goal_classification": classification,
            "platform_goal_anchor_available": anchor is not None,
            "platform_goal_feasibility": feasibility,
            "eligible_negative_evidence_candidate": False,
            "reason_codes": ["goal_blocked", classification],
        }

    def _negative_audit_proxy_record(self) -> dict:
        return {
            "scenario_id": "negative-audit-proxy",
            "pair_key": "all-all-k3",
            "action_index": 0,
            "cell": [17, 5],
            "diagnostic_decision": "eligible_negative_evidence_candidate",
            "failure_category": None,
            "eligible_negative_evidence_candidate": True,
            "platform_goal_feasibility": {
                "schema_version": "platform-goal-feasibility/v1",
                "classification": "goal_passable",
                "policy_target_cell": [17, 5],
                "execution_goal_cell": [17, 5],
                "anchor_projection": {
                    "comparison_scope": "audit_proxy_anchor_not_same_cell",
                    "scope": "audit_proxy_anchor_not_same_cell",
                    "training_use": "not_positive_evidence",
                    "same_cell_positive_evidence": False,
                },
            },
            "reason_codes": ["goal_blocked"],
        }

    def _write_candidate_generation_summary_only(self) -> Path:
        path = self.batch_root / "anchor-projection-candidate-generation-summary.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "anchor-projection-candidate-generation-summary/v1",
                    "generated_at": "2026-06-07T00:00:00Z",
                    "status": "passed",
                    "reason_codes": [],
                    "platform_goal_contract_mismatch_count": 2,
                    "trainable_anchor_projection_count": 1,
                    "nontrainable_blocked_target_count": 1,
                    "positive_training_evidence_contains_audit_proxy_anchor_count": 0,
                    "anchor_projection_coverage_diagnosis": {
                        "nontrainable_primary_reason_counts": {
                            "source_candidate_not_selected": 1
                        },
                        "projection_distance_contract_rejected_count": 1,
                    },
                    "source_selected_but_distance_rejected_count": 1,
                    "distance_contract_rejected_source_selected_count": 1,
                    "distance_contract_rejected_by_distance_bin": {
                        "count": 1,
                        "source_selected_count": 1,
                        "not_source_selected_count": 0,
                        "by_projection_distance_cells": {
                            "3": {
                                "count": 1,
                                "source_selected_count": 1,
                                "not_source_selected_count": 0,
                                "scenario_id_counts": {"trainable": 1},
                                "run_id_counts": {"all-all-k3-astar": 1},
                            }
                        },
                        "by_projection_distance_m": {
                            "1.5": {
                                "count": 1,
                                "source_selected_count": 1,
                                "not_source_selected_count": 0,
                                "scenario_id_counts": {"trainable": 1},
                                "run_id_counts": {"all-all-k3-astar": 1},
                            }
                        },
                    },
                    "source_candidate_not_selected_by_best_alternative_reason": {
                        "distance_contract_rejected": 0,
                        "higher_path_cost": 1,
                        "higher_path_cost_and_risk": 0,
                        "higher_risk": 0,
                        "lower_utility_or_coverage": 0,
                        "ranking_weight_tradeoff_or_unobserved_utility": 0,
                        "no_selected_candidate_comparison": 0,
                    },
                    "source_selection_quality_tradeoff_summary": {
                        "generated_not_source_selected_count": 1,
                        "source_selected_but_distance_rejected_count": 1,
                        "distance_contract_rejected_count": 1,
                        "source_candidate_not_selected_reason_counts": {
                            "distance_contract_rejected": 0,
                            "higher_path_cost": 1,
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
                    "context_records": [
                        {
                            "run_id": "all-all-k3-astar",
                            "scenario_id": "trainable",
                            "source_action_index": 0,
                            "generated_action_index": 2,
                            "policy_target_cell": [2, 1],
                            "execution_goal_cell": [1, 1],
                            "projected_anchor_cell": [1, 1],
                            "classification": "platform_inflated_goal_blocked",
                            "anchor_available": True,
                            "anchor_reachable": True,
                            "projected_candidate_generated": True,
                            "projected_candidate_source_selected": True,
                            "trainable": True,
                            "training_use": "trainable_anchor_projection_contrast",
                            "comparison_scope": "projected_target_anchor_contrast",
                            "projection_distance_m": 0.5,
                            "projection_distance_cells": 1.0,
                            "positive_audit_proxy": False,
                            "source_selection_quality_regression": False,
                            "reject_reasons": ["audit_proxy_scope_not_positive_evidence"],
                        },
                        {
                            "run_id": "all-all-k3-astar",
                            "scenario_id": "not-selected",
                            "source_action_index": 1,
                            "generated_action_index": 3,
                            "policy_target_cell": [4, 1],
                            "execution_goal_cell": [3, 1],
                            "projected_anchor_cell": [3, 1],
                            "classification": "platform_inflated_goal_blocked",
                            "anchor_available": True,
                            "anchor_reachable": True,
                            "projected_candidate_generated": True,
                            "projected_candidate_source_selected": False,
                            "trainable": False,
                            "training_use": "not_positive_evidence",
                            "comparison_scope": "projected_target_anchor_contrast",
                            "projection_distance_m": 0.5,
                            "projection_distance_cells": 1,
                            "positive_audit_proxy": False,
                            "source_selection_quality_regression": False,
                            "reject_reasons": ["source_candidate_not_selected"],
                        },
                    ],
                    "git_provenance": {
                        "current": self.git_snapshot,
                        "current_matches_sources": True,
                    },
                    "audit_only": True,
                    "runs_training": False,
                    "no_ppo_training": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def test_contract_splits_trainable_projection_from_audit_only_proxy(self) -> None:
        self._write_sources(
            [
                self._platform_record("audit-proxy", anchor_reachable=True),
                self._platform_record(
                    "trainable-projection",
                    anchor_reachable=True,
                    training_use="trainable_anchor_projection_contrast",
                    comparison_scope="projected_target_anchor_contrast",
                ),
            ]
        )

        completed = self._run_contract("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-evidence-contract-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["schema_version"], "anchor-projection-evidence-contract-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["platform_goal_contract_mismatch_count"], 2)
        self.assertEqual(summary["trainable_anchor_projection_count"], 1)
        self.assertEqual(summary["platform_goal_trainable_anchor_projection_count"], 1)
        self.assertEqual(summary["nontrainable_blocked_target_count"], 1)
        self.assertEqual(summary["platform_goal_nontrainable_blocked_target_count"], 1)
        self.assertEqual(summary["platform_goal_unresolved_count"], 0)
        self.assertEqual(summary["positive_training_evidence_contains_audit_proxy_anchor_count"], 0)
        decisions = {item["scenario_id"]: item for item in summary["anchor_projection_decisions"]}
        self.assertEqual(
            decisions["trainable-projection"]["contract_decision"],
            "trainable_anchor_projection_contrast",
        )
        self.assertEqual(
            decisions["audit-proxy"]["contract_decision"],
            "nontrainable_blocked_target",
        )
        self.assertIn(
            "source_training_use_not_trainable",
            decisions["audit-proxy"]["reject_reasons"],
        )
        self.assertFalse(summary["runs_training"])
        self.assertTrue(summary["audit_only"])
        self.assertTrue(summary["no_ppo_training"])

    def test_contract_can_classify_candidate_generation_summary_without_old_audit_sources(self) -> None:
        self._write_candidate_generation_summary_only()

        completed = self._run_contract("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-evidence-contract-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["contract_source"], "anchor_projection_candidate_generation_summary")
        self.assertEqual(summary["platform_goal_contract_mismatch_count"], 2)
        self.assertEqual(summary["trainable_anchor_projection_count"], 1)
        self.assertEqual(summary["nontrainable_blocked_target_count"], 1)
        self.assertEqual(summary["candidate_contract_alignment_gap_count"], 0)
        decisions = {item["scenario_id"]: item for item in summary["anchor_projection_decisions"]}
        self.assertEqual(decisions["trainable"]["sample_weight"], 1.0)
        self.assertEqual(decisions["not-selected"]["sample_weight"], 0.0)

    def test_contract_preserves_reachability_aware_candidate_diagnosis(self) -> None:
        candidate_path = self._write_candidate_generation_summary_only()
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["reachable_substitute_anchor_found_count"] = 1
        candidate["anchor_unreachable_repaired_by_reachable_substitute_count"] = 1
        candidate["true_geometry_unreachable_count"] = 1
        candidate["anchor_projection_coverage_diagnosis"].update(
            {
                "anchor_selection_status_counts": {
                    "reachable_substitute_anchor_found": 1,
                    "true_geometry_unreachable": 1,
                },
                "reachable_substitute_anchor_found_count": 1,
                "anchor_unreachable_repaired_by_reachable_substitute_count": 1,
                "true_geometry_unreachable_count": 1,
            }
        )
        candidate["context_records"][0].update(
            {
                "nearest_anchor_reachable": False,
                "anchor_selection_status": "reachable_substitute_anchor_found",
                "start_component_id": 0,
                "nearest_anchor_component_id": 1,
                "projected_anchor_component_id": 0,
                "reachable_substitute_anchor_available": True,
                "reachable_substitute_anchor_count": 5,
            }
        )
        candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")

        completed = self._run_contract("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-evidence-contract-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["reachable_substitute_anchor_found_count"], 1)
        self.assertEqual(summary["anchor_unreachable_repaired_by_reachable_substitute_count"], 1)
        self.assertEqual(summary["true_geometry_unreachable_count"], 1)
        decisions = {item["scenario_id"]: item for item in summary["anchor_projection_decisions"]}
        self.assertEqual(
            decisions["trainable"]["anchor_selection_status"],
            "reachable_substitute_anchor_found",
        )
        self.assertEqual(decisions["trainable"]["start_component_id"], 0)
        self.assertEqual(decisions["trainable"]["projected_anchor_component_id"], 0)

    def test_contract_preserves_source_selection_distance_contract_calibration(self) -> None:
        self._write_candidate_generation_summary_only()

        completed = self._run_contract("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-evidence-contract-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["source_selected_but_distance_rejected_count"], 1)
        self.assertEqual(summary["distance_contract_rejected_source_selected_count"], 1)
        self.assertEqual(summary["distance_contract_rejected_by_distance_bin"]["count"], 1)
        self.assertEqual(
            summary["source_candidate_not_selected_by_best_alternative_reason"]["higher_path_cost"],
            1,
        )
        self.assertEqual(
            summary["source_selection_quality_tradeoff_summary"][
                "distance_contract_relaxation_recommendation"
            ],
            "record_only_keep_current_training_distance_contract",
        )

    def test_contract_fails_when_platform_goal_classification_is_unresolved(self) -> None:
        self._write_sources([self._platform_record("unknown", classification="unknown_contract_mismatch")])

        completed = self._run_contract("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-evidence-contract-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["platform_goal_unresolved_count"], 1)
        self.assertIn("anchor_projection_records_unresolved", summary["contract_blockers"])
        self.assertIn("platform_goal_unresolved_count_exceeds_threshold", summary["reason_codes"])

    def test_validate_only_reports_current_git_mismatch_counts_without_writing(self) -> None:
        self._write_sources([self._platform_record("stale")], git_mismatch=True)

        completed = self._run_contract(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--validate-only",
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        validation = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(validation["status"], "validation failed")
        self.assertGreater(validation["current_git_provenance_mismatch_count"], 0)
        self.assertGreater(validation["git_provenance_mismatch_count"], 0)
        self.assertIn("current_git_provenance_mismatch", validation["reason_codes"])
        self.assertFalse((self.batch_root / "anchor-projection-evidence-contract-summary.json").exists())

    def test_validate_only_reports_missing_current_git_without_writing(self) -> None:
        self._write_sources([self._platform_record("missing-current")])
        regeneration_path = self.batch_root / "goal-blocked-evidence-regeneration-summary.json"
        regeneration = json.loads(regeneration_path.read_text(encoding="utf-8"))
        regeneration["git_provenance"].pop("current")
        regeneration_path.write_text(json.dumps(regeneration, indent=2), encoding="utf-8")

        completed = self._run_contract(
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
            "goal_blocked_evidence_regeneration_summary_current_git_provenance_missing",
            validation["reason_codes"],
        )
        self.assertFalse((self.batch_root / "anchor-projection-evidence-contract-summary.json").exists())

    def test_contract_blocks_fallback_and_safety_regression(self) -> None:
        for kwargs, expected_reason in (
            ({"fallback_count": 1}, "fallback_or_open_grid_blocks_anchor_projection_contract"),
            ({"safety_regression_count": 1}, "safety_regression_blocks_anchor_projection_contract"),
        ):
            with self.subTest(expected_reason=expected_reason):
                shutil.rmtree(self.batch_root)
                self.batch_root.mkdir(parents=True)
                self._write_sources([self._platform_record("blocked")], **kwargs)

                completed = self._run_contract("--batch-root", str(self.batch_root), "--config", str(self.config))

                self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                summary = json.loads(
                    (self.batch_root / "anchor-projection-evidence-contract-summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertIn(expected_reason, summary["contract_blockers"])

    def test_contract_blocks_negative_evidence_from_audit_proxy_scope(self) -> None:
        self._write_sources([self._negative_audit_proxy_record()])

        completed = self._run_contract("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-evidence-contract-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["negative_evidence_scope_violation_count"], 1)
        self.assertIn("negative_evidence_scope_violation", summary["contract_blockers"])


if __name__ == "__main__":
    unittest.main()
