import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class AnchorProjectionCandidateGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_anchor_projection_candidate_generation.sh"
        self.config = self.repo_root / "configs" / "anchor_projection_candidate_generation_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anchor-projection-candidates-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.script), "--batch-root", str(self.batch_root), "--config", str(self.config), *args],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _current_git(self) -> dict:
        parent = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_root,
            text=True,
        ).strip()
        submodules = {}
        for name in ("dev-platform-constraints", "model-explorer", "path-planner"):
            sha = subprocess.check_output(
                ["git", "-C", name, "rev-parse", "HEAD"],
                cwd=self.repo_root,
                text=True,
            ).strip()
            submodules[name] = {"path": name, "sha": sha}
        return {
            "parent": {"path": ".", "sha": parent},
            "submodules": submodules,
        }

    def _write_batch(
        self,
        *,
        stale_git: bool = False,
        open_grid_count: int = 0,
        safety_regression_count: int = 0,
        summary: dict | None = None,
    ) -> None:
        git = self._current_git()
        if stale_git:
            git["parent"]["sha"] = "0" * 40
        run_root = self.batch_root / "run-a"
        run_root.mkdir()
        summary_path = run_root / "path-feedback-summary.json"
        summary_path.write_text(
            json.dumps(summary or self._path_feedback_summary(), indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "batch-run-index.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-batch-run-index/v1",
                    "git": git,
                    "runs": [
                        {
                            "run_id": "run-a",
                            "status": "passed",
                            "source_paths": {
                                "summary": str(summary_path.relative_to(self.repo_root))
                                if summary_path.is_relative_to(self.repo_root)
                                else str(summary_path)
                            },
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.batch_root / "batch-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-batch-evaluation-summary/v1",
                    "failed_count": 0,
                    "open_grid_fallback_used_count": open_grid_count,
                    "safety_regression_count": safety_regression_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _path_feedback_summary(self) -> dict:
        return {
            "schema_version": "path-feedback-summary/v1",
            "scenario_count": 2,
            "open_grid_fallback_used": False,
            "tracking_safety_violation_count": 0,
            "scenarios": [
                {
                    "scenario_id": "trainable",
                    "selected_cell_before_path_feedback": [2, 1],
                    "selected_cell_after_path_feedback": [1, 1],
                    "selection_changed_by_path_feedback": True,
                    "path_feedback": {
                        "candidates": [
                            self._blocked_candidate(action_index=0),
                            self._projected_candidate(action_index=2, source_selected=True),
                            self._normal_candidate(action_index=1, path_cost=8.0, risk=0.4),
                        ]
                    },
                },
                {
                    "scenario_id": "nontrainable",
                    "selected_cell_before_path_feedback": [2, 1],
                    "selected_cell_after_path_feedback": [0, 2],
                    "selection_changed_by_path_feedback": True,
                    "path_feedback": {
                        "candidates": [
                            self._blocked_candidate(action_index=0),
                            self._projected_candidate(action_index=2, source_selected=False),
                            self._normal_candidate(action_index=1, path_cost=3.0, risk=0.2),
                        ]
                    },
                },
            ],
        }

    def _blocked_candidate(self, *, action_index: int) -> dict:
        return {
            "action_index": action_index,
            "cell": [2, 1],
            "candidate_role": "policy_target",
            "policy_target_cell": [2, 1],
            "execution_goal_cell": None,
            "platform_goal_feasibility": {
                "classification": "platform_inflated_goal_blocked",
                "policy_target_cell": [2, 1],
                "execution_goal_cell": None,
                "nearest_inflated_passable_anchor": [1, 1],
                "anchor_projection": {
                    "projected_anchor_cell": [1, 1],
                    "anchor_reachable": True,
                    "comparison_scope": "audit_proxy_anchor_not_same_cell",
                    "training_use": "not_positive_evidence",
                    "sample_weight": 0.0,
                    "reject_reason": "audit_proxy_scope_not_positive_evidence",
                    "evidence_boundary": "audit_projection_not_same_cell_positive_evidence",
                },
            },
        }

    def _projected_candidate(self, *, action_index: int, source_selected: bool) -> dict:
        training_use = "trainable_anchor_projection_contrast" if source_selected else "not_positive_evidence"
        sample_weight = 1.0 if source_selected else 0.0
        reject_reason = None if source_selected else "source_candidate_not_selected"
        selection_status = "source_selected" if source_selected else "not_source_selected"
        generation = {
            "schema_version": "anchor-projection-candidate/v1",
            "candidate_role": "projected_execution_target",
            "source_action_index": 0,
            "policy_target_cell": [2, 1],
            "execution_goal_cell": [1, 1],
            "projected_anchor_cell": [1, 1],
            "projection_distance_cells": 1,
            "projection_distance_m": 1.0,
            "anchor_reachable": True,
            "comparison_scope": "projected_target_anchor_contrast",
            "training_use": training_use,
            "sample_weight": sample_weight,
            "reject_reason": reject_reason,
            "source_selection_status": selection_status,
            "evidence_boundary": "source_selected_projected_target_candidate"
            if source_selected
            else "source_candidate_not_selected_not_positive_evidence",
        }
        if source_selected:
            generation.update(
                {
                    "source_selection_best_alternative_scope": (
                        "reachable_non_replan_candidates_including_policy_and_projected_targets"
                    ),
                    "source_selection_best_alternative_candidate_role": "policy_target",
                }
            )
        return {
            "action_index": action_index,
            "source_action_index": 0,
            "cell": [1, 1],
            "candidate_role": "projected_execution_target",
            "policy_target_cell": [2, 1],
            "execution_goal_cell": [1, 1],
            "utility": 0.8,
            "reachable": True,
            "replan_required": False,
            "path_cost": 5.0,
            "risk": 0.3,
            "candidate_generation": dict(generation),
            "platform_goal_feasibility": {
                "classification": "platform_inflated_goal_blocked",
                "policy_target_cell": [2, 1],
                "execution_goal_cell": [1, 1],
                "nearest_inflated_passable_anchor": [1, 1],
                "anchor_projection": dict(generation),
            },
        }

    def _normal_candidate(self, *, action_index: int, path_cost: float, risk: float) -> dict:
        return {
            "action_index": action_index,
            "cell": [0, 2],
            "candidate_role": "policy_target",
            "policy_target_cell": [0, 2],
            "execution_goal_cell": [0, 2],
            "utility": 0.4,
            "reachable": True,
            "replan_required": False,
            "path_cost": path_cost,
            "risk": risk,
            "platform_goal_feasibility": {
                "classification": "platform_goal_reachable",
                "policy_target_cell": [0, 2],
                "execution_goal_cell": [0, 2],
            },
        }

    def _reachability_diagnosis_summary(self) -> dict:
        substitute_generation = {
            "schema_version": "anchor-projection-candidate/v1",
            "candidate_role": "projected_execution_target",
            "source_action_index": 0,
            "policy_target_cell": [3, 2],
            "execution_goal_cell": [0, 2],
            "projected_anchor_cell": [0, 2],
            "nearest_inflated_passable_anchor": [4, 2],
            "projection_distance_cells": 3,
            "projection_distance_m": 3.0,
            "nearest_anchor_distance_cells": 1,
            "nearest_anchor_distance_m": 1.0,
            "anchor_reachable": True,
            "nearest_anchor_reachable": False,
            "anchor_selection_status": "reachable_substitute_anchor_found",
            "start_component_id": 0,
            "target_component_id": None,
            "nearest_anchor_component_id": 1,
            "projected_anchor_component_id": 0,
            "start_component_size": 5,
            "nearest_anchor_component_size": 3,
            "projected_anchor_component_size": 5,
            "reachable_substitute_anchor_available": True,
            "reachable_substitute_anchor_count": 5,
            "comparison_scope": "projected_target_anchor_contrast",
            "training_use": "trainable_anchor_projection_contrast",
            "sample_weight": 1.0,
            "reject_reason": None,
            "source_selection_status": "source_selected",
            "evidence_boundary": "source_selected_projected_target_candidate",
        }
        true_geometry_projection = {
            "projected_anchor_cell": [4, 2],
            "nearest_inflated_passable_anchor": [4, 2],
            "projection_distance_cells": 1,
            "projection_distance_m": 1.0,
            "nearest_anchor_distance_cells": 1,
            "nearest_anchor_distance_m": 1.0,
            "anchor_reachable": False,
            "nearest_anchor_reachable": False,
            "anchor_selection_status": "true_geometry_unreachable",
            "start_component_id": 0,
            "target_component_id": None,
            "nearest_anchor_component_id": 1,
            "projected_anchor_component_id": 1,
            "start_component_size": 5,
            "nearest_anchor_component_size": 3,
            "projected_anchor_component_size": 3,
            "reachable_substitute_anchor_available": False,
            "reachable_substitute_anchor_count": 0,
            "comparison_scope": "audit_proxy_anchor_not_same_cell",
            "training_use": "not_positive_evidence",
            "sample_weight": 0.0,
            "reject_reason": "anchor_not_reachable",
            "source_selection_status": "not_source_candidate",
            "evidence_boundary": "audit_projection_not_same_cell_positive_evidence",
        }
        return {
            "schema_version": "path-feedback-summary/v1",
            "scenario_count": 2,
            "open_grid_fallback_used": False,
            "tracking_safety_violation_count": 0,
            "scenarios": [
                {
                    "scenario_id": "substitute",
                    "selected_cell_before_path_feedback": [3, 2],
                    "selected_cell_after_path_feedback": [0, 2],
                    "selection_changed_by_path_feedback": True,
                    "path_feedback": {
                        "candidates": [
                            {
                                "action_index": 0,
                                "cell": [3, 2],
                                "candidate_role": "policy_target",
                                "policy_target_cell": [3, 2],
                                "execution_goal_cell": None,
                                "platform_goal_feasibility": {
                                    "classification": "platform_inflated_goal_blocked",
                                    "policy_target_cell": [3, 2],
                                    "execution_goal_cell": None,
                                    "nearest_inflated_passable_anchor": [4, 2],
                                    "anchor_projection": dict(substitute_generation, training_use="not_positive_evidence", sample_weight=0.0, reject_reason="audit_proxy_scope_not_positive_evidence", source_selection_status="not_source_candidate"),
                                },
                            },
                            {
                                "action_index": 2,
                                "source_action_index": 0,
                                "cell": [0, 2],
                                "candidate_role": "projected_execution_target",
                                "policy_target_cell": [3, 2],
                                "execution_goal_cell": [0, 2],
                                "reachable": True,
                                "replan_required": False,
                                "path_cost": 2.0,
                                "risk": 0.1,
                                "utility": 0.9,
                                "candidate_generation": dict(substitute_generation),
                                "platform_goal_feasibility": {
                                    "classification": "platform_inflated_goal_blocked",
                                    "policy_target_cell": [3, 2],
                                    "execution_goal_cell": [0, 2],
                                    "nearest_inflated_passable_anchor": [4, 2],
                                    "anchor_projection": dict(substitute_generation),
                                },
                            },
                        ]
                    },
                },
                {
                    "scenario_id": "true-geometry",
                    "selected_cell_before_path_feedback": [3, 2],
                    "selected_cell_after_path_feedback": [0, 4],
                    "selection_changed_by_path_feedback": True,
                    "path_feedback": {
                        "candidates": [
                            {
                                "action_index": 0,
                                "cell": [3, 2],
                                "candidate_role": "policy_target",
                                "policy_target_cell": [3, 2],
                                "execution_goal_cell": None,
                                "platform_goal_feasibility": {
                                    "classification": "platform_inflated_goal_blocked",
                                    "policy_target_cell": [3, 2],
                                    "execution_goal_cell": None,
                                    "nearest_inflated_passable_anchor": [4, 2],
                                    "anchor_projection": true_geometry_projection,
                                },
                            }
                        ]
                    },
                },
            ],
        }

    def test_summary_counts_trainable_source_selected_projection_without_audit_proxy_positive(self) -> None:
        self._write_batch()

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-candidate-generation-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["current_git_provenance_mismatch_count"], 0)
        self.assertEqual(summary["git_provenance_mismatch_count"], 0)
        self.assertEqual(summary["platform_goal_contract_mismatch_count"], 2)
        self.assertEqual(summary["trainable_anchor_projection_count"], 1)
        self.assertEqual(summary["nontrainable_blocked_target_count"], 1)
        self.assertEqual(summary["nontrainable_anchor_unreachable_count"], 0)
        self.assertEqual(summary["nontrainable_source_candidate_not_selected_count"], 1)
        self.assertEqual(summary["audit_proxy_positive_count"], 0)
        self.assertEqual(summary["source_selected_candidate_changed_rate"], 0.5)
        self.assertEqual(summary["positive_training_evidence_contains_audit_proxy_anchor_count"], 0)
        self.assertEqual(
            summary["trainable_anchor_projection_count"] + summary["nontrainable_blocked_target_count"],
            summary["platform_goal_contract_mismatch_count"],
        )
        diagnosis = summary["anchor_projection_coverage_diagnosis"]
        self.assertEqual(diagnosis["projected_candidate_generated_count"], 2)
        self.assertEqual(diagnosis["projected_candidate_source_selected_count"], 1)
        self.assertEqual(diagnosis["projected_candidate_not_source_selected_count"], 1)
        self.assertEqual(diagnosis["anchor_unreachable_not_generated_count"], 0)
        self.assertEqual(diagnosis["nontrainable_primary_reason_counts"], {"source_candidate_not_selected": 1})
        self.assertEqual(diagnosis["scenario_diagnosis_counts"]["nontrainable"], 1)
        self.assertEqual(diagnosis["projection_distance_cells"]["count"], 2)
        self.assertEqual(diagnosis["projection_distance_cells"]["max"], 1.0)
        self.assertEqual(diagnosis["source_selection_margin"]["count"], 1)
        self.assertEqual(diagnosis["source_selection_margin"]["max_path_cost_margin"], 2.0)
        self.assertAlmostEqual(diagnosis["source_selection_margin"]["max_risk_margin"], 0.1)
        contexts = {item["scenario_id"]: item for item in summary["context_records"]}
        self.assertEqual(
            contexts["trainable"]["source_selection_best_alternative_scope"],
            "reachable_non_replan_candidates_including_policy_and_projected_targets",
        )
        self.assertEqual(
            contexts["trainable"]["source_selection_best_alternative_candidate_role"],
            "policy_target",
        )

    def test_summary_splits_reachability_aware_anchor_outcomes(self) -> None:
        self._write_batch(summary=self._reachability_diagnosis_summary())

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-candidate-generation-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reachable_substitute_anchor_found_count"], 1)
        self.assertEqual(summary["anchor_unreachable_repaired_by_reachable_substitute_count"], 1)
        self.assertEqual(summary["true_geometry_unreachable_count"], 1)
        diagnosis = summary["anchor_projection_coverage_diagnosis"]
        self.assertEqual(diagnosis["anchor_selection_status_counts"]["reachable_substitute_anchor_found"], 1)
        self.assertEqual(diagnosis["anchor_selection_status_counts"]["true_geometry_unreachable"], 1)
        self.assertEqual(diagnosis["reachable_substitute_anchor_found_count"], 1)
        self.assertEqual(diagnosis["true_geometry_unreachable_count"], 1)
        self.assertEqual(diagnosis["anchor_unreachable_repaired_by_reachable_substitute_count"], 1)
        contexts = {item["scenario_id"]: item for item in summary["context_records"]}
        self.assertEqual(contexts["substitute"]["anchor_selection_status"], "reachable_substitute_anchor_found")
        self.assertFalse(contexts["substitute"]["nearest_anchor_reachable"])
        self.assertTrue(contexts["substitute"]["anchor_reachable"])
        self.assertEqual(contexts["substitute"]["start_component_id"], 0)
        self.assertEqual(contexts["substitute"]["projected_anchor_component_id"], 0)
        self.assertEqual(contexts["true-geometry"]["anchor_selection_status"], "true_geometry_unreachable")
        self.assertFalse(contexts["true-geometry"]["reachable_substitute_anchor_available"])

    def test_validate_only_fails_on_current_git_provenance_mismatch(self) -> None:
        self._write_batch(stale_git=True)

        completed = self._run("--validate-only")

        self.assertEqual(completed.returncode, 1)
        self.assertIn("current_git_provenance_mismatch", completed.stdout)

    def test_validate_only_fails_on_open_grid_fallback(self) -> None:
        self._write_batch(open_grid_count=1)

        completed = self._run("--validate-only")

        self.assertEqual(completed.returncode, 1)
        self.assertIn("fallback_or_open_grid_blocks_anchor_projection_candidate_generation", completed.stdout)

    def test_validate_only_fails_on_safety_regression(self) -> None:
        self._write_batch(safety_regression_count=1)

        completed = self._run("--validate-only")

        self.assertEqual(completed.returncode, 1)
        self.assertIn("safety_regression_blocks_anchor_projection_candidate_generation", completed.stdout)
