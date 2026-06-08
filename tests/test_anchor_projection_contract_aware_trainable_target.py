import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class AnchorProjectionContractAwareTrainableTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = (
            self.repo_root
            / "scripts"
            / "run_anchor_projection_contract_aware_trainable_target.sh"
        )
        self.config = (
            self.repo_root
            / "configs"
            / "anchor_projection_contract_aware_trainable_target_v1.json"
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anchor-contract-aware-target-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return subprocess.run(
            ["bash", str(self.script), "--batch-root", str(self.batch_root), "--config", str(self.config), *args],
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

    def _write_candidate_summary(
        self,
        *,
        ppo_consumable_count: int,
        nontrainable_count: int,
        distance_rejected_count: int,
        source_not_selected_count: int,
        fallback_count: int = 0,
        safety_regression_count: int = 0,
    ) -> None:
        context_records = [
            self._context(
                scenario_id=f"ppo-{index}",
                trainable=True,
                ppo_consumable=True,
                source_selected=True,
                reject_reasons=[],
            )
            for index in range(ppo_consumable_count)
        ]
        context_records.extend(
            self._context(
                scenario_id=f"blocked-{index}",
                trainable=False,
                ppo_consumable=False,
                source_selected=False,
                reject_reasons=["source_candidate_not_selected"],
            )
            for index in range(max(nontrainable_count - distance_rejected_count, 0))
        )
        context_records.extend(
            self._context(
                scenario_id=f"distance-{index}",
                trainable=False,
                ppo_consumable=False,
                source_selected=False,
                reject_reasons=[
                    "source_candidate_not_selected",
                    "projection_distance_cells_exceeds_contract",
                    "projection_distance_m_exceeds_contract",
                ],
            )
            for index in range(distance_rejected_count)
        )
        candidate = {
            "schema_version": "anchor-projection-candidate-generation-summary/v1",
            "generated_at": "2026-06-08T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "batch_root": str(self.batch_root),
            "current_git_provenance_mismatch_count": 0,
            "git_provenance_mismatch_count": 0,
            "fallback_or_open_grid_count": fallback_count,
            "open_grid_fallback_used_count": fallback_count,
            "safety_regression_count": safety_regression_count,
            "platform_goal_contract_mismatch_count": len(context_records),
            "trainable_anchor_projection_count": ppo_consumable_count,
            "candidate_contract_alignment_gap_count": 0,
            "nontrainable_blocked_target_count": nontrainable_count,
            "distance_contract_rejected_count": distance_rejected_count,
            "source_candidate_not_selected_by_best_alternative_reason": {
                "distance_contract_rejected": distance_rejected_count,
                "higher_path_cost_and_risk": 0,
                "higher_path_cost": max(source_not_selected_count - distance_rejected_count, 0),
                "higher_risk": 0,
                "lower_utility_or_coverage": 0,
                "ranking_weight_tradeoff_or_unobserved_utility": 0,
                "no_selected_candidate_comparison": 0,
            },
            "context_records": context_records,
            "git_provenance": {
                "current": self.git_snapshot,
                "current_matches_sources": True,
            },
            "audit_only": True,
            "runs_training": False,
            "no_ppo_training": True,
        }
        (self.batch_root / "anchor-projection-candidate-generation-summary.json").write_text(
            json.dumps(candidate, indent=2),
            encoding="utf-8",
        )

    def _context(
        self,
        *,
        scenario_id: str,
        trainable: bool,
        ppo_consumable: bool,
        source_selected: bool,
        reject_reasons: list[str],
    ) -> dict:
        return {
            "run_id": "run-a",
            "scenario_id": scenario_id,
            "scenario_group": "stress",
            "source_action_index": 0,
            "policy_target_cell": [2, 1],
            "execution_goal_cell": [1, 1],
            "projected_anchor_cell": [1, 1],
            "classification": "platform_inflated_goal_blocked",
            "trainable": trainable,
            "projected_candidate_generated": True,
            "projected_candidate_source_selected": source_selected,
            "source_selection_status": "source_selected" if source_selected else "not_source_selected",
            "training_use": "trainable_anchor_projection_contrast" if trainable else "not_positive_evidence",
            "comparison_scope": "projected_target_anchor_contrast",
            "reject_reasons": reject_reasons,
            "projection_distance_cells": 1 if ppo_consumable else 3,
            "projection_distance_m": 0.5 if ppo_consumable else 1.5,
            "candidate_generation": {
                "target_binding_mode": "same_action_execution_substitute" if ppo_consumable else "synthetic_projection",
                "ppo_consumable_action": ppo_consumable,
                "contract_safe": ppo_consumable,
            },
            "trainability_gate": {
                "status": "selected_trainable" if ppo_consumable else "rejected",
                "reason_codes": reject_reasons,
            },
            "positive_audit_proxy": False,
            "source_selection_quality_regression": False,
        }

    def test_summary_reports_ppo_consumable_delta_and_no_next_required_change(self) -> None:
        self._write_candidate_summary(
            ppo_consumable_count=2,
            nontrainable_count=58,
            distance_rejected_count=34,
            source_not_selected_count=46,
        )

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-contract-aware-trainable-target-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["contract_trainable_contrast_count"], 2)
        self.assertEqual(summary["ppo_consumable_trainable_target_count"], 2)
        self.assertEqual(summary["nontrainable_blocked_target_count_delta"], -2)
        self.assertEqual(summary["distance_contract_rejected_count_delta"], -2)
        self.assertEqual(summary["source_candidate_not_selected_count_delta"], -2)
        self.assertIsNone(summary["next_required_change"])
        self.assertEqual(summary["readiness_impact"]["recommended_training_blockers"], [])

    def test_summary_escalates_when_no_ppo_consumable_target_exists(self) -> None:
        self._write_candidate_summary(
            ppo_consumable_count=0,
            nontrainable_count=60,
            distance_rejected_count=36,
            source_not_selected_count=48,
        )

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-contract-aware-trainable-target-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["ppo_consumable_trainable_target_count"], 0)
        self.assertEqual(
            summary["next_required_change"],
            "action_or_target_contract_change_required",
        )
        self.assertIn(
            "anchor_projection_nontrainable_contexts_remain",
            summary["readiness_impact"]["recommended_training_blockers"],
        )
        self.assertGreater(summary["no_contract_safe_reachable_substitute_count"], 0)

    def test_summary_escalates_when_nontrainable_count_does_not_decrease(self) -> None:
        self._write_candidate_summary(
            ppo_consumable_count=2,
            nontrainable_count=60,
            distance_rejected_count=34,
            source_not_selected_count=46,
        )

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-contract-aware-trainable-target-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["ppo_consumable_trainable_target_count"], 2)
        self.assertEqual(summary["nontrainable_blocked_target_count_delta"], 0)
        self.assertEqual(
            summary["next_required_change"],
            "action_or_target_contract_change_required",
        )
        self.assertIn(
            "nontrainable_blocked_target_count_not_reduced",
            summary["main_success_gate_failures"],
        )
        self.assertIn(
            "anchor_projection_nontrainable_contexts_remain",
            summary["readiness_impact"]["recommended_training_blockers"],
        )
