import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class PlannerValidatedTrainableTargetMiningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_planner_validated_trainable_target_mining.sh"
        self.config = self.repo_root / "configs" / "planner_validated_trainable_target_mining_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="planner-validated-mining-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run(self) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return subprocess.run(
            ["bash", str(self.script), "--batch-root", str(self.batch_root), "--config", str(self.config)],
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

    def _write_candidate_summary(self, contexts: list[dict]) -> None:
        (self.batch_root / "anchor-projection-candidate-generation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "anchor-projection-candidate-generation-summary/v1",
                    "generated_at": "2026-06-08T00:00:00Z",
                    "status": "passed",
                    "reason_codes": [],
                    "current_git_provenance_mismatch_count": 0,
                    "git_provenance_mismatch_count": 0,
                    "fallback_or_open_grid_count": 0,
                    "open_grid_fallback_used_count": 0,
                    "safety_regression_count": 0,
                    "platform_goal_contract_mismatch_count": len(contexts),
                    "trainable_anchor_projection_count": sum(1 for item in contexts if item["trainable"]),
                    "ppo_consumable_trainable_target_count": sum(
                        1 for item in contexts if item["trainable"] and item["ppo_consumable_action"]
                    ),
                    "candidate_contract_alignment_gap_count": 0,
                    "nontrainable_blocked_target_count": sum(
                        1 for item in contexts if not item["trainable"]
                    ),
                    "context_records": contexts,
                    "git_provenance": {
                        "current": self.git_snapshot,
                        "current_matches_sources": True,
                    },
                    "runs_training": False,
                    "audit_only": True,
                    "no_ppo_training": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _context(
        self,
        *,
        scenario_id: str,
        trainable: bool,
        source_status: str,
        distance_cells: int,
        distance_m: float,
        ppo: bool,
        contract_safe: bool,
        planner_exception: bool = False,
        quality_regression: bool = False,
    ) -> dict:
        reject_reasons = [] if trainable else ["source_candidate_not_selected"]
        if quality_regression:
            reject_reasons = ["source_selection_quality_regression"]
        elif distance_cells > 2 or distance_m > 1.0:
            reject_reasons.extend(
                [
                    "projection_distance_cells_exceeds_contract",
                    "projection_distance_m_exceeds_contract",
                ]
            )
        return {
            "run_id": "run-a",
            "scenario_id": scenario_id,
            "scenario_group": "stress",
            "source_action_index": 0,
            "policy_target_cell": [3, 1],
            "execution_goal_cell": [0, 1],
            "projected_anchor_cell": [0, 1],
            "classification": "platform_inflated_goal_blocked",
            "trainable": trainable,
            "projected_candidate_generated": True,
            "projected_candidate_source_selected": source_status.startswith("source_selected"),
            "source_selection_status": source_status,
            "training_use": "trainable_anchor_projection_contrast" if trainable else "not_positive_evidence",
            "comparison_scope": "projected_target_anchor_contrast",
            "reject_reasons": reject_reasons,
            "projection_distance_cells": distance_cells,
            "projection_distance_m": distance_m,
            "reachable": True,
            "replan_required": False,
            "source_selection_quality_regression": quality_regression,
            "source_selection_path_cost_margin_vs_best_alternative": 0.0 if not quality_regression else 3.0,
            "source_selection_risk_margin_vs_best_alternative": 0.0 if not quality_regression else 0.5,
            "target_binding_mode": "same_action_execution_substitute",
            "ppo_consumable_action": ppo,
            "contract_safe": contract_safe,
            "default_distance_contract_safe": contract_safe,
            "planner_validated_distance_exception": planner_exception,
            "planner_validated_exception_safe": planner_exception,
            "candidate_generation": {
                "target_binding_mode": "same_action_execution_substitute",
                "ppo_consumable_action": ppo,
                "contract_safe": contract_safe,
                "default_distance_contract_safe": contract_safe,
                "planner_validated_distance_exception": planner_exception,
                "planner_validated_exception_safe": planner_exception,
            },
            "trainability_gate": {
                "status": "selected_trainable" if trainable else "rejected",
                "reason_codes": reject_reasons,
                "ppo_consumable_action": ppo,
                "contract_safe": contract_safe,
            },
        }

    def test_summary_counts_default_and_planner_validated_exception_once_per_context(self) -> None:
        contexts = [
            self._context(
                scenario_id=f"default-{index}",
                trainable=True,
                source_status="source_selected",
                distance_cells=2,
                distance_m=1.0,
                ppo=True,
                contract_safe=True,
            )
            for index in range(18)
        ]
        contexts.extend(
            self._context(
                scenario_id=f"exception-{index}",
                trainable=False,
                source_status="source_selected",
                distance_cells=3,
                distance_m=1.5,
                ppo=True,
                contract_safe=False,
                planner_exception=True,
            )
            for index in range(6)
        )
        contexts.extend(
            self._context(
                scenario_id=f"not-selected-{index}",
                trainable=False,
                source_status="not_source_selected",
                distance_cells=2,
                distance_m=1.0,
                ppo=True,
                contract_safe=True,
            )
            for index in range(30)
        )
        contexts.extend(
            self._context(
                scenario_id=f"distance-{index}",
                trainable=False,
                source_status="not_source_selected",
                distance_cells=7,
                distance_m=3.5,
                ppo=True,
                contract_safe=False,
            )
            for index in range(18)
        )
        contexts.extend(
            self._context(
                scenario_id=f"quality-{index}",
                trainable=False,
                source_status="source_selected_quality_regression",
                distance_cells=3,
                distance_m=1.5,
                ppo=True,
                contract_safe=False,
                planner_exception=True,
                quality_regression=True,
            )
            for index in range(6)
        )
        self._write_candidate_summary(contexts)

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "planner-validated-trainable-target-mining-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["default_contract_trainable_target_count"], 18)
        self.assertEqual(summary["planner_validated_distance_exception_count"], 6)
        self.assertEqual(summary["planner_validated_trainable_target_count"], 24)
        self.assertEqual(summary["nontrainable_blocked_target_count"], 54)
        self.assertEqual(summary["nontrainable_blocked_target_count_delta"], -6)
        self.assertEqual(summary["distance_contract_blocked_count"], 18)
        self.assertEqual(summary["source_selection_not_selected_count"], 30)
        self.assertEqual(summary["quality_regression_rejected_count"], 6)
        self.assertIsNone(summary["next_required_change"])
        self.assertEqual(summary["final_decision_counts"]["selected_default_contract_trainable"], 18)
        self.assertEqual(summary["final_decision_counts"]["selected_planner_validated_distance_exception"], 6)

    def test_not_source_selected_planner_repair_stays_diagnostic_and_requires_contract_change(self) -> None:
        contexts = [
            self._context(
                scenario_id=f"default-{index}",
                trainable=True,
                source_status="source_selected",
                distance_cells=2,
                distance_m=1.0,
                ppo=True,
                contract_safe=True,
            )
            for index in range(18)
        ]
        contexts.extend(
            self._context(
                scenario_id=f"not-selected-exception-{index}",
                trainable=False,
                source_status="not_source_selected",
                distance_cells=3,
                distance_m=1.5,
                ppo=True,
                contract_safe=False,
                planner_exception=True,
            )
            for index in range(60)
        )
        self._write_candidate_summary(contexts)

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "planner-validated-trainable-target-mining-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["planner_validated_trainable_target_count"], 18)
        self.assertEqual(summary["planner_validated_distance_exception_count"], 0)
        self.assertEqual(summary["source_selection_not_selected_count"], 60)
        self.assertEqual(summary["nontrainable_blocked_target_count"], 60)
        self.assertEqual(
            summary["next_required_change"],
            "source_selection_or_target_contract_change_required",
        )
