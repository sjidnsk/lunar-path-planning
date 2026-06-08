import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class AnchorProjectionDistanceContractRelaxationSafetyAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = (
            self.repo_root
            / "scripts"
            / "run_anchor_projection_distance_contract_relaxation_safety_audit.sh"
        )
        self.config = (
            self.repo_root
            / "configs"
            / "anchor_projection_distance_contract_relaxation_safety_audit_v1.json"
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anchor-distance-contract-audit-"))
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

    def _write_sources(
        self,
        *,
        distance_contexts: list[dict] | None = None,
        stale_git: bool = False,
        fallback_count: int = 0,
        safety_regression_count: int = 0,
        source_selection_quality_regression_count: int = 0,
        audit_proxy_positive_count: int = 0,
    ) -> None:
        git_snapshot = self._git_snapshot(stale_git=stale_git)
        context_records = [
            self._trainable_context("trainable-0"),
            *(distance_contexts or []),
            self._not_selected_context("not-selected-cost", distance_cells=1, distance_m=0.5),
        ]
        trainable_count = sum(1 for context in context_records if context["trainable"])
        nontrainable_count = len(context_records) - trainable_count
        distance_rejected = [
            context
            for context in context_records
            if "projection_distance_cells_exceeds_contract" in context.get("reject_reasons", [])
            or "projection_distance_m_exceeds_contract" in context.get("reject_reasons", [])
        ]
        source_selected_distance_rejected = [
            context for context in distance_rejected if context["projected_candidate_source_selected"]
        ]
        not_selected_distance_rejected = [
            context for context in distance_rejected if not context["projected_candidate_source_selected"]
        ]
        source_not_selected_reason_counts = {
            "distance_contract_rejected": len(not_selected_distance_rejected),
            "higher_path_cost_and_risk": 0,
            "higher_path_cost": 1,
            "higher_risk": 0,
            "lower_utility_or_coverage": 0,
            "ranking_weight_tradeoff_or_unobserved_utility": 0,
            "no_selected_candidate_comparison": 0,
        }
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
            "trainable_anchor_projection_count": trainable_count,
            "nontrainable_blocked_target_count": nontrainable_count,
            "source_selected_but_distance_rejected_count": len(source_selected_distance_rejected),
            "distance_contract_rejected_source_selected_count": len(source_selected_distance_rejected),
            "nontrainable_source_candidate_not_selected_count": nontrainable_count,
            "source_selection_quality_regression_count": source_selection_quality_regression_count,
            "positive_training_evidence_contains_audit_proxy_anchor_count": audit_proxy_positive_count,
            "audit_proxy_positive_count": audit_proxy_positive_count,
            "source_candidate_not_selected_by_best_alternative_reason": source_not_selected_reason_counts,
            "distance_contract_rejected_by_distance_bin": {
                "count": len(distance_rejected),
                "source_selected_count": len(source_selected_distance_rejected),
                "not_source_selected_count": len(not_selected_distance_rejected),
                "by_projection_distance_cells": self._distance_bins(distance_rejected, "projection_distance_cells"),
                "by_projection_distance_m": self._distance_bins(distance_rejected, "projection_distance_m"),
            },
            "source_selection_quality_tradeoff_summary": {
                "distance_contract_rejected_count": len(distance_rejected),
                "source_selected_but_distance_rejected_count": len(source_selected_distance_rejected),
                "distance_contract_rejected_not_source_selected_count": len(not_selected_distance_rejected),
                "source_selection_quality_regression_count": source_selection_quality_regression_count,
                "source_candidate_not_selected_reason_counts": source_not_selected_reason_counts,
                "distance_contract_relaxation_recommendation": (
                    "record_only_keep_current_training_distance_contract"
                ),
            },
            "context_records": context_records,
            "git_provenance": {
                "current": git_snapshot,
                "current_matches_sources": not stale_git,
            },
            "audit_only": True,
            "runs_training": False,
            "no_ppo_training": True,
        }
        contract = {
            "schema_version": "anchor-projection-evidence-contract-summary/v1",
            "generated_at": "2026-06-08T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "contract_source": "anchor_projection_candidate_generation_summary",
            "platform_goal_contract_mismatch_count": len(context_records),
            "trainable_anchor_projection_count": trainable_count,
            "nontrainable_blocked_target_count": nontrainable_count,
            "candidate_contract_alignment_gap_count": 0,
            "contract_blockers": [],
            "fallback_or_open_grid_count": fallback_count,
            "safety_regression_count": safety_regression_count,
            "audit_proxy_positive_count": audit_proxy_positive_count,
            "git_provenance": {
                "current": git_snapshot,
                "current_matches_sources": not stale_git,
            },
            "audit_only": True,
            "runs_training": False,
            "no_ppo_training": True,
        }
        readiness = {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "generated_at": "2026-06-08T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "application_scope": "anchor_projection_readiness_contract_review_only",
            "training_readiness_status": "needs_training_contract_refinement",
            "training_blockers": ["anchor_projection_nontrainable_contexts_remain"],
            "anchor_projection_readiness_trainable_count": trainable_count,
            "anchor_projection_candidate_contract_alignment_gap_count": 0,
            "fallback_or_open_grid_count": fallback_count,
            "safety_regression_count": safety_regression_count,
            "git_provenance": {
                "current": git_snapshot,
                "current_matches_sources": not stale_git,
            },
            "audit_only": True,
            "runs_training": False,
            "no_ppo_training": True,
        }
        (self.batch_root / "anchor-projection-candidate-generation-summary.json").write_text(
            json.dumps(candidate, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "anchor-projection-evidence-contract-summary.json").write_text(
            json.dumps(contract, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "policy-training-readiness-review-summary.json").write_text(
            json.dumps(readiness, indent=2),
            encoding="utf-8",
        )

    def _git_snapshot(self, *, stale_git: bool = False) -> dict:
        if not stale_git:
            return self.git_snapshot
        return {
            **self.git_snapshot,
            "parent": {**self.git_snapshot["parent"], "sha": "0" * 40},
        }

    def _trainable_context(self, scenario_id: str) -> dict:
        return {
            "run_id": "all-all-k3-astar",
            "scenario_id": scenario_id,
            "classification": "platform_inflated_goal_blocked",
            "projected_candidate_source_selected": True,
            "trainable": True,
            "training_use": "trainable_anchor_projection_contrast",
            "comparison_scope": "projected_target_anchor_contrast",
            "projection_distance_cells": 1,
            "projection_distance_m": 0.5,
            "path_cost_margin_vs_selected": 0.0,
            "risk_margin_vs_selected": 0.0,
            "source_selection_quality_regression": False,
            "positive_audit_proxy": False,
            "reject_reasons": [],
        }

    def _source_selected_distance_context(
        self,
        scenario_id: str,
        *,
        distance_cells: float,
        distance_m: float,
        path_margin: float = 0.0,
        risk_margin: float = 0.0,
    ) -> dict:
        return {
            **self._trainable_context(scenario_id),
            "trainable": False,
            "projection_distance_cells": distance_cells,
            "projection_distance_m": distance_m,
            "path_cost_margin_vs_selected": path_margin,
            "risk_margin_vs_selected": risk_margin,
            "reject_reasons": [
                "projection_distance_cells_exceeds_contract",
                "projection_distance_m_exceeds_contract",
            ],
        }

    def _not_selected_context(self, scenario_id: str, *, distance_cells: float, distance_m: float) -> dict:
        return {
            **self._source_selected_distance_context(
                scenario_id,
                distance_cells=distance_cells,
                distance_m=distance_m,
                path_margin=0.4,
                risk_margin=0.1,
            ),
            "projected_candidate_source_selected": False,
            "training_use": "not_positive_evidence",
            "reject_reasons": [
                "source_candidate_not_selected",
                "projection_distance_cells_exceeds_contract",
                "projection_distance_m_exceeds_contract",
            ]
            if distance_cells > 2
            else ["source_candidate_not_selected"],
        }

    def _distance_bins(self, contexts: list[dict], field: str) -> dict:
        bins: dict[str, list[dict]] = {}
        for context in contexts:
            value = context.get(field)
            key = str(int(value)) if float(value).is_integer() else str(value)
            bins.setdefault(key, []).append(context)
        return {
            key: {
                "count": len(members),
                "source_selected_count": sum(1 for item in members if item["projected_candidate_source_selected"]),
                "not_source_selected_count": sum(1 for item in members if not item["projected_candidate_source_selected"]),
                "scenario_id_counts": {item["scenario_id"]: 1 for item in members},
            }
            for key, members in sorted(bins.items(), key=lambda item: float(item[0]))
        }

    def test_keeps_current_contract_when_distance_rejected_contexts_are_not_all_safe(self) -> None:
        self._write_sources(
            distance_contexts=[
                self._source_selected_distance_context("near-selected", distance_cells=3, distance_m=1.5),
                self._source_selected_distance_context("far-selected", distance_cells=11, distance_m=5.5),
                self._not_selected_context("far-not-selected", distance_cells=7, distance_m=3.5),
            ]
        )

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (
                self.batch_root
                / "anchor-projection-distance-contract-relaxation-safety-audit-summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["schema_version"], "anchor-projection-distance-contract-relaxation-safety-audit-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["recommendation"], "keep_current_training_distance_contract")
        self.assertEqual(summary["platform_goal_contract_mismatch_count"], 5)
        self.assertEqual(summary["trainable_anchor_projection_count"], 1)
        self.assertEqual(summary["nontrainable_blocked_target_count"], 4)
        self.assertEqual(summary["distance_contract_rejected_count"], 3)
        self.assertEqual(summary["source_selected_distance_rejected_count"], 2)
        self.assertEqual(summary["not_source_selected_distance_rejected_count"], 1)
        self.assertEqual(summary["relaxation_safety"]["eligible_source_selected_distance_rejected_count"], 1)
        self.assertEqual(summary["relaxation_safety"]["ineligible_source_selected_distance_rejected_count"], 1)
        self.assertFalse(summary["relaxation_profile"]["ready_for_opt_in_relaxation"])
        self.assertTrue(summary["relaxation_profile"]["default_contract_unchanged"])
        self.assertFalse(summary["runs_training"])
        self.assertTrue(summary["no_ppo_training"])

    def test_can_run_after_candidate_generation_before_contract_and_readiness_exist(self) -> None:
        self._write_sources(
            distance_contexts=[
                self._source_selected_distance_context("near-selected", distance_cells=3, distance_m=1.5),
            ]
        )
        (self.batch_root / "anchor-projection-evidence-contract-summary.json").unlink()
        (self.batch_root / "policy-training-readiness-review-summary.json").unlink()

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (
                self.batch_root
                / "anchor-projection-distance-contract-relaxation-safety-audit-summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(
            summary["source_summaries"]["anchor_projection_evidence_contract_summary"]["optional"]
        )
        self.assertTrue(
            summary["source_summaries"]["policy_training_readiness_review_summary"]["optional"]
        )
        self.assertTrue(summary["relaxation_profile"]["ready_for_opt_in_relaxation"])

    def test_blocks_fallback_safety_and_provenance_failures(self) -> None:
        cases = (
            (
                {"fallback_count": 1},
                "fallback_or_open_grid_blocks_distance_contract_relaxation_audit",
            ),
            (
                {"safety_regression_count": 1},
                "safety_regression_blocks_distance_contract_relaxation_audit",
            ),
            (
                {"stale_git": True},
                "current_git_provenance_mismatch",
            ),
        )
        for kwargs, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                shutil.rmtree(self.batch_root)
                self.batch_root.mkdir(parents=True)
                self._write_sources(
                    distance_contexts=[
                        self._source_selected_distance_context("near-selected", distance_cells=3, distance_m=1.5),
                    ],
                    **kwargs,
                )

                completed = self._run("--validate-only")

                self.assertEqual(completed.returncode, 1)
                self.assertIn(expected_reason, completed.stdout)

    def test_reports_opt_in_profile_when_every_distance_rejected_sample_passes_safety_gates(self) -> None:
        self._write_sources(
            distance_contexts=[
                self._source_selected_distance_context("near-selected", distance_cells=3, distance_m=1.5),
            ]
        )

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (
                self.batch_root
                / "anchor-projection-distance-contract-relaxation-safety-audit-summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["recommendation"], "opt_in_distance_contract_relaxation_profile_ready")
        self.assertTrue(summary["relaxation_profile"]["ready_for_opt_in_relaxation"])
        self.assertEqual(summary["relaxation_profile"]["max_projection_distance_cells"], 3)
        self.assertEqual(summary["relaxation_profile"]["max_projection_distance_m"], 1.5)
        self.assertTrue(summary["relaxation_profile"]["default_contract_unchanged"])
        self.assertEqual(summary["readiness_impact"]["recommended_readiness_status"], "needs_training_contract_refinement")
        self.assertFalse(summary["runs_training"])


if __name__ == "__main__":
    unittest.main()
