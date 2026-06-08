import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class AnchorProjectionNontrainableContextReductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_anchor_projection_nontrainable_context_reduction.sh"
        self.config = self.repo_root / "configs" / "anchor_projection_nontrainable_context_reduction_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anchor-nontrainable-reduction-"))
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
        stale_git: bool = False,
        fallback_count: int = 0,
        safety_regression_count: int = 0,
        source_selection_quality_regression_count: int = 0,
        audit_proxy_positive_count: int = 0,
    ) -> None:
        git_snapshot = self._git_snapshot(stale_git=stale_git)
        contexts = [
            self._trainable_context("trainable-0"),
            self._source_selected_distance_context("npz_near_blocked_corridor", distance_cells=3, distance_m=1.5),
            self._source_selected_distance_context("npz_dense_rock_choke", distance_cells=11, distance_m=5.5),
            self._not_selected_context(
                "npz_dense_rock_choke",
                distance_cells=7,
                distance_m=3.5,
                path_margin=0.9,
                risk_margin=0.3,
            ),
            self._not_selected_context(
                "npz_high_risk_value_trap",
                distance_cells=1,
                distance_m=0.5,
                path_margin=2.0,
                risk_margin=0.0,
            ),
        ]
        trainable_count = sum(1 for context in contexts if context["trainable"])
        nontrainable_count = len(contexts) - trainable_count
        distance_rejected = [context for context in contexts if self._distance_rejected(context)]
        source_selected_distance_rejected = [
            context for context in distance_rejected if context["projected_candidate_source_selected"]
        ]
        not_source_selected_distance_rejected = [
            context for context in distance_rejected if not context["projected_candidate_source_selected"]
        ]
        source_not_selected_reason_counts = {
            "distance_contract_rejected": len(not_source_selected_distance_rejected),
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
            "platform_goal_contract_mismatch_count": len(contexts),
            "trainable_anchor_projection_count": trainable_count,
            "nontrainable_blocked_target_count": nontrainable_count,
            "source_selected_but_distance_rejected_count": len(source_selected_distance_rejected),
            "distance_contract_rejected_source_selected_count": len(source_selected_distance_rejected),
            "nontrainable_source_candidate_not_selected_count": 2,
            "source_selection_quality_regression_count": source_selection_quality_regression_count,
            "positive_training_evidence_contains_audit_proxy_anchor_count": audit_proxy_positive_count,
            "audit_proxy_positive_count": audit_proxy_positive_count,
            "source_candidate_not_selected_by_best_alternative_reason": source_not_selected_reason_counts,
            "distance_contract_rejected_by_distance_bin": {
                "count": len(distance_rejected),
                "source_selected_count": len(source_selected_distance_rejected),
                "not_source_selected_count": len(not_source_selected_distance_rejected),
                "by_projection_distance_cells": self._distance_bins(distance_rejected, "projection_distance_cells"),
            },
            "context_records": contexts,
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
            "platform_goal_contract_mismatch_count": len(contexts),
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
        distance_audit = {
            "schema_version": "anchor-projection-distance-contract-relaxation-safety-audit-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "recommendation": "keep_current_training_distance_contract",
            "distance_contract_rejected_count": len(distance_rejected),
            "source_selected_distance_rejected_count": len(source_selected_distance_rejected),
            "not_source_selected_distance_rejected_count": len(not_source_selected_distance_rejected),
            "relaxation_safety": {
                "eligible_source_selected_distance_rejected_count": 1,
                "ineligible_source_selected_distance_rejected_count": 1,
            },
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
        (self.batch_root / "anchor-projection-distance-contract-relaxation-safety-audit-summary.json").write_text(
            json.dumps(distance_audit, indent=2),
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
            "scenario_group": "all",
            "trainable": True,
            "projected_candidate_source_selected": True,
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
    ) -> dict:
        return {
            **self._trainable_context(scenario_id),
            "run_id": "all-all-k3-astar",
            "trainable": False,
            "projection_distance_cells": distance_cells,
            "projection_distance_m": distance_m,
            "reject_reasons": [
                "audit_proxy_scope_not_positive_evidence",
                "projection_distance_cells_exceeds_contract",
                "projection_distance_m_exceeds_contract",
            ],
        }

    def _not_selected_context(
        self,
        scenario_id: str,
        *,
        distance_cells: float,
        distance_m: float,
        path_margin: float,
        risk_margin: float,
    ) -> dict:
        reject_reasons = ["audit_proxy_scope_not_positive_evidence", "source_candidate_not_selected"]
        if distance_cells > 2:
            reject_reasons.extend(
                [
                    "projection_distance_cells_exceeds_contract",
                    "projection_distance_m_exceeds_contract",
                ]
            )
        return {
            **self._source_selected_distance_context(
                scenario_id,
                distance_cells=distance_cells,
                distance_m=distance_m,
            ),
            "run_id": "all-all-k3-channel-aware",
            "projected_candidate_source_selected": False,
            "training_use": "not_positive_evidence",
            "path_cost_margin_vs_selected": path_margin,
            "risk_margin_vs_selected": risk_margin,
            "reject_reasons": reject_reasons,
        }

    def _distance_rejected(self, context: dict) -> bool:
        return any(
            reason in context.get("reject_reasons", [])
            for reason in (
                "projection_distance_cells_exceeds_contract",
                "projection_distance_m_exceeds_contract",
            )
        )

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

    def test_summarizes_nontrainable_context_destinations_without_training_release(self) -> None:
        self._write_sources()

        completed = self._run()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "anchor-projection-nontrainable-context-reduction-summary.json").read_text(
                encoding="utf-8",
            )
        )
        self.assertEqual(summary["schema_version"], "anchor-projection-nontrainable-context-reduction-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["recommendation"], "keep_training_blocker_focus_source_selection_candidate_quality")
        self.assertEqual(summary["trainable_anchor_projection_count"], 1)
        self.assertEqual(summary["nontrainable_blocked_target_count"], 4)
        self.assertEqual(summary["source_selection_quality"]["generated_not_source_selected_count"], 2)
        self.assertEqual(summary["source_selection_quality"]["not_source_selected_distance_rejected_count"], 1)
        self.assertEqual(summary["source_selection_quality"]["reason_counts"]["higher_path_cost"], 1)
        self.assertEqual(summary["source_selection_quality_regression_count"], 0)
        self.assertEqual(summary["generated_not_source_selected_count"], 2)
        self.assertEqual(summary["distance_contract_rejected_count"], 3)
        accounting = summary["nontrainable_resolution_accounting"]
        self.assertEqual(accounting["input_nontrainable_count"], 4)
        self.assertEqual(accounting["safe_default_training_conversion_count"], 0)
        self.assertEqual(accounting["opt_in_relaxation_followup_candidate_count"], 1)
        self.assertEqual(accounting["must_remain_blocked_count"], 4)
        self.assertTrue(accounting["blocker_retained"])
        self.assertEqual(accounting["classification_counts"]["opt_in_relaxation_followup_candidate"], 1)
        self.assertEqual(accounting["classification_counts"]["blocked_source_selected_distance_too_far"], 1)
        self.assertEqual(accounting["classification_counts"]["blocked_not_source_selected_distance_rejected"], 1)
        self.assertEqual(accounting["classification_counts"]["blocked_source_candidate_not_selected_quality"], 1)
        self.assertEqual(
            summary["scenario_backend_distribution"]["npz_dense_rock_choke"]["channel-aware"]["count"],
            1,
        )
        self.assertEqual(summary["audit_proxy_positive_count"], 0)
        self.assertFalse(summary["runs_training"])
        self.assertTrue(summary["no_ppo_training"])
        self.assertTrue(summary["does_not_modify_default_astar"])
        self.assertEqual(
            summary["readiness_impact"]["recommended_readiness_status"],
            "needs_training_contract_refinement",
        )

    def test_blocks_fallback_safety_provenance_quality_and_audit_proxy_failures(self) -> None:
        cases = (
            (
                {"fallback_count": 1},
                "fallback_or_open_grid_blocks_nontrainable_context_reduction",
            ),
            (
                {"safety_regression_count": 1},
                "safety_regression_blocks_nontrainable_context_reduction",
            ),
            (
                {"stale_git": True},
                "current_git_provenance_mismatch",
            ),
            (
                {"source_selection_quality_regression_count": 1},
                "source_selection_quality_regression_blocks_nontrainable_context_reduction",
            ),
            (
                {"audit_proxy_positive_count": 1},
                "audit_proxy_positive_blocks_nontrainable_context_reduction",
            ),
        )
        for kwargs, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                shutil.rmtree(self.batch_root)
                self.batch_root.mkdir(parents=True)
                self._write_sources(**kwargs)

                completed = self._run("--validate-only")

                self.assertEqual(completed.returncode, 1)
                self.assertIn(expected_reason, completed.stdout)


if __name__ == "__main__":
    unittest.main()
