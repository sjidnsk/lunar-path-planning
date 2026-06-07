import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class GoalBlockedEvidenceRegenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_goal_blocked_evidence_regeneration.sh"
        self.config = self.repo_root / "configs" / "goal_blocked_evidence_regeneration_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="goal-blocked-regeneration-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_regeneration(self, *args: str) -> subprocess.CompletedProcess[str]:
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
        decisions: list[dict],
        *,
        git_mismatch: bool = False,
        safety_regression_count: int = 0,
        fallback_count: int = 0,
        contract_mutation: bool = False,
    ) -> None:
        git_snapshot = self._git_snapshot(git_mismatch=git_mismatch)
        needs_regeneration_count = sum(
            1 for decision in decisions if decision.get("contract_decision") == "needs_regeneration"
        )
        refinement = {
            "schema_version": "goal-blocked-training-contract-refinement-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "goal_blocked_count": needs_regeneration_count,
            "excluded_candidate_count": len(decisions),
            "negative_evidence_candidate_count": 0,
            "needs_regeneration_count": needs_regeneration_count,
            "contract_decision_counts": {
                "blocked_by_contract": 0,
                "eligible_negative_evidence": 0,
                "excluded_from_positive_training": 0,
                "needs_regeneration": needs_regeneration_count,
            },
            "contract_blockers": ["goal_blocked_records_need_regeneration"]
            if needs_regeneration_count
            else [],
            "contract_decisions": decisions,
            "safety_regression_count": safety_regression_count,
            "fallback_or_open_grid_count": fallback_count,
            "audit_only": True,
            "no_ppo_training": True,
            "runs_training": False,
            "git_provenance": {"current": git_snapshot, "current_matches_sources": not git_mismatch},
        }
        application = {
            "schema_version": "policy-robustness-application-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "safety_regression_count": safety_regression_count,
            "open_grid_fallback_used_count": fallback_count,
            "does_not_modify_ppo": not contract_mutation,
            "does_not_modify_network": not contract_mutation,
            "does_not_modify_action_space": not contract_mutation,
            "git_provenance": {"current": git_snapshot},
            "channel_aware_application": {
                "schema_version": "channel-aware-application-smoke/v1",
                "record_count": len(records),
                "records": records,
            },
        }
        review = {
            "schema_version": "policy-training-readiness-review-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "training_readiness_status": "needs_training_contract_refinement",
            "recommended_next_action": "needs_training_contract_refinement",
            "training_blockers": ["goal_blocked_candidates_excluded_from_training_positive_evidence"],
            "rejected_goal_blocked_count": needs_regeneration_count,
            "excluded_candidate_count": len(decisions),
            "safety_regression_count": safety_regression_count,
            "fallback_or_open_grid_count": fallback_count,
            "audit_only": True,
            "no_ppo_training": True,
            "runs_training": False,
            "git_provenance": {"current": git_snapshot, "current_matches_sources": not git_mismatch},
            "contract_impact": {
                "training_contract_status": "compatible_audit_only",
                "contract_mutations": ["policy_ppo_mutation"] if contract_mutation else [],
            },
        }
        files = {
            "goal-blocked-training-contract-refinement-summary.json": refinement,
            "policy-robustness-application-summary.json": application,
            "policy-training-readiness-review-summary.json": review,
        }
        for filename, payload in files.items():
            (self.batch_root / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _record(
        self,
        scenario_id: str,
        *,
        action_index: int = 0,
        reason_codes: list[str] | None = None,
        application_reason_codes: list[str] | None = None,
        comparison: dict | None = None,
        failure_taxonomy: str | None = None,
    ) -> dict:
        payload = {
            "scenario_id": scenario_id,
            "pair_key": "all-all-k3",
            "action_index": action_index,
            "cell": [10, 12],
            "recommendation": "reject",
            "application_action": "exclude_blocked_candidate_evidence",
            "reason_codes": reason_codes or ["goal_blocked"],
            "application_reason_codes": application_reason_codes
            or ["channel_aware_application_exclude_blocked_candidate_evidence"],
            "comparison": comparison
            if comparison is not None
            else {
                "path_changed": False,
                "path_cost_delta": None,
                "channel_cost_delta": None,
                "high_cost_exposure_delta": None,
                "risk_delta": None,
            },
        }
        if failure_taxonomy is not None:
            payload["failure_taxonomy"] = failure_taxonomy
            payload["failure_taxonomy_source"] = "fixture_upstream_source"
        return payload

    def _decision(
        self,
        scenario_id: str,
        *,
        action_index: int = 0,
        contract_decision: str = "needs_regeneration",
    ) -> dict:
        return {
            "scenario_id": scenario_id,
            "pair_key": "all-all-k3",
            "action_index": action_index,
            "cell": [10, 12],
            "recommendation": "reject",
            "application_action": "exclude_blocked_candidate_evidence",
            "contract_decision": contract_decision,
            "decision_basis": "goal_blocked_without_explicit_candidate_contrast",
            "reason_codes": ["goal_blocked"],
        }

    def test_regeneration_splits_goal_blocked_diagnostics_without_training(self) -> None:
        self._write_sources(
            [
                self._record("missing-contrast"),
                self._record(
                    "route-failed",
                    action_index=1,
                    reason_codes=["goal_blocked"],
                    application_reason_codes=["channel_aware_application_exclude_blocked_candidate_evidence"],
                    failure_taxonomy="route_generation_failed",
                ),
                self._record(
                    "explicit-negative",
                    action_index=2,
                    comparison={
                        "path_changed": True,
                        "path_cost_delta": 1.5,
                        "channel_cost_delta": 2.0,
                        "high_cost_exposure_delta": 0.5,
                        "risk_delta": None,
                    },
                ),
            ],
            [
                self._decision("missing-contrast"),
                self._decision("route-failed", action_index=1),
                self._decision("explicit-negative", action_index=2),
            ],
        )

        completed = self._run_regeneration("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "goal-blocked-evidence-regeneration-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["schema_version"], "goal-blocked-evidence-regeneration-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["needs_regeneration_input_count"], 3)
        self.assertEqual(summary["regenerated_record_count"], 3)
        self.assertEqual(summary["failure_taxonomy_counts"]["missing_candidate_contrast"], 1)
        self.assertEqual(summary["failure_taxonomy_counts"]["route_generation_failed"], 1)
        self.assertEqual(summary["candidate_contrast_status_counts"]["missing_candidate_contrast"], 2)
        self.assertIn(
            "upstream_goal_blocked_records_without_finite_candidate_comparison",
            summary["upstream_diagnostic_blockers"],
        )
        self.assertEqual(summary["eligible_negative_evidence_candidate_count"], 1)
        self.assertEqual(summary["still_unresolved_count"], 2)
        self.assertIn("goal_blocked_records_still_unresolved", summary["contract_blockers"])
        self.assertEqual(
            summary["recommended_next_action"],
            "needs_goal_blocked_diagnostic_refinement",
        )
        decisions = {item["scenario_id"]: item for item in summary["regenerated_records"]}
        self.assertEqual(
            decisions["explicit-negative"]["diagnostic_decision"],
            "eligible_negative_evidence_candidate",
        )
        self.assertIsNone(decisions["explicit-negative"]["failure_category"])
        self.assertTrue(summary["audit_only"])
        self.assertTrue(summary["no_ppo_training"])
        self.assertFalse(summary["runs_training"])

    def test_regeneration_recommends_contract_rerun_when_all_records_have_finite_contrast(self) -> None:
        self._write_sources(
            [
                self._record(
                    "explicit-negative",
                    comparison={
                        "path_changed": True,
                        "path_cost_delta": 1.0,
                        "channel_cost_delta": 2.0,
                        "high_cost_exposure_delta": 3.0,
                        "risk_delta": 0.0,
                    },
                )
            ],
            [self._decision("explicit-negative")],
        )

        completed = self._run_regeneration("--batch-root", str(self.batch_root), "--config", str(self.config))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "goal-blocked-evidence-regeneration-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["eligible_negative_evidence_candidate_count"], 1)
        self.assertEqual(summary["still_unresolved_count"], 0)
        self.assertEqual(summary["contract_blockers"], [])
        self.assertEqual(summary["recommended_next_action"], "rerun_goal_blocked_training_contract_refinement")

    def test_regeneration_blocks_contract_mutation_and_fallback_from_training_readiness(self) -> None:
        for kwargs, expected_reason in (
            ({"fallback_count": 1}, "fallback_or_open_grid_blocks_goal_blocked_regeneration"),
            ({"safety_regression_count": 1}, "safety_regression_blocks_goal_blocked_regeneration"),
            ({"contract_mutation": True}, "contract_mutation_blocks_goal_blocked_regeneration"),
        ):
            with self.subTest(expected_reason=expected_reason):
                shutil.rmtree(self.batch_root)
                self.batch_root.mkdir(parents=True)
                self._write_sources(
                    [self._record("blocked")],
                    [self._decision("blocked")],
                    **kwargs,
                )

                completed = self._run_regeneration("--batch-root", str(self.batch_root), "--config", str(self.config))

                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                summary = json.loads(
                    (self.batch_root / "goal-blocked-evidence-regeneration-summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(summary["failure_taxonomy_counts"]["blocked_by_contract"], 1)
                self.assertIn(expected_reason, summary["contract_blockers"])
                self.assertEqual(
                    summary["recommended_next_action"],
                    "needs_goal_blocked_diagnostic_refinement",
                )

    def test_validate_only_blocks_current_git_mismatch_without_writing_summary(self) -> None:
        self._write_sources(
            [self._record("stale")],
            [self._decision("stale")],
            git_mismatch=True,
        )

        completed = self._run_regeneration(
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
        self.assertFalse((self.batch_root / "goal-blocked-evidence-regeneration-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
