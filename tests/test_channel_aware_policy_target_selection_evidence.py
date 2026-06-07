import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class ChannelAwarePolicyTargetSelectionEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_channel_aware_policy_target_selection_evidence.sh"
        self.config = self.repo_root / "configs" / "channel_aware_policy_target_selection_evidence_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="channel-aware-policy-target-selection-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_audit(self, *args: str) -> subprocess.CompletedProcess[str]:
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
        records: list[dict],
        *,
        git_mismatch: bool = False,
        readiness_status: str = "passed",
        readiness_reason_codes: list[str] | None = None,
    ) -> None:
        source_git = self.git_snapshot
        if git_mismatch:
            source_git = {
                **self.git_snapshot,
                "parent": {**self.git_snapshot["parent"], "sha": "0" * 40},
            }
        application_records = [self._application_record(record) for record in records]
        selected_changed_count = sum(1 for record in records if record["selected_candidate_changed"])
        recommendation_counts = self._counts(record["recommendation"] for record in application_records)
        action_counts = self._counts(record["application_action"] for record in application_records)
        reason_counts = self._counts(
            reason
            for record in application_records
            for reason in record.get("reason_codes", []) + record.get("application_reason_codes", [])
        )
        for key in ("keep", "downweight", "reject", "needs_more_evidence"):
            recommendation_counts.setdefault(key, 0)
        for key in (
            "keep_quality_evidence",
            "downweight_conservative_application",
            "exclude_blocked_candidate_evidence",
            "downweight_needs_more_evidence",
        ):
            action_counts.setdefault(key, 0)

        readiness = {
            "schema_version": "channel-aware-training-readiness-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": readiness_status,
            "reason_codes": [] if readiness_status == "passed" else ["fixture_readiness_blocked"],
            "readiness_status": "needs_more_evidence" if readiness_status == "passed" else "blocked",
            "readiness_reason_codes": readiness_reason_codes
            or ["policy_target_selection_not_improved"],
            "batch_root": str(self.batch_root),
            "application_summary_path": str(self.batch_root / "policy-robustness-application-summary.json"),
            "record_count": len(records),
            "selected_candidate_changed_count": selected_changed_count,
            "selected_candidate_changed_rate": selected_changed_count / len(records),
            "recommendation_counts": dict(sorted(recommendation_counts.items())),
            "application_action_counts": dict(sorted(action_counts.items())),
            "reason_code_counts": dict(sorted(reason_counts.items())),
            "git_provenance": {"current": source_git, "current_matches_application": not git_mismatch},
        }
        application = {
            "schema_version": "policy-robustness-application-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "batch_root": str(self.batch_root),
            "git_provenance": {
                "robustness": {
                    "current": source_git,
                    "current_matches_batch": not git_mismatch,
                    "runs_match_batch": not git_mismatch,
                },
                "current": source_git,
                "current_matches_robustness": not git_mismatch,
            },
            "channel_aware_application": {
                "schema_version": "channel-aware-application-smoke/v1",
                "record_count": len(application_records),
                "recommendation_counts": dict(sorted(recommendation_counts.items())),
                "action_counts": dict(sorted(action_counts.items())),
                "reason_code_counts": dict(sorted(reason_counts.items())),
                "summary_reason_codes": ["channel_aware_application_smoke_mapped"],
                "records": application_records,
            },
        }
        comparison = {
            "schema_version": "policy-decision-selection-comparison-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "batch_root": str(self.batch_root),
            "git_provenance": {"current": source_git, "current_matches_batch": not git_mismatch},
            "channel_aware_decision_audit": {
                "schema_version": "channel-aware-decision-audit/v1",
                "paired_scenario_count": len({(record["pair_key"], record["scenario_id"]) for record in records}),
                "channel_aware_candidate_count": len(records),
                "selected_candidate_changed_count": selected_changed_count,
                "selected_candidate_changed_rate": selected_changed_count / len(records),
                "recommendation_counts": dict(sorted(recommendation_counts.items())),
                "blocker_reason_counts": self._counts(
                    record["blocker_reason"]
                    for record in records
                    if record.get("blocker_reason") not in (None, "selected")
                ),
                "records": records,
            },
        }
        (self.batch_root / "channel-aware-training-readiness-summary.json").write_text(
            json.dumps(readiness, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "policy-robustness-application-summary.json").write_text(
            json.dumps(application, indent=2),
            encoding="utf-8",
        )
        (self.batch_root / "policy-decision-selection-comparison-summary.json").write_text(
            json.dumps(comparison, indent=2),
            encoding="utf-8",
        )

    def _audit_record(
        self,
        action_index: int,
        *,
        scenario_id: str,
        recommendation: str,
        selected: bool,
        reason_codes: list[str],
        blocker_reason: str | None,
        path_cost_tradeoff: bool = False,
        selected_candidate_changed: bool = False,
    ) -> dict:
        selected_cell = [10, 10]
        return {
            "pair_key": "all-all-k3",
            "astar_run_id": "all-all-k3-astar",
            "channel_aware_run_id": "all-all-k3-channel-aware",
            "scenario_id": scenario_id,
            "scenario_group": "stress",
            "action_index": action_index,
            "cell": [action_index + 1, action_index + 2],
            "astar_selected_cell": selected_cell,
            "channel_aware_selected_cell": selected_cell if not selected_candidate_changed else [11, 10],
            "selected_candidate_changed": selected_candidate_changed,
            "present": True,
            "selected": selected,
            "quality_improvement": recommendation == "keep",
            "risk_or_high_cost_improvement": recommendation == "keep",
            "path_cost_tradeoff": path_cost_tradeoff,
            "blocker_reason": blocker_reason,
            "recommendation": recommendation,
            "reason_codes": list(reason_codes),
            "comparison": {
                "path_changed": path_cost_tradeoff,
                "path_cost_delta": 1.0 if path_cost_tradeoff else None,
                "channel_cost_delta": -2.0 if path_cost_tradeoff else None,
            },
        }

    def _application_record(self, record: dict) -> dict:
        action = {
            "keep": "keep_quality_evidence",
            "downweight": "downweight_conservative_application",
            "reject": "exclude_blocked_candidate_evidence",
            "needs_more_evidence": "downweight_needs_more_evidence",
        }[record["recommendation"]]
        weight = 1.0 if action == "keep_quality_evidence" else 0.0 if action == "exclude_blocked_candidate_evidence" else 0.5
        return {
            "pair_key": record["pair_key"],
            "astar_run_id": record["astar_run_id"],
            "channel_aware_run_id": record["channel_aware_run_id"],
            "scenario_id": record["scenario_id"],
            "scenario_group": record["scenario_group"],
            "action_index": record["action_index"],
            "cell": record["cell"],
            "selected_candidate_changed": record["selected_candidate_changed"],
            "recommendation": record["recommendation"],
            "application_action": action,
            "application_sample_weight": weight,
            "quality_improvement": record["quality_improvement"],
            "risk_or_high_cost_improvement": record["risk_or_high_cost_improvement"],
            "path_cost_tradeoff": record["path_cost_tradeoff"],
            "reason_codes": record["reason_codes"],
            "application_reason_codes": [f"channel_aware_application_{action}"],
            "comparison": record["comparison"],
        }

    def _counts(self, values) -> dict:
        result: dict[str, int] = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return result

    def test_audit_explains_zero_selection_change_without_claiming_improvement(self) -> None:
        self._write_sources(
            [
                self._audit_record(
                    0,
                    scenario_id="npz_keep_selected",
                    recommendation="keep",
                    selected=True,
                    reason_codes=["channel_aware_quality_improved", "path_cost_tradeoff"],
                    blocker_reason="selected",
                    path_cost_tradeoff=True,
                ),
                self._audit_record(
                    1,
                    scenario_id="npz_keep_selected",
                    recommendation="keep",
                    selected=False,
                    reason_codes=["channel_aware_quality_improved", "path_cost_tradeoff"],
                    blocker_reason="not_selected",
                    path_cost_tradeoff=True,
                ),
                self._audit_record(
                    0,
                    scenario_id="npz_blocked_selected",
                    recommendation="reject",
                    selected=True,
                    reason_codes=["goal_blocked"],
                    blocker_reason="goal_blocked",
                ),
                self._audit_record(
                    0,
                    scenario_id="npz_same_as_baseline",
                    recommendation="downweight",
                    selected=True,
                    reason_codes=["same_as_baseline"],
                    blocker_reason="same_as_baseline",
                ),
                self._audit_record(
                    1,
                    scenario_id="npz_blocked_selected",
                    recommendation="reject",
                    selected=False,
                    reason_codes=["goal_blocked"],
                    blocker_reason="goal_blocked",
                ),
            ]
        )

        completed = self._run_audit(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "channel-aware-policy-target-selection-evidence-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["schema_version"], "channel-aware-policy-target-selection-evidence-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["selected_candidate_changed_rate"], 0.0)
        self.assertFalse(summary["supports_policy_target_selection_improvement_claim"])
        self.assertIn("policy_target_selection_not_improved", summary["improvement_claim_reason_codes"])
        self.assertIn("selected_candidate_unchanged", summary["selection_change_explanation_codes"])
        self.assertEqual(summary["keep_selected_candidate_count"], 1)
        self.assertEqual(summary["keep_selected_candidate_rate"], 0.5)
        self.assertEqual(summary["keep_non_selected_candidate_count"], 1)
        self.assertEqual(summary["keep_non_selected_candidate_rate"], 0.5)
        self.assertEqual(summary["blocked_selected_candidate_count"], 1)
        self.assertEqual(summary["blocked_selected_candidate_rate"], 0.5)
        self.assertEqual(summary["same_as_baseline_count"], 1)
        self.assertEqual(summary["goal_blocked_count"], 2)
        self.assertEqual(summary["path_cost_tradeoff_count"], 2)
        self.assertEqual(summary["path_cost_tradeoff_interpretation"], "tradeoff_reason_not_failure")
        self.assertIn("candidate_ranking", summary["recommended_next_adjustment"])
        self.assertIn("not_until_policy_target_selection_evidence_changes", summary["training_readiness_gate_rerun_recommendation"])
        self.assertTrue(summary["audit_only"])
        self.assertTrue(summary["does_not_modify_default_astar"])
        self.assertTrue(summary["does_not_modify_ppo"])

    def test_blocked_readiness_can_bootstrap_policy_target_evidence(self) -> None:
        self._write_sources(
            [
                self._audit_record(
                    0,
                    scenario_id="npz_changed",
                    recommendation="keep",
                    selected=True,
                    reason_codes=["channel_aware_quality_improved", "path_cost_tradeoff"],
                    blocker_reason="selected",
                    path_cost_tradeoff=True,
                    selected_candidate_changed=True,
                )
            ],
            readiness_status="failed",
            readiness_reason_codes=[
                "channel_aware_selection_contrast_calibration_summary_missing",
                "channel_aware_policy_target_selection_evidence_summary_failed",
            ],
        )

        completed = self._run_audit(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "channel-aware-policy-target-selection-evidence-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["selected_candidate_changed_rate"], 1.0)
        self.assertTrue(summary["supports_policy_target_selection_improvement_claim"])
        readiness_source = summary["source_summaries"]["channel_aware_training_readiness_summary"]
        self.assertEqual(readiness_source["status"], "failed")

    def test_validate_only_reports_git_mismatch_without_writing_summary(self) -> None:
        self._write_sources(
            [
                self._audit_record(
                    0,
                    scenario_id="npz_changed",
                    recommendation="keep",
                    selected=True,
                    reason_codes=["channel_aware_quality_improved"],
                    blocker_reason="selected",
                    selected_candidate_changed=True,
                )
            ],
            git_mismatch=True,
        )

        completed = self._run_audit(
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
        self.assertFalse(
            (self.batch_root / "channel-aware-policy-target-selection-evidence-summary.json").exists()
        )
