import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class ChannelAwareTrainingReadinessGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_channel_aware_training_readiness_gate.sh"
        self.config = self.repo_root / "configs" / "channel_aware_training_readiness_gate_v1.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="channel-aware-training-readiness-"))
        self.batch_root = self.temp_dir / "batch"
        self.batch_root.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _run_gate(self, *args: str) -> subprocess.CompletedProcess[str]:
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

    def _write_application_summary(
        self,
        records: list[dict],
        *,
        status: str = "passed",
        reason_codes: list[str] | None = None,
        git_mismatch: bool = False,
    ) -> Path:
        application_git = self.git_snapshot
        if git_mismatch:
            application_git = {
                **self.git_snapshot,
                "parent": {**self.git_snapshot["parent"], "sha": "0" * 40},
            }
        recommendation_counts = self._counts(record["recommendation"] for record in records)
        action_counts = self._counts(record["application_action"] for record in records)
        reason_counts = self._counts(
            reason
            for record in records
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

        payload = {
            "schema_version": "policy-robustness-application-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": status,
            "reason_codes": list(reason_codes or []),
            "batch_root": str(self.batch_root),
            "no_large_scale_training": True,
            "no_real_world_performance_claim": True,
            "does_not_modify_ppo": True,
            "does_not_modify_network": True,
            "does_not_modify_action_space": True,
            "git_provenance": {
                "robustness": {
                    "current": self.git_snapshot,
                    "current_matches_batch": not git_mismatch,
                    "runs_match_batch": not git_mismatch,
                },
                "current": application_git,
                "current_matches_robustness": not git_mismatch,
            },
            "channel_aware_application": {
                "schema_version": "channel-aware-application-smoke/v1",
                "application_scope": "channel_aware_decision_evidence_application_smoke_only",
                "quality_signal_use": "opt_in_decision_application_evidence_only",
                "no_large_scale_training": True,
                "no_real_world_performance_claim": True,
                "does_not_modify_ppo": True,
                "does_not_modify_network": True,
                "does_not_modify_action_space": True,
                "route_replacement_default_changed": False,
                "record_count": len(records),
                "recommendation_counts": dict(sorted(recommendation_counts.items())),
                "action_counts": dict(sorted(action_counts.items())),
                "reason_code_counts": dict(sorted(reason_counts.items())),
                "summary_reason_codes": ["channel_aware_application_smoke_mapped"],
                "records": records,
            },
        }
        path = self.batch_root / "policy-robustness-application-summary.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_policy_target_evidence_summary(
        self,
        *,
        selected_candidate_changed_rate: float = 0.0,
        supports_policy_target_selection_improvement_claim: bool = False,
        status: str = "passed",
        reason_codes: list[str] | None = None,
        git_mismatch: bool = False,
    ) -> Path:
        evidence_git = self._git_snapshot(git_mismatch=git_mismatch)
        payload = {
            "schema_version": "channel-aware-policy-target-selection-evidence-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": status,
            "reason_codes": list(reason_codes or []),
            "selected_candidate_changed_rate": selected_candidate_changed_rate,
            "supports_policy_target_selection_improvement_claim": (
                supports_policy_target_selection_improvement_claim
            ),
            "git_provenance": {
                "current": evidence_git,
                "readiness": {"current": evidence_git},
            },
        }
        path = self.batch_root / "channel-aware-policy-target-selection-evidence-summary.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_policy_decision_robustness_summary(
        self,
        *,
        status: str = "passed",
        reason_codes: list[str] | None = None,
        git_mismatch: bool = False,
    ) -> Path:
        robustness_git = self._git_snapshot(git_mismatch=git_mismatch)
        payload = {
            "schema_version": "policy-decision-robustness-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": status,
            "reason_codes": list(reason_codes or []),
            "git_provenance": {
                "batch": robustness_git,
                "current": robustness_git,
                "runs_match_batch": not git_mismatch,
                "current_matches_batch": not git_mismatch,
            },
        }
        path = self.batch_root / "policy-decision-robustness-summary.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_selection_contrast_calibration_summary(
        self,
        *,
        selected_candidate_changed_rate: float = 0.25,
        selected_candidate_changed_count: int = 1,
        changed_scenario_ids: list[str] | None = None,
        safety_regression_count: int = 0,
        source_selected_candidate_changed_rate: float = 0.0,
        status: str = "passed",
        reason_codes: list[str] | None = None,
        git_mismatch: bool = False,
    ) -> Path:
        calibration_git = self._git_snapshot(git_mismatch=git_mismatch)
        payload = {
            "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
            "generated_at": "2026-06-07T00:00:00Z",
            "status": status,
            "reason_codes": list(reason_codes or []),
            "source_selected_candidate_changed_rate": source_selected_candidate_changed_rate,
            "source_supports_policy_target_selection_improvement_claim": False,
            "selected_candidate_changed_count": selected_candidate_changed_count,
            "selected_candidate_changed_rate": selected_candidate_changed_rate,
            "changed_scenario_ids": list(changed_scenario_ids or ["npz_fixture"]),
            "safety_regression_count": safety_regression_count,
            "recommended_training_readiness_action": (
                "rerun_training_readiness_gate_after_selection_contrast_calibration"
            ),
            "runs_training": False,
            "does_not_modify_default_astar": True,
            "does_not_modify_ppo": True,
            "does_not_modify_network": True,
            "does_not_modify_action_space": True,
            "does_not_modify_path_planner_route_contract": True,
            "does_not_modify_model_explorer_contract": True,
            "does_not_modify_path_planner_sidecar_contract": True,
            "no_ackermann_feasible_trajectory_claim": True,
            "git_provenance": {
                "current": calibration_git,
                "policy_target_evidence": {"current": calibration_git},
                "selection_comparison": {"current": calibration_git},
            },
        }
        path = self.batch_root / "channel-aware-selection-contrast-calibration-summary.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _git_snapshot(self, *, git_mismatch: bool = False) -> dict:
        snapshot = self.git_snapshot
        if git_mismatch:
            snapshot = {
                **self.git_snapshot,
                "parent": {**self.git_snapshot["parent"], "sha": "0" * 40},
            }
        return snapshot

    def _record(
        self,
        action_index: int,
        *,
        recommendation: str,
        application_action: str,
        sample_weight: float,
        selected_candidate_changed: bool = False,
        reason_codes: list[str] | None = None,
        application_reason_codes: list[str] | None = None,
        path_cost_tradeoff: bool = False,
    ) -> dict:
        return {
            "pair_key": "all-all-k3",
            "astar_run_id": "all-all-k3-astar",
            "channel_aware_run_id": "all-all-k3-channel-aware",
            "scenario_id": "npz_fixture",
            "scenario_group": "stress",
            "action_index": action_index,
            "cell": [action_index + 1, action_index + 2],
            "selected_candidate_changed": selected_candidate_changed,
            "recommendation": recommendation,
            "application_action": application_action,
            "application_sample_weight": sample_weight,
            "quality_improvement": application_action == "keep_quality_evidence",
            "risk_or_high_cost_improvement": application_action == "keep_quality_evidence",
            "path_cost_tradeoff": path_cost_tradeoff,
            "reason_codes": list(reason_codes or []),
            "application_reason_codes": list(application_reason_codes or []),
            "comparison": {
                "path_changed": path_cost_tradeoff,
                "path_cost_delta": 1.0 if path_cost_tradeoff else None,
                "channel_cost_delta": -2.0 if path_cost_tradeoff else None,
            },
        }

    def _counts(self, values) -> dict:
        result: dict[str, int] = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return result

    def test_gate_summarizes_training_readiness_without_claiming_selection_improvement(self) -> None:
        self._write_application_summary(
            [
                self._record(
                    0,
                    recommendation="keep",
                    application_action="keep_quality_evidence",
                    sample_weight=1.0,
                    reason_codes=["channel_aware_quality_improved", "path_cost_tradeoff"],
                    application_reason_codes=["channel_aware_application_keep_quality_evidence"],
                    path_cost_tradeoff=True,
                ),
                self._record(
                    1,
                    recommendation="downweight",
                    application_action="downweight_conservative_application",
                    sample_weight=0.5,
                    reason_codes=["channel_aware_quality_not_improved", "same_as_baseline"],
                    application_reason_codes=[
                        "channel_aware_application_downweight_conservative_application"
                    ],
                ),
                self._record(
                    2,
                    recommendation="reject",
                    application_action="exclude_blocked_candidate_evidence",
                    sample_weight=0.0,
                    reason_codes=["goal_blocked"],
                    application_reason_codes=[
                        "channel_aware_application_exclude_blocked_candidate_evidence"
                    ],
                ),
                self._record(
                    3,
                    recommendation="needs_more_evidence",
                    application_action="downweight_needs_more_evidence",
                    sample_weight=0.5,
                    reason_codes=["same_as_baseline"],
                    application_reason_codes=["channel_aware_application_downweight_needs_more_evidence"],
                ),
            ]
        )
        self._write_policy_target_evidence_summary(
            selected_candidate_changed_rate=0.0,
            supports_policy_target_selection_improvement_claim=False,
        )
        self._write_policy_decision_robustness_summary()
        self._write_selection_contrast_calibration_summary(
            selected_candidate_changed_rate=0.25,
            selected_candidate_changed_count=1,
            changed_scenario_ids=["npz_fixture"],
            safety_regression_count=0,
            source_selected_candidate_changed_rate=0.0,
        )

        completed = self._run_gate(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "channel-aware-training-readiness-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["schema_version"], "channel-aware-training-readiness-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["readiness_status"], "ready_for_calibrated_policy_application_smoke")
        self.assertEqual(
            summary["calibrated_readiness_status"],
            "ready_for_calibrated_policy_application_smoke",
        )
        self.assertEqual(summary["source_selected_candidate_changed_rate"], 0.0)
        self.assertFalse(summary["source_supports_policy_target_selection_improvement_claim"])
        self.assertFalse(summary["policy_target_selection_improvement_claimed"])
        self.assertEqual(
            summary["robustness_summary_path"],
            str(self.batch_root / "policy-decision-robustness-summary.json"),
        )
        self.assertEqual(summary["calibration_selected_candidate_changed_rate"], 0.25)
        self.assertEqual(summary["calibration_changed_scenario_ids"], ["npz_fixture"])
        self.assertEqual(summary["calibration_safety_regression_count"], 0)
        self.assertIn(
            "calibrated_target_contrast_available",
            summary["calibrated_readiness_reason_codes"],
        )
        self.assertNotIn("policy_target_selection_not_improved", summary["readiness_reason_codes"])
        self.assertNotIn("path_cost_tradeoff", summary["readiness_reason_codes"])
        self.assertEqual(summary["application_action_counts"]["keep_quality_evidence"], 1)
        self.assertEqual(summary["application_action_counts"]["downweight_conservative_application"], 1)
        self.assertEqual(summary["application_action_counts"]["exclude_blocked_candidate_evidence"], 1)
        self.assertEqual(summary["recommendation_counts"]["needs_more_evidence"], 1)
        self.assertEqual(summary["excluded_candidate_count"], 1)
        self.assertEqual(summary["positive_evidence_count"], 1)
        self.assertEqual(summary["downweighted_evidence_count"], 2)
        self.assertEqual(summary["selected_candidate_changed_rate"], 0.0)
        self.assertEqual(summary["sample_weight_distribution"]["0.0"], 1)
        self.assertEqual(summary["sample_weight_distribution"]["0.5"], 2)
        self.assertEqual(summary["sample_weight_distribution"]["1.0"], 1)
        self.assertEqual(summary["reason_code_counts"]["goal_blocked"], 1)
        self.assertEqual(summary["reason_code_counts"]["same_as_baseline"], 2)
        self.assertEqual(summary["reason_code_counts"]["path_cost_tradeoff"], 1)
        self.assertTrue(summary["no_ppo_training"])
        self.assertTrue(summary["does_not_modify_network"])
        self.assertTrue(summary["does_not_modify_action_space"])
        self.assertFalse(summary["route_replacement_default_changed"])
        self.assertEqual(
            summary["recommended_next_action"],
            "run_calibrated_policy_application_smoke_before_training",
        )
        self.assertIn("policy_application_smoke", summary["conservative_next_step"])

    def test_gate_blocks_when_calibration_summary_missing(self) -> None:
        self._write_application_summary(
            [
                self._record(
                    0,
                    recommendation="keep",
                    application_action="keep_quality_evidence",
                    sample_weight=1.0,
                    selected_candidate_changed=False,
                    reason_codes=["channel_aware_quality_improved"],
                    application_reason_codes=["channel_aware_application_keep_quality_evidence"],
                )
            ]
        )
        self._write_policy_target_evidence_summary()
        self._write_policy_decision_robustness_summary()

        completed = self._run_gate(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--validate-only",
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        validation = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(validation["status"], "validation failed")
        self.assertIn(
            "channel_aware_selection_contrast_calibration_summary_missing",
            validation["reason_codes"],
        )
        self.assertEqual(validation["readiness_status"], "blocked")
        self.assertFalse((self.batch_root / "channel-aware-training-readiness-summary.json").exists())

    def test_gate_blocks_when_robustness_summary_missing(self) -> None:
        self._write_application_summary(
            [
                self._record(
                    0,
                    recommendation="keep",
                    application_action="keep_quality_evidence",
                    sample_weight=1.0,
                    selected_candidate_changed=False,
                    reason_codes=["channel_aware_quality_improved"],
                    application_reason_codes=["channel_aware_application_keep_quality_evidence"],
                )
            ]
        )
        self._write_policy_target_evidence_summary()
        self._write_selection_contrast_calibration_summary()

        completed = self._run_gate(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
            "--validate-only",
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        validation = json.loads(completed.stdout.splitlines()[0])
        self.assertEqual(validation["status"], "validation failed")
        self.assertIn("policy_decision_robustness_summary_missing", validation["reason_codes"])
        self.assertEqual(validation["readiness_status"], "blocked")

    def test_gate_blocks_failed_or_stale_calibration_summary(self) -> None:
        for status, reason_codes, git_mismatch, expected_reason in (
            (
                "failed",
                ["calibration_failed_fixture"],
                False,
                "channel_aware_selection_contrast_calibration_summary_failed",
            ),
            (
                "passed",
                [],
                True,
                "current_git_provenance_mismatch",
            ),
        ):
            with self.subTest(status=status, git_mismatch=git_mismatch):
                shutil.rmtree(self.batch_root)
                self.batch_root.mkdir(parents=True)
                self._write_application_summary(
                    [
                        self._record(
                            0,
                            recommendation="keep",
                            application_action="keep_quality_evidence",
                            sample_weight=1.0,
                            selected_candidate_changed=False,
                            reason_codes=["channel_aware_quality_improved"],
                            application_reason_codes=["channel_aware_application_keep_quality_evidence"],
                        )
                    ]
                )
                self._write_policy_target_evidence_summary()
                self._write_policy_decision_robustness_summary()
                self._write_selection_contrast_calibration_summary(
                    status=status,
                    reason_codes=reason_codes,
                    git_mismatch=git_mismatch,
                )

                completed = self._run_gate(
                    "--batch-root",
                    str(self.batch_root),
                    "--config",
                    str(self.config),
                    "--validate-only",
                )

                self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                validation = json.loads(completed.stdout.splitlines()[0])
                self.assertEqual(validation["status"], "validation failed")
                self.assertIn(expected_reason, validation["reason_codes"])
                self.assertEqual(validation["readiness_status"], "blocked")

    def test_gate_blocks_calibration_safety_regressions_before_policy_smoke(self) -> None:
        self._write_application_summary(
            [
                self._record(
                    0,
                    recommendation="keep",
                    application_action="keep_quality_evidence",
                    sample_weight=1.0,
                    selected_candidate_changed=False,
                    reason_codes=["channel_aware_quality_improved"],
                    application_reason_codes=["channel_aware_application_keep_quality_evidence"],
                )
            ]
        )
        self._write_policy_target_evidence_summary()
        self._write_policy_decision_robustness_summary()
        self._write_selection_contrast_calibration_summary(
            selected_candidate_changed_rate=0.25,
            selected_candidate_changed_count=1,
            safety_regression_count=1,
        )

        completed = self._run_gate(
            "--batch-root",
            str(self.batch_root),
            "--config",
            str(self.config),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.batch_root / "channel-aware-training-readiness-summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["readiness_status"], "blocked")
        self.assertEqual(summary["calibrated_readiness_status"], "blocked")
        self.assertIn(
            "calibration_safety_regression_blocks_policy_smoke",
            summary["calibrated_readiness_reason_codes"],
        )
        self.assertEqual(
            summary["recommended_next_action"],
            "resolve_calibration_safety_regressions_before_policy_smoke",
        )

    def test_validate_only_reports_git_mismatch_without_writing_summary(self) -> None:
        self._write_application_summary(
            [
                self._record(
                    0,
                    recommendation="keep",
                    application_action="keep_quality_evidence",
                    sample_weight=1.0,
                    selected_candidate_changed=True,
                    reason_codes=["channel_aware_quality_improved"],
                    application_reason_codes=["channel_aware_application_keep_quality_evidence"],
                )
            ],
            git_mismatch=True,
        )
        self._write_policy_target_evidence_summary()
        self._write_policy_decision_robustness_summary()
        self._write_selection_contrast_calibration_summary()

        completed = self._run_gate(
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
        self.assertFalse((self.batch_root / "channel-aware-training-readiness-summary.json").exists())
