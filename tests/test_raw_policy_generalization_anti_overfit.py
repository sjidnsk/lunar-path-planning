import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class RawPolicyGeneralizationAntiOverfitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="raw-policy-generalization-"))
        self.source_root = self.temp_dir / "source"
        self.train_root = self.temp_dir / "train"
        self.val_root = self.temp_dir / "val"
        self.test_root = self.temp_dir / "test"
        self.candidate_root = self.temp_dir / "candidate"
        for path in (
            self.source_root,
            self.train_root,
            self.val_root,
            self.test_root,
            self.candidate_root,
        ):
            path.mkdir(parents=True)
        self.git_snapshot = self._current_git_snapshot()
        self.candidate_script = self.repo_root / "scripts" / "run_raw_policy_generalization_candidate.sh"
        self.readiness_script = self.repo_root / "scripts" / "run_policy_training_readiness_review.sh"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHON"] = str(Path("/home/kai/anaconda3/envs/lunar-explorer/bin/python"))
        return env

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

    def test_raw_align_batch_configs_are_declared_and_validate(self) -> None:
        for split in ("train", "val", "test"):
            config = self.repo_root / f"configs/path_feedback_batch_raw_policy_generalization_{split}_v1.json"
            self.assertTrue(config.is_file(), config)
            completed = subprocess.run(
                [
                    "bash",
                    str(self.repo_root / "scripts" / "run_batch_path_feedback_validation.sh"),
                    "--matrix",
                    str(config),
                    "--validate-only",
                ],
                cwd=self.repo_root,
                env=self._env(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads(config.read_text(encoding="utf-8"))
            scenario_sets = {run["scenario_set"] for run in payload["runs"]}
            self.assertEqual(scenario_sets, {f"raw_align_{split}"})

    def test_candidate_refuses_val_or_test_context_leakage(self) -> None:
        self._write_source_preconditions()
        self._write_raw_mining(self.train_root, sample_context_id="ctx-train")
        self._write_raw_mining(self.val_root, sample_context_id="ctx-train", diagnostic_only=True)
        self._write_raw_mining(self.test_root, sample_context_id="ctx-test", diagnostic_only=True)

        completed = subprocess.run(
            [
                "bash",
                str(self.candidate_script),
                "--source-root",
                str(self.source_root),
                "--train-mining-root",
                str(self.train_root),
                "--val-diagnostic-root",
                str(self.val_root),
                "--test-diagnostic-root",
                str(self.test_root),
                "--output-root",
                str(self.candidate_root),
                "--config",
                str(self.repo_root / "configs/raw_policy_generalization_candidate_v1.json"),
                "--validate-only",
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.candidate_root / "raw-policy-generalization-candidate-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("train_eval_context_leakage_detected", summary["reason_codes"])
        self.assertEqual(summary["leaked_context_id_count"], 1)

    def test_readiness_advances_after_raw_policy_generalization_passes(self) -> None:
        self._write_source_preconditions()
        generalization_summary = self.test_root / "raw-policy-generalization-evaluation-summary.json"
        generalization_summary.write_text(
            json.dumps(
                {
                    "schema_version": "raw-policy-generalization-evaluation-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "test_generalization_passed": True,
                    "test_raw_policy_regression_reduction_rate": 0.8,
                    "overfit_gap": 0.1,
                    "test_regression_count": 0,
                    "test_invalid_action_mask_count": 0,
                    "test_fallback_or_open_grid_count": 0,
                    "test_safety_regression_count": 0,
                    "test_contract_violation_count": 0,
                    "test_path_cost_regression_count": 0,
                    "test_risk_regression_count": 0,
                    "test_source_selection_regression_count": 0,
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

        completed = subprocess.run(
            [
                "bash",
                str(self.readiness_script),
                "--batch-root",
                str(self.source_root),
                "--config",
                str(self.repo_root / "configs/policy_training_readiness_review_v1.json"),
                "--raw-policy-generalization-evaluation-summary",
                str(generalization_summary),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["training_readiness_status"], "raw_policy_generalization_evaluated")
        self.assertEqual(summary["training_blockers"], [])

    def _write_raw_mining(
        self,
        root: Path,
        *,
        sample_context_id: str,
        diagnostic_only: bool = False,
    ) -> None:
        sample = {
            "schema_version": "raw-policy-regression-preference-sample/v1",
            "sample_type": "raw_policy_regression_preference_pair",
            "context_id": sample_context_id,
            "alternative_context_id": sample_context_id + "-alt",
            "sample_weight": 1.0,
            "preferred": {"context_id": sample_context_id, "candidate_features": [0.0] * 8},
            "alternative": {"context_id": sample_context_id + "-alt", "candidate_features": [1.0] * 8},
        }
        summary = {
            "schema_version": "raw-policy-regression-mining-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "raw_policy_regression_input_count": 1,
            "raw_policy_regression_preference_pair_count": 0 if diagnostic_only else 1,
            "raw_policy_regression_diagnostic_count": 1 if diagnostic_only else 0,
            "hard_positive_added_count": 0,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (root / "raw-policy-regression-mining-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (root / "raw-policy-regression-preference-samples.jsonl").write_text(
            "" if diagnostic_only else json.dumps(sample) + "\n",
            encoding="utf-8",
        )
        (root / "raw-policy-regression-diagnostics.jsonl").write_text(
            json.dumps({"context_id": sample_context_id, "split_role": "eval_diagnostic"}) + "\n",
            encoding="utf-8",
        )

    def _write_source_preconditions(self) -> None:
        common = {
            "generated_at": "2026-06-08T00:00:00Z",
            "status": "passed",
            "reason_codes": [],
            "source_selected_candidate_changed_rate": 0.0,
            "safety_regression_count": 0,
            "open_grid_fallback_used_count": 0,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (self.source_root / "batch-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-batch-evaluation-summary/v1",
                    "run_count": 1,
                    "passed_count": 1,
                    "failed_count": 0,
                    "reason_codes": [],
                    "open_grid_fallback_used_count": 0,
                    "safety_regression_count": 0,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        payloads = {
            "calibrated-policy-application-smoke-summary.json": {
                **common,
                "schema_version": "calibrated-policy-application-smoke-summary/v1",
                "calibrated_selected_candidate_changed_rate": 0.5,
                "applied_calibrated_candidate_count": 2,
                "rejected_goal_blocked_count": 0,
                "platform_goal_contract_mismatch_count": 0,
                "recommended_next_action": "ready_for_policy_training_readiness_review",
            },
            "channel-aware-training-readiness-summary.json": {
                **common,
                "schema_version": "channel-aware-training-readiness-summary/v1",
                "readiness_status": "ready_for_calibrated_policy_application_smoke",
                "calibrated_readiness_status": "ready_for_calibrated_policy_application_smoke",
                "calibration_selected_candidate_changed_rate": 0.5,
                "calibration_safety_regression_count": 0,
            },
            "channel-aware-contrast-coverage-summary.json": {
                **common,
                "schema_version": "channel-aware-contrast-coverage-summary/v1",
                "calibrated_selected_candidate_changed_rate": 0.5,
                "blocked_candidate_rate": 0.0,
                "recommended_next_action": "ready_for_calibrated_policy_application_smoke",
            },
            "channel-aware-selection-contrast-calibration-summary.json": {
                **common,
                "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
                "selected_candidate_changed_count": 2,
                "selected_candidate_changed_rate": 0.5,
                "goal_blocked_count": 0,
                "platform_goal_contract_mismatch_count": 0,
            },
        }
        for filename, payload in payloads.items():
            (self.source_root / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
