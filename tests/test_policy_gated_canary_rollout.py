import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PolicyGatedCanaryRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        model_explorer_src = self.repo_root / "model-explorer" / "src"
        if str(model_explorer_src) not in sys.path:
            sys.path.insert(0, str(model_explorer_src))
        self.temp_dir = Path(tempfile.mkdtemp(prefix="policy-canary-"))
        self.source_root = self.temp_dir / "source"
        self.candidate_root = self.temp_dir / "candidate"
        self.canary_root = self.temp_dir / "canary"
        for path in (self.source_root, self.candidate_root, self.canary_root):
            path.mkdir(parents=True)
        self.script = self.repo_root / "scripts" / "run_policy_gated_canary_rollout.sh"
        self.readiness_script = self.repo_root / "scripts" / "run_policy_training_readiness_review.sh"
        self.config = self.repo_root / "configs" / "policy_gated_canary_rollout_v1.json"
        self.readiness_config = self.repo_root / "configs" / "policy_training_readiness_review_v1.json"
        self.git_snapshot = self._current_git_snapshot()

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

    def _run_canary(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.script),
                "--source-root",
                str(self.source_root),
                "--candidate-root",
                str(self.candidate_root),
                "--batch-root",
                str(self.canary_root),
                "--config",
                str(self.config),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _run_readiness(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(self.readiness_script),
                "--batch-root",
                str(self.source_root),
                "--config",
                str(self.readiness_config),
                "--raw-policy-generalization-evaluation-summary",
                str(self.candidate_root / "raw-policy-generalization-evaluation-summary.json"),
                "--policy-gated-canary-rollout-summary",
                str(self.canary_root / "policy-gated-canary-rollout-summary.json"),
            ],
            cwd=self.repo_root,
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_artifacts(
        self,
        *,
        raw_selects_alternative: bool = True,
        alternative_quality_regression: bool = False,
        missing_context_id: bool = False,
        scenario_groups: tuple[str, ...] = ("policy_canary_unit",),
    ) -> None:
        self._write_source_preconditions()
        self._write_candidate_artifacts(raw_selects_alternative=raw_selects_alternative)
        self._write_canary_batch(
            alternative_quality_regression=alternative_quality_regression,
            missing_context_id=missing_context_id,
            scenario_groups=scenario_groups,
        )

    def _write_canary_config(
        self,
        *,
        validation_overrides: dict | None = None,
        output_filename_prefix: str = "policy-gated-canary",
    ) -> Path:
        payload = json.loads(self.config.read_text(encoding="utf-8"))
        payload["validation"].update(validation_overrides or {})
        payload["output_files"] = {
            "decisions": f"{output_filename_prefix}-decisions.jsonl",
            "rejection_report": f"{output_filename_prefix}-rejection-report.json",
            "opportunity_summary": f"{output_filename_prefix}-opportunity-summary.json",
            "summary": f"{output_filename_prefix}-rollout-summary.json",
        }
        path = self.temp_dir / f"{output_filename_prefix}-config.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _write_source_preconditions(self) -> None:
        common = {
            "generated_at": "2026-06-09T00:00:00Z",
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
                    "status": "passed",
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
                "does_not_modify_default_astar": True,
                "does_not_modify_ppo": True,
                "does_not_modify_network": True,
                "does_not_modify_action_space": True,
                "does_not_modify_model_explorer_contract": True,
                "does_not_modify_path_planner_route_contract": True,
                "does_not_modify_path_planner_sidecar_contract": True,
                "no_ackermann_feasible_trajectory_claim": True,
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
            (self.source_root / filename).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

    def _write_candidate_artifacts(self, *, raw_selects_alternative: bool) -> None:
        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.features import (
            CANDIDATE_FEATURE_NAMES,
            GLOBAL_FEATURE_NAMES,
            MISSING_INDICATOR_NAMES,
            PolicyObservation,
        )
        import torch

        observation = PolicyObservation(
            candidate_feature_names=CANDIDATE_FEATURE_NAMES,
            candidate_features=(
                tuple([0.0] * len(CANDIDATE_FEATURE_NAMES)),
                tuple([1.0] * len(CANDIDATE_FEATURE_NAMES)),
            ),
            global_feature_names=GLOBAL_FEATURE_NAMES,
            global_features=tuple([0.0] * len(GLOBAL_FEATURE_NAMES)),
            action_mask=(True, True),
            candidate_cells=((10, 10), (11, 10)),
            candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
            candidate_missing_indicators=(
                tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
                tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
            ),
        )
        network = build_policy_network(None, observation=observation, hidden_size=16)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            first = network.candidate_encoder[0]
            first.weight[0, 0] = 1.0 if raw_selects_alternative else -1.0
            network.candidate_encoder[1].weight.fill_(1.0)
            network.candidate_encoder[4].weight[0, 0] = 1.0
            network.policy_head[0].weight[0, 0] = 1.0
            network.policy_head[-1].weight[0, 0] = 1.0
        torch.save(
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
                "experimental": True,
                "model_state_dict": network.state_dict(),
                "training": {"hidden_size": 16},
            },
            self.candidate_root / "experimental-hybrid-policy-candidate.pt",
        )
        summary = {
            "schema_version": "raw-policy-generalization-candidate-summary/v1",
            "status": "passed",
            "candidate_training_status": "passed",
            "reason_codes": [],
            "experimental_checkpoint": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (self.candidate_root / "raw-policy-generalization-candidate-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (self.candidate_root / "raw-policy-generalization-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "raw-policy-generalization-evaluation-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "test_generalization_passed": True,
                    "test_raw_policy_regression_reduction_rate": 1.0,
                    "overfit_gap": 0.0,
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
        metadata = {
            "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
            "experimental": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
        }
        (self.candidate_root / "experimental-hybrid-policy-candidate-metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def _write_canary_batch(
        self,
        *,
        alternative_quality_regression: bool,
        missing_context_id: bool,
        scenario_groups: tuple[str, ...],
    ) -> None:
        run_root = self.canary_root / "canary-run"
        run_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        for index, scenario_group in enumerate(scenario_groups):
            candidates = [
                self._candidate(
                    0,
                    f"ctx-{index}-source",
                    source_selected=True,
                    missing_context_id=missing_context_id,
                ),
                self._candidate(
                    1,
                    f"ctx-{index}-alt",
                    source_selected=False,
                    missing_context_id=missing_context_id,
                    quality_regression=alternative_quality_regression,
                ),
            ]
            scenarios.append(
                {
                    "scenario_id": f"npz_policy_canary_unit_{index}",
                    "scenario_group": scenario_group,
                    "scenario_seed": 9701 + index,
                    "scenario_variant_id": f"npz_policy_canary_unit_{index}-seed-{9701 + index}",
                    "open_grid_fallback_used": False,
                    "tracking_safety_violation_count": 0,
                    "path_feedback": {
                        "candidate_count": 2,
                        "reachable_count": 2,
                        "failure_count": 0,
                        "replan_count": 0,
                        "best_by_path_cost": candidates[0],
                        "candidates": candidates,
                    },
                }
            )
        (run_root / "path-feedback-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "scenario_set": "policy_canary",
                    "diagnostic_profile": "execution",
                    "top_k": 2,
                    "open_grid_fallback_used": False,
                    "safety_regression_count": 0,
                    "candidate_contract_alignment_gap_count": 0,
                    "planner_extra_args": ["--planning-backend", "astar"],
                    "scenarios": scenarios,
                    "git_provenance": {"current": self.git_snapshot, "current_matches_sources": True},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.canary_root / "batch-evaluation-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-batch-evaluation-summary/v1",
                    "status": "passed",
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

    def _candidate(
        self,
        action_index: int,
        context_id: str,
        *,
        source_selected: bool,
        missing_context_id: bool,
        replan_required: bool = False,
        quality_regression: bool = False,
    ) -> dict:
        candidate = {
            "action_index": action_index,
            "source_action_index": action_index,
            "cell": [100 * action_index, 10],
            "candidate_role": "policy_target",
            "policy_target_cell": [100 * action_index, 10],
            "execution_goal_cell": [100 * action_index, 10],
            "reachable": True,
            "replan_required": replan_required,
            "open_grid_fallback_used": False,
            "path_cost": 10.0 - action_index,
            "risk": 0.1,
            "utility": 0.5 + action_index,
            "platform_goal_feasibility": {"contract_reachable": True},
            "candidate_generation": {
                "source_selection_status": "source_selected" if source_selected else "not_source_selected",
                "source_selection_quality_regression": quality_regression,
            },
        }
        if not missing_context_id:
            candidate.update(
                {
                    "context_id": context_id,
                    "context_id_schema_version": "policy-context-id/v1",
                    "context_id_source": "stable_semantic_fields",
                    "legacy_identity_fallback_used": False,
                }
            )
        return candidate

    def test_canary_accepts_safe_policy_changed_choice(self) -> None:
        self._write_artifacts()

        completed = self._run_canary()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.canary_root / "policy-gated-canary-rollout-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["policy_decision_count"], 1)
        self.assertEqual(summary["canary_opportunity_context_count"], 1)
        self.assertEqual(summary["policy_changed_decision_count"], 1)
        self.assertEqual(summary["canary_accepted_policy_choice_count"], 1)
        self.assertEqual(summary["canary_rejected_policy_choice_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        decisions = [
            json.loads(line)
            for line in (self.canary_root / "policy-gated-canary-decisions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(decisions[0]["decision_class"], "canary_accepted_policy_choice")

    def test_canary_rejects_failed_gate_and_refuses_readiness(self) -> None:
        self._write_artifacts(alternative_quality_regression=True)

        completed = self._run_canary()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.canary_root / "policy-gated-canary-rollout-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["canary_accepted_policy_choice_count"], 0)
        self.assertEqual(summary["canary_rejected_policy_choice_count"], 1)
        self.assertIn("source_selection_regression", summary["canary_rejection_reason_counts"])
        self.assertEqual(
            summary["next_required_change"],
            "policy_candidate_fails_canary_gate_requires_objective_or_feature_refinement",
        )

    def test_canary_requires_context_id(self) -> None:
        self._write_artifacts(missing_context_id=True)

        completed = self._run_canary()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.canary_root / "policy-gated-canary-rollout-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("context_id_missing", summary["reason_codes"])

    def test_readiness_advances_after_canary_accepts_policy_choice(self) -> None:
        self._write_artifacts()
        self.assertEqual(self._run_canary().returncode, 0)

        completed = self._run_readiness()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["training_readiness_status"], "policy_gated_canary_rollout_evaluated")
        self.assertEqual(summary["training_blockers"], [])

    def test_canary_diversity_summary_reports_family_coverage(self) -> None:
        self.config = self._write_canary_config(
            validation_overrides={
                "min_policy_decision_count": 2,
                "min_canary_opportunity_context_count": 2,
                "min_policy_changed_decision_count": 2,
                "min_canary_accepted_policy_choice_count": 2,
                "min_accepted_scenario_family_count": 2,
            },
            output_filename_prefix="policy-gated-canary-diversity",
        )
        self._write_artifacts(scenario_groups=("canary_family_a", "canary_family_b"))

        completed = self._run_canary()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.canary_root / "policy-gated-canary-diversity-rollout-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["canary_diversity_passed"])
        self.assertEqual(summary["accepted_scenario_family_count"], 2)
        self.assertEqual(
            summary["accepted_decision_family_distribution"],
            {"canary_family_a": 1, "canary_family_b": 1},
        )
        self.assertEqual(
            summary["scenario_family_summary"]["canary_family_a"]["canary_accepted_policy_choice_count"],
            1,
        )

    def test_canary_diversity_rejects_single_family_acceptance(self) -> None:
        self.config = self._write_canary_config(
            validation_overrides={
                "min_policy_decision_count": 2,
                "min_canary_opportunity_context_count": 2,
                "min_policy_changed_decision_count": 2,
                "min_canary_accepted_policy_choice_count": 2,
                "min_accepted_scenario_family_count": 2,
            },
            output_filename_prefix="policy-gated-canary-diversity",
        )
        self._write_artifacts(scenario_groups=("canary_family_a", "canary_family_a"))

        completed = self._run_canary()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.canary_root / "policy-gated-canary-diversity-rollout-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("accepted_scenario_family_count_below_threshold", summary["reason_codes"])
        self.assertEqual(summary["accepted_scenario_family_count"], 1)
        self.assertFalse(summary["canary_diversity_passed"])
        self.assertEqual(summary["next_required_change"], "scenario_family_coverage_insufficient")

    def test_readiness_advances_after_canary_diversity_passes(self) -> None:
        self.config = self._write_canary_config(
            validation_overrides={
                "min_policy_decision_count": 2,
                "min_canary_opportunity_context_count": 2,
                "min_policy_changed_decision_count": 2,
                "min_canary_accepted_policy_choice_count": 2,
                "min_accepted_scenario_family_count": 2,
            },
        )
        self._write_artifacts(scenario_groups=("canary_family_a", "canary_family_b"))
        self.assertEqual(self._run_canary().returncode, 0)

        completed = self._run_readiness()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            summary["training_readiness_status"],
            "policy_gated_canary_diversity_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])

    def test_opportunity_quality_reports_missing_acceptable_alternative_family(self) -> None:
        self.config = self._write_canary_config(
            validation_overrides={
                "min_policy_decision_count": 2,
                "min_canary_opportunity_context_count": 2,
                "min_policy_changed_decision_count": 1,
                "min_canary_accepted_policy_choice_count": 1,
                "min_scenario_family_count": 2,
                "min_family_with_acceptable_alternative_count": 2,
                "min_accepted_scenario_family_count": 1,
            },
            output_filename_prefix="policy-gated-canary-opportunity-quality",
        )
        self._write_artifacts(
            scenario_groups=("canary_family_has_safe_alt", "canary_family_missing_safe_alt"),
        )
        summary_path = self.canary_root / "canary-run" / "path-feedback-summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["scenarios"][1]["path_feedback"]["candidates"][1]["candidate_generation"][
            "source_selection_quality_regression"
        ] = True
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        completed = self._run_canary()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.canary_root / "policy-gated-canary-opportunity-quality-rollout-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["family_with_acceptable_alternative_count"], 1)
        self.assertEqual(
            summary["missing_acceptable_alternative_families"],
            ["canary_family_missing_safe_alt"],
        )
        self.assertIn("family_with_acceptable_alternative_count_below_threshold", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "canary_opportunity_generation_gap")

    def test_opportunity_quality_reports_source_aligned_missed_safe_choice(self) -> None:
        self.config = self._write_canary_config(
            validation_overrides={
                "min_policy_decision_count": 1,
                "min_canary_opportunity_context_count": 1,
                "min_policy_changed_decision_count": 1,
                "min_canary_accepted_policy_choice_count": 1,
                "min_family_with_acceptable_alternative_count": 1,
                "min_accepted_scenario_family_count": 1,
            },
            output_filename_prefix="policy-gated-canary-opportunity-quality",
        )
        self._write_artifacts(raw_selects_alternative=False)

        completed = self._run_canary()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.canary_root / "policy-gated-canary-opportunity-quality-rollout-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["family_with_acceptable_alternative_count"], 1)
        self.assertEqual(summary["source_aligned_with_acceptable_alternative_count"], 1)
        self.assertEqual(summary["missed_safe_choice_family_count"], 1)
        self.assertEqual(summary["canary_missed_opportunity_preference_pair_count"], 1)
        self.assertEqual(summary["hard_positive_added_count"], 0)
        self.assertEqual(summary["next_required_change"], "policy_safe_choice_alignment_insufficient")

    def test_readiness_advances_after_canary_opportunity_quality_passes(self) -> None:
        self.config = self._write_canary_config(
            validation_overrides={
                "min_policy_decision_count": 2,
                "min_canary_opportunity_context_count": 2,
                "min_policy_changed_decision_count": 2,
                "min_canary_accepted_policy_choice_count": 2,
                "min_family_with_acceptable_alternative_count": 2,
                "min_accepted_scenario_family_count": 2,
            },
        )
        self._write_artifacts(scenario_groups=("canary_family_a", "canary_family_b"))
        self.assertEqual(self._run_canary().returncode, 0)

        completed = self._run_readiness()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            summary["training_readiness_status"],
            "policy_gated_canary_opportunity_quality_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])

    def test_dense_choke_missing_acceptable_alternative_gets_specific_next_change(self) -> None:
        self.config = self._write_canary_config(
            validation_overrides={
                "min_policy_decision_count": 2,
                "min_canary_opportunity_context_count": 2,
                "min_policy_changed_decision_count": 1,
                "min_canary_accepted_policy_choice_count": 1,
                "min_scenario_family_count": 2,
                "min_family_with_acceptable_alternative_count": 2,
                "min_accepted_scenario_family_count": 1,
            },
            output_filename_prefix="policy-gated-canary-full-family",
        )
        self._write_artifacts(
            scenario_groups=("near_blocked_safe_alt", "dense_choke_safe_bypass"),
        )
        summary_path = self.canary_root / "canary-run" / "path-feedback-summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["scenarios"][1]["path_feedback"]["candidates"][1]["candidate_generation"][
            "source_selection_quality_regression"
        ] = True
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        completed = self._run_canary()

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.canary_root / "policy-gated-canary-full-family-rollout-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["missing_acceptable_alternative_families"], ["dense_choke_safe_bypass"])
        self.assertEqual(summary["next_required_change"], "dense_choke_opportunity_generation_gap")

    def test_readiness_advances_after_full_family_opportunity_passes(self) -> None:
        self.config = self._write_canary_config(
            validation_overrides={
                "min_policy_decision_count": 6,
                "min_canary_opportunity_context_count": 6,
                "min_policy_changed_decision_count": 6,
                "min_canary_accepted_policy_choice_count": 6,
                "min_scenario_family_count": 6,
                "min_family_with_acceptable_alternative_count": 6,
                "min_accepted_scenario_family_count": 6,
            },
        )
        self._write_artifacts(
            scenario_groups=(
                "mixed_stress_detour",
                "near_blocked_safe_alt",
                "channel_contrast",
                "high_risk_tradeoff",
                "dense_choke_safe_bypass",
                "path_complexity_benefit",
            )
        )
        self.assertEqual(self._run_canary().returncode, 0)

        completed = self._run_readiness()

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads(
            (self.source_root / "policy-training-readiness-review-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            summary["training_readiness_status"],
            "policy_gated_canary_full_family_opportunity_evaluated",
        )
        self.assertEqual(summary["training_blockers"], [])


if __name__ == "__main__":
    unittest.main()
