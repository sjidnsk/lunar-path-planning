import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealPpoCollectorDryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="quasi-real-ppo-collector-"))
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.guarded_root = self.temp_dir / "guarded"
        self.candidate_root = self.temp_dir / "candidate"
        self.quasi_real_root = self.temp_dir / "quasi"
        self.output_root = self.temp_dir / "out"
        for path in (self.guarded_root, self.candidate_root, self.quasi_real_root):
            path.mkdir(parents=True, exist_ok=True)
        self._write_quasi_real_files()
        self._write_guarded_summary()

    def test_train_teacher_aligned_decision_materializes_one_step_ppo_transition(self) -> None:
        from model_explorer.policy.rollout_io import read_rollout_episodes
        from scripts.run_quasi_real_ppo_collector_dry_run import (
            run_quasi_real_ppo_collector_dry_run,
        )

        self._write_decisions(
            [
                self._decision(
                    "ctx-train-a1",
                    scenario_id="scenario-train-a",
                    split="train",
                    controlled_choice_source="policy_teacher_aligned",
                    controlled_action_index=1,
                    log_prob=-0.25,
                    value=0.75,
                )
            ]
        )

        summary = run_quasi_real_ppo_collector_dry_run(
            guarded_teacher_following_root=self.guarded_root,
            output_root=self.output_root,
            config=self._config(min_trainable=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["episode_count"], 1)
        self.assertEqual(summary["step_count"], 1)
        self.assertEqual(summary["ppo_trainable_transition_count"], 1)
        self.assertEqual(summary["diagnostic_transition_count"], 0)
        self.assertEqual(summary["source_fallback_trainable_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_training_ready_claimed"])

        episodes = read_rollout_episodes(self.output_root / "ppo-rollout-episodes.jsonl")
        self.assertEqual(len(episodes), 1)
        transition = episodes[0].transitions[0]
        self.assertTrue(transition.done)
        self.assertEqual(transition.action_index, 1)
        self.assertAlmostEqual(transition.log_prob, -0.25)
        self.assertAlmostEqual(transition.value, 0.75)
        self.assertEqual(transition.info.extra["controlled_choice_source"], "policy_teacher_aligned")
        self.assertTrue(transition.info.extra["ppo_trainable"])
        self.assertEqual(transition.info.extra["split"], "train")
        self.assertEqual(transition.info.extra["reward_components"]["teacher_following_bonus"], 1.0)
        self.assertGreater(transition.reward, 0.0)

    def test_validation_and_test_decisions_remain_diagnostic_only(self) -> None:
        from model_explorer.policy.rollout_io import read_rollout_episodes
        from scripts.run_quasi_real_ppo_collector_dry_run import (
            run_quasi_real_ppo_collector_dry_run,
        )

        self._write_decisions(
            [
                self._decision(
                    "ctx-train-a1",
                    scenario_id="scenario-train-a",
                    split="train",
                    controlled_choice_source="policy_teacher_aligned",
                    controlled_action_index=1,
                    log_prob=-0.25,
                    value=0.75,
                ),
                self._decision(
                    "ctx-validation-a1",
                    scenario_id="scenario-validation-a",
                    split="validation",
                    controlled_choice_source="policy_teacher_aligned",
                    controlled_action_index=1,
                    log_prob=-0.35,
                    value=0.65,
                ),
                self._decision(
                    "ctx-test-a1",
                    scenario_id="scenario-test-a",
                    split="test",
                    controlled_choice_source="policy_teacher_aligned",
                    controlled_action_index=1,
                    log_prob=-0.45,
                    value=0.55,
                ),
            ]
        )

        summary = run_quasi_real_ppo_collector_dry_run(
            guarded_teacher_following_root=self.guarded_root,
            output_root=self.output_root,
            config=self._config(min_trainable=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["episode_count"], 3)
        self.assertEqual(summary["step_count"], 3)
        self.assertEqual(summary["ppo_trainable_transition_count"], 1)
        self.assertEqual(summary["diagnostic_transition_count"], 2)

        episodes = read_rollout_episodes(self.output_root / "ppo-rollout-episodes.jsonl")
        self.assertEqual(len(episodes), 1)
        transitions = self._read_jsonl(self.output_root / "ppo-rollout-transitions.jsonl")
        self.assertEqual([record["split"] for record in transitions if record["ppo_trainable"]], ["train"])
        self.assertEqual(
            sorted(record["split"] for record in transitions if record["diagnostic_only"]),
            ["test", "validation"],
        )

    def test_fallback_unsafe_and_gate_reason_decisions_cannot_be_trainable(self) -> None:
        from scripts.run_quasi_real_ppo_collector_dry_run import (
            run_quasi_real_ppo_collector_dry_run,
        )

        self._write_decisions(
            [
                self._decision(
                    "ctx-train-a1",
                    scenario_id="scenario-train-a",
                    split="train",
                    controlled_choice_source="policy_teacher_aligned",
                    controlled_action_index=1,
                    log_prob=-0.25,
                    value=0.75,
                ),
                self._decision(
                    "ctx-train-a0",
                    scenario_id="scenario-train-a",
                    split="train",
                    controlled_choice_source="teacher_fallback",
                    controlled_action_index=0,
                    decision_class="policy_changed_gate_rejected",
                    gate_reason_codes=["path_cost_regression"],
                    log_prob=-0.45,
                    value=0.55,
                ),
                self._decision(
                    "ctx-train-b1",
                    scenario_id="scenario-train-b",
                    split="train",
                    controlled_choice_source="policy_safe_disagreement",
                    controlled_action_index=1,
                    gate_reason_codes=["risk_regression"],
                    log_prob=-0.35,
                    value=0.65,
                ),
            ]
        )

        summary = run_quasi_real_ppo_collector_dry_run(
            guarded_teacher_following_root=self.guarded_root,
            output_root=self.output_root,
            config=self._config(min_trainable=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["ppo_trainable_transition_count"], 1)
        self.assertEqual(summary["diagnostic_transition_count"], 2)
        transitions = self._read_jsonl(self.output_root / "ppo-rollout-transitions.jsonl")
        rejected = [record for record in transitions if record["diagnostic_only"]]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(any("controlled_choice_source_not_trainable" in record["rejection_reason_codes"] for record in rejected))
        self.assertTrue(any("gate_reason_codes_present" in record["rejection_reason_codes"] for record in rejected))

    def test_missing_observation_logprob_value_and_non_finite_reward_fail_collector(self) -> None:
        from scripts.run_quasi_real_ppo_collector_dry_run import (
            run_quasi_real_ppo_collector_dry_run,
        )

        self._write_decisions(
            [
                self._decision(
                    "ctx-missing",
                    scenario_id="scenario-missing",
                    split="train",
                    controlled_choice_source="policy_teacher_aligned",
                    controlled_action_index=1,
                    log_prob=None,
                    value=None,
                ),
                self._decision(
                    "ctx-train-a1",
                    scenario_id="scenario-train-a",
                    split="train",
                    controlled_choice_source="policy_teacher_aligned",
                    controlled_action_index=1,
                    log_prob=-0.25,
                    value=0.75,
                    path_cost_delta=math.inf,
                ),
            ]
        )

        summary = run_quasi_real_ppo_collector_dry_run(
            guarded_teacher_following_root=self.guarded_root,
            output_root=self.output_root,
            config=self._config(min_trainable=1),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("ppo_rollout_collector_contract_invalid", summary["reason_codes"])
        self.assertIn("ppo_logprob_value_missing", summary["reason_codes"])
        self.assertIn("ppo_reward_contract_invalid", summary["reason_codes"])
        self.assertEqual(summary["invalid_action_mask_count"], 1)
        self.assertEqual(summary["missing_log_prob_count"], 1)
        self.assertEqual(summary["missing_value_count"], 1)
        self.assertEqual(summary["non_finite_reward_count"], 1)

    def test_readiness_prefers_quasi_real_collector_when_teacher_pilot_is_also_present(self) -> None:
        from scripts.run_policy_training_readiness_review import analyze_policy_training_readiness_review

        batch_root = self.temp_dir / "readiness-batch"
        batch_root.mkdir(parents=True)
        self._write_minimal_readiness_base_summaries(batch_root)
        teacher_summary = self.guarded_root / "quasi-real-guarded-teacher-following-pilot-summary.json"
        teacher_payload = json.loads(teacher_summary.read_text(encoding="utf-8"))
        teacher_payload.update(
            {
                "quasi_real_context_count": 108,
                "policy_decision_count": 108,
                "teacher_following_step_count": 108,
                "teacher_agreement_rate": 1.0,
            }
        )
        self._write_json(teacher_summary, teacher_payload)
        collector_summary = self.temp_dir / "collector-summary.json"
        self._write_json(
            collector_summary,
            {
                "schema_version": "ppo-rollout-collector-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "episode_count": 108,
                "step_count": 108,
                "ppo_trainable_transition_count": 36,
                "diagnostic_transition_count": 72,
                "source_fallback_trainable_count": 0,
                "invalid_action_mask_count": 0,
                "empty_action_mask_count": 0,
                "missing_log_prob_count": 0,
                "missing_value_count": 0,
                "non_finite_reward_count": 0,
                "state_continuity_violation_count": 0,
                "fallback_or_open_grid_count": 0,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_training_ready_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )

        summary = analyze_policy_training_readiness_review(
            batch_root=batch_root,
            smoke_path=batch_root / "calibrated-policy-application-smoke-summary.json",
            readiness_path=batch_root / "channel-aware-training-readiness-summary.json",
            coverage_path=batch_root / "channel-aware-contrast-coverage-summary.json",
            calibration_path=batch_root / "channel-aware-selection-contrast-calibration-summary.json",
            anchor_candidate_path=batch_root / "anchor-projection-candidate-generation-summary.json",
            anchor_contract_path=batch_root / "anchor-projection-evidence-contract-summary.json",
            contract_aware_target_path=batch_root / "anchor-projection-contract-aware-trainable-target-summary.json",
            planner_validated_mining_path=batch_root / "planner-validated-trainable-target-mining-summary.json",
            hybrid_training_dry_run_path=batch_root / "hybrid-policy-training-dry-run-summary.json",
            controlled_candidate_path=batch_root / "controlled-hybrid-policy-training-candidate-summary.json",
            controlled_holdout_path=batch_root / "controlled-hybrid-policy-holdout-evaluation-summary.json",
            fresh_holdout_path=batch_root / "fresh-holdout-policy-candidate-evaluation-summary.json",
            scenario_rollout_path=batch_root / "scenario-disjoint-policy-rollout-evaluation-summary.json",
            raw_strict_rollout_path=batch_root / "raw-policy-strict-rollout-evaluation-summary.json",
            raw_generalization_path=batch_root / "raw-policy-generalization-evaluation-summary.json",
            policy_canary_path=batch_root / "policy-gated-canary-rollout-summary.json",
            sequential_canary_path=batch_root / "policy-gated-sequential-canary-rollout-summary.json",
            ppo_collector_path=collector_summary,
            limited_ppo_update_smoke_path=batch_root / "limited-ppo-update-smoke-summary.json",
            iterative_ppo_mini_loop_path=batch_root / "iterative-ppo-mini-loop-stability-summary.json",
            guarded_ppo_rollout_pilot_path=batch_root / "guarded-ppo-rollout-pilot-summary.json",
            policy_training_cuda_device_support_path=batch_root / "policy-training-cuda-device-support-summary.json",
            quasi_real_map_domain_gap_path=batch_root / "quasi-real-map-domain-gap-summary.json",
            quasi_real_shadow_policy_behavior_path=batch_root / "quasi-real-shadow-policy-behavior-summary.json",
            quasi_real_shadow_alignment_path=batch_root / "quasi-real-shadow-alignment-summary.json",
            quasi_real_guarded_policy_pilot_path=batch_root / "quasi-real-guarded-policy-pilot-summary.json",
            quasi_real_safe_alternative_opportunity_path=batch_root / "quasi-real-safe-alternative-opportunity-summary.json",
            quasi_real_safe_better_opportunity_expansion_path=batch_root
            / "quasi-real-safe-better-opportunity-expansion-summary.json",
            quasi_real_teacher_equivalent_validation_path=batch_root / "quasi-real-teacher-equivalent-summary.json",
            quasi_real_teacher_distillation_path=batch_root / "quasi-real-teacher-distillation-summary.json",
            config=self._readiness_config(),
            repo_root=self.repo_root,
            quasi_real_guarded_teacher_following_pilot_path=teacher_summary,
            quasi_real_guarded_teacher_following_pilot_required=True,
            ppo_collector_required=True,
        )

        self.assertEqual(summary["training_readiness_status"], "ppo_rollout_collector_dry_run_evaluated")
        self.assertEqual(summary["training_blockers"], [])

    def test_readiness_ignores_stale_base_provenance_for_explicit_quasi_real_collector(self) -> None:
        from scripts.run_policy_training_readiness_review import analyze_policy_training_readiness_review

        batch_root = self.temp_dir / "readiness-stale-base"
        batch_root.mkdir(parents=True)
        stale_git = self._stale_git()
        self._write_minimal_readiness_base_summaries(batch_root, git_provenance=stale_git)
        teacher_summary = self.guarded_root / "quasi-real-guarded-teacher-following-pilot-summary.json"
        teacher_payload = json.loads(teacher_summary.read_text(encoding="utf-8"))
        teacher_payload.update(
            {
                "quasi_real_context_count": 108,
                "policy_decision_count": 108,
                "teacher_following_step_count": 108,
                "teacher_agreement_rate": 1.0,
                "git_provenance": stale_git,
            }
        )
        self._write_json(teacher_summary, teacher_payload)

        collector_summary = self.temp_dir / "collector-current-summary.json"
        self._write_json(collector_summary, self._collector_readiness_summary())

        summary = analyze_policy_training_readiness_review(
            batch_root=batch_root,
            smoke_path=batch_root / "calibrated-policy-application-smoke-summary.json",
            readiness_path=batch_root / "channel-aware-training-readiness-summary.json",
            coverage_path=batch_root / "channel-aware-contrast-coverage-summary.json",
            calibration_path=batch_root / "channel-aware-selection-contrast-calibration-summary.json",
            anchor_candidate_path=batch_root / "anchor-projection-candidate-generation-summary.json",
            anchor_contract_path=batch_root / "anchor-projection-evidence-contract-summary.json",
            contract_aware_target_path=batch_root / "anchor-projection-contract-aware-trainable-target-summary.json",
            planner_validated_mining_path=batch_root / "planner-validated-trainable-target-mining-summary.json",
            hybrid_training_dry_run_path=batch_root / "hybrid-policy-training-dry-run-summary.json",
            controlled_candidate_path=batch_root / "controlled-hybrid-policy-training-candidate-summary.json",
            controlled_holdout_path=batch_root / "controlled-hybrid-policy-holdout-evaluation-summary.json",
            fresh_holdout_path=batch_root / "fresh-holdout-policy-candidate-evaluation-summary.json",
            scenario_rollout_path=batch_root / "scenario-disjoint-policy-rollout-evaluation-summary.json",
            raw_strict_rollout_path=batch_root / "raw-policy-strict-rollout-evaluation-summary.json",
            raw_generalization_path=batch_root / "raw-policy-generalization-evaluation-summary.json",
            policy_canary_path=batch_root / "policy-gated-canary-rollout-summary.json",
            sequential_canary_path=batch_root / "policy-gated-sequential-canary-rollout-summary.json",
            ppo_collector_path=collector_summary,
            limited_ppo_update_smoke_path=batch_root / "limited-ppo-update-smoke-summary.json",
            iterative_ppo_mini_loop_path=batch_root / "iterative-ppo-mini-loop-stability-summary.json",
            guarded_ppo_rollout_pilot_path=batch_root / "guarded-ppo-rollout-pilot-summary.json",
            policy_training_cuda_device_support_path=batch_root / "policy-training-cuda-device-support-summary.json",
            quasi_real_map_domain_gap_path=batch_root / "quasi-real-map-domain-gap-summary.json",
            quasi_real_shadow_policy_behavior_path=batch_root / "quasi-real-shadow-policy-behavior-summary.json",
            quasi_real_shadow_alignment_path=batch_root / "quasi-real-shadow-alignment-summary.json",
            quasi_real_guarded_policy_pilot_path=batch_root / "quasi-real-guarded-policy-pilot-summary.json",
            quasi_real_safe_alternative_opportunity_path=batch_root / "quasi-real-safe-alternative-opportunity-summary.json",
            quasi_real_safe_better_opportunity_expansion_path=batch_root
            / "quasi-real-safe-better-opportunity-expansion-summary.json",
            quasi_real_teacher_equivalent_validation_path=batch_root / "quasi-real-teacher-equivalent-summary.json",
            quasi_real_teacher_distillation_path=batch_root / "quasi-real-teacher-distillation-summary.json",
            config=self._readiness_config(require_current_git_match=True),
            repo_root=self.repo_root,
            quasi_real_guarded_teacher_following_pilot_path=teacher_summary,
            quasi_real_guarded_teacher_following_pilot_required=True,
            ppo_collector_required=True,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["training_readiness_status"], "ppo_rollout_collector_dry_run_evaluated")
        self.assertNotIn("current_git_provenance_mismatch", summary["reason_codes"])

    def _write_quasi_real_files(self) -> None:
        self._write_json(
            self.quasi_real_root / "quasi-real-map-path-feedback-summary.json",
            {
                "schema_version": "path-feedback-summary/v1",
                "status": "completed",
                "scenario_count": 4,
                "candidate_count": 8,
                "reachable_count": 8,
                "open_grid_fallback_used": False,
                "scenarios": [
                    self._scenario("scenario-train-a", "train"),
                    self._scenario("scenario-train-b", "train"),
                    self._scenario("scenario-validation-a", "validation"),
                    self._scenario("scenario-test-a", "test"),
                ],
            },
        )
        self._write_jsonl(
            self.quasi_real_root / "quasi-real-map-slices.jsonl",
            [
                self._slice("scenario-train-a", "train"),
                self._slice("scenario-train-b", "train"),
                self._slice("scenario-validation-a", "validation"),
                self._slice("scenario-test-a", "test"),
            ],
        )

    def _write_guarded_summary(self) -> None:
        self._write_json(
            self.guarded_root / "quasi-real-guarded-teacher-following-pilot-summary.json",
            {
                "schema_version": "quasi-real-guarded-teacher-following-pilot-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "teacher_following_pilot_verdict": "teacher_following_pilot_validated",
                "source_root": "outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1",
                "candidate_root": str(self.candidate_root),
                "quasi_real_root": str(self.quasi_real_root),
                "decisions_path": str(
                    self.guarded_root / "quasi-real-guarded-teacher-following-decisions.jsonl"
                ),
                "quasi_real_context_count": 0,
                "policy_decision_count": 0,
                "teacher_following_step_count": 0,
                "teacher_agreement_rate": 1.0,
                "safe_disagreement_count": 0,
                "unsafe_disagreement_count": 0,
                "policy_changed_gate_rejected_count": 0,
                "roi_group_count": 4,
                "context_id_missing_count": 0,
                "invalid_action_mask_count": 0,
                "fallback_or_open_grid_count": 0,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                "runs_ppo_update": False,
                "writes_ppo_transition": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )

    def _write_decisions(self, decisions: list[dict]) -> None:
        self._write_jsonl(
            self.guarded_root / "quasi-real-guarded-teacher-following-decisions.jsonl",
            decisions,
        )
        summary_path = self.guarded_root / "quasi-real-guarded-teacher-following-pilot-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["quasi_real_context_count"] = len(decisions)
        summary["policy_decision_count"] = len(decisions)
        summary["teacher_following_step_count"] = sum(1 for row in decisions if row.get("teacher_following"))
        summary["safe_disagreement_count"] = sum(1 for row in decisions if row.get("safe_disagreement"))
        summary["unsafe_disagreement_count"] = sum(1 for row in decisions if row.get("unsafe_disagreement"))
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _scenario(self, scenario_id: str, split: str) -> dict:
        return {
            "scenario_id": scenario_id,
            "scenario_group": "smooth_high_confidence",
            "scenario_seed": 7,
            "scenario_variant_id": f"{scenario_id}-v1",
            "open_grid_fallback_used": False,
            "path_feedback": {
                "candidates": [
                    self._candidate(scenario_id, 0, source_selected=True),
                    self._candidate(scenario_id, 1, source_selected=False),
                ]
            },
            "metadata": {"split": split},
        }

    def _candidate(self, scenario_id: str, action_index: int, *, source_selected: bool) -> dict:
        return {
            "action_index": action_index,
            "source_action_index": action_index,
            "cell": [action_index, action_index + 1],
            "candidate_role": "policy_target",
            "policy_target_cell": [action_index, action_index + 1],
            "execution_goal_cell": [action_index, action_index + 1],
            "reachable": True,
            "replan_required": False,
            "path_cost": 10.0 - action_index,
            "risk": 0.1 + action_index * 0.01,
            "utility": 1.0 - action_index * 0.01,
            "context_id": f"ctx-{scenario_id.rsplit('-', 1)[0].split('scenario-')[1]}-{action_index}",
            "context_id_schema_version": "policy-context-id/v1",
            "context_id_source": "stable_semantic_fields",
            "candidate_generation": {
                "source_selection_status": "source_selected" if source_selected else "not_source_candidate",
            },
            "platform_goal_feasibility": {"contract_reachable": True},
            "open_grid_fallback_used": False,
        }

    def _slice(self, scenario_id: str, split: str) -> dict:
        return {
            "schema_version": "quasi-real-map-slice/v1",
            "scenario_id": scenario_id,
            "scenario_group": "smooth_high_confidence",
            "roi_group": "smooth_high_confidence",
            "roi_name": "smooth_high_confidence",
            "split": split,
            "map_id": "lola-test",
            "slice_id": scenario_id,
            "dataset_id": "lunar_south_pole_lro_lola_gdr_875s_20m",
            "context_id_schema_version": "policy-context-id/v1",
            "context_id_source": "stable_semantic_fields",
            "legacy_identity_fallback_used": False,
        }

    def _decision(
        self,
        context_id: str,
        *,
        scenario_id: str,
        split: str,
        controlled_choice_source: str,
        controlled_action_index: int,
        decision_class: str = "source_aligned",
        gate_reason_codes: list[str] | None = None,
        log_prob: float | None = -0.1,
        value: float | None = 0.1,
        path_cost_delta: float = 0.0,
    ) -> dict:
        gate_reasons = list(gate_reason_codes or [])
        return {
            "schema_version": "quasi-real-guarded-teacher-following-decision/v1",
            "context_id": context_id,
            "scenario_id": scenario_id,
            "roi_group": "smooth_high_confidence",
            "roi_name": "smooth_high_confidence",
            "split": split,
            "map_id": "lola-test",
            "slice_id": scenario_id,
            "source_action_index": 0,
            "teacher_action_index": 0,
            "raw_policy_action_index": controlled_action_index,
            "controlled_action_index": controlled_action_index,
            "controlled_choice_source": controlled_choice_source,
            "action_mask_valid": "invalid_action_mask" not in gate_reasons,
            "path_cost_delta": path_cost_delta,
            "risk_delta": 0.0,
            "gate_reason_codes": gate_reasons,
            "decision_class": decision_class,
            "teacher_following": controlled_choice_source == "policy_teacher_aligned",
            "safe_disagreement": controlled_choice_source == "policy_safe_disagreement",
            "unsafe_disagreement": controlled_choice_source == "teacher_fallback",
            "policy_takes_control": controlled_choice_source
            in {"policy_teacher_aligned", "policy_safe_disagreement"},
            "policy_action_log_prob": log_prob,
            "policy_value": value,
            "runs_ppo_update": False,
            "writes_ppo_transition": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
        }

    def _config(self, *, min_trainable: int) -> dict:
        return {
            "schema_version": "quasi-real-ppo-collector-dry-run-config/v1",
            "input_files": {
                "guarded_summary": "quasi-real-guarded-teacher-following-pilot-summary.json",
                "decisions": "quasi-real-guarded-teacher-following-decisions.jsonl",
                "quasi_real_path_feedback_summary": "quasi-real-map-path-feedback-summary.json",
                "quasi_real_slices": "quasi-real-map-slices.jsonl",
                "checkpoint": "experimental-hybrid-policy-candidate.pt",
            },
            "output_files": {
                "episodes": "ppo-rollout-episodes.jsonl",
                "transitions": "ppo-rollout-transitions.jsonl",
                "summary": "ppo-rollout-collector-summary.json",
                "rejection_report": "ppo-rollout-rejection-report.json",
                "reward_audit": "ppo-rollout-reward-audit.json",
            },
            "splits": {"trainable": ["train"], "diagnostic": ["validation", "test"]},
            "trainable_controlled_choice_sources": [
                "policy_teacher_aligned",
                "policy_safe_disagreement",
            ],
            "reward": {
                "teacher_following_bonus": 1.0,
                "safe_disagreement_bonus": 1.0,
                "gate_regression_penalty": 1.0,
            },
            "validation": {
                "min_ppo_trainable_transition_count": min_trainable,
                "max_source_fallback_trainable_count": 0,
                "max_invalid_action_mask_count": 0,
                "max_empty_action_mask_count": 0,
                "max_missing_log_prob_count": 0,
                "max_missing_value_count": 0,
                "max_non_finite_reward_count": 0,
            },
            "non_goals": ["no_ppo_update", "no_checkpoint_publication"],
        }

    def _collector_readiness_summary(self) -> dict:
        from git_provenance import git_snapshot

        return {
            "schema_version": "ppo-rollout-collector-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "episode_count": 108,
            "step_count": 108,
            "ppo_trainable_transition_count": 36,
            "diagnostic_transition_count": 72,
            "source_fallback_trainable_count": 0,
            "invalid_action_mask_count": 0,
            "empty_action_mask_count": 0,
            "missing_log_prob_count": 0,
            "missing_value_count": 0,
            "non_finite_reward_count": 0,
            "state_continuity_violation_count": 0,
            "fallback_or_open_grid_count": 0,
            "safety_regression_count": 0,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "performance_claimed": False,
            "formal_training_ready_claimed": False,
            "git_provenance": {
                "current": git_snapshot(self.repo_root),
                "current_matches_sources": True,
            },
        }

    def _readiness_config(self, *, require_current_git_match: bool = False) -> dict:
        return {
            "schema_version": "policy-training-readiness-review-config/v1",
            "output_files": {
                "policy_training_readiness_review_summary": "policy-training-readiness-review-summary.json"
            },
            "readiness_thresholds": {
                "require_smoke_ready_for_training_review": False,
                "min_applied_calibrated_candidate_count": 0,
                "min_calibrated_selection_rate_delta": -1.0,
                "max_rejected_goal_blocked_count": 0,
                "max_safety_regression_count": 0,
                "max_fallback_or_open_grid_count": 0,
            },
            "validation": {
                "require_current_git_match": require_current_git_match,
                "allow_dirty_current_git_match": True,
                "fail_on_input_failure": True,
            },
            "non_goals": [],
        }

    def _write_minimal_readiness_base_summaries(
        self,
        batch_root: Path,
        *,
        git_provenance: dict | None = None,
    ) -> None:
        self._write_json(
            batch_root / "calibrated-policy-application-smoke-summary.json",
            {
                "schema_version": "calibrated-policy-application-smoke-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "recommended_next_action": "ready_for_policy_training_readiness_review",
                "applied_calibrated_candidate_count": 0,
                "source_selected_candidate_changed_rate": 0.0,
                "calibrated_selection_rate": 0.0,
                "git_provenance": git_provenance or {"current_matches_sources": True},
            },
        )
        self._write_json(
            batch_root / "channel-aware-training-readiness-summary.json",
            {
                "schema_version": "channel-aware-training-readiness-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "source_selected_candidate_changed_rate": 0.0,
                "calibrated_selection_rate": 0.0,
                "git_provenance": git_provenance or {"current_matches_sources": True},
            },
        )
        self._write_json(
            batch_root / "channel-aware-contrast-coverage-summary.json",
            {
                "schema_version": "channel-aware-contrast-coverage-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "source_selected_candidate_changed_rate": 0.0,
                "calibrated_selection_rate": 0.0,
                "git_provenance": git_provenance or {"current_matches_sources": True},
            },
        )
        self._write_json(
            batch_root / "channel-aware-selection-contrast-calibration-summary.json",
            {
                "schema_version": "channel-aware-selection-contrast-calibration-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "source_selected_candidate_changed_rate": 0.0,
                "calibrated_selection_rate": 0.0,
                "git_provenance": git_provenance or {"current_matches_sources": True},
            },
        )

    def _stale_git(self) -> dict:
        return {
            "current": {
                "parent": {"path": ".", "sha": "0000000000000000000000000000000000000000", "dirty": False},
                "submodules": {
                    "dev-platform-constraints": {
                        "path": "dev-platform-constraints",
                        "sha": "0000000000000000000000000000000000000000",
                        "dirty": False,
                    },
                    "model-explorer": {
                        "path": "model-explorer",
                        "sha": "0000000000000000000000000000000000000000",
                        "dirty": False,
                    },
                    "path-planner": {
                        "path": "path-planner",
                        "sha": "0000000000000000000000000000000000000000",
                        "dirty": False,
                    },
                },
                "dirty": False,
            },
            "current_matches_sources": True,
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
