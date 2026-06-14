import json
import tempfile
import unittest
from pathlib import Path


class QuasiRealTrainableContextExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="quasi-real-context-expansion-"))
        self.horizon5_root = self.temp_dir / "horizon5"
        self.scale512_root = self.temp_dir / "scale512"
        self.source_root = self.temp_dir / "source"
        self.output_root = self.temp_dir / "expansion"
        self.batch_root = self.temp_dir / "batch"
        for path in (self.horizon5_root, self.scale512_root, self.source_root, self.batch_root):
            path.mkdir(parents=True)

    def test_expansion_passes_with_512_unique_contexts_and_passed_scale512_rerun(self) -> None:
        from scripts.run_quasi_real_trainable_context_expansion import (
            run_quasi_real_trainable_context_expansion,
        )

        self._write_input_summaries()
        self._write_source_steps([self._step(index, trainable=True) for index in range(512)])

        result = run_quasi_real_trainable_context_expansion(
            horizon5_root=self.horizon5_root,
            scale512_root=self.scale512_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            scale512_runner=self._passing_scale512_runner,
        )

        self.assertEqual(result["schema_version"], "quasi-real-trainable-context-expansion-summary/v1")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["unique_trainable_context_count"], 512)
        self.assertEqual(result["ppo_trainable_transition_count"], 512)
        self.assertEqual(result["duplicate_trainable_context_count"], 0)
        self.assertEqual(result["validation_trainable_count"], 0)
        self.assertEqual(result["test_trainable_count"], 0)
        self.assertEqual(result["source_fallback_trainable_count"], 0)
        self.assertEqual(result["teacher_fallback_trainable_count"], 0)
        self.assertEqual(result["non_empty_gate_reason_trainable_count"], 0)
        self.assertEqual(result["non_finite_reward_count"], 0)
        self.assertEqual(result["non_finite_return_count"], 0)
        self.assertEqual(result["non_finite_advantage_count"], 0)
        self.assertEqual(result["controlled_regression_count"], 0)
        self.assertEqual(result["scale512_status"], "passed")
        self.assertEqual(result["passed_seed_count"], 3)
        self.assertEqual(
            result["readiness_status"],
            "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated",
        )
        self.assertFalse(result["runs_formal_ppo_rollout"])
        self.assertFalse(result["publishes_checkpoint"])
        self.assertFalse(result["replaces_default_policy"])

        expanded_horizon5 = self.output_root / "expanded_horizon5"
        expanded_summary = json.loads(
            (expanded_horizon5 / "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(expanded_summary["horizon"], 5)
        self.assertEqual(expanded_summary["ppo_trainable_transition_count"], 512)
        self.assertEqual(expanded_summary["status"], "passed")

    def test_expansion_fails_when_real_unique_context_pool_is_below_512(self) -> None:
        from scripts.run_quasi_real_trainable_context_expansion import (
            run_quasi_real_trainable_context_expansion,
        )

        self._write_input_summaries()
        self._write_source_steps([self._step(index, trainable=True) for index in range(36)])

        result = run_quasi_real_trainable_context_expansion(
            horizon5_root=self.horizon5_root,
            scale512_root=self.scale512_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            scale512_runner=self._passing_scale512_runner,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("insufficient_quasi_real_candidate_pool", result["reason_codes"])
        self.assertEqual(result["unique_trainable_context_count"], 36)
        self.assertEqual(result["scale512_status"], "skipped")
        self.assertEqual(result["next_required_change"], "expand_quasi_real_trainable_context_source_pool")

    def test_expansion_materializes_teacher_distillation_train_slices_as_unique_trainable_contexts(self) -> None:
        from scripts.run_quasi_real_trainable_context_expansion import (
            run_quasi_real_trainable_context_expansion,
        )

        dataset_root = self.temp_dir / "teacher_distillation"
        dataset_root.mkdir()
        self._write_input_summaries()
        self._write_teacher_distillation_dataset(dataset_root, count=2)

        result = run_quasi_real_trainable_context_expansion(
            horizon5_root=self.horizon5_root,
            scale512_root=self.scale512_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(
                source_roots=[],
                materialization={
                    "enabled": True,
                    "dataset_roots": [str(dataset_root)],
                    "candidate_root": str(self.temp_dir / "candidate"),
                },
                min_count=2,
            ),
            repo_root=self.repo_root,
            scale512_runner=self._passing_scale512_runner,
            policy_evaluator=self._fake_policy_evaluator(),
        )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["unique_trainable_context_count"], 2)
        self.assertEqual(result["ppo_trainable_transition_count"], 2)
        self.assertEqual(result["materialized_trainable_context_count"], 2)
        self.assertEqual(result["materialized_missing_observation_count"], 0)
        self.assertEqual(result["missing_log_prob_count"], 0)
        self.assertEqual(result["missing_value_count"], 0)

        rows = [
            json.loads(line)
            for line in (self.output_root / "quasi-real-trainable-context-expansion-steps.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual({row["split"] for row in rows}, {"train"})
        self.assertEqual({row["controlled_choice_source"] for row in rows}, {"policy"})
        self.assertEqual({row["controlled_choice_detail"] for row in rows}, {"policy_teacher_aligned"})
        self.assertTrue(all(row["observation"]["action_mask"] for row in rows))
        self.assertEqual({row["log_prob"] for row in rows}, {-0.25})
        self.assertEqual({row["value"] for row in rows}, {0.4})

    def test_expansion_keeps_diagnostic_rows_out_of_trainable_pool(self) -> None:
        from scripts.run_quasi_real_trainable_context_expansion import (
            run_quasi_real_trainable_context_expansion,
        )

        self._write_input_summaries()
        rows = [self._step(index, trainable=True) for index in range(510)]
        rows.append(self._step(9000, trainable=True, split="validation"))
        rows.append(self._step(9001, trainable=True, split="test"))
        rows.append(self._step(9002, trainable=True, controlled_choice_source="source_fallback"))
        rows.append(self._step(9003, trainable=True, controlled_choice_source="teacher_fallback"))
        rows.append(self._step(9004, trainable=True, gate_reason_codes=["path_cost_regression"]))
        rows.append(self._step(9005, trainable=True, reward=float("nan")))
        rows.append(self._step(0, trainable=True, scenario_id="duplicate-scenario"))
        self._write_source_steps(rows)

        result = run_quasi_real_trainable_context_expansion(
            horizon5_root=self.horizon5_root,
            scale512_root=self.scale512_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            scale512_runner=self._passing_scale512_runner,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("insufficient_quasi_real_candidate_pool", result["reason_codes"])
        self.assertEqual(result["unique_trainable_context_count"], 510)
        self.assertEqual(result["duplicate_trainable_context_count"], 0)
        self.assertEqual(result["duplicate_source_trainable_context_count"], 1)
        self.assertEqual(result["validation_trainable_count"], 1)
        self.assertEqual(result["test_trainable_count"], 1)
        self.assertEqual(result["source_fallback_trainable_count"], 1)
        self.assertEqual(result["teacher_fallback_trainable_count"], 1)
        self.assertEqual(result["non_empty_gate_reason_trainable_count"], 1)
        self.assertEqual(result["non_finite_reward_count"], 1)

    def test_expansion_fails_when_scale512_rerun_fails(self) -> None:
        from scripts.run_quasi_real_trainable_context_expansion import (
            run_quasi_real_trainable_context_expansion,
        )

        self._write_input_summaries()
        self._write_source_steps([self._step(index, trainable=True) for index in range(512)])

        def failing_scale512_runner(**kwargs) -> dict:
            summary = self._passing_scale512_runner(**kwargs)
            summary["status"] = "failed"
            summary["reason_codes"] = ["scale512_seed_smoke_not_all_passed"]
            summary["readiness_status"] = "needs_training_contract_refinement"
            return summary

        result = run_quasi_real_trainable_context_expansion(
            horizon5_root=self.horizon5_root,
            scale512_root=self.scale512_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            scale512_runner=failing_scale512_runner,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("scale512_preflight_not_passed", result["reason_codes"])
        self.assertEqual(result["scale512_status"], "failed")

    def test_expansion_fails_on_split_leakage_even_when_capacity_is_met(self) -> None:
        from scripts.run_quasi_real_trainable_context_expansion import (
            run_quasi_real_trainable_context_expansion,
        )

        self._write_input_summaries()
        rows = [self._step(index, trainable=True) for index in range(512)]
        rows.append(self._step(0, trainable=True, scenario_id="duplicate-scenario"))
        rows.append(self._step(9000, trainable=True, split="validation"))
        self._write_source_steps(rows)

        result = run_quasi_real_trainable_context_expansion(
            horizon5_root=self.horizon5_root,
            scale512_root=self.scale512_root,
            output_root=self.output_root,
            batch_root=self.batch_root,
            config=self._config(),
            repo_root=self.repo_root,
            scale512_runner=self._passing_scale512_runner,
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("quasi_real_context_expansion_split_leakage", result["reason_codes"])
        self.assertEqual(result["unique_trainable_context_count"], 512)
        self.assertEqual(result["duplicate_trainable_context_count"], 0)
        self.assertEqual(result["duplicate_source_trainable_context_count"], 1)
        self.assertEqual(result["validation_trainable_count"], 1)
        self.assertEqual(result["scale512_status"], "skipped")

    def test_config_declares_context_expansion_contract(self) -> None:
        config_path = self.repo_root / "configs" / "quasi_real_trainable_context_expansion_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["schema_version"], "quasi-real-trainable-context-expansion-config/v1")
        self.assertEqual(config["validation"]["min_unique_trainable_context_count"], 512)
        self.assertEqual(config["validation"]["min_ppo_trainable_transition_count"], 512)
        self.assertTrue(config["materialization"]["enabled"])
        self.assertIn(
            "outputs/path_feedback_batch_quasi_real_teacher_distillation_dataset_v1",
            config["materialization"]["dataset_roots"],
        )
        self.assertIn("does_not_duplicate_contexts_to_fake_scale", config["non_goals"])

    def _config(
        self,
        *,
        source_roots: list[str] | None = None,
        materialization: dict | None = None,
        min_count: int = 512,
    ) -> dict:
        return {
            "schema_version": "quasi-real-trainable-context-expansion-config/v1",
            "horizon": 5,
            "discount_factor": 0.99,
            "source_roots": [str(self.source_root)] if source_roots is None else source_roots,
            "source_jsonl_globs": ["*.jsonl"],
            "materialization": materialization or {"enabled": False},
            "validation": {
                "min_unique_trainable_context_count": min_count,
                "min_ppo_trainable_transition_count": min_count,
                "min_teacher_agreement_rate": 0.95,
            },
            "scale512": {
                "output_root": str(self.output_root / "scale512_rerun"),
                "config": "configs/quasi_real_guarded_ppo_scale512_multiseed_preflight_v1.json",
            },
        }

    def _write_input_summaries(self) -> None:
        horizon5 = {
            "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "horizon": 5,
            "episode_count": 96,
            "step_count": 480,
            "ppo_trainable_transition_count": 162,
            "diagnostic_transition_count": 318,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "replay_count": 3,
            "passed_replay_count": 3,
            "readiness_status": "quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated",
        }
        (self.horizon5_root / "quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json").write_text(
            json.dumps(horizon5, indent=2),
            encoding="utf-8",
        )
        scale512 = {
            "schema_version": "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary/v1",
            "status": "failed",
            "reason_codes": ["insufficient_quasi_real_trainable_capacity"],
            "unique_trainable_context_count": 36,
            "ppo_trainable_transition_count": 36,
            "readiness_status": "needs_training_contract_refinement",
        }
        (self.scale512_root / "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json").write_text(
            json.dumps(scale512, indent=2),
            encoding="utf-8",
        )

    def _write_teacher_distillation_dataset(self, root: Path, *, count: int) -> None:
        slices = []
        scenarios = []
        for index in range(count):
            scenario_id = f"qreal_teacher_distill_{index:03d}_train_00"
            source_context = f"source-context-{index:04d}"
            alternative_context = f"alternative-context-{index:04d}"
            slices.append(
                {
                    "schema_version": "quasi-real-teacher-distillation-slice/v1",
                    "scenario_id": scenario_id,
                    "scenario_group": f"family-{index % 3}",
                    "scenario_seed": 20260612 + index,
                    "split": "train",
                    "context_id": f"distill-slice-context-{index:04d}",
                    "source_failure_context_id": source_context,
                    "failure_class": "path_cost_only_regression",
                    "path_cost_delta": 1.0,
                    "risk_delta": 0.0,
                }
            )
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_group": f"family-{index % 3}",
                    "scenario_seed": 20260612 + index,
                    "path_feedback": {
                        "candidates": [
                            self._candidate(action_index=1, context_id=source_context, path_cost=1.0),
                            self._candidate(action_index=0, context_id=alternative_context, path_cost=2.0),
                        ]
                    },
                }
            )
        (root / "quasi-real-teacher-distillation-slices.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in slices),
            encoding="utf-8",
        )
        (root / "quasi-real-teacher-distillation-path-feedback-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "quasi-real-teacher-distillation-path-feedback-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "scenario_count": len(scenarios),
                    "scenarios": scenarios,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _candidate(self, *, action_index: int, context_id: str, path_cost: float) -> dict:
        return {
            "action_index": action_index,
            "candidate_role": "policy_target",
            "policy_target_cell": [action_index + 1, action_index + 2],
            "execution_goal_cell": [action_index + 1, action_index + 2],
            "utility": 0.5,
            "reachable": True,
            "path_cost": path_cost,
            "risk": 0.1,
            "context_id": context_id,
        }

    def _fake_policy_evaluator(self):
        class FakePolicyEvaluator:
            def evaluate(self, _observation, _action_index: int) -> dict:
                return {"log_prob": -0.25, "value": 0.4}

        return FakePolicyEvaluator()

    def _write_source_steps(self, rows: list[dict]) -> None:
        (self.source_root / "quasi-real-source-steps.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _step(
        self,
        index: int,
        *,
        trainable: bool,
        split: str = "train",
        controlled_choice_source: str = "policy",
        gate_reason_codes: list[str] | None = None,
        scenario_id: str | None = None,
        reward: float = 1.0,
    ) -> dict:
        return {
            "schema_version": "quasi-real-guarded-ppo-horizon5-batch-expansion-step/v1",
            "episode_id": f"source-episode-{index // 5:04d}",
            "step_index": index % 5,
            "context_id": f"context-{index:04d}",
            "scenario_id": scenario_id or f"scenario-{index:04d}",
            "scenario_family": f"family-{index % 7}",
            "split": split,
            "controlled_choice_source": controlled_choice_source,
            "controlled_choice_detail": "policy_teacher_aligned",
            "ppo_trainable": trainable,
            "gate_reason_codes": gate_reason_codes or [],
            "controlled_regression_reason_codes": [],
            "observation": {"action_mask": [True, True, True]},
            "missing_observation": False,
            "log_prob": -0.1,
            "value": 0.2,
            "reward": reward,
            "discounted_return": reward,
            "advantage": reward - 0.2,
        }

    def _passing_scale512_runner(self, *, horizon5_root: Path, output_root: Path, trainable_steps: list[dict], **_kwargs) -> dict:
        output_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "horizon": 5,
            "ppo_trainable_transition_count": len(trainable_steps),
            "unique_trainable_context_count": len({row["context_id"] for row in trainable_steps}),
            "seed_count": 3,
            "passed_seed_count": 3,
            "controlled_regression_count": 0,
            "readiness_status": "quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated",
            "summary": str(output_root / "quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json"),
        }
        Path(summary["summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


if __name__ == "__main__":
    unittest.main()
