import json
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
class XunceHighFidelityRealMapComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        model_explorer_src = str(self.repo_root / "model-explorer" / "src")
        if model_explorer_src not in sys.path:
            sys.path.insert(0, model_explorer_src)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-hifi-comparison-"))
        self.expansion_root = self.temp_dir / "expansion"
        self.output_root = self.temp_dir / "comparison"
        self.config_path = self.temp_dir / "config.json"
        self.xunce_checkpoint = self.temp_dir / "xunce.pt"
        self.incumbent_checkpoint = self.temp_dir / "incumbent.pt"
        self._write_xunce_checkpoint()
        self._write_incumbent_checkpoint()
        self._write_expansion_evidence()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_default_fixture_runs_true_checkpoint_inference_without_proxy_selection(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_comparison import run_xunce_high_fidelity_real_map_comparison

        summary = run_xunce_high_fidelity_real_map_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertFalse(summary["xunce_candidate_advantage_established"])
        self.assertEqual(summary["required_scenario_count"], 24)
        self.assertEqual(summary["failed_required_scenario_count"], 0)
        self.assertEqual(summary["xunce_worse_than_incumbent_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertLessEqual(summary["xunce_guard_fallback_rate"], 0.05)
        self.assertEqual(summary["next_required_change"], "xunce_research_iteration_required")
        self.assertTrue(summary["true_model_inference_executed"])
        self.assertFalse(summary["proxy_selection_used"])
        self.assertTrue(summary["xunce_checkpoint_loaded"])
        self.assertTrue(summary["incumbent_checkpoint_loaded"])
        self.assertEqual(summary["model_inference_mask_violation_count"], 0)
        self.assertGreater(summary["model_inference_finite_output_count"], 0)
        self.assertGreater(summary["xunce_parameter_count"], 0)
        self.assertGreater(summary["incumbent_parameter_count"], 0)
        self.assertGreaterEqual(summary["xunce_median_inference_latency_ms"], 0.0)
        self.assertGreaterEqual(summary["incumbent_median_inference_latency_ms"], 0.0)
        self.assertGreaterEqual(summary["latency_ratio_vs_incumbent"], 0.0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])

        for filename in (
            "xunce-high-fidelity-real-map-comparison-summary.json",
            "xunce-high-fidelity-real-map-comparison-manifest.json",
            "xunce-high-fidelity-real-map-scenario-results.jsonl",
            "xunce-high-fidelity-policy-decisions.jsonl",
            "xunce-high-fidelity-roi-family-summary.json",
            "xunce-vs-incumbent-audit.json",
            "xunce-efficiency-audit.json",
            "xunce-source-match-audit.json",
            "xunce-high-fidelity-boundary-audit.json",
            "xunce-high-fidelity-rejection-report.json",
            "xunce-high-fidelity-real-map-comparison-report.md",
            "xunce-high-fidelity-model-inference-audit.json",
            "xunce-high-fidelity-model-inference-results.jsonl",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

        audit = json.loads((self.output_root / "xunce-high-fidelity-model-inference-audit.json").read_text(encoding="utf-8"))
        self.assertTrue(audit["true_model_inference_executed"])
        self.assertFalse(audit["proxy_selection_used"])
        result_rows = [
            json.loads(line)
            for line in (self.output_root / "xunce-high-fidelity-model-inference-results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(result_rows), 24)
        self.assertEqual(result_rows[0]["input_source"], "high_fidelity_scenario_adapter/v1")
        self.assertIn("logits", result_rows[0]["xunce"])
        self.assertIn("action_probs", result_rows[0]["xunce"])
        self.assertIn("value", result_rows[0]["incumbent"])
        self.assertIn("selected_action_index", result_rows[0]["incumbent"])

    def test_missing_xunce_checkpoint_routes_to_sandbox_candidate_fix(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_comparison import run_xunce_high_fidelity_real_map_comparison

        self.xunce_checkpoint.unlink()
        summary = run_xunce_high_fidelity_real_map_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_xunce_candidate_checkpoint", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_sandbox_candidate_preflight")

    def test_missing_incumbent_checkpoint_routes_to_incumbent_fix(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_comparison import run_xunce_high_fidelity_real_map_comparison

        self.incumbent_checkpoint.unlink()
        summary = run_xunce_high_fidelity_real_map_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_incumbent_policy_checkpoint", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_incumbent_policy_checkpoint")

    def test_unsupported_incumbent_checkpoint_routes_to_incumbent_fix(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_comparison import run_xunce_high_fidelity_real_map_comparison

        self.incumbent_checkpoint.write_bytes(b"not-a-model-explorer-policy-checkpoint")
        summary = run_xunce_high_fidelity_real_map_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("incumbent_checkpoint_format_unsupported", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_incumbent_policy_checkpoint")

    def test_regression_prevents_authorization(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_comparison import run_xunce_high_fidelity_real_map_comparison

        self._write_expansion_evidence(xunce_regresses=True)
        summary = run_xunce_high_fidelity_real_map_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertFalse(summary["xunce_candidate_advantage_established"])
        self.assertGreater(summary["controlled_regression_count"], 0)
        self.assertEqual(summary["next_required_change"], "xunce_research_iteration_required")

    def test_fallback_rate_above_threshold_fails_comparison(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_comparison import run_xunce_high_fidelity_real_map_comparison

        self._write_expansion_evidence(force_guard_fallback=True)
        summary = run_xunce_high_fidelity_real_map_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("xunce_guard_fallback_rate_exceeded", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_high_fidelity_guard_fallback")

    def test_parameter_budget_blocks_advantage_claim_without_failing_evidence(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_comparison import run_xunce_high_fidelity_real_map_comparison

        self._write_config(max_xunce_parameter_count=1)
        summary = run_xunce_high_fidelity_real_map_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertFalse(summary["xunce_candidate_advantage_established"])
        self.assertEqual(summary["next_required_change"], "xunce_research_iteration_required")

    def _write_xunce_checkpoint(self) -> None:
        import torch

        from scripts.xunce_full_network_common import XunceFullNetworkV1

        config = self._xunce_model_config()
        model = XunceFullNetworkV1(
            candidate_feature_count=config["candidate_feature_count"],
            edge_feature_count=config["edge_feature_count"],
            memory_feature_count=config["memory_feature_count"],
            context_feature_count=config["context_feature_count"],
            missing_indicator_count=config["missing_indicator_count"],
            hidden_dim=config["hidden_dim"],
            message_passing_layers=config["message_passing_layers"],
            dropout=0.0,
        )
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 0.0)
        torch.save(
            {
                "schema_version": "xunce-controlled-training-candidate-checkpoint/v1",
                "model_state_dict": model.state_dict(),
                "metadata": {
                    "schema_version": "xunce-controlled-training-checkpoint-metadata/v1",
                    "architecture": "xunce_full_network_v1",
                    **config,
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "writes_research_checkpoint": True,
                    "publishes_checkpoint": False,
                    "checkpoint_publication_approved": False,
                    "default_policy_replacement_approved": False,
                },
            },
            self.xunce_checkpoint,
        )

    def _write_incumbent_checkpoint(self) -> None:
        import torch

        from model_explorer.policy.features import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, MISSING_INDICATOR_NAMES
        from model_explorer.policy.torch_policy import MaskedCandidatePolicyNetwork

        network = MaskedCandidatePolicyNetwork(
            candidate_feature_count=len(CANDIDATE_FEATURE_NAMES),
            global_feature_count=len(GLOBAL_FEATURE_NAMES),
            hidden_size=16,
            missing_indicator_count=len(MISSING_INDICATOR_NAMES),
        )
        for parameter in network.parameters():
            torch.nn.init.constant_(parameter, 0.0)
        torch.save(
            {
                "state_dict": network.state_dict(),
                "hidden_size": 16,
                "candidate_feature_names": tuple(CANDIDATE_FEATURE_NAMES),
                "global_feature_names": tuple(GLOBAL_FEATURE_NAMES),
                "candidate_missing_indicator_names": tuple(MISSING_INDICATOR_NAMES),
                "metadata": {
                    "format": "model-explorer-masked-policy",
                    "version": 2,
                    "format_version": "model-explorer-masked-policy/v2",
                    "architecture": "mlp_v1",
                    "architecture_config": {"hidden_dim": 16, "dropout": 0.0},
                    "candidate_feature_names": tuple(CANDIDATE_FEATURE_NAMES),
                    "global_feature_names": tuple(GLOBAL_FEATURE_NAMES),
                    "candidate_missing_indicator_names": tuple(MISSING_INDICATOR_NAMES),
                    "action_count": 2,
                },
            },
            self.incumbent_checkpoint,
        )

    @staticmethod
    def _xunce_model_config() -> dict[str, int]:
        return {
            "seed": 17,
            "candidate_feature_count": 8,
            "edge_feature_count": 5,
            "memory_feature_count": 6,
            "context_feature_count": 7,
            "missing_indicator_count": 3,
            "hidden_dim": 32,
            "message_passing_layers": 2,
            "candidate_count": 6,
        }

    def _write_config(self, *, max_xunce_parameter_count: int = 10_000_000) -> None:
        payload = {
            "schema_version": "xunce-high-fidelity-real-map-comparison-config/v1",
            "source_roi_expansion_root": str(self.expansion_root),
            "xunce_candidate_checkpoint": str(self.xunce_checkpoint),
            "incumbent_policy_checkpoint": str(self.incumbent_checkpoint),
            "required_scenario_count": 24,
            "max_xunce_guard_fallback_rate": 0.05,
            "max_xunce_parameter_count": max_xunce_parameter_count,
            "max_latency_ratio_vs_candidate_attention": 1.5,
            "max_median_inference_latency_ms": 5.0,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_expansion_evidence(self, *, xunce_regresses: bool = False, force_guard_fallback: bool = False) -> None:
        self.expansion_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        splits = ["train", "validation", "test"]
        for index in range(24):
            group = f"roi_{index // 3}"
            split = splits[index % 3]
            scenario_id = f"scenario_{index:03d}"
            xunce_cost = 20.0 if xunce_regresses and index == 0 else 8.0 + index * 0.01
            incumbent_cost = 10.0 + index * 0.01
            if force_guard_fallback and index < 2:
                xunce_cost = None
                incumbent_cost = None
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_group": group,
                    "roi_group": group,
                    "selected_cell_after_path_feedback": [1, 1],
                    "selected_cell_before_path_feedback": [2, 2],
                    "selected_path_cost_after_feedback": 8.0 + index * 0.01,
                    "selected_path_cost_before_feedback": incumbent_cost,
                    "coverage_rate_delta": 0.1,
                    "open_grid_fallback_used": False,
                    "tracking_safety_violation_count": 1 if xunce_regresses and index == 0 else 0,
                    "path_feedback": {
                        "candidates": [
                            {"cell": [1, 1], "reachable": xunce_cost is not None, "path_cost": xunce_cost, "risk": 0.1, "policy": "xunce"},
                            {"cell": [2, 2], "reachable": incumbent_cost is not None, "path_cost": incumbent_cost, "risk": 0.1, "policy": "incumbent"},
                        ]
                    },
                }
            )
            slices.append(
                {
                    "schema_version": "quasi-real-map-slice/v1",
                    "scenario_id": scenario_id,
                    "scenario_group": group,
                    "roi_name": group,
                    "split": split,
                    "context_id": f"context-{scenario_id}",
                    "legacy_identity_fallback_used": False,
                    "contract": str(self.expansion_root / f"{scenario_id}.contract.json"),
                    "sidecar": str(self.expansion_root / f"{scenario_id}.sidecar.json"),
                }
            )
        (self.expansion_root / "xunce-high-fidelity-real-map-slices.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in slices) + "\n",
            encoding="utf-8",
        )
        self._write_json(
            self.expansion_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            {
                "schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "next_required_change": "xunce_high_fidelity_real_map_policy_comparison",
                "slice_count": 24,
                "roi_group_count": 8,
                "context_id_missing_count": 0,
                "legacy_identity_fallback_count": 0,
                "fallback_or_open_grid_count": 0,
                "domain_gap_verdict": "acceptable_for_high_fidelity_comparison",
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "runs_new_ppo_update": False,
                "modifies_network": False,
                "modifies_action_space": False,
                "modifies_default_astar": False,
                "real_world_release_approved": False,
                "real_world_performance_claimed": False,
            },
        )
        self._write_json(
            self.expansion_root / "xunce-high-fidelity-path-feedback-audit.json",
            {
                "schema_version": "xunce-high-fidelity-path-feedback-audit/v1",
                "scenario_count": 24,
                "candidate_count": 48,
                "reachable_count": 48,
                "fallback_or_open_grid_count": 0,
                "open_grid_fallback_used": False,
                "scenarios": scenarios,
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
