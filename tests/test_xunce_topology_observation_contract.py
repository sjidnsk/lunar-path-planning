import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceTopologyObservationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-topology-contract-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage2_root = self.repo_root / "outputs" / "stage2"
        self.output_root = self.repo_root / "outputs" / "stage3"
        self.config_path = self.repo_root / "configs" / "xunce_topology_observation_contract_v1.json"
        self.features_path = self.repo_root / "model-explorer" / "src" / "model_explorer" / "policy" / "features.py"
        self.torch_policy_path = self.repo_root / "model-explorer" / "src" / "model_explorer" / "policy" / "torch_policy.py"
        self.training_path = self.repo_root / "model-explorer" / "src" / "model_explorer" / "policy" / "training.py"
        self.architectures_path = self.repo_root / "model-explorer" / "src" / "model_explorer" / "policy" / "architectures.py"
        self._write_config()
        self._write_stage2()
        self._write_policy_sources()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_contract_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_topology_observation_contract import run_xunce_topology_observation_contract

        summary = run_xunce_topology_observation_contract(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["topology_observation_contract_passed"])
        self.assertTrue(summary["contract_audit_passed"])
        self.assertTrue(summary["compatibility_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertEqual(summary["base_observation_schema_version"], "policy-observation/v1.1")
        self.assertEqual(summary["next_required_change"], "topology_feature_extraction_audit")
        self.assertEqual(summary["candidate_topology_field_count"], 9)
        self.assertEqual(summary["candidate_edge_field_count"], 6)
        self.assertEqual(summary["coverage_memory_field_count"], 7)
        self.assertTrue(summary["additive_only"])
        self.assertTrue(summary["all_new_fields_optional"])
        self.assertTrue(summary["all_new_fields_have_missing_indicators"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "xunce-topology-observation-contract-summary.json",
            "xunce-topology-observation-contract-manifest.json",
            "xunce-topology-observation-contract.json",
            "xunce-topology-observation-contract-audit.json",
            "xunce-topology-observation-compatibility-audit.json",
            "xunce-topology-observation-boundary-audit.json",
            "xunce-topology-observation-contract-rejection-report.json",
            "xunce-topology-observation-contract-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_failed_stage2_routes_to_literature_bottleneck_fix(self) -> None:
        from scripts.run_xunce_topology_observation_contract import run_xunce_topology_observation_contract

        self._write_stage2({"status": "failed", "next_required_change": "fix_xunce_network_literature_bottleneck_review"})
        summary = run_xunce_topology_observation_contract(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("literature_bottleneck_review_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_network_literature_bottleneck_review")

    def test_non_additive_contract_fails(self) -> None:
        from scripts.run_xunce_topology_observation_contract import run_xunce_topology_observation_contract

        self._write_config(contract_updates={"additive_only": False})
        summary = run_xunce_topology_observation_contract(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("topology_contract_not_additive", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_topology_observation_contract")

    def test_missing_indicator_required_for_new_fields(self) -> None:
        from scripts.run_xunce_topology_observation_contract import run_xunce_topology_observation_contract

        fields = self._candidate_fields()
        fields[0].pop("missing_indicator")
        self._write_config(candidate_fields=fields)
        summary = run_xunce_topology_observation_contract(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("topology_contract_missing_indicators", summary["reason_codes"])
        self.assertFalse(summary["all_new_fields_have_missing_indicators"])

    def test_base_schema_mutation_fails(self) -> None:
        from scripts.run_xunce_topology_observation_contract import run_xunce_topology_observation_contract

        self._write_config(base_schema="policy-observation/v2")
        summary = run_xunce_topology_observation_contract(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("base_observation_schema_changed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_topology_observation_contract")

    def test_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_topology_observation_contract import run_xunce_topology_observation_contract

        self._write_stage2({"runs_new_ppo_update": True})
        summary = run_xunce_topology_observation_contract(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("topology_observation_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])

    def _write_config(
        self,
        *,
        base_schema: str = "policy-observation/v1.1",
        contract_updates: dict | None = None,
        candidate_fields: list[dict] | None = None,
    ) -> None:
        contract = {
            "additive_only": True,
            "preserve_action_mask": True,
            "preserve_candidate_order": True,
            "optional_by_default": True,
            "require_missing_indicators": True,
        }
        if contract_updates:
            contract.update(contract_updates)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "xunce-topology-observation-contract-config/v1",
                    "source_network_literature_bottleneck_review_root": str(self.stage2_root),
                    "features_path": str(self.features_path),
                    "torch_policy_path": str(self.torch_policy_path),
                    "training_path": str(self.training_path),
                    "architectures_path": str(self.architectures_path),
                    "base_observation_schema_version": base_schema,
                    "extension_schema_version": "xunce-topology-observation-extension/v1",
                    "contract": contract,
                    "candidate_topology_fields": candidate_fields or self._candidate_fields(),
                    "candidate_edge_fields": self._edge_fields(),
                    "coverage_memory_fields": self._memory_fields(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_stage2(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "xunce-network-literature-bottleneck-review-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "topology_observation_contract",
            "literature_bottleneck_review_passed": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
        if updates:
            payload.update(updates)
        self.stage2_root.mkdir(parents=True, exist_ok=True)
        (self.stage2_root / "xunce-network-literature-bottleneck-review-summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_policy_sources(self) -> None:
        self.features_path.parent.mkdir(parents=True, exist_ok=True)
        self.features_path.write_text(
            'OBSERVATION_SCHEMA_VERSION = "policy-observation/v1.1"\n'
            "class PolicyObservation:\n"
            "    action_mask = ()\n"
            "    candidate_missing_indicator_names = ()\n"
            "    candidate_missing_indicators = ()\n",
            encoding="utf-8",
        )
        self.torch_policy_path.write_text("class TorchPolicyScorer:\n    def score(self, observation):\n        return observation.action_mask\n", encoding="utf-8")
        self.training_path.write_text(
            "def load_policy_checkpoint(path):\n"
            "    checkpoint = {}\n"
            "    checkpoint.get('candidate_missing_indicator_names', ())\n"
            "    return checkpoint['state_dict']\n",
            encoding="utf-8",
        )
        self.architectures_path.write_text('SUPPORTED_ARCHITECTURES = ("mlp_v1", "mlp_missing_v1", "candidate_attention_v1")\n', encoding="utf-8")

    def _candidate_fields(self) -> list[dict]:
        return [
            self._field("frontier_cluster_id", "id"),
            self._field("roi_group_id", "id"),
            self._field("new_coverage_cell_count", "numeric"),
            self._field("coverage_overlap_count", "numeric"),
            self._field("bfs_distance_from_current", "numeric"),
            self._field("path_bottleneck_score", "numeric"),
            self._field("revisit_path_cell_count", "numeric"),
            self._field("budget_fraction_cost", "numeric"),
            self._field("fallback_risk", "numeric"),
        ]

    def _edge_fields(self) -> list[dict]:
        return [
            self._field("same_frontier_cluster", "bool"),
            self._field("same_roi_group", "bool"),
            self._field("bfs_distance_between_candidates", "numeric"),
            self._field("coverage_overlap_ratio", "numeric"),
            self._field("shared_bottleneck", "bool"),
            self._field("mutual_redundancy_score", "numeric"),
        ]

    def _memory_fields(self) -> list[dict]:
        return [
            self._field("coverage_rate", "numeric"),
            self._field("remaining_budget_fraction", "numeric"),
            self._field("recent_path_cost_trend", "numeric"),
            self._field("recent_new_coverage_trend", "numeric"),
            self._field("revisit_rate", "numeric"),
            self._field("fallback_rate", "numeric"),
            self._field("roi_group_completion_ratio", "numeric"),
        ]

    def _field(self, name: str, field_type: str) -> dict:
        return {
            "name": name,
            "type": field_type,
            "optional": True,
            "default": None,
            "missing_indicator": f"{name}_missing",
        }
