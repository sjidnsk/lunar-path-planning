import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceNetworkLiteratureBottleneckReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-literature-bottleneck-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage1_root = self.repo_root / "outputs" / "stage1"
        self.network_root = self.repo_root / "outputs" / "network"
        self.multi_map_root = self.repo_root / "outputs" / "multi-map"
        self.real_map_root = self.repo_root / "outputs" / "real-map"
        self.output_root = self.repo_root / "outputs" / "stage2"
        self.config_path = self.repo_root / "configs" / "xunce_network_literature_bottleneck_review_v1.json"
        self.architectures_path = self.repo_root / "model-explorer" / "src" / "model_explorer" / "policy" / "architectures.py"
        self.features_path = self.repo_root / "model-explorer" / "src" / "model_explorer" / "policy" / "features.py"
        self._write_config()
        self._write_sources()
        self._write_policy_sources()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_current_evidence_passes_with_no_release_blocking_network_bottleneck(self) -> None:
        from scripts.run_xunce_network_literature_bottleneck_review import (
            run_xunce_network_literature_bottleneck_review,
        )

        summary = run_xunce_network_literature_bottleneck_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["literature_bottleneck_review_passed"])
        self.assertTrue(summary["literature_map_audit_passed"])
        self.assertTrue(summary["project_bottleneck_audit_passed"])
        self.assertTrue(summary["research_scope_audit_passed"])
        self.assertEqual(summary["next_required_change"], "topology_observation_contract")
        self.assertEqual(summary["literature_reference_count"], 8)
        self.assertFalse(summary["coverage_or_generalization_bottleneck"])
        self.assertFalse(summary["fallback_or_policy_regression_bottleneck"])
        self.assertTrue(summary["topology_representation_gap"])
        self.assertTrue(summary["no_current_release_blocking_network_bottleneck"])
        self.assertEqual(summary["primary_bottleneck_decision"], "no_current_release_blocking_network_bottleneck")
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "xunce-network-literature-bottleneck-review-summary.json",
            "xunce-network-literature-bottleneck-review-manifest.json",
            "xunce-network-literature-map-audit.json",
            "xunce-project-bottleneck-audit.json",
            "xunce-network-research-scope-audit.json",
            "xunce-network-literature-bottleneck-rejection-report.json",
            "xunce-network-literature-bottleneck-review-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_insufficient_literature_references_fails(self) -> None:
        from scripts.run_xunce_network_literature_bottleneck_review import (
            run_xunce_network_literature_bottleneck_review,
        )

        self._write_config(literature_references=self._literature_references()[:2])
        summary = run_xunce_network_literature_bottleneck_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("insufficient_literature_references", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_network_literature_bottleneck_review")

    def test_failed_stage1_routes_to_current_head_fix(self) -> None:
        from scripts.run_xunce_network_literature_bottleneck_review import (
            run_xunce_network_literature_bottleneck_review,
        )

        self._write_sources(stage1_updates={"status": "failed", "next_required_change": "fix_current_head_evidence_refresh"})
        summary = run_xunce_network_literature_bottleneck_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("current_head_evidence_refresh_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_current_head_evidence_refresh")

    def test_policy_regression_is_attributed_without_opening_boundaries(self) -> None:
        from scripts.run_xunce_network_literature_bottleneck_review import (
            run_xunce_network_literature_bottleneck_review,
        )

        self._write_sources(
            network_updates={
                "policy_worse_than_baseline_count": 1,
                "controlled_regression_count": 1,
                "policy_guard_fallback_rate": 0.2,
            }
        )
        summary = run_xunce_network_literature_bottleneck_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["fallback_or_policy_regression_bottleneck"])
        self.assertEqual(summary["primary_bottleneck_decision"], "fallback_or_policy_regression_bottleneck")
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

    def test_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_network_literature_bottleneck_review import (
            run_xunce_network_literature_bottleneck_review,
        )

        self._write_sources(stage1_updates={"publishes_checkpoint": True})
        summary = run_xunce_network_literature_bottleneck_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("research_scope_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_network_literature_bottleneck_review")

    def _write_config(self, literature_references: list[dict] | None = None) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "xunce-network-literature-bottleneck-review-config/v1",
                    "source_current_head_evidence_refresh_root": str(self.stage1_root),
                    "source_network_readiness_root": str(self.network_root),
                    "source_multi_map_root": str(self.multi_map_root),
                    "source_real_map_multi_roi_root": str(self.real_map_root),
                    "policy_architectures_path": str(self.architectures_path),
                    "policy_features_path": str(self.features_path),
                    "min_literature_reference_count": 6,
                    "max_policy_guard_fallback_rate": 0.05,
                    "require_current_head_refresh_passed": True,
                    "require_closed_boundaries": True,
                    "literature_references": literature_references or self._literature_references(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_sources(
        self,
        *,
        stage1_updates: dict | None = None,
        network_updates: dict | None = None,
        multi_map_updates: dict | None = None,
        real_map_updates: dict | None = None,
    ) -> None:
        boundary = {
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
        stage1 = {
            "schema_version": "xunce-current-head-evidence-refresh-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "network_literature_bottleneck_review",
            "current_head_evidence_refresh_passed": True,
            **boundary,
        }
        network = {
            "schema_version": "network-architecture-upgrade-readiness-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "global_99_release_governance_preflight",
            "network_upgrade_recommended": False,
            "aggregate_achieved_coverage_rate": 0.9963513964901461,
            "min_scenario_achieved_coverage_rate": 0.9925,
            "policy_better_than_baseline_count": 14,
            "policy_worse_than_baseline_count": 0,
            "controlled_regression_count": 0,
            "policy_guard_fallback_rate": 0.031578947368421054,
            **boundary,
        }
        multi_map = {
            "schema_version": "global-99-multi-map-generalization-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "network_architecture_upgrade_readiness_review",
            "aggregate_achieved_coverage_rate": 0.9963513964901461,
            "min_scenario_achieved_coverage_rate": 0.9925,
            "failed_required_scenario_count": 0,
            "policy_worse_than_baseline_count": 0,
            "controlled_regression_count": 0,
            "policy_guard_fallback_count": 3,
            "policy_guided_decision_count": 95,
            **boundary,
        }
        real_map = {
            "schema_version": "global-99-real-map-multi-roi-generalization-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "default_policy_candidate_authorization_preflight",
            "passed_required_scenario_count": 12,
            "failed_required_scenario_count": 0,
            "slice_count": 12,
            "roi_group_count": 4,
            "split_coverage_complete": True,
            **boundary,
        }
        for payload, updates in (
            (stage1, stage1_updates),
            (network, network_updates),
            (multi_map, multi_map_updates),
            (real_map, real_map_updates),
        ):
            if updates:
                payload.update(updates)
        for root in (self.stage1_root, self.network_root, self.multi_map_root, self.real_map_root):
            root.mkdir(parents=True, exist_ok=True)
        (self.stage1_root / "xunce-current-head-evidence-refresh-summary.json").write_text(
            json.dumps(stage1, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.network_root / "network-architecture-upgrade-readiness-summary.json").write_text(
            json.dumps(network, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.multi_map_root / "global-99-multi-map-generalization-summary.json").write_text(
            json.dumps(multi_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.real_map_root / "global-99-real-map-multi-roi-generalization-summary.json").write_text(
            json.dumps(real_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_policy_sources(self) -> None:
        self.architectures_path.parent.mkdir(parents=True, exist_ok=True)
        self.architectures_path.write_text(
            'SUPPORTED_ARCHITECTURES = ("mlp_v1", "mlp_missing_v1", "candidate_attention_v1")\n',
            encoding="utf-8",
        )
        self.features_path.write_text(
            'OBSERVATION_SCHEMA_VERSION = "policy-observation/v1.1"\n',
            encoding="utf-8",
        )

    def _literature_references(self) -> list[dict]:
        return [
            {"id": "deep_sets", "title": "Deep Sets", "url": "https://arxiv.org/abs/1703.06114", "family": "set", "project_relevance": "candidate set invariance"},
            {"id": "set_transformer", "title": "Set Transformer", "url": "https://arxiv.org/abs/1810.00825", "family": "set_attention", "project_relevance": "candidate set attention"},
            {"id": "attention_routing", "title": "Attention, Learn to Solve Routing Problems!", "url": "https://arxiv.org/abs/1803.08475", "family": "routing_attention", "project_relevance": "waypoint ranking"},
            {"id": "gat", "title": "Graph Attention Networks", "url": "https://arxiv.org/abs/1710.10903", "family": "graph_attention", "project_relevance": "candidate topology edges"},
            {"id": "decision_transformer", "title": "Decision Transformer", "url": "https://arxiv.org/abs/2106.01345", "family": "sequence_memory", "project_relevance": "trajectory conditioning"},
            {"id": "mamba", "title": "Mamba", "url": "https://arxiv.org/abs/2312.00752", "family": "sequence_memory", "project_relevance": "efficient memory token research"},
            {"id": "active_neural_slam", "title": "Active Neural SLAM", "url": "https://arxiv.org/abs/2004.05155", "family": "exploration", "project_relevance": "modular learning around planning"},
            {"id": "neural_astar", "title": "Neural A*", "url": "https://arxiv.org/abs/2009.07476", "family": "planning_diagnostic", "project_relevance": "learned heuristic diagnostic only"},
        ]
