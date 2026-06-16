import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceTopologyFeatureExtractionAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-feature-audit-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage3_root = self.repo_root / "outputs" / "stage3"
        self.output_root = self.repo_root / "outputs" / "stage4"
        self.config_path = self.repo_root / "configs" / "xunce_topology_feature_extraction_audit_v1.json"
        self._write_stage3()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_topology_feature_extraction_audit import run_xunce_topology_feature_extraction_audit

        summary = run_xunce_topology_feature_extraction_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "topology_aware_coverage_graph_proto")
        self.assertTrue(summary["feature_extraction_audit_passed"])
        self.assertTrue(summary["field_audit_passed"])
        self.assertTrue(summary["determinism_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertGreater(summary["candidate_count"], 0)
        self.assertGreater(summary["edge_count"], 0)
        self.assertEqual(summary["candidate_missing_field_count"], 0)
        self.assertEqual(summary["edge_missing_field_count"], 0)
        self.assertEqual(summary["memory_missing_field_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "xunce-topology-feature-extraction-audit-summary.json",
            "xunce-topology-feature-extraction-audit-manifest.json",
            "xunce-topology-feature-extraction-candidates.jsonl",
            "xunce-topology-feature-extraction-edges.jsonl",
            "xunce-topology-feature-extraction-memory.json",
            "xunce-topology-feature-extraction-field-audit.json",
            "xunce-topology-feature-extraction-determinism-audit.json",
            "xunce-topology-feature-extraction-boundary-audit.json",
            "xunce-topology-feature-extraction-rejection-report.json",
            "xunce-topology-feature-extraction-audit-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_candidate_rows_include_contract_fields_and_missing_indicators(self) -> None:
        from scripts.run_xunce_topology_feature_extraction_audit import run_xunce_topology_feature_extraction_audit

        summary = run_xunce_topology_feature_extraction_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )
        rows = self._read_jsonl(self.output_root / "xunce-topology-feature-extraction-candidates.jsonl")
        first = rows[0]

        self.assertEqual(summary["candidate_topology_field_count"], 9)
        for field in self._candidate_field_names():
            self.assertIn(field, first["features"])
            self.assertIn(f"{field}_missing", first["missing_indicators"])
            self.assertFalse(first["missing_indicators"][f"{field}_missing"])
        self.assertGreaterEqual(first["features"]["new_coverage_cell_count"], 0)
        self.assertGreaterEqual(first["features"]["bfs_distance_from_current"], 0)
        self.assertGreaterEqual(first["features"]["budget_fraction_cost"], 0.0)

    def test_edge_rows_include_pairwise_fields_and_stable_ordering(self) -> None:
        from scripts.run_xunce_topology_feature_extraction_audit import run_xunce_topology_feature_extraction_audit

        run_xunce_topology_feature_extraction_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )
        edges = self._read_jsonl(self.output_root / "xunce-topology-feature-extraction-edges.jsonl")
        pair_keys = [edge["candidate_pair_key"] for edge in edges]

        self.assertEqual(pair_keys, sorted(pair_keys))
        for field in self._edge_field_names():
            self.assertIn(field, edges[0]["features"])
            self.assertIn(f"{field}_missing", edges[0]["missing_indicators"])
            self.assertFalse(edges[0]["missing_indicators"][f"{field}_missing"])

    def test_memory_token_fields_are_finite_and_bounded(self) -> None:
        from scripts.run_xunce_topology_feature_extraction_audit import run_xunce_topology_feature_extraction_audit

        run_xunce_topology_feature_extraction_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )
        memory = json.loads((self.output_root / "xunce-topology-feature-extraction-memory.json").read_text(encoding="utf-8"))

        for field in self._memory_field_names():
            self.assertIn(field, memory["features"])
            self.assertIn(f"{field}_missing", memory["missing_indicators"])
            self.assertFalse(memory["missing_indicators"][f"{field}_missing"])
            self.assertTrue(math.isfinite(float(memory["features"][field])), field)
        self.assertGreaterEqual(memory["features"]["coverage_rate"], 0.0)
        self.assertLessEqual(memory["features"]["coverage_rate"], 1.0)
        self.assertGreaterEqual(memory["features"]["remaining_budget_fraction"], 0.0)
        self.assertLessEqual(memory["features"]["remaining_budget_fraction"], 1.0)
        self.assertGreaterEqual(memory["features"]["revisit_rate"], 0.0)
        self.assertLessEqual(memory["features"]["revisit_rate"], 1.0)

    def test_missing_stage3_routes_to_contract_fix(self) -> None:
        from scripts.run_xunce_topology_feature_extraction_audit import run_xunce_topology_feature_extraction_audit

        shutil.rmtree(self.stage3_root)
        summary = run_xunce_topology_feature_extraction_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_topology_observation_contract_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_topology_observation_contract")

    def test_no_frontier_fails_extraction_audit(self) -> None:
        from scripts.run_xunce_topology_feature_extraction_audit import run_xunce_topology_feature_extraction_audit

        scenario = self._scenario_fixture()
        scenario["current_covered_cells"] = [[x, y] for y in range(6) for x in range(8)]
        self._write_config(scenario=scenario)
        summary = run_xunce_topology_feature_extraction_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("no_frontier_candidates", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_topology_feature_extraction_audit")

    def test_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_topology_feature_extraction_audit import run_xunce_topology_feature_extraction_audit

        self._write_stage3(summary_updates={"publishes_checkpoint": True})
        summary = run_xunce_topology_feature_extraction_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("topology_feature_extraction_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])

    def _write_config(self, *, scenario: dict | None = None) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "xunce-topology-feature-extraction-audit-config/v1",
                    "source_topology_observation_contract_root": str(self.stage3_root),
                    "target_coverage_rate": 0.99,
                    "path_budget_m": 1000.0,
                    "coverage_radius_cells": 1,
                    "revisit_penalty_weight": 1.0,
                    "new_coverage_weight": 4.0,
                    "scenario": scenario or self._scenario_fixture(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_stage3(self, *, summary_updates: dict | None = None) -> None:
        self.stage3_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "xunce-topology-observation-contract-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "topology_feature_extraction_audit",
            "topology_observation_contract_passed": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
        if summary_updates:
            summary.update(summary_updates)
        (self.stage3_root / "xunce-topology-observation-contract-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        contract = {
            "schema_version": "xunce-topology-observation-contract/v1",
            "base_observation_schema_version": "policy-observation/v1.1",
            "extension_schema_version": "xunce-topology-observation-extension/v1",
            "candidate_topology_fields": [self._field(name, "numeric") for name in self._candidate_field_names()],
            "candidate_edge_fields": [self._field(name, "numeric") for name in self._edge_field_names()],
            "coverage_memory_fields": [self._field(name, "numeric") for name in self._memory_field_names()],
        }
        for name in ("frontier_cluster_id", "roi_group_id"):
            for field in contract["candidate_topology_fields"]:
                if field["name"] == name:
                    field["type"] = "id"
        for name in ("same_frontier_cluster", "same_roi_group", "shared_bottleneck"):
            for field in contract["candidate_edge_fields"]:
                if field["name"] == name:
                    field["type"] = "bool"
        (self.stage3_root / "xunce-topology-observation-contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _scenario_fixture(self) -> dict:
        return {
            "scenario_id": "xunce_feature_fixture",
            "grid": {"width": 8, "height": 6, "resolution_m": 10.0, "origin": [0.0, 0.0]},
            "start_cell": [0, 0],
            "current_cell": [2, 1],
            "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 8, "y1": 6},
            "blocked_rectangles": [{"x0": 3, "y0": 2, "x1": 4, "y1": 5}],
            "unsafe_rectangles": [{"x0": 7, "y0": 5, "x1": 8, "y1": 6}],
            "current_covered_cells": [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [1, 2]],
            "previous_path_cells": [[0, 0], [1, 0], [2, 0], [2, 1]],
            "recent_history": [
                {"path_cost_m": 20.0, "new_coverage_cell_count": 4, "fallback_used": False},
                {"path_cost_m": 30.0, "new_coverage_cell_count": 3, "fallback_used": False},
            ],
        }

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _candidate_field_names(self) -> list[str]:
        return [
            "frontier_cluster_id",
            "roi_group_id",
            "new_coverage_cell_count",
            "coverage_overlap_count",
            "bfs_distance_from_current",
            "path_bottleneck_score",
            "revisit_path_cell_count",
            "budget_fraction_cost",
            "fallback_risk",
        ]

    def _edge_field_names(self) -> list[str]:
        return [
            "same_frontier_cluster",
            "same_roi_group",
            "bfs_distance_between_candidates",
            "coverage_overlap_ratio",
            "shared_bottleneck",
            "mutual_redundancy_score",
        ]

    def _memory_field_names(self) -> list[str]:
        return [
            "coverage_rate",
            "remaining_budget_fraction",
            "recent_path_cost_trend",
            "recent_new_coverage_trend",
            "revisit_rate",
            "fallback_rate",
            "roi_group_completion_ratio",
        ]

    def _field(self, name: str, field_type: str) -> dict:
        return {
            "name": name,
            "type": field_type,
            "optional": True,
            "default": None,
            "missing_indicator": f"{name}_missing",
        }
