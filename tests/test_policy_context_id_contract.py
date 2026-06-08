import sys
import unittest
from pathlib import Path


class PolicyContextIdContractTests(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        model_src = repo_root / "model-explorer" / "src"
        if str(model_src) not in sys.path:
            sys.path.insert(0, str(model_src))

    def _fields(self) -> dict:
        return {
            "scenario_id": "holdout-near-blocked-a",
            "scenario_group": "holdout_near_blocked",
            "scenario_seed": 8601,
            "scenario_variant_id": "holdout-near-blocked-a-seed-8601",
            "diagnostic_profile": "execution",
            "planning_backend": "path_planner_route",
            "top_k": 3,
            "sample_type": "path_feedback_candidate",
            "candidate_role": "policy_target",
            "source_action_index": 1,
            "policy_target_cell": [18, 6],
            "execution_goal_cell": [18, 6],
            "target_binding_mode": "policy_target",
        }

    def test_context_id_is_stable_for_canonical_semantic_fields(self) -> None:
        from model_explorer.policy.context_id import (
            POLICY_CONTEXT_ID_SCHEMA_VERSION,
            build_policy_context_id,
            policy_context_id_metadata,
        )

        fields = self._fields()
        shuffled = dict(reversed(list(fields.items())))

        context_id = build_policy_context_id(fields)
        metadata = policy_context_id_metadata(shuffled)

        self.assertRegex(context_id or "", r"^[0-9a-f]{64}$")
        self.assertEqual(context_id, metadata["context_id"])
        self.assertEqual(metadata["context_id_schema_version"], POLICY_CONTEXT_ID_SCHEMA_VERSION)
        self.assertEqual(metadata["context_id_source"], "stable_semantic_fields")
        self.assertFalse(metadata["legacy_identity_fallback_used"])

    def test_context_id_changes_when_target_action_backend_or_seed_changes(self) -> None:
        from model_explorer.policy.context_id import build_policy_context_id

        baseline = build_policy_context_id(self._fields())
        mutations = [
            {"source_action_index": 2},
            {"policy_target_cell": [19, 6]},
            {"execution_goal_cell": [17, 6]},
            {"target_binding_mode": "same_action_execution_substitute"},
            {"planning_backend": "channel_aware_astar"},
            {"scenario_seed": 8602},
        ]

        for mutation in mutations:
            fields = {**self._fields(), **mutation}
            self.assertNotEqual(baseline, build_policy_context_id(fields), mutation)

    def test_missing_required_fields_do_not_create_fake_context_id(self) -> None:
        from model_explorer.policy.context_id import (
            build_policy_context_id,
            policy_context_id_metadata,
        )

        fields = self._fields()
        del fields["scenario_seed"]

        self.assertIsNone(build_policy_context_id(fields))
        metadata = policy_context_id_metadata(fields)
        self.assertIsNone(metadata["context_id"])
        self.assertEqual(metadata["missing_context_id_fields"], ["scenario_seed"])
        self.assertTrue(metadata["context_id_missing"])


if __name__ == "__main__":
    unittest.main()
