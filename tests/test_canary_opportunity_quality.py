import json
import subprocess
import unittest
from pathlib import Path


class CanaryOpportunityQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_generator_exposes_policy_canary_opportunity_quality_families(self) -> None:
        script = self.repo_root / "dev-platform-constraints" / "scripts" / "generate_npz_validation_maps.py"

        completed = subprocess.run(
            [
                "/home/kai/anaconda3/envs/lunar-explorer/bin/python",
                str(script),
                "--scenario-set",
                "policy_canary_opportunity_quality",
                "--dry-run",
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        groups = {item["scenario_group"] for item in payload["scenarios"]}
        self.assertGreaterEqual(
            groups,
            {
                "mixed_stress_detour",
                "near_blocked_safe_alt",
                "channel_contrast",
                "high_risk_tradeoff",
                "dense_choke_safe_bypass",
                "path_complexity_benefit",
            },
        )
        seeds = [item["seed"] for item in payload["scenarios"]]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_opportunity_quality_closure_script_references_required_roots(self) -> None:
        script = self.repo_root / "scripts" / "run_canary_opportunity_quality_closure.sh"

        completed = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        text = script.read_text(encoding="utf-8")
        self.assertIn("path_feedback_batch_canary_opportunity_quality_clean_src_v1", text)
        self.assertIn("path_feedback_batch_canary_opportunity_quality_candidate_v1", text)
        self.assertIn("path_feedback_batch_policy_gated_canary_opportunity_quality_v1", text)
        self.assertIn("run_raw_policy_generalization_closure.sh", text)
        self.assertIn("run_policy_gated_canary_rollout.sh", text)


if __name__ == "__main__":
    unittest.main()
