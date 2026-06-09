import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class CanaryDiversitySafeTakeoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_generator_exposes_policy_canary_diversity_families(self) -> None:
        script = self.repo_root / "dev-platform-constraints" / "scripts" / "generate_npz_validation_maps.py"

        completed = subprocess.run(
            [
                "/home/kai/anaconda3/envs/lunar-explorer/bin/python",
                str(script),
                "--scenario-set",
                "policy_canary_diversity",
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
                "high_risk_tradeoff",
                "dense_choke_safe_bypass",
                "channel_contrast",
                "path_complexity_benefit",
            },
        )
        seeds = [item["seed"] for item in payload["scenarios"]]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_policy_canary_diversity_exports_safe_alternative_candidate_mix(self) -> None:
        dev_root = self.repo_root / "dev-platform-constraints"
        generator = dev_root / "scripts" / "generate_npz_validation_maps.py"
        exporter = dev_root / "scripts" / "export_path_planner_sidecars.py"
        root = Path(tempfile.mkdtemp(prefix="policy-canary-diversity-export-"))
        scenario_config = root / "npz_validation_scenarios.json"

        generated = subprocess.run(
            [
                "/home/kai/anaconda3/envs/lunar-explorer/bin/python",
                str(generator),
                "--scenario-set",
                "policy_canary_diversity",
                "--output-dir",
                str(root / "maps"),
                "--scenario-config",
                str(scenario_config),
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)

        exported = subprocess.run(
            [
                "/home/kai/anaconda3/envs/lunar-explorer/bin/python",
                str(exporter),
                "--scenario-config",
                str(scenario_config),
                "--output-dir",
                str(root / "exports"),
                "--top-k",
                "3",
            ],
            cwd=dev_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(exported.returncode, 0, exported.stdout + exported.stderr)

        scenarios = json.loads(scenario_config.read_text(encoding="utf-8"))["scenarios"]
        for scenario in scenarios:
            with self.subTest(scenario=scenario["scenario_id"]):
                contract = json.loads(
                    (root / "exports" / f"{scenario['scenario_id']}.contract.json").read_text(
                        encoding="utf-8"
                    )
                )
                reachable_values = {bool(goal["reachable"]) for goal in contract["top_goals"]}
                self.assertEqual(reachable_values, {False, True})

    def test_canary_diversity_closure_script_references_required_roots(self) -> None:
        script = self.repo_root / "scripts" / "run_canary_diversity_safe_takeover_closure.sh"

        completed = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        text = script.read_text(encoding="utf-8")
        self.assertIn("path_feedback_batch_canary_diversity_clean_src_v1", text)
        self.assertIn("path_feedback_batch_canary_diversity_candidate_v1", text)
        self.assertIn("path_feedback_batch_policy_gated_canary_diversity_v1", text)
        self.assertIn("run_raw_policy_generalization_closure.sh", text)
        self.assertIn("run_policy_gated_canary_rollout.sh", text)


if __name__ == "__main__":
    unittest.main()
